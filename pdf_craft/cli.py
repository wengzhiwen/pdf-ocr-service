"""
CLI entrypoints for pdf-craft.

RunJPLib OCR bundle generator:
    pdf-craft runjplib-ocr \
      --input /path/to/pdf_or_dir \
      --output-dir /path/to/output \
      --ocr-size gundam \
      --workers 2

This command outputs a timestamped zip containing:
    manifest.json
    items/<item_id>/original.pdf
    items/<item_id>/original.md

The bundle is designed for RunJPLib's OCR import workflow so that
processing continues from the translation step.

Auto-fix behavior:
    If a PDF cannot be opened, the CLI will try to repair it via Ghostscript
    and retry OCR. Use --no-auto-fix-pdf to disable.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import subprocess
import shutil
import sys
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .transform import Transform
from .pdf import pdf_pages_count
from .error import PDFError


_OCR_SIZES = ("tiny", "small", "base", "large", "gundam")
_WORKER_TRANSFORMER = None
_WORKER_OPTIONS = None


def _slugify(value: str) -> str:
    safe = []
    for ch in value.strip().lower():
        if ch.isalnum():
            safe.append(ch)
        elif ch in (" ", "-", "_", "."):
            safe.append("_")
    slug = "".join(safe).strip("_")
    return slug or "item"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _collect_pdfs(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input path not found: {input_path}")

    if recursive:
        candidates = input_path.rglob("*")
    else:
        candidates = input_path.iterdir()

    pdf_files = [path for path in candidates if path.is_file() and path.suffix.lower() == ".pdf"]
    return sorted(pdf_files)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_manifest(manifest_path: Path, items: list[dict], errors: list[str]) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "pdf-craft",
        "version": "local-dev",
        "items": items,
    }
    if errors:
        manifest["errors"] = errors
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")


def _zip_dir(root: Path, output_zip: Path) -> None:
    import zipfile

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zip_handle:
        for file_path in root.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(root)
                zip_handle.write(file_path, arcname)


def _parse_device_ids(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def _write_error_log(item_dir: Path, message: str, exc: Exception | None = None) -> None:
    try:
        item_dir.mkdir(parents=True, exist_ok=True)
        log_path = item_dir / "error.log"
        lines = [
            f"timestamp: {datetime.now(timezone.utc).isoformat()}",
            f"message: {message}",
        ]
        if exc:
            lines.append("traceback:")
            lines.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass


def _print_results_summary(results: list[dict]) -> None:
    if not results:
        return

    successes = [item for item in results if item.get("status") == "success"]
    failures = [item for item in results if item.get("status") == "failed"]

    print("[runjplib-ocr] Batch summary:", flush=True)
    for item in successes:
        print(f"  [OK] {item.get('filename')}", flush=True)
    for item in failures:
        print(f"  [FAIL] {item.get('filename')}", flush=True)

def _is_pdf_open_error(exc: Exception) -> bool:
    checked = set()
    current: Exception | None = exc
    while current and current not in checked:
        checked.add(current)
        if isinstance(current, PDFError):
            message = str(current)
            if "Failed to open PDF document" in message or "Failed to parse PDF document" in message:
                return True
        else:
            message = str(current)
            if "Failed to open PDF document" in message or "Failed to parse PDF document" in message:
                return True
        current = current.__cause__ or current.__context__
    return False


def _fix_pdf_with_gs(source_path: Path, fixed_path: Path) -> bool:
    gs_path = shutil.which("gs")
    if not gs_path:
        _print_gs_install_hint()
        return False

    fixed_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        gs_path,
        "-sDEVICE=pdfwrite",
        "-dNOPAUSE",
        "-dBATCH",
        "-dSAFER",
        f"-sOutputFile={fixed_path}",
        str(source_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print("[runjplib-ocr] Ghostscript fix failed.", file=sys.stderr, flush=True)
        if result.stdout:
            print(result.stdout, file=sys.stderr, flush=True)
        return False
    return True


def _print_gs_install_hint() -> None:
    print("[runjplib-ocr] Ghostscript (gs) not found; auto-fix disabled.", file=sys.stderr, flush=True)
    print("[runjplib-ocr] Install Ghostscript:", file=sys.stderr, flush=True)
    print("  Ubuntu/Debian: sudo apt-get install ghostscript", file=sys.stderr, flush=True)
    print("  macOS: brew install ghostscript", file=sys.stderr, flush=True)
    print("  Conda: conda install -c conda-forge ghostscript", file=sys.stderr, flush=True)


def _init_worker(options: dict, device_ids: list[str], device_counter) -> None:
    global _WORKER_TRANSFORMER
    global _WORKER_OPTIONS
    _WORKER_OPTIONS = options

    if device_ids:
        with device_counter.get_lock():
            index = device_counter.value
            device_counter.value += 1
        device_id = device_ids[index % len(device_ids)]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id

    _WORKER_TRANSFORMER = Transform(
        models_cache_path=options.get("models_cache_path"),
        local_only=options.get("local_only", False),
    )


def _process_pdf_internal(
    pdf_path: Path,
    item_dir: Path,
    university_name: str,
    original_filename: str,
    transformer: Transform,
    options: dict,
) -> dict:
    _ensure_dir(item_dir)

    original_pdf_path = item_dir / "original.pdf"
    original_md_path = item_dir / "original.md"
    assets_dir = item_dir / "assets"
    analysis_dir = item_dir / "_analysis"

    shutil.copy2(pdf_path, original_pdf_path)

    pdf_for_ocr = original_pdf_path
    try:
        transformer.transform_markdown(
            pdf_path=pdf_for_ocr,
            markdown_path=original_md_path,
            markdown_assets_path=assets_dir,
            analysing_path=analysis_dir,
            ocr_size=options["ocr_size"],
            includes_footnotes=options["includes_footnotes"],
            ignore_pdf_errors=options["ignore_pdf_errors"],
            generate_plot=options["generate_plot"],
            toc_assumed=options["toc_assumed"],
        )
    except Exception as exc:
        if options.get("auto_fix_pdf") and _is_pdf_open_error(exc):
            fixed_pdf_path = analysis_dir / "original.fixed.pdf"
            print(f"[runjplib-ocr] Auto-fix PDF: {original_filename}", flush=True)
            if _fix_pdf_with_gs(original_pdf_path, fixed_pdf_path):
                pdf_for_ocr = fixed_pdf_path
                transformer.transform_markdown(
                    pdf_path=pdf_for_ocr,
                    markdown_path=original_md_path,
                    markdown_assets_path=assets_dir,
                    analysing_path=analysis_dir,
                    ocr_size=options["ocr_size"],
                    includes_footnotes=options["includes_footnotes"],
                    ignore_pdf_errors=options["ignore_pdf_errors"],
                    generate_plot=options["generate_plot"],
                    toc_assumed=options["toc_assumed"],
                )
            else:
                raise exc
        else:
            raise

    if not options["include_assets"] and assets_dir.exists():
        shutil.rmtree(assets_dir, ignore_errors=True)
    if not options["keep_analysis"] and analysis_dir.exists():
        shutil.rmtree(analysis_dir, ignore_errors=True)

    try:
        page_count = pdf_pages_count(pdf_for_ocr)
    except Exception:
        page_count = None

    item = {
        "item_id": item_dir.name,
        "university_name": university_name,
        "filename": original_filename,
        "page_count": page_count,
        "paths": {
            "original_pdf": str(Path("items") / item_dir.name / "original.pdf"),
            "original_md": str(Path("items") / item_dir.name / "original.md"),
        },
        "checksums": {
            "original_pdf": _sha256(original_pdf_path),
            "original_md": _sha256(original_md_path),
        },
    }
    return item


def _process_pdf_task(task: dict) -> dict:
    global _WORKER_TRANSFORMER
    global _WORKER_OPTIONS
    transformer = _WORKER_TRANSFORMER
    options = _WORKER_OPTIONS

    if transformer is None or options is None:
        options = task["options"]
        transformer = Transform(
            models_cache_path=options.get("models_cache_path"),
            local_only=options.get("local_only", False),
        )

    pdf_path = Path(task["pdf_path"])
    item_dir = Path(task["item_dir"])
    university_name = task["university_name"]
    original_filename = task["original_filename"]

    proc_name = multiprocessing.current_process().name
    print(f"[runjplib-ocr:{proc_name}] OCR start: {original_filename}", flush=True)

    try:
        item = _process_pdf_internal(
            pdf_path=pdf_path,
            item_dir=item_dir,
            university_name=university_name,
            original_filename=original_filename,
            transformer=transformer,
            options=options,
        )
    except Exception as exc:
        error_msg = f"Failed to OCR {original_filename}: {exc}"
        _write_error_log(item_dir, error_msg, exc)
        print(f"[runjplib-ocr:{proc_name}] {error_msg}", file=sys.stderr, flush=True)
    return {"status": "error", "error": error_msg, "filename": original_filename}

    print(f"[runjplib-ocr:{proc_name}] OCR done: {original_filename}", flush=True)
    return {"status": "ok", "item": item}


def _run_runjplib_ocr(args: argparse.Namespace) -> int:
    try:
        import cryptography  # noqa: F401
    except Exception:
        print("[runjplib-ocr] Missing dependency: cryptography>=3.1", file=sys.stderr, flush=True)
        print("[runjplib-ocr] Install with:", file=sys.stderr, flush=True)
        print("  conda run -n pdf-craft python -m pip install \"cryptography>=3.1\"", file=sys.stderr, flush=True)
        return 1

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    _ensure_dir(output_dir)

    pdf_files = _collect_pdfs(input_path, args.recursive)
    if not pdf_files:
        raise ValueError(f"No PDF files found under {input_path}")

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    if args.auto_fix_pdf and not shutil.which("gs"):
        _print_gs_install_hint()

    if args.university_name and len(pdf_files) != 1:
        raise ValueError("--university-name can only be used with a single PDF input")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    unique_suffix = uuid.uuid4().hex[:6]
    staging_root = output_dir / f"runjplib-ocr-{timestamp}-{unique_suffix}"
    items_dir = staging_root / "items"
    _ensure_dir(items_dir)

    items: list[dict] = []
    errors: list[str] = []
    results: list[dict] = []

    run_options = {
        "models_cache_path": args.models_cache_path,
        "local_only": args.local_only,
        "ocr_size": args.ocr_size,
        "includes_footnotes": args.includes_footnotes,
        "ignore_pdf_errors": args.ignore_pdf_errors,
        "generate_plot": args.generate_plot,
        "toc_assumed": args.toc_assumed,
        "include_assets": args.include_assets,
        "keep_analysis": args.keep_analysis,
        "auto_fix_pdf": args.auto_fix_pdf,
    }

    tasks: list[dict] = []
    for pdf_path in pdf_files:
        university_name = args.university_name or pdf_path.stem
        item_slug = _slugify(university_name or pdf_path.stem)
        item_id = f"{item_slug}-{uuid.uuid4().hex[:8]}"
        item_dir = items_dir / item_id
        tasks.append({
            "pdf_path": str(pdf_path),
            "item_dir": str(item_dir),
            "university_name": university_name,
            "original_filename": pdf_path.name,
        })

    if args.workers <= 1:
        transformer = Transform(
            models_cache_path=args.models_cache_path,
            local_only=args.local_only,
        )

        for task in tasks:
            pdf_path = Path(task["pdf_path"])
            item_dir = Path(task["item_dir"])
            print(f"[runjplib-ocr] OCR start: {task['original_filename']}", flush=True)
            try:
                item = _process_pdf_internal(
                    pdf_path=pdf_path,
                    item_dir=item_dir,
                    university_name=task["university_name"],
                    original_filename=task["original_filename"],
                    transformer=transformer,
                    options=run_options,
                )
            except Exception as exc:
                error_msg = f"Failed to OCR {task['original_filename']}: {exc}"
                errors.append(error_msg)
                _write_error_log(item_dir, error_msg, exc)
                print(f"[runjplib-ocr] {error_msg}", file=sys.stderr, flush=True)
                results.append({
                    "filename": task["original_filename"],
                    "status": "failed",
                    "error": error_msg,
                })
                if not args.continue_on_error:
                    raise
                continue

            items.append(item)
            results.append({
                "filename": task["original_filename"],
                "status": "success",
            })
            print(f"[runjplib-ocr] OCR done: {task['original_filename']}", flush=True)
    else:
        device_ids = _parse_device_ids(args.device_ids)
        ctx = multiprocessing.get_context("spawn")
        device_counter = ctx.Value("i", 0)
        for task in tasks:
            task["options"] = run_options

        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(run_options, device_ids, device_counter),
        ) as executor:
            futures = [executor.submit(_process_pdf_task, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "ok":
                    items.append(result["item"])
                    results.append({
                        "filename": result["item"]["filename"],
                        "status": "success",
                    })
                else:
                    errors.append(result["error"])
                    results.append({
                        "filename": result.get("filename", "unknown"),
                        "status": "failed",
                        "error": result["error"],
                    })
                    if not args.continue_on_error:
                        raise RuntimeError(result["error"])

    if not items:
        raise RuntimeError("No OCR results were generated.")

    manifest_path = staging_root / "manifest.json"
    _write_manifest(manifest_path, items, errors)

    output_zip = output_dir / f"runjplib-ocr-{timestamp}.zip"
    if output_zip.exists():
        output_zip = output_dir / f"runjplib-ocr-{timestamp}-{unique_suffix}.zip"
    _zip_dir(staging_root, output_zip)

    if not args.keep_staging:
        shutil.rmtree(staging_root, ignore_errors=True)

    print(f"[runjplib-ocr] Output zip: {output_zip}", flush=True)
    _print_results_summary(results)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pdf-craft", description="PDF Craft CLI utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    runjplib_parser = subparsers.add_parser("runjplib-ocr", help="Generate RunJPLib OCR zip bundle.")
    runjplib_parser.add_argument("--input", required=True, help="PDF file or directory containing PDFs.")
    runjplib_parser.add_argument("--output-dir", required=True, help="Directory to store output zip bundle.")
    runjplib_parser.add_argument("--university-name", help="University name for single PDF input.")
    runjplib_parser.add_argument("--recursive", action="store_true", help="Scan directories recursively.")
    runjplib_parser.add_argument("--ocr-size", default="gundam", choices=_OCR_SIZES, help="OCR model size.")
    runjplib_parser.add_argument("--models-cache-path", default=None, help="OCR model cache path.")
    runjplib_parser.add_argument("--local-only", action="store_true", help="Disable model downloads.")
    runjplib_parser.add_argument("--includes-footnotes", action="store_true", help="Keep footnotes.")
    runjplib_parser.add_argument("--ignore-pdf-errors", action="store_true", help="Continue on PDF render errors.")
    runjplib_parser.add_argument("--generate-plot", action="store_true", help="Generate debug plots.")
    runjplib_parser.add_argument("--toc-assumed", action="store_true", help="Assume PDF contains TOC.")
    runjplib_parser.add_argument("--include-assets", action="store_true", help="Keep extracted assets.")
    runjplib_parser.add_argument("--keep-analysis", action="store_true", help="Keep analysis artifacts.")
    runjplib_parser.add_argument("--keep-staging", action="store_true", help="Keep staging folder after zip.")
    runjplib_parser.add_argument("--continue-on-error", action="store_true", help="Skip failed PDFs.")
    runjplib_parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes.")
    runjplib_parser.add_argument("--device-ids", help="Comma-separated GPU device IDs to assign per worker.")
    runjplib_parser.add_argument("--no-auto-fix-pdf", dest="auto_fix_pdf", action="store_false", help="Disable auto-fix when PDF cannot be opened.")
    runjplib_parser.set_defaults(auto_fix_pdf=True)
    runjplib_parser.set_defaults(func=_run_runjplib_ocr)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"[runjplib-ocr] Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
