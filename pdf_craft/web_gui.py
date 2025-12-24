import json
import multiprocessing
import queue
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

try:
    from .error import PDFError
    from .pdf import pdf_pages_count
    from .transform import Transform
except ImportError:  # Allows running as a script: python pdf_craft/web_gui.py
    from pdf_craft.error import PDFError
    from pdf_craft.pdf import pdf_pages_count
    from pdf_craft.transform import Transform


_OCR_SIZES = ("tiny", "small", "base", "large", "gundam")
# Use an absolute task root so downloads are stable regardless of CWD.
TASK_ROOT = Path(__file__).resolve().parent.parent / "task"
MAX_CONCURRENT = 2


@dataclass
class Task:
    task_id: str
    created_at: str
    status: str
    original_name: str
    input_path: str
    task_dir: str
    options: dict[str, Any]
    message: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    output_zip: str | None = None
    pid: int | None = None

    def to_dict(self, queue_position: int | None = None) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "original_name": self.original_name,
            "message": self.message,
            "output_ready": bool(self.output_zip and Path(self.output_zip).exists()),
            "queue_position": queue_position,
        }


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _zip_dir(root: Path, output_zip: Path) -> None:
    import zipfile

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zip_handle:
        for file_path in root.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(root)
                zip_handle.write(file_path, arcname)


def _write_manifest(manifest_path: Path, items: list[dict[str, Any]], errors: list[str]) -> None:
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "pdf-craft-web",
        "version": "local-dev",
        "items": items,
    }
    if errors:
        manifest["errors"] = errors
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_task_meta(task_dir: Path, task: "Task") -> None:
    meta = {
        "task_id": task.task_id,
        "created_at": task.created_at,
        "original_name": task.original_name,
        "input_path": task.input_path,
        "options": task.options,
    }
    (task_dir / "task.json").write_text(
        json.dumps(meta, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _load_task_meta(task_dir: Path) -> dict[str, Any] | None:
    meta_path = task_dir / "task.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_task_result(task_dir: Path) -> dict[str, Any] | None:
    result_path = task_dir / "result.json"
    if not result_path.exists():
        return None
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _default_options() -> dict[str, Any]:
    base_dir = Path(__file__).resolve().parent.parent
    return {
        "university_name": None,
        "ocr_size": "gundam",
        "models_cache_path": str(base_dir / "models-cache"),
        "local_only": True,
        "includes_footnotes": False,
        "ignore_pdf_errors": False,
        "generate_plot": False,
        "toc_assumed": False,
        "include_assets": True,
        "keep_analysis": True,
        "keep_staging": True,
        "device_ids": None,
        "auto_fix_pdf": True,
    }


def _write_error_log(item_dir: Path, message: str, exc: Exception | None = None) -> None:
    try:
        item_dir.mkdir(parents=True, exist_ok=True)
        log_path = item_dir / "error.log"
        lines = [
            f"timestamp: {datetime.utcnow().isoformat()}Z",
            f"message: {message}",
        ]
        if exc:
            import traceback

            lines.append("traceback:")
            lines.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass


def _is_pdf_open_error(exc: Exception) -> bool:
    checked = set()
    current: Exception | None = exc
    while current and current not in checked:
        checked.add(current)
        message = str(current)
        if isinstance(current, PDFError):
            if "Failed to open PDF document" in message or "Failed to parse PDF document" in message:
                return True
        else:
            if "Failed to open PDF document" in message or "Failed to parse PDF document" in message:
                return True
        current = current.__cause__ or current.__context__
    return False


def _fix_pdf_with_gs(source_path: Path, fixed_path: Path) -> bool:
    import shutil
    import subprocess

    gs_path = shutil.which("gs")
    if not gs_path:
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
    return result.returncode == 0


def _process_pdf_internal(
    pdf_path: Path,
    item_dir: Path,
    university_name: str,
    original_filename: str,
    transformer: Transform,
    options: dict[str, Any],
    aborted: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    _ensure_dir(item_dir)

    original_pdf_path = item_dir / "original.pdf"
    original_md_path = item_dir / "original.md"
    assets_dir = item_dir / "assets"
    analysis_dir = item_dir / "_analysis"

    import shutil

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
            aborted=aborted,
        )
    except Exception as exc:
        if options.get("auto_fix_pdf") and _is_pdf_open_error(exc):
            fixed_pdf_path = analysis_dir / "original.fixed.pdf"
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
                    aborted=aborted,
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


def _parse_device_ids(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def _process_task_payload(payload: dict[str, Any], transformer: Transform) -> dict[str, Any]:
    task_dir = Path(payload["task_dir"])
    input_pdf = Path(payload["input_pdf"])
    options = payload["options"]
    original_name = payload["original_name"]
    abort_path = Path(payload.get("abort_path", ""))

    result_path = task_dir / "result.json"

    try:
        if abort_path and abort_path.exists():
            result = {"status": "stopped", "error": "Stopped by user"}
            result_path.write_text(json.dumps(result), encoding="utf-8")
            return result

        staging_root = task_dir / "staging"
        items_dir = staging_root / "items"
        output_dir = task_dir / "output"
        _ensure_dir(items_dir)
        _ensure_dir(output_dir)

        university_name = options.get("university_name") or input_pdf.stem
        item_slug = _slugify(university_name or input_pdf.stem)
        item_id = f"{item_slug}-{uuid.uuid4().hex[:8]}"
        item_dir = items_dir / item_id

        items: list[dict[str, Any]] = []
        errors: list[str] = []

        try:
            item = _process_pdf_internal(
                pdf_path=input_pdf,
                item_dir=item_dir,
                university_name=university_name,
                original_filename=original_name,
                transformer=transformer,
                options=options,
                aborted=abort_path.exists if abort_path else (lambda: False),
            )
            items.append(item)
        except InterruptedError:
            result = {"status": "stopped", "error": "Stopped by user"}
            result_path.write_text(json.dumps(result), encoding="utf-8")
            return result
        except Exception as exc:
            error_msg = f"Failed to OCR {original_name}: {exc}"
            errors.append(error_msg)
            _write_error_log(item_dir, error_msg, exc)
            raise

        manifest_path = staging_root / "manifest.json"
        _write_manifest(manifest_path, items, errors)

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        output_zip = output_dir / f"runjplib-ocr-{timestamp}.zip"
        _zip_dir(staging_root, output_zip)

        if not options.get("keep_staging", True):
            import shutil

            shutil.rmtree(staging_root, ignore_errors=True)

        result = {"status": "completed", "output_zip": str(output_zip)}
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return result
    except Exception as exc:
        result = {"status": "failed", "error": str(exc)}
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return result


def _worker_loop(
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    worker_index: int,
) -> None:
    # Keep a single transformer per worker to avoid reloads between tasks.
    transformer: Transform | None = None

    while True:
        payload = task_queue.get()
        if payload is None or payload.get("command") == "shutdown":
            return

        options = payload["options"]
        if transformer is None:
            device_ids = _parse_device_ids(options.get("device_ids"))
            if device_ids:
                import os

                os.environ["CUDA_VISIBLE_DEVICES"] = device_ids[worker_index % len(device_ids)]

            transformer = Transform(
                models_cache_path=options.get("models_cache_path"),
                local_only=options.get("local_only", False),
            )

        result = _process_task_payload(payload, transformer)
        result_queue.put({"task_id": payload["task_id"], **result})


class TaskManager:
    def __init__(self, max_concurrent: int = MAX_CONCURRENT, task_root: Path = TASK_ROOT) -> None:
        self._ctx = multiprocessing.get_context("spawn")
        self._max_concurrent = max_concurrent
        self._task_root = task_root
        self._task_root.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, Task] = {}
        self._queue: deque[str] = deque()
        self._running: dict[str, int] = {}
        self._worker_busy: dict[int, str | None] = {}
        self._worker_queues: dict[int, multiprocessing.Queue] = {}
        self._workers: dict[int, multiprocessing.Process] = {}
        self._result_queue: multiprocessing.Queue = self._ctx.Queue()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._shutdown = False
        self._dispatch_thread: threading.Thread | None = None
        self._result_thread: threading.Thread | None = None
        self._started = False
        self._load_existing_tasks()

    def start(self) -> None:
        with self._condition:
            if self._started:
                return
            for index in range(self._max_concurrent):
                self._start_worker(index)
            self._dispatch_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
            self._result_thread = threading.Thread(target=self._result_loop, daemon=True)
            self._dispatch_thread.start()
            self._result_thread.start()
            self._started = True

    def create_task(self, file_storage, options: dict[str, Any]) -> Task:
        original_name = secure_filename(file_storage.filename or "upload.pdf")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        task_id = f"{timestamp}-{uuid.uuid4().hex[:6]}"
        task_dir = self._task_root / task_id
        input_dir = task_dir / "input"
        _ensure_dir(input_dir)
        input_path = input_dir / original_name
        file_storage.save(input_path)

        options = dict(options)
        options["original_name"] = original_name

        task = Task(
            task_id=task_id,
            created_at=datetime.now().isoformat(timespec="seconds"),
            status="queued",
            original_name=original_name,
            input_path=str(input_path),
            task_dir=str(task_dir),
            options=options,
        )
        _write_task_meta(task_dir, task)
        with self._condition:
            self._tasks[task_id] = task
            self._queue.append(task_id)
            self._condition.notify_all()
        return task

    def list_tasks(self) -> list[Task]:
        with self._lock:
            return list(self._tasks.values())

    def get_task(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def stop_task(self, task_id: str) -> None:
        with self._condition:
            if task_id in self._running:
                worker_index = self._running[task_id]
                task = self._tasks.get(task_id)
                if task:
                    task.message = "Stopping..."
                abort_path = Path(task.task_dir) / "abort.flag" if task else None
                if abort_path:
                    abort_path.write_text("stop", encoding="utf-8")
                self._condition.notify_all()
                return

            if task_id in self._queue:
                self._queue.remove(task_id)
                task = self._tasks.get(task_id)
                if task:
                    task.status = "stopped"
                    task.message = "Stopped by user"
                    task.finished_at = datetime.now().isoformat(timespec="seconds")
                self._condition.notify_all()

    def restart_task(self, task_id: str) -> None:
        with self._condition:
            task = self._tasks.get(task_id)
            if not task:
                return
            if task.status == "running":
                return
            abort_path = Path(task.task_dir) / "abort.flag"
            if abort_path.exists():
                abort_path.unlink()
            task.status = "queued"
            task.message = ""
            task.started_at = None
            task.finished_at = None
            task.output_zip = None
            if task_id not in self._queue:
                self._queue.append(task_id)
            self._condition.notify_all()

    def _dispatch_loop(self) -> None:
        while True:
            with self._condition:
                if self._shutdown:
                    return
                self._ensure_workers()
                self._dispatch_tasks()
                self._condition.wait(timeout=0.5)

    def _result_loop(self) -> None:
        while True:
            try:
                result = self._result_queue.get(timeout=0.5)
            except queue.Empty:
                with self._condition:
                    if self._shutdown:
                        return
                continue

            with self._condition:
                task_id = result.get("task_id")
                task = self._tasks.get(task_id)
                if task and task.status != "stopped":
                    task.finished_at = datetime.now().isoformat(timespec="seconds")
                    if result.get("status") == "completed":
                        task.status = "completed"
                        task.output_zip = result.get("output_zip")
                        task.message = "Done"
                    elif result.get("status") == "stopped":
                        task.status = "stopped"
                        task.message = "Stopped by user"
                    else:
                        task.status = "failed"
                        task.message = result.get("error", "Failed")

                worker_index = self._running.pop(task_id, None)
                if worker_index is not None:
                    self._worker_busy[worker_index] = None
                self._condition.notify_all()

    def _ensure_workers(self) -> None:
        for index in range(self._max_concurrent):
            process = self._workers.get(index)
            if process and process.is_alive():
                continue

            task_id = self._worker_busy.get(index)
            if task_id:
                task = self._tasks.get(task_id)
                if task and task.status == "running":
                    task.status = "failed"
                    task.message = "Worker exited"
                    task.finished_at = datetime.now().isoformat(timespec="seconds")
                self._running.pop(task_id, None)
                self._worker_busy[index] = None

            self._start_worker(index)

    def _dispatch_tasks(self) -> None:
        for index in range(self._max_concurrent):
            if not self._queue:
                break
            if self._worker_busy.get(index) is not None:
                continue
            task_id = self._queue.popleft()
            task = self._tasks.get(task_id)
            if not task or task.status != "queued":
                continue
            self._assign_task(task, index)

    def _assign_task(self, task: Task, worker_index: int) -> None:
        abort_path = Path(task.task_dir) / "abort.flag"
        if abort_path.exists():
            abort_path.unlink()
        payload = {
            "task_id": task.task_id,
            "task_dir": task.task_dir,
            "input_pdf": task.input_path,
            "options": task.options,
            "original_name": task.original_name,
            "abort_path": str(abort_path),
        }
        self._worker_busy[worker_index] = task.task_id
        self._running[task.task_id] = worker_index
        task.status = "running"
        task.started_at = datetime.now().isoformat(timespec="seconds")
        task.pid = self._workers[worker_index].pid
        self._worker_queues[worker_index].put(payload)

    def _load_existing_tasks(self) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        for task_dir in sorted(self._task_root.iterdir()):
            if not task_dir.is_dir():
                continue
            task_id = task_dir.name
            meta = _load_task_meta(task_dir)
            result = _load_task_result(task_dir)
            input_dir = task_dir / "input"
            input_path: str | None = None
            original_name: str | None = None
            options: dict[str, Any] = _default_options()

            if meta:
                original_name = meta.get("original_name") or original_name
                input_path = meta.get("input_path") or input_path
                options = meta.get("options") or options

            if input_path is None and input_dir.exists():
                input_files = sorted(path for path in input_dir.iterdir() if path.is_file())
                if input_files:
                    input_path = str(input_files[0])
                    original_name = input_files[0].name

            if not input_path or not original_name:
                continue

            created_at = self._guess_created_at(task_id, task_dir, meta)
            status = "failed"
            message = "Server restarted before completion"
            output_zip: str | None = None
            finished_at: str | None = None

            if result:
                status = result.get("status", "failed")
                if status == "completed":
                    message = "Done"
                    output_zip = result.get("output_zip")
                elif status == "stopped":
                    message = "Stopped by user"
                else:
                    message = result.get("error", "Failed")

                result_path = task_dir / "result.json"
                if result_path.exists():
                    finished_at = datetime.fromtimestamp(result_path.stat().st_mtime).isoformat(timespec="seconds")

            if output_zip:
                output_path = Path(output_zip)
                if not output_path.is_absolute():
                    output_zip = str(base_dir / output_path)

            task = Task(
                task_id=task_id,
                created_at=created_at,
                status=status,
                original_name=original_name,
                input_path=input_path,
                task_dir=str(task_dir),
                options=options,
                message=message,
                finished_at=finished_at,
                output_zip=output_zip,
            )
            self._tasks[task_id] = task

    def _guess_created_at(self, task_id: str, task_dir: Path, meta: dict[str, Any] | None) -> str:
        if meta:
            created_at = meta.get("created_at")
            if isinstance(created_at, str) and created_at:
                return created_at
        try:
            prefix = task_id.split("-", 2)[:2]
            if len(prefix) == 2:
                parsed = datetime.strptime("-".join(prefix), "%Y%m%d-%H%M%S")
                return parsed.isoformat(timespec="seconds")
        except Exception:
            pass
        return datetime.fromtimestamp(task_dir.stat().st_mtime).isoformat(timespec="seconds")

    def _start_worker(self, index: int) -> None:
        task_queue = self._ctx.Queue()
        process = self._ctx.Process(target=_worker_loop, args=(task_queue, self._result_queue, index))
        process.start()
        self._worker_queues[index] = task_queue
        self._workers[index] = process
        self._worker_busy[index] = None

    def _restart_worker(self, index: int) -> None:
        process = self._workers.get(index)
        if process and process.is_alive():
            process.terminate()
            process.join(timeout=5)
        self._start_worker(index)


app = Flask(__name__)
_manager: TaskManager | None = None
_manager_lock = threading.Lock()


def _ensure_manager() -> TaskManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = TaskManager()
                _manager.start()
    return _manager


def _parse_options(form) -> dict[str, Any]:
    def _bool(name: str) -> bool:
        return form.get(name) == "on"

    options = _default_options()
    options.update({
        "university_name": form.get("university_name") or None,
        "ocr_size": form.get("ocr_size") or "gundam",
        "includes_footnotes": _bool("includes_footnotes"),
        "ignore_pdf_errors": _bool("ignore_pdf_errors"),
        "generate_plot": _bool("generate_plot"),
        "toc_assumed": _bool("toc_assumed"),
        "device_ids": form.get("device_ids") or None,
    })
    return options


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", ocr_sizes=_OCR_SIZES)


@app.route("/upload", methods=["POST"])
def upload():
    file_storage = request.files.get("pdf")
    if not file_storage or not file_storage.filename:
        return redirect(url_for("index"))
    if not file_storage.filename.lower().endswith(".pdf"):
        return redirect(url_for("index"))

    options = _parse_options(request.form)
    _ensure_manager().create_task(file_storage, options)
    return redirect(url_for("index"))


@app.route("/task/<task_id>/stop", methods=["POST"])
def stop_task(task_id: str):
    _ensure_manager().stop_task(task_id)
    return redirect(url_for("index"))


@app.route("/task/<task_id>/start", methods=["POST"])
def start_task(task_id: str):
    _ensure_manager().restart_task(task_id)
    return redirect(url_for("index"))


@app.route("/task/<task_id>/download", methods=["GET"])
def download_task(task_id: str):
    task = _ensure_manager().get_task(task_id)
    if not task or not task.output_zip:
        return redirect(url_for("index"))
    output_path = Path(task.output_zip)
    if not output_path.exists():
        return redirect(url_for("index"))
    return send_file(output_path, as_attachment=True, download_name=output_path.name)


@app.route("/api/tasks", methods=["GET"])
def api_tasks():
    manager = _ensure_manager()
    tasks = manager.list_tasks()
    with manager._lock:
        queue_positions = {task_id: idx + 1 for idx, task_id in enumerate(manager._queue)}
    payload = [
        task.to_dict(queue_position=queue_positions.get(task.task_id))
        for task in sorted(tasks, key=lambda t: t.created_at, reverse=True)
    ]
    return jsonify(payload)


def main() -> None:
    _ensure_manager()
    app.run(host="0.0.0.0", port=5201, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
