async function fetchTasks() {
  const response = await fetch("/api/tasks");
  if (!response.ok) {
    return [];
  }
  return response.json();
}

function badgeClass(status) {
  return `badge ${status}`;
}

function formatMeta(task) {
  const parts = [];
  if (task.queue_position) {
    parts.push(`队列位置: ${task.queue_position}`);
  }
  if (task.started_at) {
    parts.push(`开始: ${task.started_at}`);
  }
  if (task.finished_at) {
    parts.push(`结束: ${task.finished_at}`);
  }
  return parts.join(" · ");
}

function buildTaskRow(task) {
  const container = document.createElement("div");
  container.className = "task";

  const name = document.createElement("div");
  name.innerHTML = `<div class="name">${task.original_name}</div><div class="meta">${formatMeta(task)}</div>`;

  const status = document.createElement("div");
  status.innerHTML = `<span class="${badgeClass(task.status)}">${task.status}</span>`;

  const message = document.createElement("div");
  message.textContent = task.message || "";

  const created = document.createElement("div");
  created.innerHTML = `<div class="meta">创建: ${task.created_at}</div>`;

  const actions = document.createElement("div");
  actions.className = "task-actions";

  if (task.status === "running" || task.status === "queued") {
    const stopBtn = document.createElement("button");
    stopBtn.textContent = "停止";
    stopBtn.addEventListener("click", () => postAction(`/task/${task.task_id}/stop`));
    actions.appendChild(stopBtn);
  } else {
    const startBtn = document.createElement("button");
    startBtn.textContent = "再开";
    startBtn.addEventListener("click", () => postAction(`/task/${task.task_id}/start`));
    actions.appendChild(startBtn);
  }

  if (task.output_ready) {
    const download = document.createElement("a");
    download.textContent = "下载";
    download.href = `/task/${task.task_id}/download`;
    actions.appendChild(download);
  }

  container.appendChild(name);
  container.appendChild(status);
  container.appendChild(message);
  container.appendChild(created);
  container.appendChild(actions);
  return container;
}

async function refresh() {
  const tasks = await fetchTasks();
  const container = document.getElementById("tasks");
  container.innerHTML = "";

  if (!tasks.length) {
    const empty = document.createElement("div");
    empty.className = "tasks-empty";
    empty.textContent = "暂无任务";
    container.appendChild(empty);
    return;
  }

  tasks.forEach((task) => {
    container.appendChild(buildTaskRow(task));
  });
}

async function postAction(url) {
  await fetch(url, { method: "POST" });
  refresh();
}

const refreshBtn = document.getElementById("refresh");
if (refreshBtn) {
  refreshBtn.addEventListener("click", refresh);
}

refresh();
setInterval(refresh, 4000);
