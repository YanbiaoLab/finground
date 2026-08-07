const APP_NAME = "finground";
const BROWSER_ID_STORAGE_KEY = "finground-browser-user-id";
const MAX_FILE_BYTES = 100 * 1024 * 1024;
const DEFAULT_MESSAGE_PLACEHOLDER = "给 FinGround 发消息…";

const state = {
  userId: getBrowserUserId(),
  sessionId: null,
  sessionState: {},
  attachment: null,
  tasks: [],
  taskProgress: {},
  running: false,
  taskDrawerOpen: false,
  openedTasksForRun: false,
  activeAssistantContent: null,
  activeProgress: null,
  hasAttachmentQuestionSuggestion: false,
};

const ids = [
  "sidebar", "menuButton", "newChatButton", "historyList", "welcome", "messages", "messageInput",
  "fileInput", "attachButton", "sendButton", "attachmentPreview", "attachmentType", "attachmentName",
  "attachmentSize", "removeAttachmentButton", "tasksButton", "tasksButtonLabel",
  "taskDrawer", "closeTasksButton", "taskProgress", "progressValue", "progressTitle", "progressMeta",
  "taskList", "conversation", "toast", "sampleReports", "sampleReportList",
];
const elements = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));

function escapeHtml(value = "") {
  const node = document.createElement("div");
  node.textContent = String(value);
  return node.innerHTML;
}

function getBrowserUserId() {
  try {
    const existing = localStorage.getItem(BROWSER_ID_STORAGE_KEY);
    if (existing) return existing;
  } catch (_) {}

  const browserId = `browser-${crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`}`;
  try {
    localStorage.setItem(BROWSER_ID_STORAGE_KEY, browserId);
  } catch (_) {
    return browserId;
  }
  return browserId;
}

function renderInlineMarkdown(value = "") {
  const codeSpans = [];
  const source = String(value).replace(/`([^`\n]+)`/g, (_, code) => {
    const token = `\uE000${codeSpans.length}\uE001`;
    codeSpans.push(code);
    return token;
  });
  let html = escapeHtml(source)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>");
  codeSpans.forEach((code, index) => {
    html = html.replace(`\uE000${index}\uE001`, `<code>${escapeHtml(code)}</code>`);
  });
  return html;
}

function splitTableRow(line) {
  let source = line.trim();
  if (source.startsWith("|")) source = source.slice(1);
  if (source.endsWith("|")) source = source.slice(0, -1);
  const cells = [];
  let cell = "";
  for (let index = 0; index < source.length; index += 1) {
    if (source[index] === "\\" && source[index + 1] === "|") {
      cell += "|";
      index += 1;
    } else if (source[index] === "|") {
      cells.push(cell.trim());
      cell = "";
    } else {
      cell += source[index];
    }
  }
  cells.push(cell.trim());
  return cells;
}

function tableAlignments(line) {
  const cells = splitTableRow(line);
  if (!cells.length || !cells.every((cell) => /^:?-{3,}:?$/.test(cell))) return null;
  return cells.map((cell) => {
    if (cell.startsWith(":") && cell.endsWith(":")) return "center";
    if (cell.endsWith(":")) return "right";
    return "left";
  });
}

function normalizedTableRow(cells, columnCount) {
  if (cells.length <= columnCount) {
    return [...cells, ...Array(columnCount - cells.length).fill("")];
  }
  return [...cells.slice(0, columnCount - 1), cells.slice(columnCount - 1).join(" | ")];
}

function renderTable(lines, start) {
  const headers = splitTableRow(lines[start]);
  const alignments = tableAlignments(lines[start + 1]);
  const statusColumn = headers.findIndex((header) => /^(状态|status)$/i.test(header.trim()));
  const kpiColumn = headers.findIndex((header) => /^kpi$/i.test(header.trim()));
  const rows = [];
  let index = start + 2;
  while (index < lines.length && lines[index].trim() && lines[index].includes("|")) {
    rows.push(normalizedTableRow(splitTableRow(lines[index]), headers.length));
    index += 1;
  }
  const cell = (value, column, tag) => {
    const alignment = alignments?.[column] || "left";
    let content = renderInlineMarkdown(value);
    let className = `align-${alignment}`;
    if (tag === "td" && column === statusColumn) {
      const status = value.trim().toLowerCase();
      const knownStatus = ["found", "absent", "ambiguous", "explicit_zero"].includes(status);
      content = `<span class="status-badge ${knownStatus ? status : "neutral"}">${content}</span>`;
      className += " status-column";
    }
    if (tag === "td" && column === kpiColumn) className += " kpi-column";
    return `<${tag} class="${className}">${content}</${tag}>`;
  };
  const head = headers.map((header, column) => cell(header, column, "th")).join("");
  const body = rows.map((row) => `<tr>${row.map((value, column) => cell(value, column, "td")).join("")}</tr>`).join("");
  return {
    html: `<div class="markdown-table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`,
    nextIndex: index,
  };
}

function renderMarkdown(value = "") {
  const lines = String(value).replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  const isTableStart = (index) => index + 1 < lines.length
    && lines[index].includes("|")
    && tableAlignments(lines[index + 1]) !== null;
  const isBlockStart = (index) => /^```|^#{1,6}\s+|^\s*[-*+]\s+|^\s*\d+\.\s+/.test(lines[index])
    || isTableStart(index);

  for (let index = 0; index < lines.length;) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }
    if (line.startsWith("```")) {
      const code = [];
      index += 1;
      while (index < lines.length && !lines[index].startsWith("```")) {
        code.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      blocks.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }
    if (isTableStart(index)) {
      const table = renderTable(lines, index);
      blocks.push(table.html);
      index = table.nextIndex;
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      const level = Math.min(heading[1].length + 1, 6);
      blocks.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      index += 1;
      continue;
    }
    const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
    if (unordered) {
      const items = [];
      while (index < lines.length) {
        const item = lines[index].match(/^\s*[-*+]\s+(.+)$/);
        if (!item) break;
        items.push(`<li>${renderInlineMarkdown(item[1])}</li>`);
        index += 1;
      }
      blocks.push(`<ul>${items.join("")}</ul>`);
      continue;
    }
    const ordered = line.match(/^\s*\d+\.\s+(.+)$/);
    if (ordered) {
      const items = [];
      while (index < lines.length) {
        const item = lines[index].match(/^\s*\d+\.\s+(.+)$/);
        if (!item) break;
        items.push(`<li>${renderInlineMarkdown(item[1])}</li>`);
        index += 1;
      }
      blocks.push(`<ol>${items.join("")}</ol>`);
      continue;
    }
    const paragraph = [line];
    index += 1;
    while (index < lines.length && lines[index].trim() && !isBlockStart(index)) {
      paragraph.push(lines[index]);
      index += 1;
    }
    blocks.push(`<p>${paragraph.map(renderInlineMarkdown).join("<br>")}</p>`);
  }
  return blocks.join("");
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.remove("hidden");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => elements.toast.classList.add("hidden"), 3600);
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail || body.error || message;
    } catch (_) { /* non-JSON response */ }
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}

function sessionUrl(sessionId) {
  return `/apps/${APP_NAME}/users/${encodeURIComponent(state.userId)}/sessions/${encodeURIComponent(sessionId)}`;
}

async function createSession() {
  return request(`/apps/${APP_NAME}/users/${encodeURIComponent(state.userId)}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId: crypto.randomUUID() }),
  });
}

function ensureConversationVisible() {
  elements.welcome.classList.add("hidden");
  elements.messages.classList.remove("hidden");
}

function scrollToLatest() {
  requestAnimationFrame(() => { elements.messages.scrollTop = elements.messages.scrollHeight; });
}

function fileBadge(fileName) {
  return fileName.toLowerCase().endsWith(".pdf") ? "PDF" : "MD";
}

function addUserMessage(text, file = null) {
  ensureConversationVisible();
  const article = document.createElement("article");
  article.className = "message user";
  const body = document.createElement("div");
  body.className = "message-body";
  if (file) {
    const fileNode = document.createElement("div");
    fileNode.className = "message-file";
    fileNode.innerHTML = `<span class="attachment-type">${fileBadge(file.name)}</span><span>${escapeHtml(file.name)}</span>`;
    body.append(fileNode);
  }
  const content = document.createElement("div");
  content.className = "message-content";
  content.innerHTML = renderMarkdown(text);
  body.append(content);
  article.append(body);
  elements.messages.append(article);
  scrollToLatest();
}

function addAssistantMessage({ thinking = false } = {}) {
  ensureConversationVisible();
  const article = document.createElement("article");
  article.className = "message assistant";
  article.innerHTML = `<div class="message-avatar">FG</div><div class="message-body"><div class="assistant-label">FinGround</div><div class="message-content"></div></div>`;
  const content = article.querySelector(".message-content");
  if (thinking) content.innerHTML = `<div class="thinking"><span class="thinking-dots"><i></i><i></i><i></i></span><span>正在理解你的请求…</span></div>`;
  elements.messages.append(article);
  scrollToLatest();
  return {
    article,
    body: article.querySelector(".message-body"),
    content,
    text: "",
    events: [],
  };
}

function renderAssistantMessage(message) {
  message.text = message.events.map((entry) => entry.text).filter(Boolean).join("\n\n");
  message.content.innerHTML = renderMarkdown(message.text);
  scrollToLatest();
}

function appendAssistantText(message, addition) {
  if (!addition) return;
  message.events.push({ id: null, author: "root_agent", partial: false, text: addition });
  renderAssistantMessage(message);
}

function updateAssistantEvent(message, event) {
  const addition = eventText(event);
  if (!addition) return;
  const entry = {
    id: event.id || null,
    author: event.author,
    partial: Boolean(event.partial),
    text: addition,
  };

  if (event.partial) {
    const last = message.events.at(-1);
    if (last?.partial && last.author === entry.author) {
      last.id = entry.id;
      last.text += entry.text;
    } else {
      message.events.push(entry);
    }
    renderAssistantMessage(message);
    return;
  }

  let replaceIndex = entry.id
    ? message.events.findIndex((candidate) => candidate.id === entry.id)
    : -1;
  if (replaceIndex < 0) {
    const lastIndex = message.events.length - 1;
    const last = message.events[lastIndex];
    if (last?.partial && last.author === entry.author) replaceIndex = lastIndex;
  }
  if (replaceIndex >= 0) message.events[replaceIndex] = entry;
  else message.events.push(entry);
  renderAssistantMessage(message);
}

function addHistoricalAssistant(text) {
  const message = addAssistantMessage();
  appendAssistantText(message, text);
}

function attachmentQuestionSuggestions(fileName) {
  const year = fileName.match(/(?:19|20)\d{2}/)?.[0];
  const period = year ? `FY${year}` : "报告期内";
  return [
    `这份年报 ${period} 的 revenue 是多少？请给出数值、单位、页码和原文证据。`,
    `请提取这份年报 ${period} 的 revenue、operating_income 和 net_income，并附上单位、页码和原文证据。`,
    `请根据 revenue 和 gross_profit 计算 ${period} 的毛利率，并附上计算过程、单位、页码和原文证据。`,
    `请比较 ${period} 的 net_income 与 operating_cash_flow，说明两者差异，并提供数值、单位、页码和原文证据。`,
    `请根据 ${period} 的 operating_income 和 revenue 计算营业利润率，并附上计算过程、单位、页码和原文证据。`,
  ];
}

function setAttachment(file) {
  if (!file) return;
  const lowerName = file.name.toLowerCase();
  const supported = lowerName.endsWith(".pdf") || lowerName.endsWith(".md") || lowerName.endsWith(".markdown");
  if (!supported) return showToast("目前支持 PDF 或 Markdown 附件");
  if (file.size > MAX_FILE_BYTES) return showToast("附件不能超过 100 MB");
  state.attachment = file;
  elements.attachmentType.textContent = fileBadge(file.name);
  elements.attachmentName.textContent = file.name;
  elements.attachmentSize.textContent = `${formatBytes(file.size)} · 将随下一条消息发送`;
  elements.attachmentPreview.classList.remove("hidden");
  if (!elements.messageInput.value.trim()) {
    const suggestions = attachmentQuestionSuggestions(file.name);
    const suggestion = suggestions[Math.floor(Math.random() * suggestions.length)];
    elements.messageInput.placeholder = suggestion;
    state.hasAttachmentQuestionSuggestion = true;
  }
  updateSendState();
}

function clearAttachment() {
  state.attachment = null;
  state.hasAttachmentQuestionSuggestion = false;
  elements.fileInput.value = "";
  elements.messageInput.placeholder = DEFAULT_MESSAGE_PLACEHOLDER;
  elements.attachmentPreview.classList.add("hidden");
  updateSendState();
}

function sampleReportTitle(name) {
  return name.replace(/\.pdf$/i, "").replaceAll("_", " ");
}

function renderSampleReports(reports) {
  elements.sampleReportList.innerHTML = "";
  for (const report of reports) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sample-report";
    button.dataset.name = report.name;
    button.dataset.url = report.url;
    button.dataset.size = String(report.size);
    button.innerHTML = `<span class="sample-report-icon">PDF</span><span class="sample-report-copy"><strong></strong><small></small></span><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 10h10m-4-4 4 4-4 4" /></svg>`;
    button.querySelector("strong").textContent = sampleReportTitle(report.name);
    button.querySelector("small").textContent = `${formatBytes(report.size)} · 点击选用`;
    elements.sampleReportList.append(button);
  }
  elements.sampleReports.classList.toggle("hidden", reports.length === 0);
}

async function loadSampleReports() {
  try {
    const reports = await request("/_finground/sample-reports");
    renderSampleReports(Array.isArray(reports) ? reports : []);
  } catch (_) {
    elements.sampleReports.classList.add("hidden");
  }
}

async function selectSampleReport(button) {
  if (state.running) return showToast("当前回复完成后再选择示例年报");
  const size = Number(button.dataset.size);
  if (size > MAX_FILE_BYTES) return showToast("示例年报不能超过 100 MB");
  button.disabled = true;
  button.classList.add("loading");
  try {
    const response = await fetch(button.dataset.url);
    if (!response.ok) throw new Error(`示例年报加载失败 (${response.status})`);
    const blob = await response.blob();
    const file = new File([blob], button.dataset.name, { type: "application/pdf" });
    setAttachment(file);
    elements.messageInput.focus();
    showToast("示例年报已加入附件，请输入你的问题");
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
    button.classList.remove("loading");
  }
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1]);
    reader.readAsDataURL(file);
  });
}

function updateSendState() {
  elements.sendButton.disabled = state.running || (!elements.messageInput.value.trim() && !state.attachment);
}

function resizeComposer() {
  elements.messageInput.style.height = "auto";
  elements.messageInput.style.height = `${Math.min(elements.messageInput.scrollHeight, 150)}px`;
}

function normalizedTasks(session) {
  return Object.values(session.state?.tasks || {}).sort((a, b) => Number(a.id) - Number(b.id));
}

function updateTasksFromSessionState({ reveal = false } = {}) {
  state.tasks = normalizedTasks({ state: state.sessionState });
  for (const task of state.tasks) {
    if (task.status !== "in_progress") delete state.taskProgress[task.id];
  }
  renderTasks();
  if (reveal && state.tasks.length && !state.openedTasksForRun && window.innerWidth > 980) {
    state.openedTasksForRun = true;
    setTaskDrawer(true);
  }
}

function renderTasks() {
  const total = state.tasks.length;
  const completed = state.tasks.filter((task) => task.status === "completed").length;
  const active = state.tasks.filter((task) => task.status === "in_progress").length;
  const pending = state.tasks.filter((task) => task.status === "pending").length;
  const percent = total ? Math.round((completed / total) * 100) : 0;
  elements.tasksButton.classList.toggle("hidden", total === 0);
  elements.tasksButtonLabel.textContent = active ? `${active} 个任务执行中` : `${completed}/${total} 个任务`;
  elements.progressValue.textContent = `${percent}%`;
  elements.taskProgress.querySelector(".progress-ring").style.setProperty("--progress", `${percent * 3.6}deg`);
  elements.progressTitle.textContent = total ? (completed === total ? "全部任务已完成" : active ? "正在执行" : "等待继续") : "尚无任务";
  elements.progressMeta.textContent = total ? `${completed} 已完成 · ${active} 进行中 · ${pending} 待处理` : "复杂工作会在这里显示实时状态";
  elements.taskList.innerHTML = total ? state.tasks.map((task) => {
    const icon = task.status === "completed" ? "✓" : task.status === "pending" ? "·" : "";
    const detail = task.status === "completed"
      ? "已完成"
      : task.status === "in_progress"
        ? (state.taskProgress[task.id] || task.activeForm || "正在执行")
        : task.metadata?.error ? "等待重试" : "等待执行";
    return `<div class="task-item"><span class="task-state ${escapeHtml(task.status)}">${icon}</span><div><strong>${escapeHtml(task.subject)}</strong><span>${escapeHtml(detail)}</span></div></div>`;
  }).join("") : `<div class="task-empty">发送一个复杂任务后，执行计划会出现在这里。</div>`;
  renderInlineProgress(percent, completed, total, active);
}

function renderInlineProgress(percent, completed, total, active) {
  if (!state.activeAssistantContent || !total) return;
  if (!state.activeProgress) {
    state.activeProgress = document.createElement("div");
    state.activeProgress.className = "inline-progress";
    state.activeAssistantContent.body.append(state.activeProgress);
  }
  const title = completed === total ? "任务已全部完成" : active ? `正在执行 ${active} 个任务` : "正在规划任务";
  state.activeProgress.innerHTML = `<div class="inline-progress-head">${completed === total ? "" : '<i class="mini-spinner"></i>'}<strong>${title}</strong><span>${completed}/${total}</span></div><div class="inline-track"><span style="width:${percent}%"></span></div>`;
  scrollToLatest();
}

function setTaskDrawer(open) {
  state.taskDrawerOpen = open;
  document.querySelector(".conversation-layout").classList.toggle("tasks-open", open);
  elements.taskDrawer.setAttribute("aria-hidden", String(!open));
}

async function saveConversationTitle(title) {
  await request(sessionUrl(state.sessionId), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stateDelta: { conversation_title: title.replace(/\s+/g, " ").slice(0, 60) } }),
  });
}

async function refreshSession() {
  if (!state.sessionId) return null;
  const session = await request(sessionUrl(state.sessionId));
  state.sessionState = session.state || {};
  updateTasksFromSessionState({ reveal: state.running });
  return session;
}

function eventText(event) {
  if (event.author !== "root_agent" || !event.content?.parts) return "";
  return event.content.parts.filter((part) => part.text && !part.thought).map((part) => part.text).join("");
}

function processEventStateDelta(event) {
  const stateDelta = event.actions?.stateDelta;
  if (!stateDelta || typeof stateDelta !== "object") return;
  state.sessionState = { ...state.sessionState, ...stateDelta };
  if (Object.hasOwn(stateDelta, "tasks")) {
    updateTasksFromSessionState({ reveal: state.running });
  }
}

function workerTaskId(event) {
  if (!["kpi_worker", "report_qa_worker"].includes(event.author) || !event.isolationScope) return null;
  const taskId = event.isolationScope.split("/").filter(Boolean).at(-1);
  return state.tasks.some((task) => task.id === taskId) ? taskId : null;
}

function workerPhaseForPart(part) {
  const tool = part.functionCall?.name || part.functionResponse?.name;
  if (tool === "GetKpiKnowledge") return "正在读取 KPI 规则";
  if (tool === "PrepareReportQuestion") return "正在理解问题";
  if (tool === "SearchReport") return "正在搜索年报";
  if (tool === "ReadReportChunks") return "正在核验证据";
  if (tool === "finish_task") return "结果已返回，等待汇总";
  return null;
}

function processWorkerProgressEvent(event) {
  const taskId = workerTaskId(event);
  if (!taskId) return;
  let phase = event.output ? "结果已返回，等待汇总" : null;
  for (const part of event.content?.parts || []) {
    phase = workerPhaseForPart(part) || phase;
  }
  if (!phase) return;
  state.taskProgress[taskId] = phase;
  renderTasks();
}

function processRunEvent(event) {
  processEventStateDelta(event);
  processWorkerProgressEvent(event);
  updateAssistantEvent(state.activeAssistantContent, event);
}

async function streamRun(parts) {
  const response = await fetch("/run_sse", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      appName: APP_NAME,
      userId: state.userId,
      sessionId: state.sessionId,
      streaming: true,
      newMessage: { role: "user", parts },
    }),
  });
  if (!response.ok) throw new Error((await response.text()) || `请求失败 (${response.status})`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const consumeRecords = () => {
    const records = buffer.split(/\r?\n\r?\n/);
    buffer = records.pop() || "";
    for (const record of records) {
      const payload = record.split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (!payload) continue;
      const event = JSON.parse(payload);
      if (event.error) throw new Error(event.error);
      processRunEvent(event);
    }
  };
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      buffer += decoder.decode();
      if (buffer.trim()) buffer += "\n\n";
      consumeRecords();
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    consumeRecords();
  }
}

async function sendMessage() {
  const typedText = elements.messageInput.value.trim();
  const attachment = state.attachment;
  if (state.running || (!typedText && !attachment)) return;
  const messageText = typedText || "请阅读这份文件，并告诉我你能提供哪些帮助。";
  state.running = true;
  state.openedTasksForRun = false;
  state.taskProgress = {};
  state.activeProgress = null;
  updateSendState();
  addUserMessage(messageText, attachment);
  elements.messageInput.value = "";
  resizeComposer();
  state.activeAssistantContent = addAssistantMessage({ thinking: true });

  try {
    if (!state.sessionId) {
      const session = await createSession();
      state.sessionId = session.id;
      state.sessionState = session.state || {};
      history.replaceState(null, "", `/?session=${encodeURIComponent(session.id)}`);
      await saveConversationTitle(messageText);
    }
    const parts = [{ text: messageText }];
    if (attachment) {
      const data = await fileToBase64(attachment);
      parts.push({ inlineData: { mimeType: fileBadge(attachment.name) === "PDF" ? "application/pdf" : "text/markdown", displayName: attachment.name, data } });
    }
    clearAttachment();
    await streamRun(parts);
    await refreshSession();
    if (!state.activeAssistantContent.text) appendAssistantText(state.activeAssistantContent, "任务已执行完成，但没有生成可显示的文本回复。");
    await loadHistory();
  } catch (error) {
    appendAssistantText(state.activeAssistantContent, `抱歉，本次请求没有完成。\n\n${error.message}`);
    showToast(`请求失败：${error.message}`);
  } finally {
    state.running = false;
    state.activeAssistantContent = null;
    state.activeProgress = null;
    updateSendState();
    elements.messageInput.focus();
  }
}

function cleanUserText(event) {
  if (event.author !== "user" || !event.content?.parts) return "";
  return event.content.parts.map((part) => part.text || "").filter((text) => text && !text.startsWith("[Uploaded annual report artifact:")) .join("\n").trim();
}

function renderSession(session) {
  elements.messages.innerHTML = "";
  state.taskProgress = {};
  state.sessionState = session.state || {};
  state.tasks = normalizedTasks(session);
  for (const event of session.events || []) {
    if (event.partial) continue;
    const userText = cleanUserText(event);
    if (userText) {
      addUserMessage(userText);
      continue;
    }
    const text = eventText(event);
    if (text) addHistoricalAssistant(text);
  }
  if (!elements.messages.children.length && session.state?.conversation_title) {
    addUserMessage(session.state.conversation_title);
    addHistoricalAssistant("这条对话没有保存可显示的回复。你可以重新发送问题。");
  }
  if (elements.messages.children.length) ensureConversationVisible();
  else {
    elements.messages.classList.add("hidden");
    elements.welcome.classList.remove("hidden");
  }
  renderTasks();
  scrollToLatest();
}

function sessionTitle(session) {
  if (session.state?.conversation_title) return session.state.conversation_title;
  for (const event of session.events || []) {
    const text = cleanUserText(event);
    if (text) return text.replace(/\s+/g, " ").slice(0, 34);
  }
  return session.state?.report?.report_ref || "新对话";
}

async function loadHistory() {
  try {
    const sessions = await request(`/apps/${APP_NAME}/users/${encodeURIComponent(state.userId)}/sessions`);
    sessions.sort((a, b) => (b.lastUpdateTime || 0) - (a.lastUpdateTime || 0));
    elements.historyList.innerHTML = sessions.length ? sessions.slice(0, 20).map((session) => {
      const taskCount = normalizedTasks(session).length;
      return `<button class="history-item ${session.id === state.sessionId ? "active" : ""}" type="button" data-session="${escapeHtml(session.id)}"><strong>${escapeHtml(sessionTitle(session))}</strong><span>${taskCount ? `${taskCount} 个任务` : "普通对话"}</span></button>`;
    }).join("") : `<div class="history-empty">还没有对话记录</div>`;
  } catch (_) {
    elements.historyList.innerHTML = `<div class="history-empty">暂时无法载入历史对话</div>`;
  }
}

async function openSession(sessionId) {
  if (state.running) return showToast("当前回复完成后再切换对话");
  try {
    const session = await request(sessionUrl(sessionId));
    state.sessionId = session.id;
    history.replaceState(null, "", `/?session=${encodeURIComponent(session.id)}`);
    renderSession(session);
    await loadHistory();
    elements.sidebar.classList.remove("open");
  } catch (error) {
    showToast(`无法打开对话：${error.message}`);
  }
}

function newChat() {
  if (state.running) return showToast("当前回复完成后再新建对话");
  state.sessionId = null;
  state.sessionState = {};
  state.tasks = [];
  state.taskProgress = {};
  state.attachment = null;
  state.hasAttachmentQuestionSuggestion = false;
  state.taskDrawerOpen = false;
  elements.messages.innerHTML = "";
  elements.messages.classList.add("hidden");
  elements.welcome.classList.remove("hidden");
  elements.messageInput.value = "";
  elements.messageInput.placeholder = DEFAULT_MESSAGE_PLACEHOLDER;
  elements.fileInput.value = "";
  elements.attachmentPreview.classList.add("hidden");
  setTaskDrawer(false);
  renderTasks();
  history.replaceState(null, "", "/");
  loadHistory();
  elements.messageInput.focus();
}

function bindEvents() {
  elements.messageInput.addEventListener("input", () => {
    if (state.hasAttachmentQuestionSuggestion && elements.messageInput.value) {
      state.hasAttachmentQuestionSuggestion = false;
      elements.messageInput.placeholder = DEFAULT_MESSAGE_PLACEHOLDER;
    }
    resizeComposer();
    updateSendState();
  });
  elements.messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendMessage(); }
  });
  elements.sendButton.addEventListener("click", sendMessage);
  elements.attachButton.addEventListener("click", () => elements.fileInput.click());
  elements.fileInput.addEventListener("change", () => setAttachment(elements.fileInput.files[0]));
  elements.removeAttachmentButton.addEventListener("click", clearAttachment);
  elements.newChatButton.addEventListener("click", newChat);
  elements.menuButton.addEventListener("click", () => elements.sidebar.classList.toggle("open"));
  elements.tasksButton.addEventListener("click", () => setTaskDrawer(!state.taskDrawerOpen));
  elements.closeTasksButton.addEventListener("click", () => setTaskDrawer(false));
  elements.sampleReportList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-url]");
    if (button) selectSampleReport(button);
  });
  elements.historyList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-session]");
    if (button) openSession(button.dataset.session);
  });
}

async function initialize() {
  bindEvents();
  renderTasks();
  await Promise.all([loadHistory(), loadSampleReports()]);
  const sessionId = new URLSearchParams(location.search).get("session");
  if (sessionId) await openSession(sessionId);
  elements.messageInput.focus();
}

initialize();
