// No framework, no build step - the UI must stay readable and hackable.

const state = {
  mode: "simple",
  busy: false,
  pendingPlan: null,
  info: null,
};

const $ = (selector) => document.querySelector(selector);

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function el(html) {
  const template = document.createElement("template");
  template.innerHTML = html.trim();
  return template.content.firstElementChild;
}

function scrollDown() {
  const transcript = $("#transcript");
  transcript.scrollTop = transcript.scrollHeight;
}

function addUser(text) {
  $("#empty")?.remove();
  const node = el(`<div class="msg user"><div class="bubble">${esc(text)}</div></div>`);
  $("#transcript").append(node);
  scrollDown();
}

function addAgent(text) {
  $("#empty")?.remove();
  const node = el(`<div class="msg agent"><div class="bubble">${esc(text)}</div></div>`);
  $("#transcript").append(node);
  scrollDown();
}

const ICONS = {
  status: "·", intent: "◎", plan: "☰", plan_pending: "⏸", file: "＋", scaffold: "▣",
  tests: "✓", review: "☑", deploy_plan: "▲", progress: "›", step: "›", error: "!", done: "★", end: "",
};

function eventCard(event) {
  const type = event.type;
  const icon = ICONS[type] || "•";
  let tone = "";
  if (type === "error") tone = "bad";
  if (type === "tests" || type === "done") tone = event.data?.ok === false ? "bad" : "ok";
  if (type === "review") tone = event.data?.ok ? "ok" : "warn";
  if (type === "plan_pending" || type === "deploy_plan") tone = "warn";

  const details = renderDetails(event);
  const node = el(`
    <div class="event ${tone}">
      <span class="icon">${icon}</span>
      <div class="body">
        <div class="label">${esc(event.message || type)}</div>
        <div class="detail"></div>
      </div>
    </div>
  `);
  const detailHost = node.querySelector(".detail");
  if (event.type === "file" && state.mode !== "power") {
    node.remove();
    return null;
  }
  detailHost.append(details.fragment);
  if (details.text) {
    const p = document.createElement("div");
    p.textContent = details.text;
    detailHost.prepend(p);
  }
  if (state.mode === "power" && event.data && Object.keys(event.data).length) {
    const raw = document.createElement("details");
    raw.innerHTML = `<summary>raw event</summary><pre>${esc(JSON.stringify(event.data, null, 2))}</pre>`;
    detailHost.append(raw);
  }
  $("#transcript").append(node);
  scrollDown();
  return node;
}

function renderDetails(event) {
  const data = event.data || {};
  const fragment = document.createDocumentFragment();
  let text = "";

  if (event.type === "error" && data.plain) {
    text = data.plain;
    if (Array.isArray(data.next_steps) && data.next_steps.length) {
      const list = document.createElement("ul");
      list.className = "clean";
      data.next_steps.slice(0, 4).forEach((step) => {
        const item = document.createElement("li");
        item.textContent = step;
        list.append(item);
      });
      fragment.append(list);
    }
  }

  if (event.type === "plan") {
    const list = document.createElement("div");
    list.className = "steps-list";
    (data.steps || []).forEach((step, index) => {
      const row = document.createElement("div");
      row.className = "step-row";
      row.innerHTML = `<span class="idx">${index + 1}</span><span>${esc(step)}</span>`;
      list.append(row);
    });
    fragment.append(list);
    if (data.awaiting_approval) {
      const actions = document.createElement("div");
      actions.className = "plan-actions";
      const approve = document.createElement("button");
      approve.className = "primary small";
      approve.textContent = "Approve and build";
      approve.onclick = () => approvePlan(actions);
      const reject = document.createElement("button");
      reject.className = "ghost small";
      reject.textContent = "Not now";
      reject.onclick = async () => {
        await fetch("/api/plan/reject", { method: "POST" });
        state.pendingPlan = null;
        actions.innerHTML = `<span class="muted small">Plan rejected - nothing was written.</span>`;
      };
      actions.append(approve, reject);
      fragment.append(actions);
    }
  }

  if (event.type === "tests") {
    const failed = data.failed || [];
    text = data.ok
      ? `${data.command || "tests"} passed in ${Math.round(data.duration || 0)}s`
      : `${failed.length || "some"} failing: ${failed.slice(0, 3).join(", ")}`;
  }

  if (event.type === "review") {
    const counts = data.counts || {};
    const parts = Object.entries(counts).filter(([, n]) => n).map(([k, n]) => `${n} ${k}`);
    text = `score ${data.score}/100${parts.length ? " · " + parts.join(", ") : " · no findings"}`;
    const findings = data.findings || [];
    if (findings.length) {
      const list = document.createElement("ul");
      list.className = "clean";
      findings.slice(0, 4).forEach((finding) => {
        const item = document.createElement("li");
        item.textContent = `${finding.path}:${finding.line} — ${finding.message}`;
        list.append(item);
      });
      fragment.append(list);
    }
  }

  if (event.type === "deploy_plan") {
    text = `provider: ${data.provider || "auto"}${data.warnings?.length ? " · " + data.warnings[0] : ""}`;
  }

  if (event.type === "scaffold") {
    text = `${(data.files || []).length} files written${(data.warnings || []).length ? " · " + data.warnings[0] : ""}`;
  }

  if (event.type === "step" || event.type === "progress") {
    text = data.detail ? String(data.detail).slice(0, 240) : "";
  }

  return { fragment, text };
}

async function stream(url, body, onEvent) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!response.ok || !response.body) {
    const message = await response.text().catch(() => "request failed");
    onEvent({ type: "error", message: "The agent hit an error", data: { plain: message.slice(0, 400) } });
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() || "";
    for (const chunk of chunks) {
      const line = chunk.split("\n").find((row) => row.startsWith("data: "));
      if (!line) continue;
      try {
        onEvent(JSON.parse(line.slice(6)));
      } catch (error) {
        console.warn("bad event", error, chunk);
      }
    }
  }
}

async function sendMessage(text) {
  if (!text.trim() || state.busy) return;
  addUser(text);
  setBusy(true);
  const requireApproval = $("#plan-first").checked;
  await stream("/api/chat", { message: text, mode: state.mode, require_approval: requireApproval }, (event) => {
    if (event.type === "end") return;
    if (event.type === "plan_pending") {
      state.pendingPlan = event.data?.id || "pending";
      eventCard(event);
      return;
    }
    if (event.type === "done") {
      addAgent(event.message);
      refreshInfo();
      return;
    }
    if (event.type === "file" && state.mode !== "power") return;
    eventCard(event);
  });
  setBusy(false);
}

async function approvePlan(container) {
  container.innerHTML = `<span class="muted small">Building…</span>`;
  setBusy(true);
  await stream("/api/plan/continue", {}, (event) => {
    if (event.type === "end") return;
    if (event.type === "done") { addAgent(event.message); refreshInfo(); return; }
    if (event.type === "file" && state.mode !== "power") return;
    eventCard(event);
  });
  setBusy(false);
}

async function deploy() {
  const button = $("#deploy-btn");
  button.disabled = true;
  button.textContent = "Deploying…";
  eventCard({ type: "status", message: "Deploying: building artifacts and verifying the app", data: {} });
  try {
    const response = await fetch("/api/deploy", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dry_run: false, publish: true }),
    });
    const data = await response.json();
    if (data.url) {
      const link = $("#live-link");
      link.hidden = false;
      link.href = data.url;
      link.textContent = data.url;
      addAgent(`Deployed. Live at ${data.url}`);
      state.previewUrl = data.url;
    }
    if (data.plain || data.plain_success) {
      addAgent(data.plain || data.plain_success);
    }
    eventCard({
      type: data.ok ? "done" : "error",
      message: data.ok ? `Deployed with ${data.provider}` : (data.human?.title || "Deploy failed"),
      data: data.ok ? { url: data.url } : (data.human || data.human_error || { plain: data.error }),
    });
    if (data.rendered) {
      const node = el(`<details class="event ok"><summary>deploy log</summary><pre>${esc(data.rendered)}</pre></details>`);
      $("#transcript").append(node);
    }
    const sc = $("#status-card");
    if (sc) sc.textContent = data.ok ? `✓ Live: ${data.url || "artifact ready"} • status/index.html updated` : `✗ ${data.human?.plain || data.error || "deploy failed"}`;
  } catch (error) {
    eventCard({ type: "error", message: "Deploy failed", data: { plain: String(error) } });
  } finally {
    button.disabled = false;
    button.textContent = "Deploy";
    refreshInfo();
    scrollDown();
  }
}

async function ship() {
  const input = $("#input").value.trim() || $("#input").placeholder;
  const button = $("#ship-btn");
  if (!input || input.length < 5) {
    eventCard({ type: "error", message: "Tell me what to build first", data: { plain: "Type a sentence like 'a task app with login' then hit Ship it." } });
    return;
  }
  button.disabled = true;
  const orig = button.textContent;
  button.textContent = "Shipping…";
  addUser(input);
  $("#input").value = "";
  eventCard({ type: "status", message: "Ship it: scaffolding → testing → reviewing → deploying (one click, no decisions)", data: {} });
  try {
    const response = await fetch("/api/ship", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request: input, provider: null, dry_run: false, force: false }),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      const human = data.human || { title: data.error || "Ship failed", plain: data.error || "unknown", next_steps: [] };
      eventCard({ type: "error", message: human.title || "Ship needs work", data: human });
      if (data.verify) {
        const v = data.verify;
        eventCard({ type: "tests", message: v.ok ? "Tests passed" : "Tests failed", data: v });
      }
      if (data.review) {
        const r = data.review;
        eventCard({ type: "review", message: `Review score ${r.score}/100`, data: r });
      }
      if (data.plain) addAgent(data.plain);
    } else {
      if (data.url) {
        const link = $("#live-link");
        link.hidden = false;
        link.href = data.url;
        link.textContent = data.url;
        // Auto open preview
        startPreview();
      }
      addAgent(data.plain_success || data.plain || `Shipped! ${data.url ? "Live at " + data.url : "Artifact ready."}`);
      eventCard({ type: "done", message: data.ok ? `Shipped with ${data.provider || data.deploy?.provider || "archive"}` : "Shipped", data: { url: data.url, plain: data.plain_success } });
      const sc = $("#status-card");
      if (sc) sc.textContent = data.url ? `✓ Live at ${data.url}` : "✓ Artifact ready — see status/index.html";
    }
    if (data.deploy?.rendered || data.rendered) {
      const rendered = data.deploy?.rendered || data.rendered;
      const node = el(`<details class="event ok"><summary>deploy log</summary><pre>${esc(rendered)}</pre></details>`);
      $("#transcript").append(node);
    }
  } catch (error) {
    eventCard({ type: "error", message: "Ship failed", data: { plain: String(error), next_steps: ["Check that your description is a sentence", "Try again, or use `agent deploy --dry-run`"] } });
    addAgent("Sorry — I couldn't ship that. Try a simpler description, or check the deploy log above. Your idea is safe; just re-hit Ship it.");
  } finally {
    button.disabled = false;
    button.textContent = orig;
    refreshInfo();
    scrollDown();
  }
}

async function startPreview() {
  const panel = $("#preview-panel");
  panel.hidden = false;
  const frame = $("#preview-frame");
  frame.src = "about:blank";
  frame.srcdoc = "<p style='font:15px system-ui;padding:24px'>Starting the project…</p>";
  try {
    const response = await fetch("/api/preview/start", { method: "POST" });
    const data = await response.json();
    if (!data.running) {
      frame.srcdoc = `<p style='font:15px system-ui;padding:24px'>Preview unavailable: ${esc(data.detail)}</p>`;
      return;
    }
    frame.removeAttribute("srcdoc");
    frame.src = `/preview/?t=${Date.now()}`;
    $("#preview-open").href = "/preview/";
  } catch (error) {
    frame.srcdoc = `<p style='font:15px system-ui;padding:24px'>Preview error: ${esc(String(error))}</p>`;
  }
}

function setBusy(busy) {
  state.busy = busy;
  const send = $("#send");
  send.disabled = busy;
  send.textContent = busy ? "Working…" : "Send";
  $("#input").disabled = false;
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  const pill = $("#mode-pill");
  if (mode === "power") {
    pill.textContent = "power mode · raw events";
    $("#subtitle").textContent = "Tool calls and raw events are visible.";
  } else {
    pill.textContent = state.info?.runtime?.mode === "model" ? "simple mode · model" : "simple mode · local workflow";
    $("#subtitle").textContent = "Describe it in plain English — a plan comes first.";
  }
  pill.classList.toggle("model", mode === "simple" && state.info?.runtime?.mode === "model");
}

function projectCard(project) {
  const card = $("#project-card");
  if (!project) {
    card.innerHTML = `<h2>Project</h2><p class="muted small">No recognizable project in this folder yet. Start one below.</p>`;
    return;
  }
  const rows = [
    ["Type", project.kind],
    ["Stack", `${project.language} / ${project.framework}`],
    ["Port", project.kind === "worker" || project.kind === "cli" ? "—" : project.port],
    ["Files", project.file_count],
    ["Tests", project.test_cmd || "none detected"],
  ];
  card.innerHTML = `<h2>Project</h2><dl class="kv">${rows
    .map(([key, value]) => `<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`)
    .join("")}</dl>`;
}

function templateList(templates) {
  const host = $("#templates");
  host.innerHTML = "";
  templates.forEach((template) => {
    const button = el(`
      <button class="template" type="button">
        <strong>${esc(template.name)}</strong>
        <span>${esc(template.tagline)}</span>
      </button>
    `);
    button.onclick = () => {
      $("#input").value = template.description;
      $("#input").focus();
    };
    host.append(button);
  });
}

async function refreshInfo() {
  try {
    const response = await fetch("/api/info");
    const info = await response.json();
    state.info = info;
    projectCard(info.project);
    templateList(info.runtime.templates || []);
    const pill = $("#mode-pill");
    pill.textContent = info.runtime.mode === "model" ? "model connected" : "local workflow · no model needed";
    pill.classList.toggle("model", info.runtime.mode === "model");
    const docker = info.tooling?.docker ? "docker" : "no docker";
    const git = info.tooling?.git ? "git" : "no git";
    $("#footer").textContent = `${info.runtime.workspace} · ${git} · ${docker}`;
    const deployHint = $("#deploy-hint");
    const usable = Object.entries(info.providers || {}).filter(([, value]) => value.ok).map(([name]) => name);
    deployHint.textContent = usable.length
      ? `Ready: ${usable.join(", ")}`
      : "No host configured — a portable build artifact will be produced.";
  } catch (error) {
    console.warn("info failed", error);
  }
}

async function loadHistory() {
  try {
    const response = await fetch("/api/session");
    const data = await response.json();
    (data.history || []).slice(-20).forEach((turn) => {
      if (turn.role === "user") addUser(turn.content);
      else addAgent(turn.content);
    });
  } catch (error) {
    console.warn("history failed", error);
  }
}

document.querySelectorAll(".mode").forEach((button) => {
  button.onclick = () => setMode(button.dataset.mode);
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.onclick = () => {
    $("#input").value = chip.dataset.prompt;
    $("#input").focus();
  };
});

$("#composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = $("#input");
  const text = input.value;
  input.value = "";
  sendMessage(text);
});

$("#input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#composer").requestSubmit();
  }
});

$("#deploy-btn").onclick = deploy;
$("#ship-btn").onclick = ship;
$("#preview-btn").onclick = startPreview;
$("#preview-close").onclick = async () => {
  $("#preview-panel").hidden = true;
  $("#preview-frame").src = "about:blank";
  await fetch("/api/preview/stop", { method: "POST" });
};
$("#reset").onclick = async () => {
  await fetch("/api/session/reset", { method: "POST" });
  $("#transcript").innerHTML = "";
  location.reload();
};

(async function init() {
  await refreshInfo();
  await loadHistory();
  setMode("simple");
})();
