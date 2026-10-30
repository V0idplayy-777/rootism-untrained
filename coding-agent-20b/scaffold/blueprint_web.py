"""Frontend assets for blueprint apps: a small, tasteful, dependency-free SPA.

The UI is deliberately vanilla (no build step, no CDN): it must run from the
same container as the API, on a phone, and after the agent is uninstalled. What
"taste" means here is written down in :mod:`quality.taste` and is measurable:

* a design-token layer (colour, spacing, radius, type) rather than ad-hoc values;
* light and dark schemes driven by ``prefers-color-scheme``;
* a real responsive layout (sidebar becomes a top bar; tables become cards);
* empty, loading, error and success states for every data view;
* accessible controls - labels, focus rings, ``aria-live`` errors, skip link;
* safe rendering: every value goes through ``textContent``, never ``innerHTML``.

``render_web_assets`` returns ``path -> content`` for the generator, exactly
like the Python builders do.
"""

from __future__ import annotations

import json

from .blueprints import AppBlueprint, EntitySpec, FieldSpec

ICON_GLYPHS = {
    "check": "✓",
    "note": "▤",
    "bookmark": "◆",
    "bowl": "◍",
    "person": "◉",
    "receipt": "▤",
    "spark": "✦",
    "book": "▭",
    "box": "▢",
    "briefcase": "▣",
    "bolt": "ϟ",
    "leaf": "❧",
    "folder": "▣",
    "layers": "≡",
    "play": "▷",
    "map": "◇",
    "pin": "◈",
    "paw": "❋",
    "heart": "♡",
    "calendar": "▦",
    "flag": "⚑",
    "bug": "✳",
    "chat": "◌",
    "home": "⌂",
    "wrench": "⚒",
    "car": "◫",
    "people": "◉",
    "building": "▥",
    "inbox": "⌄",
    "ticket": "▤",
    "wallet": "▱",
    "tag": "◇",
    "chart": "▤",
    "clock": "◷",
    "seed": "❧",
    "help": "?",
    "quiz": "▤",
    "funnel": "▽",
    "deal": "◆",
    "basket": "◍",
    "tool": "⚒",
    "board": "▦",
    "dot": "•",
}


def _field_config(blueprint: AppBlueprint, field_spec: FieldSpec) -> dict:
    data = {
        "key": field_spec.name,
        "label": field_spec.display,
        "type": field_spec.type,
        "required": bool(field_spec.required),
        "help": field_spec.help,
        "choices": list(field_spec.choices),
        "min": field_spec.min_value,
        "max": field_spec.max_value,
        "maxLength": field_spec.max_length or (20000 if field_spec.type == "longtext" else 0),
    }
    if field_spec.type == "ref":
        target = blueprint.entity(field_spec.ref)
        data["refPlural"] = target.table if target else field_spec.ref + "s"
        data["refDisplay"] = target.display if target else "title"
        data["refLabel"] = target.label if target else field_spec.ref.title()
    return data


def _list_fields(entity: EntitySpec) -> list[str]:
    chosen = [entity.display]
    for spec in entity.fields:
        if spec.name == entity.display or spec.type == "longtext":
            continue
        if spec.type in {"enum", "bool", "date", "int", "timestamp", "ref"}:
            chosen.append(spec.name)
        if len(chosen) >= 4:
            break
    return chosen


def app_config(blueprint: AppBlueprint, module: str, title: str, auth: bool, accent: str) -> dict:
    entities = []
    for entity in blueprint.entities:
        fields = [_field_config(blueprint, spec) for spec in entity.fields]
        entities.append(
            {
                "name": entity.name,
                "plural": entity.table,
                "label": entity.label,
                "pluralLabel": entity.label + "s",
                "display": entity.display,
                "icon": entity.icon,
                "fields": fields,
                "list": _list_fields(entity),
                "searchable": list(entity.searchable) or [entity.display],
                "empty": {
                    "title": "No " + entity.table.replace("_", " ") + " yet",
                    "body": "Add the first " + entity.label.lower() + " and it will show up here. "
                    "Everything is stored in your own database.",
                    "action": "New " + entity.label.lower(),
                },
            }
        )
    feature: dict = {"kind": blueprint.feature if blueprint.feature != "none" else "none"}
    if blueprint.feature == "board":
        target = next(
            (entity for entity in blueprint.entities if entity.field(blueprint.feature_field or "status")),
            blueprint.entities[0],
        )
        status = target.field(blueprint.feature_field or "status") or target.fields[0]
        feature.update(
            {
                "entity": target.table,
                "field": status.name,
                "columns": [
                    {"value": choice, "label": choice.replace("_", " ").title()} for choice in status.choices
                ],
            }
        )
    elif blueprint.feature == "schedule":
        target = next(
            (entity for entity in blueprint.entities if entity.field("start_at") and entity.field("end_at")),
            blueprint.entities[0],
        )
        feature.update({"entity": target.table, "start": "start_at", "end": "end_at"})
    elif blueprint.feature == "team":
        feature.update({"entity": blueprint.entities[0].table})
    elif blueprint.feature == "stats":
        target = blueprint.entities[0]
        feature.update({"entity": target.table, "field": blueprint.feature_field})
    return {
        "title": title,
        "auth": auth,
        "accent": accent,
        "module": module,
        "tokenKey": module + ".token",
        "features": feature,
        "entities": entities,
    }


INDEX_TMPL = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="theme-color" content="@@ACCENT@@" />
  <meta name="description" content="@@DESCRIPTION@@" />
  <title>@@TITLE@@</title>
  <link rel="stylesheet" href="/assets/app.css" />
</head>
<body>
  <a class="skip-link" href="#view">Skip to content</a>
  <div class="shell">
    <aside class="sidebar" id="sidebar" aria-label="Main navigation">
      <div class="brand">
        <span class="brand-mark" aria-hidden="true">@@MARK@@</span>
        <span class="brand-name">@@TITLE@@</span>
      </div>
      <nav class="nav" id="nav" aria-label="Sections"></nav>
      <div class="sidebar-foot" id="sidebar-foot"></div>
    </aside>
    <main class="view" id="view" tabindex="-1">
      <div class="skeleton-stack" aria-hidden="true">
        <div class="skeleton skeleton-title"></div>
        <div class="skeleton skeleton-line"></div>
        <div class="skeleton skeleton-line short"></div>
        <div class="skeleton skeleton-block"></div>
      </div>
    </main>
  </div>
  <div class="toast" id="toast" role="status" aria-live="polite"></div>
  <div class="modal-root" id="modal-root"></div>
  <noscript>
    <p class="noscript">This interface needs JavaScript. The API still works -
      read <a href="/docs">/docs</a> or call <a href="/healthz">/healthz</a>.</p>
  </noscript>
  <script src="/assets/config.js"></script>
  <script src="/assets/app.js"></script>
</body>
</html>
"""

APP_CSS = r""":root {
  --accent: @@ACCENT@@;
  --accent-strong: color-mix(in srgb, var(--accent) 82%, black);
  --accent-soft: color-mix(in srgb, var(--accent) 14%, white);
  --ink: #17181c;
  --ink-soft: #4b5058;
  --ink-faint: #6f7683;
  --surface: #ffffff;
  --surface-2: #f5f6f8;
  --surface-3: #eceef2;
  --line: #dfe3e9;
  --line-strong: #c8ced8;
  --positive: #0f7a4d;
  --positive-soft: #e2f5ec;
  --warn: #9a6100;
  --warn-soft: #fdf1d8;
  --danger: #b3261e;
  --danger-soft: #fde8e6;
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 18px;
  --radius-pill: 999px;
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;
  --space-7: 48px;
  --font-sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  --text-xs: 0.75rem;
  --text-sm: 0.875rem;
  --text-md: 1rem;
  --text-lg: 1.25rem;
  --text-xl: clamp(1.4rem, 1.2rem + 0.8vw, 1.8rem);
  --shadow-sm: 0 1px 2px rgb(16 20 30 / 6%);
  --shadow-md: 0 8px 24px rgb(16 20 30 / 8%);
  --shadow-lg: 0 24px 64px rgb(16 20 30 / 18%);
  --sidebar-width: 264px;
  --focus: 0 0 0 3px color-mix(in srgb, var(--accent) 35%, transparent);
}

@media (prefers-color-scheme: dark) {
  :root {
    --ink: #eef0f4;
    --ink-soft: #b8bec9;
    --ink-faint: #8e96a3;
    --surface: #14161b;
    --surface-2: #1b1e25;
    --surface-3: #242833;
    --line: #2c313b;
    --line-strong: #3a404d;
    --accent-soft: color-mix(in srgb, var(--accent) 22%, #14161b);
    --positive-soft: #12302a;
    --warn-soft: #33270f;
    --danger-soft: #3a1a18;
    --shadow-sm: 0 1px 2px rgb(0 0 0 / 40%);
    --shadow-md: 0 8px 24px rgb(0 0 0 / 45%);
    --shadow-lg: 0 24px 64px rgb(0 0 0 / 60%);
  }
}

*,
*::before,
*::after {
  box-sizing: border-box;
}

html {
  -webkit-text-size-adjust: 100%;
}

body {
  margin: 0;
  min-height: 100dvh;
  background: var(--surface-2);
  color: var(--ink);
  font-family: var(--font-sans);
  font-size: var(--text-md);
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}

h1, h2, h3, p {
  margin: 0;
}

h1 {
  font-size: var(--text-xl);
  letter-spacing: -0.02em;
  line-height: 1.2;
}

h2 {
  font-size: var(--text-lg);
  letter-spacing: -0.01em;
}

a {
  color: var(--accent-strong);
}

button, input, select, textarea {
  font: inherit;
  color: inherit;
}

:focus-visible {
  outline: none;
  box-shadow: var(--focus);
  border-radius: var(--radius-sm);
}

.skip-link {
  position: absolute;
  left: var(--space-4);
  top: -60px;
  z-index: 100;
  background: var(--surface);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-sm);
  box-shadow: var(--shadow-md);
  transition: top 120ms ease;
}

.skip-link:focus {
  top: var(--space-4);
}

.shell {
  display: grid;
  grid-template-columns: var(--sidebar-width) minmax(0, 1fr);
  min-height: 100dvh;
}

.sidebar {
  display: flex;
  flex-direction: column;
  gap: var(--space-5);
  padding: var(--space-5) var(--space-4);
  background: var(--surface);
  border-right: 1px solid var(--line);
  position: sticky;
  top: 0;
  height: 100dvh;
}

.brand {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: 0 var(--space-2);
  font-weight: 650;
  letter-spacing: -0.01em;
}

.brand-mark {
  display: inline-grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border-radius: var(--radius-md);
  background: var(--accent);
  color: #fff;
  font-size: 1rem;
}

.brand-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.nav {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  overflow-y: auto;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-sm);
  color: var(--ink-soft);
  text-decoration: none;
  font-size: var(--text-sm);
  font-weight: 500;
  transition: background 120ms ease, color 120ms ease;
}

.nav-item:hover {
  background: var(--surface-2);
  color: var(--ink);
}

.nav-item[aria-current="page"] {
  background: var(--accent-soft);
  color: var(--accent-strong);
}

.sidebar-foot {
  margin-top: auto;
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  padding-top: var(--space-4);
  border-top: 1px solid var(--line);
  font-size: var(--text-xs);
  color: var(--ink-faint);
}

.view {
  padding: var(--space-6) clamp(var(--space-4), 4vw, var(--space-7));
  max-width: 1180px;
  width: 100%;
}

.page-head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: var(--space-4);
  margin-bottom: var(--space-3);
  flex-wrap: wrap;
}

.page-head p {
  color: var(--ink-faint);
  font-size: var(--text-sm);
  margin-top: var(--space-1);
}

.toolbar {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  margin: var(--space-4) 0;
  flex-wrap: wrap;
}

.search {
  position: relative;
  flex: 1 1 260px;
  max-width: 420px;
}

.search input {
  width: 100%;
  padding: var(--space-2) var(--space-3) var(--space-2) var(--space-6);
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-pill);
  background: var(--surface);
  font-size: var(--text-sm);
}

.search::before {
  content: "⌕";
  position: absolute;
  left: var(--space-3);
  top: 50%;
  transform: translateY(-50%);
  color: var(--ink-faint);
}

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-4);
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--ink);
  font-size: var(--text-sm);
  font-weight: 550;
  cursor: pointer;
  text-decoration: none;
  transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
}

.btn:hover {
  background: var(--surface-2);
}

.btn:active {
  transform: translateY(1px);
}

.btn-primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}

.btn-primary:hover {
  background: var(--accent-strong);
}

.btn-ghost {
  border-color: transparent;
  background: transparent;
}

.btn-danger {
  color: var(--danger);
  border-color: var(--danger-soft);
  background: var(--danger-soft);
}

.btn-sm {
  padding: var(--space-1) var(--space-3);
  font-size: var(--text-xs);
}

.icon-button {
  display: inline-grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  background: transparent;
  cursor: pointer;
  color: var(--ink-soft);
}

.icon-button:hover {
  background: var(--surface-2);
}

.icon {
  font-family: var(--font-sans);
  font-size: 0.95em;
  line-height: 1;
}

.table-wrap {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-sm);
  overflow: hidden;
}

.table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--text-sm);
}

.table th {
  text-align: left;
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--ink-faint);
  padding: var(--space-3) var(--space-4);
  border-bottom: 1px solid var(--line);
  background: var(--surface-2);
  font-weight: 600;
}

.table td {
  padding: var(--space-3) var(--space-4);
  border-bottom: 1px solid var(--line);
  vertical-align: middle;
}

.table tbody tr:last-child td {
  border-bottom: 0;
}

.table tbody tr:hover {
  background: var(--surface-2);
}

.cell-strong {
  font-weight: 600;
}

.cell-muted {
  color: var(--ink-faint);
}

.row-actions {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-1);
}

.badge, .pill {
  display: inline-flex;
  align-items: center;
  gap: var(--space-1);
  padding: 2px var(--space-2);
  border-radius: var(--radius-pill);
  font-size: var(--text-xs);
  font-weight: 600;
  background: var(--surface-3);
  color: var(--ink-soft);
  white-space: nowrap;
}

.badge[data-tone="positive"], .pill[data-tone="positive"] {
  background: var(--positive-soft);
  color: var(--positive);
}

.badge[data-tone="warn"], .pill[data-tone="warn"] {
  background: var(--warn-soft);
  color: var(--warn);
}

.badge[data-tone="danger"], .pill[data-tone="danger"] {
  background: var(--danger-soft);
  color: var(--danger);
}

.badge[data-tone="accent"], .pill[data-tone="accent"] {
  background: var(--accent-soft);
  color: var(--accent-strong);
}

.empty-state {
  display: grid;
  justify-items: center;
  gap: var(--space-3);
  text-align: center;
  padding: var(--space-7) var(--space-4);
  background: var(--surface);
  border: 1px dashed var(--line-strong);
  border-radius: var(--radius-lg);
}

.empty-mark {
  display: grid;
  place-items: center;
  width: 48px;
  height: 48px;
  border-radius: var(--radius-pill);
  background: var(--accent-soft);
  color: var(--accent-strong);
  font-size: 1.2rem;
}

.empty-state p {
  color: var(--ink-faint);
  max-width: 44ch;
}

.error-banner {
  display: flex;
  gap: var(--space-3);
  align-items: flex-start;
  padding: var(--space-3) var(--space-4);
  border: 1px solid var(--danger-soft);
  background: var(--danger-soft);
  color: var(--danger);
  border-radius: var(--radius-md);
  margin-bottom: var(--space-4);
}

.skeleton-stack {
  display: grid;
  gap: var(--space-3);
  max-width: 720px;
}

.skeleton {
  background: linear-gradient(90deg, var(--surface-3) 25%, var(--surface-2) 37%, var(--surface-3) 63%);
  background-size: 400% 100%;
  border-radius: var(--radius-sm);
  animation: shimmer 1.4s ease infinite;
}

.skeleton-title { height: 30px; width: 38%; }
.skeleton-line { height: 14px; width: 72%; }
.skeleton-line.short { width: 48%; }
.skeleton-block { height: 180px; border-radius: var(--radius-lg); }

@keyframes shimmer {
  0% { background-position: 100% 0; }
  100% { background-position: 0 0; }
}

.dialog {
  width: min(560px, calc(100vw - 32px));
  max-height: calc(100dvh - 64px);
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  padding: var(--space-5);
  background: var(--surface);
  color: var(--ink);
  box-shadow: var(--shadow-lg);
}

.dialog::backdrop {
  background: rgb(10 12 16 / 55%);
  backdrop-filter: blur(2px);
}

.dialog-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-3);
  margin-bottom: var(--space-4);
}

.dialog-head p {
  color: var(--ink-faint);
  font-size: var(--text-sm);
  margin-top: var(--space-1);
}

.form {
  display: grid;
  gap: var(--space-4);
}

.field {
  display: grid;
  gap: var(--space-1);
}

.field label {
  font-size: var(--text-sm);
  font-weight: 600;
}

.field .hint {
  font-size: var(--text-xs);
  color: var(--ink-faint);
}

.field input,
.field select,
.field textarea {
  width: 100%;
  padding: var(--space-2) var(--space-3);
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-sm);
  background: var(--surface);
  font-size: var(--text-sm);
}

.field textarea {
  min-height: 110px;
  resize: vertical;
}

.field input[aria-invalid="true"],
.field select[aria-invalid="true"],
.field textarea[aria-invalid="true"] {
  border-color: var(--danger);
}

.check {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.check input {
  width: 18px;
  height: 18px;
  accent-color: var(--accent);
}

.field-error {
  color: var(--danger);
  font-size: var(--text-xs);
}

.form-errors {
  color: var(--danger);
  font-size: var(--text-sm);
}

.form-errors:empty {
  display: none;
}

.form-actions {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-2);
  padding-top: var(--space-2);
}

.toast {
  position: fixed;
  right: var(--space-5);
  bottom: var(--space-5);
  z-index: 60;
  padding: var(--space-3) var(--space-4);
  border-radius: var(--radius-md);
  background: var(--ink);
  color: var(--surface);
  box-shadow: var(--shadow-lg);
  font-size: var(--text-sm);
  opacity: 0;
  transform: translateY(8px);
  pointer-events: none;
  transition: opacity 160ms ease, transform 160ms ease;
  max-width: min(420px, calc(100vw - 32px));
}

.toast[data-visible="true"] {
  opacity: 1;
  transform: translateY(0);
}

.toast[data-kind="error"] {
  background: var(--danger);
  color: #fff;
}

.board {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: var(--space-4);
  align-items: start;
}

.board-column {
  background: var(--surface-2);
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  padding: var(--space-3);
  min-height: 180px;
}

.board-column h2 {
  font-size: var(--text-sm);
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--space-1) var(--space-2) var(--space-3);
  color: var(--ink-soft);
}

.board-card {
  display: grid;
  gap: var(--space-2);
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  padding: var(--space-3);
  margin-bottom: var(--space-2);
  box-shadow: var(--shadow-sm);
}

.board-card h3 {
  font-size: var(--text-sm);
  font-weight: 600;
}

.board-card .meta {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1);
}

.board-card .row-actions {
  justify-content: flex-start;
}

.board-empty {
  color: var(--ink-faint);
  font-size: var(--text-xs);
  padding: var(--space-3) var(--space-2);
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: var(--space-4);
  margin-bottom: var(--space-5);
}

.stat-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  padding: var(--space-4);
  box-shadow: var(--shadow-sm);
}

.stat-card .label {
  color: var(--ink-faint);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.stat-card .value {
  font-size: clamp(1.6rem, 1.2rem + 1.2vw, 2.2rem);
  font-weight: 700;
  letter-spacing: -0.02em;
}

.panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  padding: var(--space-4);
  margin-bottom: var(--space-4);
  box-shadow: var(--shadow-sm);
}

.bar-row {
  display: grid;
  grid-template-columns: 130px 1fr 48px;
  gap: var(--space-3);
  align-items: center;
  padding: var(--space-1) 0;
  font-size: var(--text-sm);
}

.bar-track {
  height: 10px;
  background: var(--surface-3);
  border-radius: var(--radius-pill);
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  background: var(--accent);
  border-radius: var(--radius-pill);
}

.trend {
  display: flex;
  align-items: flex-end;
  gap: var(--space-1);
  height: 90px;
  padding-top: var(--space-2);
}

.trend-bar {
  flex: 1;
  min-width: 4px;
  background: var(--accent-soft);
  border-radius: 3px 3px 0 0;
}

.auth {
  min-height: 100dvh;
  display: grid;
  place-items: center;
  padding: var(--space-5);
  background: radial-gradient(circle at 20% 10%, var(--accent-soft), transparent 55%), var(--surface-2);
}

.auth-card {
  width: min(420px, 100%);
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  box-shadow: var(--shadow-md);
  display: grid;
  gap: var(--space-4);
}

.auth-card .brand {
  padding: 0;
}

.tabs {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-1);
  background: var(--surface-2);
  border-radius: var(--radius-sm);
  padding: var(--space-1);
}

.tabs button {
  border: 0;
  background: transparent;
  padding: var(--space-2);
  border-radius: var(--radius-sm);
  font-size: var(--text-sm);
  font-weight: 600;
  color: var(--ink-faint);
  cursor: pointer;
}

.tabs button[aria-selected="true"] {
  background: var(--surface);
  color: var(--ink);
  box-shadow: var(--shadow-sm);
}

.muted {
  color: var(--ink-faint);
}

.stack {
  display: grid;
  gap: var(--space-4);
}

.row {
  display: flex;
  gap: var(--space-2);
  align-items: center;
  flex-wrap: wrap;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.noscript {
  position: fixed;
  inset: auto var(--space-4) var(--space-4);
  background: var(--warn-soft);
  color: var(--warn);
  padding: var(--space-3);
  border-radius: var(--radius-md);
}

@media (max-width: 900px) {
  .shell {
    grid-template-columns: minmax(0, 1fr);
  }

  .sidebar {
    position: static;
    height: auto;
    border-right: 0;
    border-bottom: 1px solid var(--line);
    padding: var(--space-3) var(--space-4);
    flex-direction: row;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-3);
  }

  .brand-name {
    font-size: var(--text-sm);
  }

  .nav {
    flex-direction: row;
    overflow-x: auto;
    width: 100%;
    padding-bottom: var(--space-1);
  }

  .nav-item {
    white-space: nowrap;
  }

  .sidebar-foot {
    display: none;
  }

  .view {
    padding: var(--space-5) var(--space-4);
  }
}

@media (max-width: 760px) {
  .table-wrap {
    background: transparent;
    border: 0;
    box-shadow: none;
    overflow: visible;
  }

  .table, .table tbody, .table tr, .table td {
    display: block;
    width: 100%;
  }

  .table thead {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
  }

  .table tr {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    margin-bottom: var(--space-3);
    padding: var(--space-2) var(--space-3);
    box-shadow: var(--shadow-sm);
  }

  .table td {
    display: flex;
    justify-content: space-between;
    gap: var(--space-3);
    border: 0;
    padding: var(--space-2) 0;
    text-align: right;
  }

  .table td::before {
    content: attr(data-label);
    color: var(--ink-faint);
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    text-align: left;
  }

  .table td:first-child {
    font-size: var(--text-md);
  }

  .row-actions {
    justify-content: flex-end;
  }

  .bar-row {
    grid-template-columns: 90px 1fr 40px;
  }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.001ms !important;
    scroll-behavior: auto !important;
  }
}
"""

APP_JS = r"""/* Blueprint app frontend: no build step, no dependencies, textContent only. */
(function () {
  "use strict";

  var config = window.APP_CONFIG;
  var state = {
    token: window.localStorage.getItem(config.tokenKey),
    viewer: null,
    route: null,
    search: {},
    requestId: 0
  };

  var ICONS = @@ICONS@@;

  function el(tag, attrs) {
    var node = document.createElement(tag);
    var settings = attrs || {};
    Object.keys(settings).forEach(function (key) {
      var value = settings[key];
      if (value === null || value === undefined || value === false) {
        return;
      }
      if (key === "class") {
        node.className = value;
      } else if (key === "text") {
        node.textContent = String(value);
      } else if (key === "value") {
        node.value = value;
      } else if (key.indexOf("on") === 0 && typeof value === "function") {
        node.addEventListener(key.slice(2).toLowerCase(), value);
      } else if (key === "dataset") {
        Object.keys(value).forEach(function (name) {
          node.dataset[name] = value[name];
        });
      } else if (value === true) {
        node.setAttribute(key, "");
      } else {
        node.setAttribute(key, String(value));
      }
    });
    var children = Array.prototype.slice.call(arguments, 2);
    children.forEach(function append(child) {
      if (child === null || child === undefined || child === false) {
        return;
      }
      if (Array.isArray(child)) {
        child.forEach(append);
        return;
      }
      node.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
    });
    return node;
  }

  function icon(name) {
    return el("span", { class: "icon", "aria-hidden": "true", text: ICONS[name] || ICONS.dot });
  }

  function titleCase(value) {
    return String(value || "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, function (letter) { return letter.toUpperCase(); });
  }

  function describeError(detail) {
    if (!detail) {
      return "Something went wrong. Try again.";
    }
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail)) {
      return detail
        .map(function (item) {
          var where = Array.isArray(item.loc) ? item.loc[item.loc.length - 1] : "";
          return (where ? where + ": " : "") + (item.msg || "invalid value");
        })
        .join("; ");
    }
    return String(detail);
  }

  function ApiError(status, detail) {
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.message = describeError(detail);
  }
  ApiError.prototype = Object.create(Error.prototype);

  function api(path, options) {
    var settings = options || {};
    var headers = Object.assign({ "Content-Type": "application/json" }, settings.headers || {});
    if (state.token) {
      headers.Authorization = "Bearer " + state.token;
    }
    return fetch("/api" + path, Object.assign({}, settings, { headers: headers })).then(function (response) {
      if (response.status === 401 && state.token) {
        state.token = null;
        window.localStorage.removeItem(config.tokenKey);
      }
      if (response.status === 204) {
        return null;
      }
      return response.json().catch(function () { return {}; }).then(function (data) {
        if (!response.ok) {
          throw new ApiError(response.status, data.detail || "Request failed (" + response.status + ").");
        }
        return data;
      });
    });
  }

  function authRequest(kind, email, password) {
    return fetch("/auth/" + kind, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: email, password: password })
    }).then(function (response) {
      return response.json().catch(function () { return {}; }).then(function (data) {
        if (!response.ok) {
          throw new ApiError(response.status, data.detail || "Could not sign you in.");
        }
        return data;
      });
    });
  }

  var toastTimer = null;
  function toast(message, kind) {
    var node = document.getElementById("toast");
    node.textContent = message;
    node.dataset.kind = kind || "ok";
    node.dataset.visible = "true";
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(function () {
      node.dataset.visible = "false";
    }, 3600);
  }

  function navigate(hash) {
    window.location.hash = hash;
  }

  function currentRoute() {
    var raw = window.location.hash.replace(/^#\/?/, "");
    var parts = raw.split("/");
    var first = parts[0];
    var second = parts[1];
    if (first === "overview") {
      return { view: "overview" };
    }
    if (first === "shared") {
      return { view: "shared" };
    }
    var entity = config.entities.filter(function (item) { return item.plural === first; })[0];
    if (entity) {
      return { view: "entity", entity: entity, id: second || null };
    }
    return { view: "entity", entity: config.entities[0], id: null };
  }

  function entityByPlural(plural) {
    return config.entities.filter(function (item) { return item.plural === plural; })[0];
  }

  function fieldByKey(entity, key) {
    return entity.fields.filter(function (item) { return item.key === key; })[0];
  }

  function labelFor(entity, key) {
    if (key === entity.display) {
      return entity.label;
    }
    var field = fieldByKey(entity, key);
    return field ? field.label : titleCase(key);
  }

  function toneFor(value) {
    var text = String(value || "").toLowerCase();
    if (["done", "yes", "won", "active", "hired", "delivered", "fixed", "paid", "published", "true", "great", "1"].indexOf(text) !== -1) {
      return "positive";
    }
    if (["high", "critical", "rejected", "lost", "danger", "blocked", "void"].indexOf(text) !== -1) {
      return "danger";
    }
    if (["doing", "in_progress", "interview", "pending", "waiting", "medium", "maybe", "review", "scheduled"].indexOf(text) !== -1) {
      return "warn";
    }
    return "neutral";
  }

  function formatValue(entity, row, key) {
    var field = fieldByKey(entity, key);
    var value = row[key];
    if (value === null || value === undefined || value === "") {
      return el("span", { class: "cell-muted", text: "—" });
    }
    if (!field) {
      return el("span", { text: String(value) });
    }
    if (field.type === "bool") {
      var positive = value === 1 || value === true;
      return el("span", { class: "pill", dataset: { tone: positive ? "positive" : "neutral" }, text: positive ? "Yes" : "No" });
    }
    if (field.type === "enum") {
      return el("span", { class: "badge", dataset: { tone: toneFor(value) }, text: titleCase(value) });
    }
    if (field.type === "ref") {
      var label = row[field.key.replace(/_id$/, "") + "_label"];
      return el("span", { text: label || String(value) });
    }
    if (field.type === "timestamp") {
      var parsed = new Date(value);
      return el("span", { class: "cell-muted", text: isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString() });
    }
    if (field.type === "date") {
      return el("span", { class: "cell-muted", text: String(value) });
    }
    if (field.type === "int") {
      return el("span", { text: Number(value).toLocaleString() });
    }
    var text = String(value);
    return el("span", { text: text.length > 90 ? text.slice(0, 87) + "..." : text });
  }

  function errorBanner(error, retry) {
    return el(
      "div",
      { class: "error-banner", role: "alert" },
      el("div", {},
        el("strong", { text: error.status === 0 ? "Network problem" : "Could not load this view" }),
        el("p", { text: describeError(error.detail || error.message) })
      ),
      el("button", { class: "btn btn-sm", type: "button", onclick: retry, text: "Retry" })
    );
  }

  function emptyState(entity, onCreate) {
    return el(
      "div",
      { class: "empty-state" },
      el("span", { class: "empty-mark", text: ICONS[entity.icon] || ICONS.dot }),
      el("h2", { text: entity.empty.title }),
      el("p", { text: entity.empty.body }),
      el("button", { class: "btn btn-primary", type: "button", onclick: onCreate, text: entity.empty.action })
    );
  }

  function skeletonRows() {
    var wrap = el("div", { class: "skeleton-stack", "aria-hidden": "true" });
    for (var index = 0; index < 4; index += 1) {
      wrap.appendChild(el("div", { class: "skeleton skeleton-line" }));
    }
    wrap.appendChild(el("div", { class: "skeleton skeleton-block" }));
    return wrap;
  }

  function renderNav(route) {
    var nav = document.getElementById("nav");
    nav.replaceChildren();
    config.entities.forEach(function (entity) {
      var active = route.view === "entity" && route.entity.plural === entity.plural;
      nav.appendChild(
        el("a", {
          class: "nav-item",
          href: "#/" + entity.plural,
          "aria-current": active ? "page" : null
        }, icon(entity.icon), el("span", { text: entity.pluralLabel }))
      );
    });
    if (config.features.kind === "stats") {
      nav.appendChild(
        el("a", {
          class: "nav-item",
          href: "#/overview",
          "aria-current": route.view === "overview" ? "page" : null
        }, icon("chart"), el("span", { text: "Overview" }))
      );
    }
    if (config.features.kind === "team") {
      nav.appendChild(
        el("a", {
          class: "nav-item",
          href: "#/shared",
          "aria-current": route.view === "shared" ? "page" : null
        }, icon("people"), el("span", { text: "Shared with me" }))
      );
    }
    var foot = document.getElementById("sidebar-foot");
    foot.replaceChildren();
    if (config.auth && state.viewer) {
      foot.appendChild(el("span", { text: state.viewer.email }));
      foot.appendChild(
        el("button", {
          class: "btn btn-sm",
          type: "button",
          text: "Sign out",
          onclick: function () {
            state.token = null;
            window.localStorage.removeItem(config.tokenKey);
            state.viewer = null;
            renderAuth("Signed out.");
          }
        })
      );
    } else {
      foot.appendChild(el("span", { class: "badge", dataset: { tone: "accent" }, text: "Local mode" }));
      foot.appendChild(el("span", { text: "No account needed." }));
    }
  }

  function openModal(title, subtitle, content) {
    var root = document.getElementById("modal-root");
    var dialog = el("dialog", { class: "dialog", "aria-label": title });
    var close = el("button", { class: "icon-button", type: "button", "aria-label": "Close", text: "×" });
    close.addEventListener("click", function () { dialog.close(); });
    dialog.appendChild(
      el("div", { class: "dialog-head" },
        el("div", {}, el("h2", { text: title }), subtitle ? el("p", { text: subtitle }) : null),
        close
      )
    );
    dialog.appendChild(content);
    dialog.addEventListener("close", function () { dialog.remove(); });
    root.appendChild(dialog);
    if (typeof dialog.showModal === "function") {
      dialog.showModal();
    } else {
      dialog.setAttribute("open", "");
    }
    return dialog;
  }

  function toLocalInput(value) {
    if (!value) {
      return "";
    }
    var date = new Date(value);
    if (isNaN(date.getTime())) {
      return "";
    }
    var offset = date.getTime() - date.getTimezoneOffset() * 60000;
    return new Date(offset).toISOString().slice(0, 16);
  }

  function buildControl(entity, field, value) {
    if (field.type === "bool") {
      var box = el("input", { type: "checkbox", name: field.key, id: "field-" + field.key });
      box.checked = value === 1 || value === true;
      return el("div", { class: "check" }, box, el("label", { for: "field-" + field.key, text: field.label }));
    }
    if (field.type === "enum") {
      var select = el("select", { name: field.key, id: "field-" + field.key, required: field.required });
      field.choices.forEach(function (choice) {
        var option = el("option", { value: choice, text: titleCase(choice) });
        if (String(value || "") === String(choice)) {
          option.selected = true;
        }
        select.appendChild(option);
      });
      return select;
    }
    if (field.type === "longtext") {
      var area = el("textarea", { name: field.key, id: "field-" + field.key, maxlength: field.maxLength || null });
      area.value = value || "";
      return area;
    }
    if (field.type === "ref") {
      return el("select", { name: field.key, id: "field-" + field.key, required: field.required, "data-ref": field.refPlural });
    }
    var type = "text";
    if (field.type === "int") { type = "number"; }
    if (field.type === "date") { type = "date"; }
    if (field.type === "timestamp") { type = "datetime-local"; }
    var input = el("input", {
      type: type,
      name: field.key,
      id: "field-" + field.key,
      required: field.required,
      min: field.min === null ? null : field.min,
      max: field.max === null ? null : field.max,
      maxlength: field.maxLength || null,
      autocomplete: "off"
    });
    input.value = field.type === "timestamp" ? toLocalInput(value) : (value || "");
    return input;
  }

  function fillRefOptions(form, entity) {
    var refs = entity.fields.filter(function (field) { return field.type === "ref"; });
    return Promise.all(refs.map(function (field) {
      var select = form.querySelector('[name="' + field.key + '"]');
      if (!select) {
        return Promise.resolve();
      }
      return api("/" + field.refPlural + "?limit=200").then(function (rows) {
        var current = select.value;
        select.replaceChildren();
        select.appendChild(el("option", { value: "", text: "Choose " + field.refLabel.toLowerCase() + "..." }));
        rows.forEach(function (row) {
          select.appendChild(el("option", { value: row.id, text: row[field.refDisplay] || row.id }));
        });
        if (current) {
          select.value = current;
        }
      }).catch(function (error) {
        toast("Could not load " + field.refLabel + " options: " + error.message, "error");
      });
    }));
  }

  function openForm(entity, record) {
    var editing = Boolean(record);
    var form = el("form", { class: "form", novalidate: "novalidate" });
    var errors = el("div", { class: "form-errors", role: "alert" });
    form.appendChild(errors);
    entity.fields.forEach(function (field) {
      var value = record ? record[field.key] : null;
      var control = buildControl(entity, field, value);
      if (field.type !== "bool") {
        var fieldNode = el("div", { class: "field" },
          el("label", { for: "field-" + field.key },
            field.label,
            field.required ? el("span", { class: "muted", text: " *" }) : null
          ),
          control,
          field.help ? el("span", { class: "hint", text: field.help }) : null,
          el("span", { class: "field-error", id: "error-" + field.key })
        );
        form.appendChild(fieldNode);
      } else {
        form.appendChild(el("div", { class: "field" }, control, el("span", { class: "field-error", id: "error-" + field.key })));
      }
    });
    var submit = el("button", { class: "btn btn-primary", type: "submit", text: editing ? "Save changes" : "Create " + entity.label.toLowerCase() });
    var cancel = el("button", { class: "btn", type: "button", text: "Cancel", onclick: function () { dialog.close(); } });
    form.appendChild(el("div", { class: "form-actions" }, cancel, submit));
    var dialog = openModal(editing ? "Edit " + entity.label.toLowerCase() : "New " + entity.label.toLowerCase(), entity.empty.body, form);
    fillRefOptions(form, entity);
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      errors.textContent = "";
      entity.fields.forEach(function (field) {
        var slot = document.getElementById("error-" + field.key);
        if (slot) { slot.textContent = ""; }
        var control = form.querySelector('[name="' + field.key + '"]');
        if (control) { control.removeAttribute("aria-invalid"); }
      });
      var payload = {};
      entity.fields.forEach(function (field) {
        var control = form.querySelector('[name="' + field.key + '"]');
        if (!control) { return; }
        if (field.type === "bool") {
          payload[field.key] = control.checked;
        } else if (field.type === "timestamp") {
          if (control.value) { payload[field.key] = new Date(control.value).toISOString(); }
          else if (!field.required) { payload[field.key] = null; }
        } else if (field.type === "int") {
          if (control.value !== "") { payload[field.key] = Number(control.value); }
        } else if (control.value !== "") {
          payload[field.key] = control.value;
        }
      });
      submit.disabled = true;
      var request = editing
        ? api("/" + entity.plural + "/" + record.id, { method: "PATCH", body: JSON.stringify(payload) })
        : api("/" + entity.plural, { method: "POST", body: JSON.stringify(payload) });
      request.then(function () {
        dialog.close();
        toast(editing ? entity.label + " updated" : entity.label + " created");
        render();
      }).catch(function (error) {
        submit.disabled = false;
        errors.textContent = error.message;
        if (Array.isArray(error.detail)) {
          error.detail.forEach(function (item) {
            var where = Array.isArray(item.loc) ? item.loc[item.loc.length - 1] : "";
            var slot = document.getElementById("error-" + where);
            if (slot) {
              slot.textContent = item.msg || "invalid value";
              var control = form.querySelector('[name="' + where + '"]');
              if (control) { control.setAttribute("aria-invalid", "true"); }
            }
          });
        }
      });
    });
  }

  function deleteRecord(entity, record) {
    var label = record[entity.display] || entity.label;
    if (!window.confirm("Delete " + label + "? This cannot be undone.")) {
      return;
    }
    api("/" + entity.plural + "/" + record.id, { method: "DELETE" }).then(function () {
      toast(entity.label + " deleted");
      render();
    }).catch(function (error) {
      toast(error.message, "error");
    });
  }

  function shareRecord(entity, record) {
    var form = el("form", { class: "form" });
    var input = el("input", { type: "email", name: "email", id: "share-email", required: true, placeholder: "teammate@example.com" });
    var errors = el("div", { class: "form-errors", role: "alert" });
    var submit = el("button", { class: "btn btn-primary", type: "submit", text: "Share" });
    form.appendChild(el("div", { class: "field" },
      el("label", { for: "share-email", text: "Teammate email" }),
      input,
      el("span", { class: "hint", text: "They will be able to read and edit this item, but not delete it." })
    ));
    form.appendChild(errors);
    form.appendChild(el("div", { class: "form-actions" },
      el("button", { class: "btn", type: "button", text: "Cancel", onclick: function () { dialog.close(); } }),
      submit
    ));
    var dialog = openModal("Share " + entity.label.toLowerCase(), "Sharing is per item and never grants deletion.", form);
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      submit.disabled = true;
      api("/" + entity.plural + "/" + record.id + "/share", {
        method: "POST",
        body: JSON.stringify({ email: input.value.trim() })
      }).then(function () {
        dialog.close();
        toast("Shared with " + input.value.trim());
        render();
      }).catch(function (error) {
        submit.disabled = false;
        errors.textContent = error.message;
      });
    });
  }

  function rowActions(entity, record) {
    var actions = el("div", { class: "row-actions" },
      el("button", { class: "btn btn-sm", type: "button", text: "Edit", onclick: function () { openForm(entity, record); } })
    );
    if (config.features.kind === "team" && entity.plural === config.features.entity) {
      actions.appendChild(el("button", { class: "btn btn-sm", type: "button", text: "Share", onclick: function () { shareRecord(entity, record); } }));
    }
    actions.appendChild(el("button", { class: "btn btn-sm btn-danger", type: "button", text: "Delete", onclick: function () { deleteRecord(entity, record); } }));
    return actions;
  }

  function dataTable(entity, rows) {
    var head = el("tr", {}, el("th", { scope: "col", text: entity.label }));
    entity.list.forEach(function (key) {
      if (key !== entity.display) {
        head.appendChild(el("th", { scope: "col", text: labelFor(entity, key) }));
      }
    });
    head.appendChild(el("th", { scope: "col", class: "sr-only", text: "Actions" }));
    var body = el("tbody");
    rows.forEach(function (row) {
      var tr = el("tr", {});
      tr.appendChild(el("td", { "data-label": entity.label }, el("span", { class: "cell-strong", text: row[entity.display] || "Untitled" })));
      entity.list.forEach(function (key) {
        if (key !== entity.display) {
          tr.appendChild(el("td", { "data-label": labelFor(entity, key) }, formatValue(entity, row, key)));
        }
      });
      tr.appendChild(el("td", { "data-label": "Actions" }, rowActions(entity, row)));
      body.appendChild(tr);
    });
    return el("div", { class: "table-wrap" }, el("table", { class: "table" }, el("thead", {}, head), body));
  }

  function loadTable(host, entity, query, offset, append) {
    var requestId = ++state.requestId;
    if (!append) {
      host.replaceChildren(skeletonRows());
    }
    var path = "/" + entity.plural + "?limit=50&offset=" + (offset || 0);
    if (query) {
      path += "&q=" + encodeURIComponent(query);
    }
    api(path).then(function (rows) {
      if (requestId !== state.requestId) {
        return;
      }
      var table = rows.length ? dataTable(entity, rows) : emptyState(entity, function () { openForm(entity, null); });
      if (append) {
        host.replaceChildren(table);
      } else {
        host.replaceChildren(table);
      }
      if (rows.length === 50) {
        var more = el("button", {
          class: "btn btn-sm",
          type: "button",
          text: "Load more",
          style: "margin-top:12px",
          onclick: function () { loadTable(host, entity, query, (offset || 0) + 50, true); }
        });
        host.appendChild(more);
      }
    }).catch(function (error) {
      if (requestId !== state.requestId) {
        return;
      }
      host.replaceChildren(errorBanner(error, function () { loadTable(host, entity, query, offset, append); }));
    });
  }

  function loadBoard(host, entity) {
    host.replaceChildren(skeletonRows());
    api("/" + entity.plural + "?limit=200").then(function (rows) {
      if (!rows.length) {
        host.replaceChildren(emptyState(entity, function () { openForm(entity, null); }));
        return;
      }
      var fieldKey = config.features.field;
      var board = el("div", { class: "board" });
      config.features.columns.forEach(function (column) {
        var cards = rows.filter(function (row) { return String(row[fieldKey]) === String(column.value); });
        var columnNode = el("section", { class: "board-column", "aria-label": column.label });
        columnNode.appendChild(el("h2", {}, el("span", { text: column.label }), el("span", { class: "badge", text: String(cards.length) })));
        if (!cards.length) {
          columnNode.appendChild(el("p", { class: "board-empty", text: "Nothing in " + column.label.toLowerCase() + " yet." }));
        }
        cards.forEach(function (record) {
          var card = el("article", { class: "board-card" });
          card.appendChild(el("h3", { text: record[entity.display] || "Untitled" }));
          var meta = el("div", { class: "meta" });
          entity.list.forEach(function (key) {
            if (key === entity.display) { return; }
            var value = record[key];
            if (value === null || value === undefined || value === "") { return; }
            meta.appendChild(formatValue(entity, record, key));
          });
          card.appendChild(meta);
          var move = el("select", { "aria-label": "Move " + (record[entity.display] || "card") });
          config.features.columns.forEach(function (option) {
            var optionNode = el("option", { value: option.value, text: option.label });
            if (String(option.value) === String(record[fieldKey])) { optionNode.selected = true; }
            move.appendChild(optionNode);
          });
          move.addEventListener("change", function () {
            api("/" + entity.plural + "/" + record.id + "/status", {
              method: "PATCH",
              body: JSON.stringify({ status: move.value })
            }).then(function () {
              toast("Moved to " + titleCase(move.value));
              loadBoard(host, entity);
            }).catch(function (error) {
              toast(error.message, "error");
              loadBoard(host, entity);
            });
          });
          card.appendChild(move);
          card.appendChild(rowActions(entity, record));
          columnNode.appendChild(card);
        });
        board.appendChild(columnNode);
      });
      host.replaceChildren(board);
    }).catch(function (error) {
      host.replaceChildren(errorBanner(error, function () { loadBoard(host, entity); }));
    });
  }

  function renderEntity(route) {
    var entity = route.entity;
    var view = document.getElementById("view");
    view.replaceChildren();
    var head = el("header", { class: "page-head" },
      el("div", {},
        el("h1", { text: entity.pluralLabel }),
        el("p", { text: "Create, search and edit " + entity.pluralLabel.toLowerCase() + ". Every change is scoped to your account." })
      ),
      el("button", { class: "btn btn-primary", type: "button", text: "New " + entity.label.toLowerCase(), onclick: function () { openForm(entity, null); } })
    );
    var searchInput = el("input", {
      type: "search",
      placeholder: "Search " + entity.pluralLabel.toLowerCase(),
      "aria-label": "Search " + entity.pluralLabel.toLowerCase(),
      value: state.search[entity.plural] || ""
    });
    searchInput.addEventListener("input", function () {
      state.search[entity.plural] = searchInput.value;
      window.clearTimeout(searchInput._timer);
      searchInput._timer = window.setTimeout(function () {
        loadTable(host, entity, searchInput.value, 0, false);
      }, 220);
    });
    var toolbar = el("div", { class: "toolbar" },
      el("div", { class: "search" }, searchInput),
      el("span", { class: "muted", text: "Showing up to 50 per page" })
    );
    var host = el("div", {});
    view.appendChild(head);
    view.appendChild(toolbar);
    view.appendChild(host);
    if (config.features.kind === "board" && entity.plural === config.features.entity) {
      loadBoard(host, entity);
    } else {
      loadTable(host, entity, state.search[entity.plural] || "", 0, false);
    }
  }

  function renderShared() {
    var entity = entityByPlural(config.features.entity);
    var view = document.getElementById("view");
    view.replaceChildren();
    view.appendChild(el("header", { class: "page-head" },
      el("div", {},
        el("h1", { text: "Shared with me" }),
        el("p", { text: "Items other people have shared with you. You can edit them, but only the owner can delete them." })
      )
    ));
    var host = el("div", {});
    view.appendChild(host);
    host.replaceChildren(skeletonRows());
    api("/shared").then(function (rows) {
      host.replaceChildren(rows.length ? dataTable(entity, rows) : el("div", { class: "empty-state" },
        el("span", { class: "empty-mark", text: ICONS.people }),
        el("h2", { text: "Nothing shared yet" }),
        el("p", { text: "When a teammate shares an item with your email, it appears here." })
      ));
    }).catch(function (error) {
      host.replaceChildren(errorBanner(error, renderShared));
    });
  }

  function renderOverview() {
    var entity = entityByPlural(config.features.entity);
    var view = document.getElementById("view");
    view.replaceChildren();
    view.appendChild(el("header", { class: "page-head" },
      el("div", {},
        el("h1", { text: "Overview" }),
        el("p", { text: "Totals and trends computed directly from your data, not cached somewhere else." })
      )
    ));
    var host = el("div", {});
    view.appendChild(host);
    host.replaceChildren(skeletonRows());
    api("/" + entity.plural + "/stats").then(function (data) {
      var grid = el("div", { class: "stats-grid" });
      grid.appendChild(el("div", { class: "stat-card" },
        el("span", { class: "label", text: "Total " + entity.pluralLabel.toLowerCase() }),
        el("span", { class: "value", text: Number(data.total || 0).toLocaleString() })
      ));
      if (data.sum_field) {
        grid.appendChild(el("div", { class: "stat-card" },
          el("span", { class: "label", text: titleCase(data.sum_field) + " (total)" }),
          el("span", { class: "value", text: Number(data.sum_value || 0).toLocaleString() })
        ));
      }
      grid.appendChild(el("div", { class: "stat-card" },
        el("span", { class: "label", text: "Groups" }),
        el("span", { class: "value", text: String((data.by_group || []).length) })
      ));
      var panel = el("section", { class: "panel" });
      panel.appendChild(el("h2", { text: "By " + titleCase(data.group_field || "group") }));
      var counts = (data.by_group || []).map(function (row) { return Number(row.count || 0); });
      var max = Math.max.apply(null, counts.concat([1]));
      (data.by_group || []).forEach(function (row) {
        panel.appendChild(el("div", { class: "bar-row" },
          el("span", { text: titleCase(row.key || "unknown") }),
          el("div", { class: "bar-track" }, el("div", { class: "bar-fill", style: "width:" + Math.round((Number(row.count) / max) * 100) + "%" })),
          el("span", { class: "muted", text: String(row.count) })
        ));
      });
      var trendPanel = el("section", { class: "panel" });
      trendPanel.appendChild(el("h2", { text: "Last 14 days" }));
      var trend = (data.trend || []).slice().reverse();
      var trendMax = Math.max.apply(null, trend.map(function (row) { return Number(row.count || 0); }).concat([1]));
      var chart = el("div", { class: "trend" });
      trend.forEach(function (row) {
        var bar = el("div", { class: "trend-bar", style: "height:" + Math.max(6, Math.round((Number(row.count) / trendMax) * 100)) + "%", title: row.day + ": " + row.count });
        chart.appendChild(bar);
      });
      trendPanel.appendChild(chart);
      var host2 = el("div", { class: "stack" }, grid, panel, trendPanel);
      host.replaceChildren(host2);
    }).catch(function (error) {
      host.replaceChildren(errorBanner(error, renderOverview));
    });
  }

  function renderAuth(message) {
    document.getElementById("view").replaceChildren();
    var root = el("div", { class: "auth" });
    var mode = "login";
    var errors = el("div", { class: "form-errors", role: "alert", text: message || "" });
    var email = el("input", { type: "email", id: "auth-email", name: "email", autocomplete: "email", required: true });
    var password = el("input", { type: "password", id: "auth-password", name: "password", autocomplete: "current-password", required: true, minlength: 8 });
    var submit = el("button", { class: "btn btn-primary", type: "submit", text: "Sign in" });
    var loginTab = el("button", { type: "button", role: "tab", "aria-selected": "true", text: "Sign in" });
    var registerTab = el("button", { type: "button", role: "tab", "aria-selected": "false", text: "Create account" });
    function setMode(next) {
      mode = next;
      loginTab.setAttribute("aria-selected", next === "login" ? "true" : "false");
      registerTab.setAttribute("aria-selected", next === "register" ? "true" : "false");
      submit.textContent = next === "login" ? "Sign in" : "Create account";
      password.autocomplete = next === "login" ? "current-password" : "new-password";
    }
    loginTab.addEventListener("click", function () { setMode("login"); });
    registerTab.addEventListener("click", function () { setMode("register"); });
    var form = el("form", { class: "form" },
      el("div", { class: "tabs", role: "tablist" }, loginTab, registerTab),
      errors,
      el("div", { class: "field" }, el("label", { for: "auth-email", text: "Email" }), email),
      el("div", { class: "field" }, el("label", { for: "auth-password", text: "Password" }), password,
        el("span", { class: "hint", text: "At least 8 characters. Stored as a scrypt hash, never in plain text." })),
      submit
    );
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      submit.disabled = true;
      errors.textContent = "";
      authRequest(mode === "login" ? "login" : "register", email.value.trim(), password.value).then(function (data) {
        state.token = data.token;
        window.localStorage.setItem(config.tokenKey, data.token);
        return bootstrap();
      }).catch(function (error) {
        submit.disabled = false;
        errors.textContent = error.message;
      });
    });
    var card = el("div", { class: "auth-card" },
      el("div", { class: "brand" }, el("span", { class: "brand-mark", text: ICONS[config.entities[0].icon] || ICONS.dot }), el("span", { class: "brand-name", text: config.title })),
      el("h1", { text: mode === "login" ? "Welcome back" : "Create your account" }),
      el("p", { class: "muted", text: "Your data stays in this app's database. No third-party identity provider is in the loop." }),
      form
    );
    root.appendChild(card);
    document.querySelector(".shell").replaceChildren(root);
  }

  function render() {
    var route = currentRoute();
    state.route = route;
    renderNav(route);
    if (route.view === "overview") {
      renderOverview();
    } else if (route.view === "shared") {
      renderShared();
    } else {
      renderEntity(route);
    }
    document.getElementById("view").focus();
  }

  function bootstrap() {
    if (config.auth && !state.token) {
      renderAuth("");
      return Promise.resolve();
    }
    return api("/me").then(function (viewer) {
      state.viewer = viewer;
      render();
    }).catch(function (error) {
      if (config.auth && error.status === 401) {
        renderAuth("Sign in to continue.");
        return;
      }
      renderAuth(error.message);
    });
  }

  window.addEventListener("hashchange", render);
  document.addEventListener("DOMContentLoaded", bootstrap);
  if (document.readyState !== "loading") {
    bootstrap();
  }
})();
"""


def render_web_assets(
    blueprint: AppBlueprint,
    module: str,
    title: str,
    auth: bool,
    accent: str,
) -> dict[str, str]:
    """Return ``web/`` assets for one blueprint."""

    config = app_config(blueprint, module, title, auth, accent)
    config_js = "/* Generated from the application blueprint. */\nwindow.APP_CONFIG = " + json.dumps(
        config, indent=2
    ) + ";\n"
    index = (
        INDEX_TMPL.replace("@@ACCENT@@", accent)
        .replace("@@DESCRIPTION@@", blueprint.prompt.replace('"', "&quot;"))
        .replace("@@TITLE@@", title)
        .replace("@@MARK@@", ICON_GLYPHS.get(blueprint.primary.icon, ICON_GLYPHS["dot"]))
    )
    icons = json.dumps(ICON_GLYPHS, ensure_ascii=False)
    return {
        "web/index.html": index,
        "web/app.css": APP_CSS.replace("@@ACCENT@@", accent),
        "web/config.js": config_js,
        "web/app.js": APP_JS.replace("@@ICONS@@", icons),
    }
