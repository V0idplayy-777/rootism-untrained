"""Plausible-but-worse variants used for preference pairs and repair traces.

A preference pair is only worth training on if the rejected answer is something
a model would actually produce and the reason it is worse is *measurable*. Each
function here takes a rendered project and degrades exactly one dimension:

* :func:`apply_legacy_ui` - the 2018 Bootstrap look: fixed 960px container, no
  tokens, no responsive rules, no empty/loading/error states, no dark scheme and
  weakened accessibility. The screenshots differ; so do the :mod:`quality.taste`
  scores.
* :func:`apply_broken_owner_scope` - authentication that does not authorise:
  tokens are verified, but every query returns every user's rows. The generated
  isolation tests fail with real assertions.
* :func:`apply_incomplete_entities` - relations as loose id columns: no foreign
  keys, no cascade, no parent validation. Entity design looks fine until you
  delete a parent or paste a bogus id.

Nothing here is used by the default build path.
"""

from __future__ import annotations

import re

from .blueprints import AppBlueprint

LEGACY_CSS = r"""/* Bootstrap 3 era layout. Kept deliberately plain. */
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 13px;
  color: #333333;
  background-color: #f7f7f7;
  margin: 0;
}

.container {
  width: 960px;
  margin: 0 auto;
  padding: 20px;
}

h1 { font-size: 24px; margin: 0 0 10px; }
h2 { font-size: 18px; margin: 0 0 8px; }

.panel {
  background: #ffffff;
  border: 1px solid #dddddd;
  border-radius: 4px;
  padding: 15px;
  margin-bottom: 15px;
}

.btn {
  display: inline-block;
  padding: 6px 12px;
  background: #ffffff;
  border: 1px solid #cccccc;
  border-radius: 4px;
  color: #333333;
  cursor: pointer;
}

.btn-primary {
  background: #337ab7;
  border-color: #2e6da4;
  color: #ffffff;
}

.btn-danger {
  background: #d9534f;
  border-color: #d43f3a;
  color: #ffffff;
}

.btn-sm { padding: 3px 8px; font-size: 12px; }
.btn-ghost { background: transparent; }

.table {
  width: 100%;
  border-collapse: collapse;
  background: #ffffff;
}

.table th {
  background: #f0f0f0;
  border-bottom: 2px solid #dddddd;
  padding: 8px;
  text-align: left;
}

.table td {
  border-bottom: 1px solid #eeeeee;
  padding: 8px;
}

.table-wrap { border: 1px solid #dddddd; }

.badge, .pill {
  display: inline-block;
  padding: 2px 6px;
  background: #eeeeee;
  border-radius: 4px;
  font-size: 11px;
}

.icon { display: none; }
.muted, .cell-muted { color: #888888; }
.cell-strong { font-weight: bold; }
.row-actions { text-align: right; }

.empty-state {
  background: #ffffff;
  border: 1px solid #dddddd;
  padding: 20px;
  text-align: center;
}

.empty-mark { display: none; }

.toast {
  position: fixed;
  right: 20px;
  bottom: 20px;
  background: #333333;
  color: #ffffff;
  padding: 10px;
}

.dialog {
  width: 500px;
  border: 1px solid #999999;
  background: #ffffff;
  padding: 15px;
}

.field { margin-bottom: 10px; }
.field label { display: block; font-weight: bold; }
.field input, .field select, .field textarea {
  width: 100%;
  border: 1px solid #cccccc;
  padding: 4px;
}
.field textarea { height: 90px; }
.form-actions { text-align: right; }
.form-errors { color: #a94442; }
.field-error { color: #a94442; }
.hint { color: #999999; }

.auth {
  width: 420px;
  margin: 60px auto;
}

.auth-card {
  background: #ffffff;
  border: 1px solid #dddddd;
  padding: 20px;
}

.sidebar {
  background: #ffffff;
  border-bottom: 1px solid #dddddd;
  padding: 10px 20px;
}

.shell { width: 960px; margin: 0 auto; }
.view { padding: 20px; }
.nav { display: inline-block; margin-left: 20px; }
.nav-item { margin-right: 10px; color: #337ab7; }
.brand { font-weight: bold; }
.brand-mark { color: #337ab7; }
.sidebar-foot { display: none; }

.board { display: block; }
.board-column { display: inline-block; width: 30%; vertical-align: top; margin-right: 2%; }
.board-card { background: #ffffff; border: 1px solid #dddddd; padding: 8px; margin-bottom: 8px; }
.stats-grid { display: block; }
.stat-card { display: inline-block; width: 30%; border: 1px solid #dddddd; padding: 10px; margin-right: 2%; }
.bar-track { background: #eeeeee; height: 10px; }
.bar-fill { background: #337ab7; height: 10px; }
.trend { height: 60px; }
.trend-bar { display: inline-block; width: 6%; background: #337ab7; }
.skip-link, .skeleton, .skeleton-stack, .noscript { display: none; }
"""


def _legacy_index(index: str) -> str:
    index = index.replace('<html lang="en">', "<html>")
    index = index.replace('<meta name="viewport" content="width=device-width, initial-scale=1" />', "")
    index = index.replace('<meta name="theme-color" content="#4f46e5" />', "")
    index = index.replace('<a class="skip-link" href="#view">Skip to content</a>', "")
    index = re.sub(r"<noscript>.*?</noscript>", "", index, flags=re.S)
    index = index.replace('<div class="shell">', '<div class="shell"><div class="container">')
    index = index.replace("  </div>\n  <div class=\"toast\"", "  </div></div>\n  <div class=\"toast\"")
    return index


def apply_legacy_ui(files: dict[str, str], title: str) -> dict[str, str]:
    """Replace the design system and the shell with a 2018 template."""

    updated = dict(files)
    updated["web/app.css"] = LEGACY_CSS
    if "web/index.html" in updated:
        updated["web/index.html"] = _legacy_index(updated["web/index.html"])
    _ = title
    return updated


def apply_broken_owner_scope(
    files: dict[str, str], blueprint: AppBlueprint, *, auth: bool
) -> dict[str, str]:
    """Authenticate users but return every row to every user."""

    if not auth:
        return dict(files)
    updated = dict(files)
    module = next((path.split("/")[0] for path in files if path.endswith("/store.py")), "")
    if not module:
        return updated
    path = module + "/store.py"
    source = updated[path]
    replacement = (
        "def _scope(viewer: dict[str, str], alias: str = \"t\") -> tuple[str, list[Any]]:\n"
        '    """Authentication exists here; authorisation does not."""\n'
        "    _ = viewer, alias\n"
        "    return \"1 = 1\", []\n"
    )
    source = re.sub(
        r"def _scope\(viewer: dict\[str, str\], alias: str = \"t\"\) -> tuple\[str, list\[Any\]\]:\n(?:.|\n)*?\n\n\ndef ",
        replacement + "\n\ndef ",
        source,
        count=1,
    )
    updated[path] = source
    _ = blueprint
    return updated


def apply_incomplete_entities(files: dict[str, str], blueprint: AppBlueprint) -> dict[str, str]:
    """Drop foreign keys, relation indexes and parent validation."""

    updated = dict(files)
    for path, content in list(updated.items()):
        if path.endswith((".sql", "/db.py")):
            content = re.sub(r"\s+REFERENCES\s+\w+\(id\)\s+ON DELETE\s+\w+", "", content)
            content = re.sub(r"CREATE INDEX IF NOT EXISTS idx_\w+_\w+_id ON \w+\(\w+_id\);\n", "", content)
            updated[path] = content
    module = next((path.split("/")[0] for path in files if path.endswith("/store.py")), "")
    if module:
        path = module + "/store.py"
        source = updated[path]
        source = re.sub(r"^\s+_require_parent\(.*\)\n", "", source, flags=re.M)
        updated[path] = source
    return updated
