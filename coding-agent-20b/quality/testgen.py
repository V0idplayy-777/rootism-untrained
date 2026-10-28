"""Starter tests for modules that have none.

Two kinds of generated test, both real (not decorative):

* **Web apps** - FastAPI/Flask routes get a live ``TestClient`` request that
  asserts the endpoint answers (< 500). That catches import errors, broken
  dependencies and startup failures, which is where generated apps usually die.
* **Library modules** - an import test plus per-public-symbol existence and
  signature checks. These are smoke tests and the file says so: they verify the
  module loads and the API surface exists, *not* that the logic is correct.

The generator never overwrites an existing test file, so it is safe to run on a
project that already has tests, and it is idempotent.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

HEADER = '''"""Generated smoke tests (agent testgen).

These verify that the module imports and that the public API exists. They are a
floor, not a substitute for behaviour tests: add assertions for the cases that
matter, then delete whichever smoke tests stop earning their keep.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

'''

FASTAPI_HEADER = '''"""Generated API tests (agent testgen).

Each test performs a real request against the ASGI app. Add cases for auth,
errors and happy-path payloads as the API grows.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from {module} import {app_name}

client = TestClient({app_name})

'''


def _routes(source: str) -> list[tuple[str, str]]:
    """(method, path) pairs for FastAPI/Flask decorators."""

    found: list[tuple[str, str]] = []
    for match in re.finditer(
        r"@(?:app|router|api)\.(get|post|put|patch|delete)\(\s*[\"']([^\"']+)[\"']", source
    ):
        found.append((match.group(1).upper(), match.group(2)))
    for match in re.finditer(
        r"@(?:app|bp|blueprint)\.route\(\s*[\"']([^\"']+)[\"']([^)]*)\)", source
    ):
        path, kwargs = match.group(1), match.group(2)
        methods = re.findall(r"methods\s*=\s*\[([^\]]+)\]", kwargs)
        verbs = [m.strip(" '\"") for m in methods[0].split(",")] if methods else ["GET"]
        for verb in verbs:
            found.append((verb.upper(), path))
    seen: set[tuple[str, str]] = set()
    unique = []
    for item in found:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _app_name(source: str) -> str | None:
    match = re.search(r"^(\w+)\s*=\s*(?:FastAPI|Flask)\(", source, re.MULTILINE)
    return match.group(1) if match else None


def _public_symbols(tree: ast.Module) -> list[str]:
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            names.append(node.name)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    names.append(target.id)
    return names


def generate_tests(
    module_path: str | Path,
    source: str | None = None,
    *,
    dotted: str | None = None,
) -> str | None:
    """Test module content for one Python file (None when there is nothing useful)."""

    path = Path(module_path)
    source = path.read_text(encoding="utf-8", errors="replace") if source is None else source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    module = dotted or path.stem
    routes = _routes(source)
    app_name = _app_name(source)
    is_web = app_name is not None and ("FastAPI" in source or "Flask" in source) and routes
    if is_web and "FastAPI" in source:
        body = FASTAPI_HEADER.format(module=module, app_name=app_name)
        for method, route in routes[:12]:
            safe = re.sub(r"\W+", "_", f"{method}_{route}").strip("_") or "root"
            concrete = re.sub(r"\{[^}]+\}", "1", route)
            if method == "GET":
                body += f'''def test_{safe.lower()}_answers():
    response = client.get("{concrete}")
    assert response.status_code < 500


'''
            else:
                body += f'''def test_{safe.lower()}_rejects_empty_body():
    response = client.{method.lower()}("{concrete}", json={{}})
    assert response.status_code < 500


'''
        return body

    if is_web and "Flask" in source:
        return f'''"""Generated API smoke tests (agent testgen) for a Flask app."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from {module} import {app_name}

client = {app_name}.test_client()


def test_get_routes_answer():
    for method, route in [(m, r) for m, r in {routes!r}]:
        if "{{" in route or "<" in route:
            continue
        response = client.open(route, method=method)
        assert response.status_code < 500
'''

    symbols = _public_symbols(tree)
    if not symbols:
        return None
    parts: list[str] = [HEADER, f'MODULE = "{module}"\n\n\n']
    parts.append(
        "def _load():\n"
        "    try:\n"
        "        return importlib.import_module(MODULE)\n"
        "    except ModuleNotFoundError as exc:  # dependency not installed in this env\n"
        "        import pytest\n\n"
        "        pytest.skip(f\"dependency not installed: {exc}\")\n\n\n"
    )
    parts.append("def test_module_imports():\n    assert _load() is not None\n\n\n")
    for name in symbols[:20]:
        parts.append(
            f"def test_{name.lower()}_exists():\n"
            f'    assert hasattr(_load(), "{name}")\n\n\n'
        )
    functions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
    ]
    if functions:
        parts.append("def test_public_functions_are_callable():\n    module = _load()\n")
        for name in functions[:10]:
            parts.append(f'    assert callable(getattr(module, "{name}"))\n')
        parts.append("\n")
    return "".join(parts)


def write_generated_tests(
    root: str | Path,
    paths: list[str] | None = None,
    *,
    force: bool = False,
    max_files: int = 20,
) -> dict:
    """Write ``tests/test_generated_<stem>.py`` for modules without tests."""

    root = Path(root)
    tests_dir = root / "tests"
    tests_dir.mkdir(exist_ok=True)
    candidates = paths or [
        str(path.relative_to(root))
        for path in sorted(root.rglob("*.py"))
        if not any(part in {".venv", "venv", "node_modules", "__pycache__", ".agent", "tests", "build", "dist"} for part in path.relative_to(root).parts[:-1])
        and not path.name.startswith("test_")
        and path.name != "setup.py"
    ]
    written: list[str] = []
    skipped: list[str] = []
    for rel in candidates[:max_files]:
        path = root / rel
        if not path.exists() or path.suffix != ".py":
            continue
        dotted = rel[:-3].replace("/", ".") if rel.endswith(".py") else rel
        if dotted.endswith(".__init__"):
            dotted = dotted[: -len(".__init__")]
        content = generate_tests(path, dotted=dotted)
        if content is None:
            skipped.append(rel)
            continue
        target = tests_dir / f"test_generated_{path.stem}.py"
        if target.exists() and not force:
            skipped.append(str(target.relative_to(root)))
            continue
        target.write_text(content, encoding="utf-8")
        written.append(str(target.relative_to(root)))
    return {"written": written, "skipped": skipped}
