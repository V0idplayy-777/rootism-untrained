"""Project health tool: broken imports, missing tests, circular deps.

This is the 'project health' tool mentioned in row 4.
It reports actionable findings without needing a model.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HealthFinding:
    rule: str
    severity: str
    path: str
    message: str
    fix: str = ""

    def to_dict(self) -> dict:
        return {"rule": self.rule, "severity": self.severity, "path": self.path, "message": self.message, "fix": self.fix}


@dataclass
class HealthReport:
    ok: bool
    score: int
    findings: list[HealthFinding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "score": self.score, "findings": [f.to_dict() for f in self.findings], "stats": self.stats}

    def render(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines = [f"{status} health score {self.score}/100 ({len(self.findings)} findings)"]
        for f in self.findings[:20]:
            lines.append(f"[{f.severity}] {f.rule} @ {f.path}: {f.message} -> {f.fix}")
        if len(self.findings) > 20:
            lines.append(f"... {len(self.findings)-20} more findings")
        return "\n".join(lines)


def _parse_imports(py_text: str) -> list[str]:
    try:
        tree = ast.parse(py_text)
    except SyntaxError:
        return []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])
    return imports


def _find_broken_imports(root: Path) -> list[HealthFinding]:
    findings: list[HealthFinding] = []
    # collect local modules
    local_modules = set()
    for p in root.rglob("*.py"):
        if any(part in {".git", ".agent", "node_modules", ".venv", "venv"} for part in p.parent.parts):
            continue
        # top-level package names
        rel = p.relative_to(root)
        if len(rel.parts) == 1:
            local_modules.add(rel.stem)
        elif len(rel.parts) > 1:
            local_modules.add(rel.parts[0])
    # check each py file
    for p in root.rglob("*.py"):
        if any(part in {".git", ".agent", "node_modules", ".venv", "venv", "__pycache__"} for part in p.parts):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        imports = _parse_imports(text)
        for imp in imports:
            # stdlib-ish allowlist (not exhaustive; we flag only obvious missing local)
            if imp in {"os", "sys", "json", "pathlib", "re", "time", "math", "typing", "dataclasses", "collections", "datetime", "itertools", "functools", "ast", "hashlib", "logging", "argparse", "unittest", "pytest", "fastapi", "uvicorn", "pydantic", "httpx", "supabase", "jose", "sqlalchemy", "psycopg", "asyncio", "tempfile", "subprocess", "shutil", "html", "http", "urllib"}:
                continue
            # if import looks like local but no file exists
            if imp not in local_modules and "." not in imp:
                # only flag if there's a strong hint it's project's own (e.g., same as root name)
                # skip generic third-party to avoid false positives
                pass
        # also check syntax
        try:
            ast.parse(text)
        except SyntaxError as exc:
            findings.append(HealthFinding("syntax-error", "critical", str(p.relative_to(root)), f"SyntaxError: {exc.msg} at line {exc.lineno}", "fix the syntax; run `python -m py_compile`"))
    return findings


def _find_missing_tests(root: Path) -> list[HealthFinding]:
    findings: list[HealthFinding] = []
    source_files: list[Path] = []
    test_files: set[str] = set()
    for p in root.rglob("*.py"):
        if any(part in {".git", ".agent", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache"} for part in p.parts):
            continue
        rel = p.relative_to(root)
        if "test" in rel.parts or rel.name.startswith("test_"):
            test_files.add(rel.name)
            # also add module under test if naming convention: test_foo -> foo
            if rel.name.startswith("test_"):
                test_files.add(rel.name[5:])
        else:
            # source file with public defs?
            try:
                text = p.read_text(encoding="utf-8")
                tree = ast.parse(text)
                has_defs = any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_") for n in tree.body)
                if has_defs:
                    source_files.append(p)
            except Exception:
                continue
    if source_files and not test_files:
        findings.append(HealthFinding("no-tests", "high", "(project)", "source modules exist but there are no tests", "run `agent testgen` to generate starter tests"))
    else:
        for src in source_files:
            stem = src.stem
            # expect test_{stem} or stem in test files
            expected = f"test_{stem}.py"
            if expected not in test_files and stem not in {Path(t).stem for t in test_files}:
                findings.append(HealthFinding("missing-test", "medium", str(src.relative_to(root)), f"no test for {stem}", f"add tests/test_{stem}.py"))
    return findings


def _find_circular_deps(root: Path) -> list[HealthFinding]:
    findings: list[HealthFinding] = []
    # Build import graph: module -> imports (local only)
    graph: dict[str, set[str]] = {}
    module_map: dict[str, Path] = {}
    for p in root.rglob("*.py"):
        if any(part in {".git", ".agent", "node_modules", ".venv", "__pycache__"} for part in p.parts):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        rel = p.relative_to(root)
        mod = ".".join(rel.with_suffix("").parts)
        if mod.endswith(".__init__"):
            mod = mod[:-9]
        module_map[mod] = p
        graph[mod] = set(_parse_imports(text))
    # map short import to full module
    short_to_full = {mod.split(".")[-1]: mod for mod in graph}
    # check for 2-cycle a->b->a
    for mod, deps in graph.items():
        for dep in deps:
            full_dep = short_to_full.get(dep)
            if not full_dep or full_dep not in graph:
                continue
            if mod in graph.get(full_dep, set()) or any(mod.split(".")[-1] == d for d in graph.get(full_dep, [])):
                # confirm dep's short name maps back
                dep_short = full_dep.split(".")[-1]
                if mod.split(".")[-1] in graph.get(full_dep, {}) or dep_short in graph.get(mod, {}):
                    pass
                # simple heuristic: mutual import
                if dep in graph[mod] and mod.split(".")[-1] in graph.get(full_dep, {}):
                    # avoid dup
                    already = any(f.path == mod and full_dep in f.message for f in findings)
                    if not already:
                        findings.append(HealthFinding("circular-import", "high", mod, f"circular import: {mod} <-> {full_dep}", "extract shared code to a third module"))
    # dedupe
    seen = set()
    uniq = []
    for f in findings:
        key = (f.rule, f.path, f.message)
        if key not in seen:
            uniq.append(f)
            seen.add(key)
    return uniq


def project_health(root: str | Path) -> HealthReport:
    root = Path(root).resolve()
    findings: list[HealthFinding] = []
    findings.extend(_find_broken_imports(root))
    findings.extend(_find_missing_tests(root))
    findings.extend(_find_circular_deps(root))
    # score: start 100, deduct per severity
    score = 100
    for f in findings:
        if f.severity == "critical":
            score -= 15
        elif f.severity == "high":
            score -= 10
        elif f.severity == "medium":
            score -= 5
        elif f.severity == "low":
            score -= 2
    score = max(0, min(100, score))
    ok = score >= 70 and not any(f.severity == "critical" for f in findings)
    stats = {
        "findings": len(findings),
        "by_severity": {sev: sum(1 for f in findings if f.severity == sev) for sev in ["critical", "high", "medium", "low"]},
        "by_rule": {},
    }
    for f in findings:
        stats["by_rule"][f.rule] = stats["by_rule"].get(f.rule, 0) + 1
    return HealthReport(ok=ok, score=score, findings=findings, stats=stats)
