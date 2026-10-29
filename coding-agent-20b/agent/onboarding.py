"""Systematic repo onboarding: architecture summary + entry-point map.

The agent must produce a short architecture summary + entry-point map before
it is allowed to edit. This module generates that artefact deterministically
without a model, so onboarding works offline and is testable.

It also supports workspace constraints: work only inside this subdirectory/package.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


IGNORE_DIRS = {".git", ".hg", ".svn", ".agent", "node_modules", ".venv", "venv", "__pycache__", ".next", "dist", "build"}
ENTRY_HINTS = ["main.py", "app.py", "server.py", "index.js", "index.ts", "app.js", "main.go", "main.rs", "Cargo.toml", "package.json", "pyproject.toml", "manage.py", "wsgi.py", "asgi.py"]
IMPORTANT_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb", ".php"}


@dataclass
class EntryPoint:
    path: str
    kind: str  # cli, web, api, worker, test, config
    why: str

    def to_dict(self) -> dict:
        return {"path": self.path, "kind": self.kind, "why": self.why}


@dataclass
class OnboardingReport:
    root: str
    summary: str
    entry_points: list[EntryPoint] = field(default_factory=list)
    architecture: dict = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "summary": self.summary,
            "entry_points": [e.to_dict() for e in self.entry_points],
            "architecture": self.architecture,
            "constraints": self.constraints,
            "warnings": self.warnings,
        }

    def render(self, max_chars: int = 4000) -> str:
        lines = [f"# Onboarding: {self.root}", "", self.summary, ""]
        lines.append("## Entry points")
        for ep in self.entry_points[:12]:
            lines.append(f"- `{ep.path}` ({ep.kind}): {ep.why}")
        if self.architecture:
            lines.append("")
            lines.append("## Architecture")
            for k, v in self.architecture.items():
                lines.append(f"- {k}: {v}")
        if self.constraints:
            lines.append("")
            lines.append("## Constraints")
            for c in self.constraints:
                lines.append(f"- {c}")
        if self.warnings:
            lines.append("")
            lines.append("## Warnings")
            for w in self.warnings:
                lines.append(f"- {w}")
        text = "\n".join(lines)
        return text[:max_chars]

    def save(self, root: str | Path) -> Path:
        target = Path(root) / ".agent" / "onboarding.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        # also write markdown for prompt injection
        md = Path(root) / ".agent" / "onboarding.md"
        md.write_text(self.render() + "\n", encoding="utf-8")
        return target


def _detect_entry_points(root: Path) -> list[EntryPoint]:
    entries: list[EntryPoint] = []
    # explicit hints
    for hint in ENTRY_HINTS:
        p = root / hint
        if p.exists():
            kind = "web" if hint in ("app.py", "main.py", "server.py", "index.js") else "config"
            if hint == "package.json":
                # check scripts
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    scripts = data.get("scripts", {})
                    if "start" in scripts:
                        kind = "web"
                    if "test" in scripts:
                        entries.append(EntryPoint(str(p.relative_to(root)), "test", f"scripts: {', '.join(scripts)}"))
                except Exception:
                    pass
            entries.append(EntryPoint(str(p.relative_to(root)), kind, f"detected {hint}"))
    # scan for common patterns
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in IGNORE_DIRS for part in rel.parts[:-1]):
            continue
        name = path.name
        # flask/fastapi entry
        if name == "main.py" and "app" in str(rel) or name == "wsgi.py":
            entries.append(EntryPoint(str(rel), "web", "possible web entry"))
        # cli entry
        if name == "cli.py" or (name.endswith(".py") and "cli" in name.lower()):
            entries.append(EntryPoint(str(rel), "cli", "cli module"))
        # worker
        if "worker" in name.lower() or "jobs" in name.lower():
            entries.append(EntryPoint(str(rel), "worker", "background job/worker"))
    # dedupe by path
    seen = set()
    uniq = []
    for e in entries:
        if e.path not in seen:
            uniq.append(e)
            seen.add(e.path)
    return uniq[:20]


def _architecture_summary(root: Path) -> dict:
    exts: dict[str, int] = {}
    dirs: set[str] = set()
    has_tests = False
    has_docker = False
    has_ci = False
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in IGNORE_DIRS for part in rel.parts[:-1]):
            continue
        ext = path.suffix.lower()
        if ext in IMPORTANT_EXTS:
            exts[ext] = exts.get(ext, 0) + 1
        if len(rel.parts) > 1:
            dirs.add(rel.parts[0])
        if "test" in path.name.lower() or "tests" in rel.parts:
            has_tests = True
        if path.name.lower().startswith("dockerfile"):
            has_docker = True
        if ".github" in rel.parts:
            has_ci = True
    return {
        "languages": ", ".join(f"{k}:{v}" for k, v in sorted(exts.items(), key=lambda x: -x[1])[:6]) or ["unknown"],
        "top_level": sorted(dirs)[:15],
        "file_count": sum(exts.values()),
        "has_tests": has_tests,
        "has_docker": has_docker,
        "has_ci": has_ci,
    }


def _summarize_repo(root: Path, entry_points: list[EntryPoint], arch: dict) -> str:
    parts = []
    if arch.get("languages"):
        parts.append(f"Languages: {arch['languages']}.")
    if arch.get("top_level"):
        parts.append(f"Top-level dirs: {', '.join(arch['top_level'][:8])}.")
    if entry_points:
        parts.append(f"Entry points: {', '.join(e.path for e in entry_points[:5])}.")
    if not parts:
        parts.append("Empty or unknown structure.")
    # try to read README for intent
    readme = root / "README.md"
    if readme.exists():
        try:
            text = readme.read_text(encoding="utf-8")[:500]
            first = text.splitlines()[0] if text else ""
            if first:
                parts.append(f"README: {first[:120]}")
        except Exception:
            pass
    return " ".join(parts)


def onboarding(root: str | Path, subdir: str | None = None) -> OnboardingReport:
    root_path = Path(root).resolve()
    constraint_paths: list[str] = []
    if subdir:
        constraint_paths.append(f"Work only inside `{subdir}` (other paths are read-only)")
        target = root_path / subdir
        if not target.exists():
            constraint_paths.append(f"WARNING: constraint dir {subdir} does not exist")
    entry_points = _detect_entry_points(root_path)
    arch = _architecture_summary(root_path)
    summary = _summarize_repo(root_path, entry_points, arch)
    warnings = []
    if not arch.get("has_tests"):
        warnings.append("No tests detected — consider adding a starter test suite.")
    if not entry_points:
        warnings.append("No entry point detected — define one (main.py, package.json start, etc).")
    return OnboardingReport(
        root=str(root_path),
        summary=summary,
        entry_points=entry_points,
        architecture=arch,
        constraints=constraint_paths,
        warnings=warnings,
    )


def ensure_onboarding(root: str | Path, subdir: str | None = None, force: bool = False) -> OnboardingReport:
    """Ensure onboarding report exists; create if missing or force."""
    root_path = Path(root).resolve()
    target = root_path / ".agent" / "onboarding.json"
    if target.exists() and not force:
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            return OnboardingReport(
                root=data.get("root", str(root_path)),
                summary=data.get("summary", ""),
                entry_points=[EntryPoint(**e) for e in data.get("entry_points", [])],
                architecture=data.get("architecture", {}),
                constraints=data.get("constraints", []),
                warnings=data.get("warnings", []),
            )
        except Exception:
            pass
    report = onboarding(root_path, subdir=subdir)
    report.save(root_path)
    return report
