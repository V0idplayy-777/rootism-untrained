"""Persistent project map: the file tree plus a summary and symbols per file.

`RepoIndex` (repo_context.py) is built for *retrieval now*. This module is the
long-lived memory an agent needs across steps and sessions:

* it is small and serialisable (``.agent/project_map.json``),
* it refreshes incrementally - only files whose size/mtime changed are re-read,
* it records layout facts (monorepo roots, where tests/docs/config live,
  detected conventions) so the agent stops rediscovering them every turn,
* it renders a budgeted tree for the prompt, with symbols inline for the files
  that matter and one-line summaries for the rest.

Symbols come from ``repo_context.extract_symbols`` so there is exactly one
parser for signatures in this codebase.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .repo_context import extract_symbols

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".agent",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    ".nuxt",
    "dist",
    "build",
    "coverage",
    "site-packages",
    ".idea",
    ".vscode",
}
SOURCE_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".java": "java",
    ".kt": "kotlin",
    ".php": "php",
    ".cs": "csharp",
    ".swift": "swift",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".sql": "sql",
    ".sh": "shell",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".toml": "config",
    ".yaml": "config",
    ".yml": "config",
    ".json": "config",
}
DOC_SUFFIX = (".md", ".rst", ".txt")
MAX_FILE_BYTES = 512 * 1024


@dataclass
class FileEntry:
    path: str
    language: str
    size: int
    mtime: float
    hash: str
    summary: str = ""
    symbols: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    role: str = "source"  # source | test | docs | config | asset

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "language": self.language,
            "size": self.size,
            "mtime": self.mtime,
            "hash": self.hash,
            "summary": self.summary,
            "symbols": list(self.symbols),
            "imports": list(self.imports),
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileEntry":
        return cls(
            path=data["path"],
            language=data.get("language", ""),
            size=int(data.get("size", 0)),
            mtime=float(data.get("mtime", 0)),
            hash=data.get("hash", ""),
            summary=data.get("summary", ""),
            symbols=list(data.get("symbols") or []),
            imports=list(data.get("imports") or []),
            role=data.get("role", "source"),
        )


def _language(path: Path) -> str:
    return SOURCE_EXT.get(path.suffix.lower(), "")


def _role(rel: str) -> str:
    name = Path(rel).name
    parts = Path(rel).parts
    if any(part in {"tests", "test", "__tests__", "spec"} for part in parts[:-1]) or name.startswith("test_") or name.endswith(
        (".test.js", ".test.ts", ".spec.js", ".spec.ts", "_test.py")
    ):
        return "test"
    if name.lower().startswith(("readme", "changelog", "contributing", "license")) or Path(rel).suffix in DOC_SUFFIX:
        return "docs"
    if name in {"package.json", "pyproject.toml", "requirements.txt", "Dockerfile", "Makefile", "go.mod", "Cargo.toml"} or Path(rel).suffix in {".toml", ".yaml", ".yml", ".ini", ".cfg"}:
        return "config"
    if Path(rel).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".woff", ".woff2", ".mp4"}:
        return "asset"
    return "source"


def summarize_text(text: str, language: str, max_chars: int = 160) -> str:
    """One line that says what the file is *for* (docstring/comment/imports)."""

    if not text.strip():
        return "empty file"
    if language == "python":
        try:
            tree = ast.parse(text)
            docstring = ast.get_docstring(tree)
            if docstring:
                return _first_sentence(docstring, max_chars)
        except SyntaxError:
            pass
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("#", "//", "/*", "*", "--")):
            comment = stripped.lstrip("#/*- ").strip()
            if comment:
                return _first_sentence(comment, max_chars)
        elif stripped.startswith(('"""', "'''")):
            return _first_sentence(stripped.strip("\"'"), max_chars)
        elif not stripped.startswith(("import ", "from ", "using ", "require", "package ", '{"', "export {")):
            return _first_sentence(stripped, max_chars)
    return f"{language or 'text'} file"


def _first_sentence(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    match = re.split(r"(?<=[.!?])\s", cleaned, maxsplit=1)
    sentence = match[0] if match else cleaned
    return sentence[:max_chars]


@dataclass
class ProjectMap:
    root: str
    files: dict[str, FileEntry] = field(default_factory=dict)
    generated_at: float = 0.0
    layout: dict = field(default_factory=dict)
    version: int = 1

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, root: str | Path) -> "ProjectMap":
        root = Path(root).resolve()
        project = cls(root=str(root))
        for rel, path in project._walk():
            entry = project._entry(rel, path)
            if entry is not None:
                project.files[rel] = entry
        project.generated_at = time.time()
        project.layout = project._layout()
        return project

    def refresh(self) -> list[str]:
        """Re-read only changed/new files; drop deleted ones. Returns changed paths."""

        seen: set[str] = set()
        changed: list[str] = []
        for rel, path in self._walk():
            seen.add(rel)
            existing = self.files.get(rel)
            try:
                stat = path.stat()
            except OSError:
                continue
            if existing and existing.size == stat.st_size and abs(existing.mtime - stat.st_mtime) < 1e-6:
                continue
            entry = self._entry(rel, path)
            if entry is not None:
                self.files[rel] = entry
                changed.append(rel)
        for rel in list(self.files):
            if rel not in seen:
                del self.files[rel]
                changed.append(rel)
        if changed:
            self.generated_at = time.time()
            self.layout = self._layout()
        return sorted(changed)

    def _walk(self):
        root = Path(self.root)
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if any(part in SKIP_DIRS for part in rel.parts[:-1]):
                continue
            if path.name.endswith((".pyc", ".pyo", ".log", ".lock")):
                continue
            yield str(rel), path

    def _entry(self, rel: str, path: Path) -> FileEntry | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        if stat.st_size > MAX_FILE_BYTES:
            return None
        language = _language(path)
        entry = FileEntry(
            path=rel,
            language=language,
            size=stat.st_size,
            mtime=stat.st_mtime,
            hash="",
            role=_role(rel),
        )
        if not language:
            return entry
        try:
            raw = path.read_bytes()
        except OSError:
            return entry
        entry.hash = hashlib.sha256(raw).hexdigest()[:16]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
        entry.summary = summarize_text(text, language)
        try:
            symbols = extract_symbols(rel, text, language)
        except Exception:  # noqa: BLE001 - parser must never break the map
            symbols = []
        entry.symbols = [f"{symbol.kind} {symbol.name}" for symbol in symbols][:40]
        entry.imports = _imports(text, language)[:30]
        return entry

    def _layout(self) -> dict:
        paths = list(self.files)
        dirs = sorted({str(Path(path).parent) for path in paths})
        conventions = {
            "src_layout": any(path.startswith("src/") for path in paths),
            "tests_dir": any(path.startswith("tests/") for path in paths),
            "docs_dir": any(path.startswith("docs/") for path in paths),
            "tests_colocated": any(Path(path).name.startswith("test_") for path in paths),
            "docker": any(Path(path).name.lower().startswith("dockerfile") for path in paths),
            "ci": any(path.startswith(".github/workflows/") for path in paths),
            "typescript": any(Path(path).suffix in {".ts", ".tsx"} for path in paths),
            "package_json": "package.json" in paths,
            "pyproject": "pyproject.toml" in paths,
            "requirements": any(Path(path).name.startswith("requirements") for path in paths),
        }
        return {
            "file_count": len(paths),
            "languages": _count_by(entry.language for entry in self.files.values()),
            "roles": _count_by(entry.role for entry in self.files.values()),
            "top_level": sorted({path.split("/")[0] for path in paths if "/" in path} | {path for path in paths if "/" not in path})[:30],
            "directories": dirs[:40],
            "monorepo": _is_monorepo(Path(self.root), paths),
            "conventions": conventions,
        }

    # ------------------------------------------------------------------- query
    def find_symbol(self, name: str, limit: int = 10) -> list[FileEntry]:
        lowered = name.lower()
        matches = []
        for entry in self.files.values():
            for symbol in entry.symbols:
                if lowered in symbol.lower():
                    matches.append(entry)
                    break
        matches.sort(key=lambda entry: (entry.path.count("/"), entry.path))
        return matches[:limit]

    def file(self, path: str) -> FileEntry | None:
        return self.files.get(path.lstrip("./"))

    def changed_since(self, snapshot: dict[str, str]) -> list[str]:
        changed = []
        for path, entry in self.files.items():
            if snapshot.get(path) != entry.hash:
                changed.append(path)
        changed.extend(path for path in snapshot if path not in self.files)
        return sorted(set(changed))

    def snapshot_hashes(self) -> dict[str, str]:
        return {path: entry.hash for path, entry in self.files.items() if entry.hash}

    def render(self, max_chars: int = 2400, focus: list[str] | None = None) -> str:
        """Budgeted tree for a prompt: focused files expanded, the rest one line."""

        if not self.files:
            return "project map: (empty)"
        focus = [path for path in (focus or []) if path in self.files]
        lines = [
            f"project map: {len(self.files)} files · languages "
            + ", ".join(f"{lang}:{count}" for lang, count in list(self.layout.get("languages", {}).items())[:5]),
        ]
        if self.layout.get("monorepo"):
            lines.append("layout: monorepo (multiple packages)")
        if focus:
            lines.append("focused files:")
            for path in focus[:6]:
                entry = self.files[path]
                symbols = ", ".join(entry.symbols[:8])
                lines.append(f"  {path} - {entry.summary}" + (f" [{symbols}]" if symbols else ""))
            lines.append("tree:")
        for path in sorted(self.files)[:200]:
            if path in focus:
                continue
            entry = self.files[path]
            symbols = ", ".join(entry.symbols[:3])
            suffix = f"  ({symbols})" if symbols else ""
            lines.append(f"  {path}{suffix}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max_chars - 40].rsplit("\n", 1)[0] + "\n  ... truncated"
        return text

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "root": self.root,
            "generated_at": self.generated_at,
            "layout": self.layout,
            "files": {path: entry.to_dict() for path, entry in sorted(self.files.items())},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectMap":
        project = cls(root=data.get("root", ""), generated_at=float(data.get("generated_at", 0)), layout=dict(data.get("layout") or {}))
        project.files = {path: FileEntry.from_dict(entry) for path, entry in (data.get("files") or {}).items()}
        return project

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else Path(self.root) / ".agent" / "project_map.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path, refresh: bool = False) -> "ProjectMap":
        target = Path(path)
        project = cls.from_dict(json.loads(target.read_text(encoding="utf-8")))
        if refresh and project.root:
            project.refresh()
        return project

    @classmethod
    def open(cls, root: str | Path, path: str | Path | None = None) -> "ProjectMap":
        """Load the cached map when it exists (and refresh it), else build one."""

        root = Path(root)
        default = root / ".agent" / "project_map.json"
        target = Path(path) if path else default
        if target.exists():
            try:
                return cls.load(target, refresh=True)
            except (OSError, json.JSONDecodeError, KeyError):
                pass
        return cls.build(root)

    def stats(self) -> dict:
        return {
            "files": len(self.files),
            "generated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.generated_at or time.time())),
            "languages": self.layout.get("languages", {}),
            "roles": self.layout.get("roles", {}),
            "monorepo": bool(self.layout.get("monorepo")),
            "conventions": self.layout.get("conventions", {}),
        }


def _imports(text: str, language: str) -> list[str]:
    if language == "python":
        found: list[str] = []
        for match in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", text, re.MULTILINE):
            found.append(match.group(1))
        return found
    found = [match.group(1) for match in re.finditer(r"(?:from|require\()\s*[\"']([^\"']+)[\"']", text)]
    return found


def _count_by(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def _is_monorepo(root: Path, paths: list[str]) -> bool:
    if (root / "go.work").exists() or (root / "pnpm-workspace.yaml").exists() or (root / "nx.json").exists() or (root / "turbo.json").exists():
        return True
    package_paths = [path for path in paths if Path(path).name == "package.json"]
    return len(package_paths) > 1


def build_project_map(root: str | Path, save: bool = True) -> ProjectMap:
    project = ProjectMap.build(root)
    if save:
        project.save()
    return project
