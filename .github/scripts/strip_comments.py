#!/usr/bin/env python3
"""Strip Python comments without downloading anything at workflow runtime.

The old ``uncomment`` PyPI wrapper downloads a platform-specific Rust binary
from a GitHub release the first time it runs.  That download is a network
dependency and has been failing in CI with HTTP 504, so this script uses the
standard-library ``tokenize`` module instead.  It removes ``#`` comments while
leaving string literals (including docstrings), shebangs, encoding cookies and
tooling directives such as ``noqa`` / ``type: ignore`` / ``pragma`` intact.

Usage:
    python .github/scripts/strip_comments.py . --exclude ".github/**"
    python .github/scripts/strip_comments.py . --check
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import io
import os
import re
import sys
import tokenize
from pathlib import Path

PYTHON_SUFFIXES = frozenset({".py", ".pyi", ".pyw"})
DEFAULT_EXCLUDES = (".github/**", ".git/**")
DEFAULT_EXCLUDED_DIRS = frozenset({
    ".git", ".hg", ".mypy_cache", ".nox", ".pytest_cache", ".ruff_cache",
    ".tox", ".venv", "__pycache__", "build", "dist", "node_modules",
    "site-packages", "venv",
})
KEEP_DIRECTIVE = re.compile(
    r"(?:"
    r"noqa\b|type:|pragma\b|"
    r"(?:pylint|flake8|mypy|ruff|pyright|isort|fmt)\s*[:=]|nosec\b|"
    r"TODO\b|FIXME\b|XXX\b|HACK\b|NOTE\b|"
    r"copyright\b|licen[cs]e|spdx\b|"
    r"#\s*:|"                                  # Sphinx ``#:`` attribute docs
    r"^#\s*!|"                                # shebang
    r"coding[:=]"                             # PEP 263 encoding cookie
    r")",
    re.IGNORECASE,
)


def _split_lines(source: str) -> list[str]:
    """Split on ``\\n`` only, matching tokenize's row numbering."""
    parts = source.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    return ""


def _keep_comment(comment: str, row: int) -> bool:
    if row <= 2 and ("#!" in comment or "coding" in comment.lower()):
        return True
    return bool(KEEP_DIRECTIVE.search(comment))


def strip_python_source(source: str) -> tuple[str, int]:
    """Return (stripped_source, number_of_removed_comments)."""
    lines = _split_lines(source)
    removed = 0
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT or token.start[0] != token.end[0]:
            continue
        if _keep_comment(token.string, token.start[0]):
            continue
        row, column = token.start
        index = row - 1
        if index >= len(lines):
            continue
        line = lines[index]
        prefix = line[:column]
        if prefix.strip():
            lines[index] = prefix.rstrip(" \t") + _line_ending(line)
        else:
            lines[index] = ""
        removed += 1
    return "".join(lines), removed


def _normalise(path: str) -> str:
    path = path.replace("\\", "/")
    return path[2:] if path.startswith("./") else path


def _matches(relative_path: str, pattern: str) -> bool:
    relative_path = _normalise(relative_path)
    pattern = _normalise(pattern).rstrip("/")
    if fnmatch.fnmatchcase(relative_path, pattern):
        return True
    if pattern.endswith("/**"):
        base = pattern[:-3]
        if relative_path == base or relative_path.startswith(base + "/"):
            return True
    return False


def _is_excluded(relative_path: str, patterns: tuple[str, ...]) -> bool:
    return any(_matches(relative_path, pattern) for pattern in patterns)


def _decode(raw: bytes) -> tuple[str, str]:
    encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    return raw.decode(encoding), encoding


def _process_file(path: Path, *, check: bool) -> str:
    try:
        raw = path.read_bytes()
        source, encoding = _decode(raw)
        stripped, removed = strip_python_source(source)
    except (OSError, SyntaxError, UnicodeError, tokenize.TokenError) as exc:
        print(f"warning: skipped {path}: {exc}", file=sys.stderr)
        return "skipped"
    if removed == 0:
        return "unchanged"
    try:
        ast.parse(stripped, filename=str(path))
    except SyntaxError as exc:
        print(f"warning: skipped {path}: stripped source is invalid: {exc}", file=sys.stderr)
        return "skipped"
    new_raw = stripped.encode(encoding)
    if new_raw == raw:
        return "unchanged"
    if check:
        print(str(path))
        return "would-change"
    try:
        path.write_bytes(new_raw)
    except OSError as exc:
        print(f"warning: skipped {path}: {exc}", file=sys.stderr)
        return "skipped"
    return "changed"


def _iter_python_files(root: Path, patterns: tuple[str, ...]):
    if root.is_file():
        if root.suffix.lower() in PYTHON_SUFFIXES:
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        relative_dir = Path(dirpath).relative_to(root).as_posix()
        relative_dir = "" if relative_dir == "." else relative_dir
        kept_dirs = []
        for name in dirnames:
            candidate = f"{relative_dir}/{name}" if relative_dir else name
            if name in DEFAULT_EXCLUDED_DIRS or _is_excluded(candidate, patterns):
                continue
            kept_dirs.append(name)
        dirnames[:] = kept_dirs
        for name in filenames:
            relative = f"{relative_dir}/{name}" if relative_dir else name
            if Path(name).suffix.lower() not in PYTHON_SUFFIXES:
                continue
            if _is_excluded(relative, patterns):
                continue
            yield Path(dirpath) / name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["."], help="files or directories to process")
    parser.add_argument(
        "--exclude", action="append", default=[], metavar="GLOB",
        help="relative glob to skip (repeatable); .github/** and .git/** are always skipped",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="do not write files; exit 1 if any file would be rewritten",
    )
    args = parser.parse_args(argv)

    patterns = tuple(DEFAULT_EXCLUDES) + tuple(args.exclude)
    changed = skipped = 0
    for raw_path in args.paths:
        root = Path(raw_path)
        if not root.exists():
            print(f"warning: skipped {root}: path does not exist", file=sys.stderr)
            skipped += 1
            continue
        for path in _iter_python_files(root, patterns):
            outcome = _process_file(path, check=args.check)
            if outcome == "changed":
                changed += 1
            elif outcome == "would-change":
                changed += 1
            elif outcome == "skipped":
                skipped += 1

    verb = "would rewrite" if args.check else "rewrote"
    print(f"{verb} {changed} Python file(s); skipped {skipped}")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
