"""Python CLI starter: argparse, testable core, JSON output, packaging."""

from __future__ import annotations

from ..addons import env_example, gitignore
from ..spec import ProjectOptions, TemplateSpec

PYPROJECT = '''[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "__NAME__"
version = "0.1.0"
description = "__DESCRIPTION__"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
dev = ["pytest==8.2.2", "ruff==0.5.1"]

[project.scripts]
__MODULE__ = "__MODULE__.cli:main"

[tool.setuptools.packages.find]
include = ["__MODULE__*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
'''

INIT = '''"""__DISPLAY_NAME__ - command-line tool."""

__version__ = "0.1.0"
'''

CORE = '''"""Pure logic for the CLI. Everything here is importable and testable."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class FileStats:
    path: str
    lines: int
    words: int
    chars: int

    def to_dict(self) -> dict:
        return asdict(self)


def greet(name: str) -> str:
    """A placeholder command that proves the wiring works."""

    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError("name must not be empty")
    return f"Hello, {cleaned}!"


def count_file(path: str | Path) -> FileStats:
    """Count lines/words/chars in one text file."""

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"no such file: {target}")
    if target.stat().st_size > 20 * 1024 * 1024:
        raise ValueError(f"file is too large to count: {target}")
    text = target.read_text(encoding="utf-8", errors="replace")
    return FileStats(
        path=str(target),
        lines=text.count("\\n") + (1 if text and not text.endswith("\\n") else 0),
        words=len(text.split()),
        chars=len(text),
    )


def summarize(paths: list[str]) -> dict:
    """Aggregate stats, turning per-file failures into a report instead of a crash."""

    files: list[dict] = []
    errors: list[dict] = []
    for path in paths:
        try:
            files.append(count_file(path).to_dict())
        except (FileNotFoundError, ValueError, OSError) as exc:
            errors.append({"path": path, "error": str(exc)})
    total = {
        "lines": sum(item["lines"] for item in files),
        "words": sum(item["words"] for item in files),
        "chars": sum(item["chars"] for item in files),
    }
    return {"files": files, "errors": errors, "total": total}
'''

CLI = '''"""Command-line entry point: `__MODULE__ --help`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .core import greet, summarize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="__MODULE__",
        description="__DESCRIPTION__",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hello = subparsers.add_parser("greet", help="print a greeting")
    hello.add_argument("name", nargs="?", default="world")

    count = subparsers.add_parser("count", help="count lines/words/chars in files")
    count.add_argument("paths", nargs="+", type=Path)
    count.set_defaults(command="count")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Return an exit code: 0 success, 1 runtime error, 2 usage error."""

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help/--version/usage errors: return, don't raise
        return int(exc.code or 0)

    try:
        if args.command == "greet":
            message = greet(args.name)
            if args.json:
                print(json.dumps({"message": message}))
            else:
                print(message)
            return 0

        if args.command == "count":
            report = summarize([str(path) for path in args.paths])
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                for item in report["files"]:
                    print(f"{item['path']}: {item['lines']} lines, {item['words']} words")
                for error in report["errors"]:
                    print(f"{error['path']}: {error['error']}", file=sys.stderr)
                print(
                    f"total: {report['total']['lines']} lines, "
                    f"{report['total']['words']} words, {report['total']['chars']} chars"
                )
            return 1 if report["errors"] else 0
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''

TEST_CORE = '''"""Core logic tests - fast, no filesystem unless the test needs it."""

from __future__ import annotations

import pytest

from __MODULE__.core import count_file, greet, summarize


def test_greet():
    assert greet(" Ada ") == "Hello, Ada!"


def test_greet_rejects_empty():
    with pytest.raises(ValueError):
        greet("   ")


def test_count_file(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("one two\\nthree\\n", encoding="utf-8")
    stats = count_file(path)
    assert stats.lines == 2
    assert stats.words == 3


def test_count_file_missing():
    with pytest.raises(FileNotFoundError):
        count_file("does-not-exist.txt")


def test_summarize_reports_errors_without_crashing(tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("a b c", encoding="utf-8")
    report = summarize([str(good), "missing.txt"])
    assert len(report["files"]) == 1
    assert report["errors"][0]["path"] == "missing.txt"
    assert report["total"]["words"] == 3
'''

TEST_CLI = '''"""CLI contract: exit codes and JSON output (no subprocess needed)."""

from __future__ import annotations

import json

from __MODULE__.cli import main


def test_greet_exit_zero(capsys):
    assert main(["greet", "Ada"]) == 0
    assert capsys.readouterr().out.strip() == "Hello, Ada!"


def test_json_output(capsys):
    assert main(["--json", "greet", "Ada"]) == 0
    assert json.loads(capsys.readouterr().out)["message"] == "Hello, Ada!"


def test_count_success(tmp_path, capsys):
    path = tmp_path / "f.txt"
    path.write_text("hello world", encoding="utf-8")
    assert main(["count", str(path)]) == 0
    assert "2 words" in capsys.readouterr().out


def test_count_missing_file_exits_one(capsys):
    assert main(["count", "nope.txt"]) == 1
    assert "no such file" in capsys.readouterr().err


def test_usage_error_exits_two():
    assert main([]) == 2
'''


CONFTEST = """import sys
from pathlib import Path

# Make the package importable when running `pytest` from the project root.
sys.path.insert(0, str(Path(__file__).parent))
"""


def _tokens(text: str, options: ProjectOptions) -> str:
    """Replace __TOKEN__ markers; generated code keeps its own braces."""

    return (
        text.replace("__NAME__", options.name)
        .replace("__MODULE__", options.module)
        .replace("__DISPLAY_NAME__", options.display_name)
        .replace("__DESCRIPTION__", options.description or f"{options.display_name} command-line tool")
        .replace("__PORT__", str(options.port))
    )


def build(options: ProjectOptions) -> dict[str, str]:
    return {
        "pyproject.toml": _tokens(PYPROJECT, options),
        f"{options.module}/__init__.py": _tokens(INIT, options),
        f"{options.module}/core.py": CORE,
        f"{options.module}/cli.py": _tokens(CLI, options),
        "tests/test_core.py": _tokens(TEST_CORE, options),
        "tests/test_cli.py": _tokens(TEST_CLI, options),
        "conftest.py": CONFTEST,
        ".env.example": env_example(options, include_secret=False),
        ".gitignore": gitignore(),
    }


SPEC = TemplateSpec(
    id="cli-tool",
    name="Python CLI tool",
    tagline="Argparse command with testable core, JSON output and packaging",
    description=(
        "A command-line tool you can install and run from anywhere: argument parsing, "
        "sensible exit codes, `--json` output for scripting, and logic separated from the "
        "CLI so it can be tested without spawning processes."
    ),
    stack="Python 3.10+ · argparse · setuptools",
    kind="cli",
    tags=["cli", "python", "tool", "script", "library"],
    supports_auth=False,
    supports_database=False,
    default_port=0,
    build=build,
    good_for=["automation scripts", "file utilities", "dev tools", "data cleanup"],
    run_command="python -m venv .venv && source .venv/bin/activate\npip install -e '.[dev]'\n__MODULE__ --help\npython -m pytest -q",
    test_command="python -m pytest -q",
    preview="cli",
    next_steps=[
        "pip install -e '.[dev]' and run `<tool> --help`",
        "python -m pytest -q",
        "Add a subparser in cli.py plus a pure function in core.py for each new command",
    ],
)
