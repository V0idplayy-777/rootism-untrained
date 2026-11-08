"""Regression tests for the network-free Python comment stripper."""

from __future__ import annotations

import ast
import runpy
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "strip_comments.py"
MODULE = runpy.run_path(str(SCRIPT), run_name="strip_comments_module")

SAMPLE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A docstring with # inside it."""
# a plain standalone comment
x = 1  # a trailing comment
value = "# not a comment"  # noqa: E501
block = """a # hash inside a string"""
# type: ignore
# NOTE: keep this note
# TODO: keep this todo
# SPDX-License-Identifier: MIT
#: keep this attribute doc
kept = 2  # pragma: no cover
'''


def test_strip_python_source_preserves_code_strings_and_directives():
    stripped, removed = MODULE["strip_python_source"](SAMPLE)
    assert removed > 0
    assert "# a plain standalone comment" not in stripped
    assert "x = 1\n" in stripped
    assert "# a trailing comment" not in stripped
    assert '"# not a comment"' in stripped
    assert "# noqa: E501" in stripped
    assert '"""A docstring with # inside it."""' in stripped
    assert '"""a # hash inside a string"""' in stripped
    for directive in (
        "#!/usr/bin/env python3",
        "# -*- coding: utf-8 -*-",
        "# type: ignore",
        "# NOTE: keep this note",
        "# TODO: keep this todo",
        "# SPDX-License-Identifier: MIT",
        "#: keep this attribute doc",
        "# pragma: no cover",
    ):
        assert directive in stripped
    ast.parse(stripped)


def test_strip_python_source_is_idempotent_and_preserves_line_endings():
    first, _ = MODULE["strip_python_source"](SAMPLE)
    second, removed_again = MODULE["strip_python_source"](first)
    assert first == second
    assert removed_again == 0

    crlf = "x = 1  # comment\r\ny = 2\r\n"
    stripped, _ = MODULE["strip_python_source"](crlf)
    assert stripped == "x = 1\r\ny = 2\r\n"

    no_newline = "x = 1  # comment"
    stripped, _ = MODULE["strip_python_source"](no_newline)
    assert stripped == "x = 1"


def test_cli_check_and_in_place_strip(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("# comment\nx = 1  # trailing\n", encoding="utf-8")

    before = subprocess.run(
        [sys.executable, str(SCRIPT), str(sample), "--check"],
        capture_output=True, text=True, check=False,
    )
    assert before.returncode == 1
    assert sample.read_text(encoding="utf-8") == "# comment\nx = 1  # trailing\n"

    applied = subprocess.run(
        [sys.executable, str(SCRIPT), str(sample)],
        capture_output=True, text=True, check=False,
    )
    assert applied.returncode == 0
    assert sample.read_text(encoding="utf-8") == "x = 1\n"

    after = subprocess.run(
        [sys.executable, str(SCRIPT), str(sample), "--check"],
        capture_output=True, text=True, check=False,
    )
    assert after.returncode == 0


def test_exclusion_glob_matches_directories_and_files():
    matches = MODULE["_matches"]
    assert matches(".github", ".github/**")
    assert matches(".github/workflows/ci.yml", ".github/**")
    assert matches("nested/.github/scripts/tool.py", "**/.github/**")
    assert not matches("coding-agent-20b/model/model.py", ".github/**")


def test_iter_python_files_skips_excluded_and_cache_directories(tmp_path):
    (tmp_path / "coding-agent-20b").mkdir()
    (tmp_path / "coding-agent-20b" / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".github" / "scripts").mkdir(parents=True)
    (tmp_path / ".github" / "scripts" / "skip.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "skip.py").write_text("x = 1\n", encoding="utf-8")

    found = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in MODULE["_iter_python_files"](tmp_path, (".github/**", ".git/**"))
    )
    assert found == ["coding-agent-20b/keep.py"]
