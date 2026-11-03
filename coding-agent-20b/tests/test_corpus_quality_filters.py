"""Hard quality floors for real-code ingestion.

These tests use ``QualityScorer`` directly so every rejection names the filter
that fired. That matters in production: ``DataProcessor`` otherwise collapses
all of them into ``filtered_hard_quality`` and a too-broad heuristic can silently
remove an entire slice (small test files are the important regression case).
"""

from __future__ import annotations

import pytest

from dataset.production_processor import (
    DataProcessor,
    QualityScorer,
    code_line_ratio,
    long_token_ratio,
    single_line_repeat_ratio,
)


def module(name: str = "alpha") -> str:
    return f'''"""A small but real module."""

from collections import Counter


def frequencies(values):
    """Count values without losing their original order."""
    counts = Counter(values)
    ordered = sorted(counts)
    return [(value, counts[value]) for value in ordered]


def labels(values):
    pairs = frequencies(values)
    return [f"{{name}}:{{count}}" for name, count in pairs]


RESULT = labels(["{name}", "beta", "{name}"])
'''


def check(content: str, path: str = "pkg/module.py", language: str = "python", **kwargs):
    return QualityScorer.passes_hard_filters(content, language, path, **kwargs)


def test_real_module_passes_the_stricter_floors():
    ok, reason = check(module())
    assert (ok, reason) == (True, "ok")


@pytest.mark.parametrize(
    "path",
    [
        "pkg/node_modules/library/index.py",
        "pkg/third-party/library/index.py",
        "pkg/site-packages/library/index.py",
        "pkg/fixtures/generated_snapshot.py",
        "pkg/__snapshots__/example.py",
    ],
)
def test_vendored_generated_and_snapshot_paths_are_rejected(path):
    assert check(module(), path=path) == (False, "vendored_dir")


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("package.json", "non_code_file"),
        ("pnpm-lock.yaml", "minified_or_generated_name"),
        ("pyproject.toml", "non_code_file"),
        ("Dockerfile", "non_code_file"),
    ],
)
def test_packaging_and_lock_files_are_not_code_documents(name, reason):
    assert check(module(), path=name) == (False, reason)


def test_third_party_file_header_is_rejected_even_outside_a_vendor_directory():
    content = "# This file was copied from AnotherProject.\n" + module()
    assert check(content) == (False, "third_party_notice")


def test_a_tiny_backbone_file_is_rejected_but_a_focused_test_is_not():
    tiny_test = '''import pytest


def test_answer():
    """One behaviour, tested directly."""
    assert 2 + 2 == 4
    assert "a".upper() == "A"
'''
    ok, reason = check(tiny_test, path="tests/test_answer.py")
    assert (ok, reason) == (True, "ok")

    tiny_module = '''def answer():
    value = 40
    value += 2
    return value
'''
    assert check(tiny_module) == (False, "too_few_code_lines")


def test_prose_does_not_pass_as_source_code():
    prose = "\n".join(
        f"This paragraph explains background topic {index} in ordinary words"
        for index in range(12)
    )
    assert code_line_ratio(prose) == 0.0
    assert check(prose, path="src/notes.js", language="javascript") == (False, "prose_heavy")


def test_wrapped_identifier_soup_is_caught_even_with_short_lines():
    blobs = [f"blob_{index}_" + "aB3" * 20 for index in range(12)]
    content = "PAYLOAD = [\n" + "".join(f'    "{blob}",\n' for blob in blobs) + "]\n"
    assert max(map(len, content.splitlines())) < 100
    assert long_token_ratio(content) > 0.30
    assert check(content) == (False, "long_token_soup")


def test_repeated_assignment_boilerplate_is_rejected():
    content = "def generated():\n" + "    value = 1\n" * 24 + "    return value\n"
    assert len(content.splitlines()) >= 20
    assert single_line_repeat_ratio(content) > 0.25
    assert check(content) == (False, "repeated_line_boilerplate")


def test_delimiter_repetition_is_not_mistaken_for_boilerplate():
    content = """function makeRecords(values) {
  const output = [];
  for (const value of values) {
    const record = {
      value: value,
      valid: true,
      source: 'input',
    };
    output.push(record);
  }
  return output;
}

module.exports = { makeRecords };
"""
    assert check(content, path="src/records.js", language="javascript") == (True, "ok")


def test_documentation_does_not_have_to_look_like_code():
    content = """# Architecture

This document explains how requests travel through the service.
It describes retries, failure ownership, observability, and deployment.
Each section is prose rather than executable source code.

## Operations

Operators should read the runbook before changing the retry budget.
"""
    assert check(content, path="docs/architecture.md", language="markdown") == (True, "ok")


def test_min_quality_zero_keeps_the_existing_relaxed_mode():
    content = 'def f(x):\n    """doc"""\n    # comment\n    return x + 1\n' * 3
    processor = DataProcessor(min_quality=0.0, languages=["python"])
    assert processor.process_document(content, "a.py", "MIT") is not None

    strict = DataProcessor(min_quality=0.2, languages=["python"])
    assert strict.process_document(content, "a.py", "MIT") is None


def test_relaxed_mode_still_rejects_invalid_python_and_unknown_licenses():
    processor = DataProcessor(min_quality=0.0, languages=["python"])
    assert processor.process_document("def broken(:\n" * 4, "bad.py", "MIT") is None
    assert processor.process_document(module(), "good.py", "Proprietary") is None
