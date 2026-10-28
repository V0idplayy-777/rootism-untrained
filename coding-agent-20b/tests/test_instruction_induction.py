"""Real-code instruction induction: shapes, licence gate, provenance."""

import json

import pytest

from dataset.instruction_induction import (
    InstructionInduction,
    iter_corpus_files,
)


def write_shard(tmp_path, name, rows):
    path = tmp_path / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


GOOD_MODULE = '''"""Serialize task runs to a durable JSONL event log.

The writer buffers events in memory, flushes on close or when the buffer
exceeds `flush_every` events, and guarantees whole-line writes so a crashed
process leaves a prefix that replay can resume from.
"""

import json


class EventLog:
    def __init__(self, path, flush_every=100):
        self.path = path
        self.flush_every = flush_every
        self._buffer = []

    def append(self, event):
        """Add one event; flushes when the buffer is full."""
        self._buffer.append(event)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        """Write buffered events as whole JSON lines."""
        with open(self.path, "a", encoding="utf-8") as f:
            for event in self._buffer:
                f.write(json.dumps(event) + "\\n")
        self._buffer = []

    def replay(self):
        """Yield every complete event line, skipping torn tails."""
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
'''

TEST_FILE = '''"""Tests for the retry helper."""

from utils import retry_once


def test_retry_once_succeeds_second_call():
    calls = []

    @retry_once
    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("transient")
        return "ok"

    assert flaky() == "ok"


def test_retry_once_raises_after_second_failure():
    @retry_once
    def always_fails():
        raise RuntimeError("nope")

    try:
        always_fails()
        raised = False
    except RuntimeError:
        raised = True
    assert raised
'''


@pytest.fixture()
def corpus(tmp_path):
    rows = [
        {"path": "pkg/eventlog.py", "content": GOOD_MODULE, "license": "MIT"},
        {"path": "tests/test_retry.py", "content": TEST_FILE, "license": "Apache-2.0"},
        # rejected: non-permissive licence never reaches induction
        {
            "path": "pkg/proprietary.py",
            "content": '"""A module with a docstring long enough to pass filters."""\n\ndef x():\n    pass\n',
            "license": "GPL-3.0",
        },
        # rejected: no docstring -> most shapes skip it anyway
        {"path": "pkg/undocumented.py", "content": "a = 1\n", "license": "MIT"},
    ]
    write_shard(tmp_path, "python-0", rows)
    write_shard(tmp_path, "python-1", [
        {"path": "pkg/other.py", "content": GOOD_MODULE, "license": "BSD-3-Clause"}
    ])
    # non-python shard is ignored for language="python"
    write_shard(tmp_path, "go-0", [
        {"path": "x.go", "content": "package x\n", "license": "MIT"}
    ])
    return tmp_path


def test_iter_corpus_files_filters_by_language_and_licence(corpus):
    files = list(iter_corpus_files(corpus, "python", max_files=100))
    paths = {f["path"] for f in files}
    assert "pkg/eventlog.py" in paths
    assert "pkg/proprietary.py" not in paths  # licence gate is fail-closed
    assert all("x.go" != p for p in paths)


def test_docstring_to_code_shape(corpus):
    rows = InstructionInduction(corpus, max_files=10).docstring_to_code()
    assert rows
    row = rows[0]
    assert row["category"] == "induced_docstring_to_code"
    assert "EventLog" in row["output"] or "task runs" in row["instruction"]
    assert row["metadata"]["license"] == "MIT"
    assert row["metadata"]["source_path"].endswith(".py")
    assert row["metadata"]["induction"] == "docstring_to_code"


def test_signature_to_body_shape(corpus):
    rows = InstructionInduction(corpus, max_files=10).signature_to_body()
    assert rows
    fns = {r["metadata"]["function"] for r in rows}
    assert "append" in fns  # public, documented
    assert "flush" in fns
    assert not any(f.startswith("_") for f in fns)  # private excluded
    body = next(r for r in rows if r["metadata"]["function"] == "flush")["output"]
    assert "open(" in body  # the answer is the real implementation


def test_code_to_explanation_is_extractive(corpus):
    rows = InstructionInduction(corpus, max_files=10).code_to_explanation()
    assert rows
    row = rows[0]
    assert row["category"] == "induced_code_to_explanation"
    # only facts checkable in the source appear
    assert "EventLog" in row["output"] or "append" in row["output"]
    assert "verifiable" in row["output"]


def test_test_to_contract_names_tested_units(corpus):
    rows = InstructionInduction(corpus, max_files=10).test_to_contract()
    assert rows
    tested = {name for r in rows for name in r["metadata"]["tested"]}
    assert any("retry_once" in name for name in tested)


def test_generate_mixed_and_deterministic(corpus):
    ind = InstructionInduction(corpus, max_files=10)
    a = ind.generate(count=12)
    b = InstructionInduction(corpus, max_files=10).generate(count=12)
    assert a == b  # deterministic given the same corpus
    cats = {r["category"] for r in a}
    assert cats  # at least one induction shape fires on this tiny corpus
    for row in a:
        assert row["metadata"]["license"] in {
            "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC",
            "MPL-2.0", "Unlicense", "CC0-1.0", "0BSD", "Zlib",
        }


def test_provenance_never_leaks_rejected_licences(corpus):
    rows = InstructionInduction(corpus, max_files=10).generate(count=50)
    for row in rows:
        assert row["metadata"]["source_path"] != "pkg/proprietary.py"
        assert row["metadata"]["license"] != "GPL-3.0"


def test_stats(corpus):
    stats = InstructionInduction(corpus, max_files=10).stats()
    assert stats["files"] >= 2
    assert "MIT" in stats["licenses"]
