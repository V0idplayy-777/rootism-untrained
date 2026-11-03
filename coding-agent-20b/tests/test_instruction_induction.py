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
        {
            "path": "pkg/proprietary.py",
            "content": '"""A module with a docstring long enough to pass filters."""\n\ndef x():\n    pass\n',
            "license": "GPL-3.0",
        },
        {"path": "pkg/undocumented.py", "content": "a = 1\n", "license": "MIT"},
    ]
    write_shard(tmp_path, "python-0", rows)
    write_shard(tmp_path, "python-1", [
        {"path": "pkg/other.py", "content": GOOD_MODULE, "license": "BSD-3-Clause"}
    ])
    write_shard(tmp_path, "go-0", [
        {"path": "x.go", "content": "package x\n", "license": "MIT"}
    ])
    return tmp_path


def test_iter_corpus_files_filters_by_language_and_licence(corpus):
    files = list(iter_corpus_files(corpus, "python", max_files=100))
    paths = {f["path"] for f in files}
    assert "pkg/eventlog.py" in paths
    assert "pkg/proprietary.py" not in paths
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
    assert "append" in fns
    assert "flush" in fns
    assert not any(f.startswith("_") for f in fns)
    body = next(r for r in rows if r["metadata"]["function"] == "flush")["output"]
    assert "open(" in body


def test_code_to_explanation_is_extractive(corpus):
    rows = InstructionInduction(corpus, max_files=10).code_to_explanation()
    assert rows
    row = rows[0]
    assert row["category"] == "induced_code_to_explanation"
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
    assert a == b
    cats = {r["category"] for r in a}
    assert cats
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


EXAMPLE_MODULE = '''"""Rotate a signing key and keep old signatures verifiable.

The helper keeps the previous key for one rotation so tokens minted with it
still validate, then drops it. Every rotation is appended to an audit trail.
"""

import hmac
import json
import time
from hashlib import sha256


class KeyRing:
    """Two-slot signing key ring with an audit trail."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.current = b"k1"
        self.previous = None

    def rotate(self, key: bytes) -> bool:
        """Swap in a new signing key.

        Example:
            >>> ring.rotate(b"k2")
            True
        Returns:
            True when the new key is active.
        Raises:
            ValueError: when the key is empty.
        """
        if not key:
            raise ValueError("key must not be empty")
        self.previous = self.current
        self.current = key
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"at": time.time()}) + "\\n")
                handle.flush()
        except OSError:
            pass
        return True

    def sign(self, payload: bytes) -> str:
        """Sign a payload with the current key and return the hex digest."""
        return hmac.new(self.current, payload, sha256).hexdigest()
'''


KEYRING_TEST = '''"""Behavioural tests for the signing key ring."""

import pytest

from pkg.keyring import KeyRing


def test_rotate_rejects_empty_key(tmp_path):
    ring = KeyRing(str(tmp_path / "audit.jsonl"))
    with pytest.raises(ValueError):
        ring.rotate(b"")


def test_rotate_keeps_previous_key(tmp_path):
    ring = KeyRing(str(tmp_path / "audit.jsonl"))
    ring.rotate(b"k2")
    assert ring.previous == b"k1"
'''


@pytest.fixture()
def example_corpus(tmp_path):
    write_shard(
        tmp_path,
        "python-0",
        [
            {"path": "pkg/keyring.py", "content": EXAMPLE_MODULE, "license": "MIT"},
            {"path": "tests/test_keyring.py", "content": KEYRING_TEST, "license": "MIT"},
        ],
    )
    return tmp_path


def test_example_to_implementation_quotes_the_example(example_corpus):
    rows = InstructionInduction(example_corpus, max_files=5).docstring_example_to_implementation()
    assert rows
    row = rows[0]
    assert row["category"] == "induced_example_to_implementation"
    assert "ring.rotate" in row["instruction"]
    assert "yield" not in row["output"].split("def ")[0]
    assert row["metadata"]["function"] == "rotate"
    assert row["difficulty"] in {"beginner", "intermediate", "advanced", "expert"}


def test_imports_to_module_lists_exact_imports(example_corpus):
    rows = InstructionInduction(example_corpus, max_files=5).imports_to_module()
    assert rows
    row = rows[0]
    assert row["category"] == "induced_imports_to_module"
    for module in ("hmac", "json", "time", "hashlib"):
        assert module in row["metadata"]["imports"]
        assert module in row["instruction"]
    assert row["output"].startswith('"""')


def test_error_branches_to_tests_covers_each_handler(example_corpus):
    rows = InstructionInduction(example_corpus, max_files=5).error_branches_to_tests()
    assert rows
    row = rows[0]
    assert row["category"] == "induced_error_branches_to_tests"
    assert ["OSError"] in [list(h) for h in row["metadata"]["handlers"]]
    assert "OSError" in row["output"]
    assert "pytest.raises" in row["output"]


def test_contract_to_tests_mirrors_documented_promises(example_corpus):
    rows = InstructionInduction(example_corpus, max_files=5).docstring_to_contract_tests()
    assert rows
    row = rows[0]
    assert row["category"] == "induced_contract_to_tests"
    assert any(p.startswith("Raises:") for p in row["metadata"]["promises"])
    assert "ValueError" in row["instruction"]
    assert "record must be" not in row["instruction"]


def test_generate_reaches_at_least_eight_task_types(example_corpus):
    rows = InstructionInduction(example_corpus, max_files=5).generate(count=12)
    assert len({r["category"] for r in rows}) >= 8
    stats = InstructionInduction(example_corpus, max_files=5).stats()
    assert len(stats["shapes"]) >= 8
