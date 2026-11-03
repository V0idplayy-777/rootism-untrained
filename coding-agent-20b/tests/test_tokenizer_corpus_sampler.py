"""``scripts/sample_tokenizer_corpus.py``: the licence-gated tokenizer corpus.

These tests pin the properties the tokenizer trainer depends on: a fail-closed
licence gate, per-language quotas, deterministic dedup, atomic shards and a
corpus hash that actually changes when the content changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.sample_tokenizer_corpus import CorpusSampler

CODE = '''"""Bounded retry with linear backoff."""

import time


class Retrier:
    """Run a callable, retrying OSError with a linear delay."""

    def __init__(self, attempts=3, delay=0.1):
        self.attempts = attempts
        self.delay = delay

    def call(self, fn):
        """Invoke fn until it stops raising OSError."""
        for attempt in range(self.attempts):
            try:
                return fn()
            except OSError:
                time.sleep(self.delay * (attempt + 1))
        raise RuntimeError("retries exhausted")
'''


PY_A = CODE

PY_B = '''"""Line-delimited JSON event log with whole-line writes."""

import json
import os


class EventLog:
    """Append-only log that never leaves a partial record behind."""

    def __init__(self, path):
        self.path = path
        self.handle = None

    def open(self):
        """Open the underlying file in append mode."""
        self.handle = open(self.path, "a", encoding="utf-8")
        return self

    def write(self, event):
        """Serialise one event and flush it to disk immediately."""
        if self.handle is None:
            raise RuntimeError("log is not open")
        self.handle.write(json.dumps(event) + "\\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self):
        """Release the handle."""
        if self.handle is not None:
            self.handle.close()
            self.handle = None
'''

PY_C = '''"""Token bucket rate limiter driven by a monotonic clock."""

import time


class TokenBucket:
    """Allow bursts up to ``capacity`` and refill at ``rate`` per second."""

    def __init__(self, capacity, rate, clock=time.monotonic):
        self.capacity = float(capacity)
        self.rate = float(rate)
        self.clock = clock
        self.tokens = float(capacity)
        self.updated = clock()

    def _refill(self):
        now = self.clock()
        elapsed = now - self.updated
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.updated = now

    def allow(self, cost=1.0):
        """Consume ``cost`` tokens if available."""
        self._refill()
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def wait_time(self, cost=1.0):
        """Seconds until ``cost`` tokens are available."""
        self._refill()
        if self.tokens >= cost:
            return 0.0
        return (cost - self.tokens) / self.rate
'''

PY_D = '''"""Pluggable backoff strategies: constant, linear and exponential."""

import random


def constant(delay):
    """Always wait ``delay`` seconds."""

    def strategy(attempt):
        return delay

    return strategy


def linear(base, cap=None):
    """Wait ``base * attempt``, optionally capped."""

    def strategy(attempt):
        value = base * max(1, attempt)
        return min(value, cap) if cap is not None else value

    return strategy


def exponential(base, factor=2.0, cap=None, jitter=0.0, rng=None):
    """Wait ``base * factor ** (attempt - 1)`` with optional jitter."""
    source = rng or random.Random(0)

    def strategy(attempt):
        value = base * (factor ** max(0, attempt - 1))
        if cap is not None:
            value = min(value, cap)
        if jitter:
            value = value * (1.0 + source.uniform(-jitter, jitter))
        return max(0.0, value)

    return strategy
'''


def item(content: str = CODE, path: str = "pkg/retry.py", license: str = "MIT", **extra) -> dict:
    return {"content": content, "path": path, "license": license, "language": "python", **extra}


def test_accept_rejects_empty_short_and_unlicensed():
    sampler = CorpusSampler("unused", languages=["python"], min_lines=5)
    assert sampler.accept({"content": ""})[0] is False
    assert sampler.accept({"content": "x = 1\n"})[1] == "size"
    assert sampler.accept(item(license="GPL-3.0"))[1] == "license"
    assert sampler.accept(item(license=""))[1] == "license"
    ok, lang = sampler.accept(item())
    assert ok is True and lang == "python"


def test_accept_rejects_other_languages():
    sampler = CorpusSampler("unused", languages=["python"], min_lines=5)
    assert sampler.accept(item(language="rust"))[1] == "language"


def test_accept_is_a_pure_peek_that_does_not_consume_the_dedup_slot():
    """Probing a document must not make the later ``add`` reject it."""

    sampler = CorpusSampler("unused", languages=["python"], min_lines=5)
    assert sampler.accept(item())[0] is True
    assert sampler.accept(item())[0] is True, "accept must be repeatable"
    assert sampler.add(item()) is True, "a document that was only probed must still be storable"


def test_add_rejects_exact_duplicates():
    sampler = CorpusSampler("unused", languages=["python"], min_lines=5)
    assert sampler.add(item()) is True
    assert sampler.add(item()) is False, "identical content is a duplicate"
    assert sampler.counts["python"] == 1


def test_unknown_license_policy_is_explicit():
    strict = CorpusSampler("u1", languages=["python"], min_lines=5)
    assert strict.accept(item(license=""))[1] == "license"

    lenient = CorpusSampler("u2", languages=["python"], min_lines=5, allow_unknown_license=True)
    assert lenient.accept(item(license=""))[0] is True


def test_near_duplicates_are_rejected_at_the_similarity_threshold():
    """A one-line edit in a 20-line file is still a near-duplicate."""

    sampler = CorpusSampler("unused", languages=["python"], min_lines=5)
    assert sampler.add(item(path="a.py")) is True
    assert sampler.add(item(path="b.py", content=CODE + "\n# tiny change\n")) is False
    assert sampler.dedup.stats()["near_duplicates"] >= 1


def test_add_respects_the_per_language_quota(tmp_path):
    sampler = CorpusSampler(tmp_path, languages=["python"], per_language_target_docs=2, min_lines=5)
    assert sampler.add(item(path="a.py", content=PY_A)) is True
    assert sampler.add(item(path="b.py", content=PY_B)) is True
    assert sampler.add(item(path="c.py", content=PY_C)) is False, "quota reached"
    assert sampler.counts["python"] == 2
    assert sampler.done() is True


def test_flush_writes_atomic_shards_and_manifest(tmp_path):
    sampler = CorpusSampler(tmp_path, languages=["python"], per_language_target_docs=5, shard_size=2, min_lines=5)
    for name, body in (("a.py", PY_A), ("b.py", PY_B), ("c.py", PY_C)):
        assert sampler.add(item(path=name, content=body)) is True
    sampler.flush()
    assert sampler.add(item(path="d.py", content=PY_D)) is True
    sampler.flush()

    shards = sorted(tmp_path.glob("python-*.jsonl"))
    assert len(shards) >= 2
    assert not list(tmp_path.glob("*.tmp")), "shard writes must be atomic"

    rows = [json.loads(line) for shard in shards for line in shard.read_text().splitlines()]
    assert len(rows) == 4
    assert len({row["path"] for row in rows}) == 4, "no document is written twice"
    assert all(row["license"] == "MIT" for row in rows)
    assert all(len(row["_sha256"]) == 64 for row in rows)

    manifest = sampler.manifest()
    assert manifest["documents"] == 4
    assert manifest["per_language"] == {"python": 4}
    assert manifest["shards"] == len(shards)
    assert len(manifest["corpus_sha256"]) == 64


def test_corpus_hash_is_content_addressed(tmp_path):
    def build(tag: str) -> str:
        sampler = CorpusSampler(tmp_path / tag, languages=["python"], min_lines=5, shard_size=10)
        sampler.add(item(path="a.py", content=PY_A if tag == "one" else PY_B))
        return sampler.manifest()["corpus_sha256"]

    assert build("one") == build("one")
    assert build("one") != build("two"), "different content must change the corpus hash"


def test_nothing_is_written_for_a_rejected_batch(tmp_path):
    sampler = CorpusSampler(tmp_path, languages=["python"], min_lines=5)
    assert sampler.add(item(license="AGPL-3.0")) is False
    sampler.flush()
    assert list(tmp_path.glob("*.jsonl")) == []
    assert sampler.manifest()["documents"] == 0
