"""End-to-end contract for ``scripts/build_corpus_v1.py``.

The builder is the only path from generated slices to pretraining shards, so
these tests check the properties the training configs rely on:

1. real documents are routed into the right backbone slice and non-permissive
   licenses never reach a shard;
2. held-out content is refused, not silently dropped after the fact;
3. the synthetic share of the *built* corpus stays under 5%;
4. the manifest is complete enough to audit the run (per-slice tokens, realized
   weights, isolation counters, SHA-256).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model.tokenizer import CodeTokenizer
from scripts.build_corpus_v1 import build, real_documents, synthetic_documents
from dataset.production_processor import HeldoutRegistry, LicenseFilter


CODE = '''"""Retry helper with linear backoff."""

import time


def retry(fn, attempts=3, delay=0.1):
    for attempt in range(attempts):
        try:
            return fn()
        except OSError:
            time.sleep(delay * (attempt + 1))
    raise RuntimeError("retries exhausted")
'''

TEST = '''from pkg.retry import retry


def test_retry_succeeds_second_time():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise OSError("boom")
        return "ok"

    assert retry(flaky) == "ok"
'''

DOC = "# Retry\n\nLinear backoff from a 100ms base delay.\n"
DIFF = "diff --git a/pkg/retry.py b/pkg/retry.py\n--- a/pkg/retry.py\n+++ b/pkg/retry.py\n@@\n-x = 1\n+x = 2\n"


def write_corpus(tmp_path: Path, rows: list[dict]) -> Path:
    real = tmp_path / "real"
    real.mkdir()
    with (real / "shard-0.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return real


def real_rows() -> list[dict]:
    return [
        {"content": CODE, "language": "python", "path": "pkg/retry.py", "license": "MIT", "source": "gh"},
        {"content": TEST, "language": "python", "path": "tests/test_retry.py", "license": "Apache-2.0", "source": "gh"},
        {"content": DOC, "language": "markdown", "path": "docs/retry.md", "license": "BSD-3-Clause", "source": "gh"},
        {"content": DIFF, "language": "diff", "path": "pr:12", "license": "MIT", "source": "gh"},
        {"content": CODE, "language": "python", "path": "pkg/vendored.py", "license": "GPL-3.0", "source": "gh"},
        {"content": CODE, "language": "python", "path": "pkg/nolicense.py", "source": "gh"},
    ]


def test_real_documents_route_into_slices_and_drop_bad_licenses(tmp_path):
    real = write_corpus(tmp_path, real_rows())
    docs = list(real_documents(real, LicenseFilter()))
    slices = {doc.slice_name for doc in docs}
    paths = {doc.path for doc in docs}
    assert slices == {"the-stack-code", "tests", "docs-and-comments", "github-prs"}
    assert "pkg/vendored.py" not in paths       # GPL is fail-closed
    assert "pkg/nolicense.py" not in paths      # no license is fail-closed
    assert all(doc.license for doc in docs)


def test_heldout_document_cannot_enter_the_shards(tmp_path):
    real = write_corpus(tmp_path, real_rows())
    registry = HeldoutRegistry()
    registry.register_text(CODE, "pkg/retry.py")
    builder_registry = HeldoutRegistry()
    builder_registry.register_text(CODE, "pkg/retry.py")

    from dataset.production_processor import Document, ProductionShardBuilder

    builder = ProductionShardBuilder(
        tmp_path / "out", CodeTokenizer(), shard_size_tokens=8192, heldout=builder_registry
    )
    doc = Document(content=CODE, language="python", source="gh", license="MIT", path="pkg/retry.py")
    with pytest.raises(ValueError, match="held-out"):
        builder.add_document(doc)


def test_synthetic_documents_carry_provenance_and_multifile_flags():
    docs = list(synthetic_documents(apps=1))
    assert docs, "synthetic slices must produce documents"
    slices = {doc.slice_name for doc in docs}
    assert {"security-sft", "repair-trace-sft", "app-sft", "app-repair-sft"} <= slices
    for doc in docs:
        assert doc.license == "CC0-1.0"
        assert doc.metadata["provenance"] == "synthetic"
        assert doc.content.strip()
        assert ":" in doc.path
    app_docs = [doc for doc in docs if doc.slice_name == "app-sft"]
    assert app_docs and all(doc.is_multi_file for doc in app_docs)


def test_real_build_caps_synthetic_share(tmp_path):
    real = write_corpus(tmp_path, real_rows() * 120)
    out = tmp_path / "shards"
    manifest = build(
        out,
        CodeTokenizer(),
        real_dir=real,
        apps=2,
        seq_length=256,
        shard_size_tokens=4096,
    )

    synthetic = manifest["synthetic"]
    assert synthetic["share"] <= synthetic["cap_share"]
    assert synthetic["within_cap"] is True
    assert manifest["synthetic"]["tokens"] > 0, "the cap must not starve the slices"
    assert manifest["total_packed_tokens"] > 0
    assert manifest["total_packed_tokens"] % 256 == 0, "shards pack whole sequences"

    # isolation invariants
    isolation = manifest["isolation"]
    assert isolation["heldout_overlap_documents"] == 0
    assert isolation["train_val_overlap_documents"] == 0

    # provenance survives into the manifest
    assert manifest["sources"]["the-stack-code"]["documents"] >= 1
    assert "tests" in manifest["sources"]
    assert manifest["licenses"]
    assert "GPL-3.0" not in manifest["licenses"]

    # files on disk match the manifest
    for split in ("train", "val"):
        for shard in manifest["splits"][split]["shards"]:
            path = out / shard["path"]
            assert path.exists()
            assert path.stat().st_size == shard["tokens"] * 4  # uint32
    corpus_manifest = json.loads((out / "corpus_manifest.json").read_text())
    assert corpus_manifest["corpus_version"] == "corpus-v1"
    assert corpus_manifest["builder"] == "scripts/build_corpus_v1.py"
    assert corpus_manifest["mixture"]["security-sft"] == 0.03


def test_a_real_backbone_too_small_for_one_row_is_reported_not_hidden(tmp_path):
    real = write_corpus(tmp_path, real_rows())
    manifest = build(
        tmp_path / "shards-tiny",
        CodeTokenizer(),
        real_dir=real,
        apps=1,
        seq_length=256,
        shard_size_tokens=4096,
    )
    synthetic = manifest["synthetic"]
    assert synthetic["within_cap"] is True          # nothing was admitted
    assert synthetic["tokens"] == 0
    assert synthetic["allowance_starved"] is True   # and the run says so
    assert synthetic["note"]


def test_cap_share_is_respected_when_lowered(tmp_path):
    real = write_corpus(tmp_path, real_rows() * 120)
    manifest = build(
        tmp_path / "shards-strict",
        CodeTokenizer(),
        real_dir=real,
        apps=2,
        seq_length=256,
        shard_size_tokens=4096,
        synthetic_cap_share=0.01,
    )
    assert manifest["synthetic"]["share"] <= 0.01
    assert manifest["synthetic"]["cap_share"] == 0.01


def test_manifest_reports_missing_slices(tmp_path):
    real = write_corpus(tmp_path, real_rows())
    manifest = build(
        tmp_path / "shards-missing",
        CodeTokenizer(),
        real_dir=real,
        apps=1,
        seq_length=256,
        shard_size_tokens=4096,
        include_apps=False,
    )
    assert "recovery-sft" in manifest["missing_slices"]
    assert "project-scaffold-sft" in manifest["missing_slices"]
    assert "the-stack-code" not in manifest["missing_slices"]
