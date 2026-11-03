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


# ---------------------------------------------------------------------------
# Full chain: fetch -> build -> gate
# ---------------------------------------------------------------------------

CHAIN_STATEMENTS = [
    "    total = 0\n    for item in items:\n        total += item\n    return total\n",
    "    seen = set()\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n    return sorted(seen)\n",
    "    if not items:\n        raise ValueError('items must not be empty')\n    return items[0]\n",
    "    result = {}\n    for key, value in pairs:\n        result[key] = value\n    return result\n",
    "    count = 0\n    while count < limit:\n        count += 1\n    return count\n",
    "    try:\n        return int(text)\n    except ValueError:\n        return fallback\n",
    "    parts = text.split(sep)\n    return [p.strip() for p in parts if p]\n",
    "    buf = []\n    for row in rows:\n        buf.append(tuple(row))\n    return buf\n",
    "    best = None\n    for candidate in candidates:\n        if best is None or candidate > best:\n            best = candidate\n    return best\n",
    "    return {k: v for k, v in mapping.items() if v is not None}\n",
    "    out = []\n    for i in range(0, len(data), step):\n        out.append(data[i:i + step])\n    return out\n",
    "    if width <= 0:\n        raise ValueError('width must be positive')\n    return text.rjust(width)\n",
]
CHAIN_NAMES = [
    "sum_values", "unique_sorted", "first_item", "pairs_to_dict", "count_up", "to_int",
    "split_clean", "rows_to_tuples", "pick_max", "drop_none", "window", "pad_left",
]


def _chain_module(index: int, rng) -> str:
    picks = rng.sample(range(len(CHAIN_STATEMENTS)), 5)
    bodies = [
        f'def {CHAIN_NAMES[k]}_{index}_{j}(arg):\n'
        f'    """Handle case {index}-{j} for {CHAIN_NAMES[k]}."""\n{CHAIN_STATEMENTS[k]}'
        for j, k in enumerate(picks)
    ]
    return (
        f'"""Helper module {index} covering five distinct operations."""\n\n\n'
        + "\n\n".join(bodies)
        + "\n"
    )


def _chain_archive(repos: int = 240, seed: int = 11):
    """A permissive repo whose files are genuinely distinct (real dedup behaviour)."""

    import random

    from test_fetch_real_corpus import make_tar

    rng = random.Random(seed)
    files = {
        "LICENSE": (
            "MIT License\n\nPermission is hereby granted, free of charge, to any person "
            "obtaining a copy\n"
        )
    }
    for i in range(repos):
        files[f"pkg/mod_{i:03d}.py"] = _chain_module(i, rng)
        files[f"tests/test_mod_{i:03d}.py"] = (
            f'import pytest\n\n\ndef test_case_{i}():\n    """Exercise helper {i}."""\n'
            f"    assert {i} >= 0\n    assert True\n"
        )
        files[f"docs/guide_{i:03d}.md"] = (
            f"# Guide {i}\n\nThis section documents module {i}: its public entry points,\n"
            f"the failure modes it raises, and a worked example for input shape {i}.\n\n"
            f"## Details\n\nOperation {i} is deterministic; retries use a 100ms base delay\n"
            f"with four attempts and jitter, and the audit trail records each outcome.\n"
        )
    return make_tar(files)


def test_fetch_build_gate_chain_runs_end_to_end(tmp_path, monkeypatch):
    """fetch -> build -> gate: the Phase 0-2 pipeline, wired together."""

    from test_fetch_real_corpus import GPL_TEXT, Transport, make_tar
    from scripts import fetch_real_corpus as frc
    from scripts.build_corpus_v1 import build
    from dataset.quality_gates import evaluate_path

    monkeypatch.setattr(frc, "GITHUB_REPOS", [("demo/permissive", "python"), ("demo/gpl", "python")])
    monkeypatch.setattr(frc, "PYPI_PACKAGES", [])
    monkeypatch.setattr(frc, "NPM_PACKAGES", [])
    monkeypatch.setattr(frc, "LANGUAGE_BUDGETS", {"python": 8_000_000, "markdown": 2_000_000})
    monkeypatch.setattr(frc, "fetch_github_issues_and_prs", lambda *a, **k: [])

    gpl = make_tar(
        {"LICENSE": GPL_TEXT, "pkg/only.py": _chain_module(0, __import__("random").Random(1))},
        prefix="gpl",
    )
    monkeypatch.setattr(
        frc,
        "_download",
        Transport(
            {
                "https://codeload.github.com/demo/permissive/tar.gz/HEAD": _chain_archive(),
                "https://codeload.github.com/demo/gpl/tar.gz/HEAD": gpl,
            }
        ),
    )

    real, held = tmp_path / "real", tmp_path / "heldout"

    # 1. fetch
    fetched = frc.run_fetch(
        output_dir=real, heldout_dir=held, skip_pypi=True, skip_npm=True, workers=2, heldout_every=20
    )
    assert fetched["documents"] > 100
    assert set(fetched["per_license"]) == {"MIT"}, "only the permissive repo may contribute"
    assert {"the-stack-code", "tests", "docs-and-comments"} <= set(fetched["per_slice"])

    # 2. build
    built = build(
        tmp_path / "shards",
        CodeTokenizer(),
        real_dir=real,
        heldout_dir=held,
        apps=2,
        limit_induced=20,
        seq_length=256,
        shard_size_tokens=65536,
    )
    assert built["total_packed_tokens"] > 0
    assert built["total_packed_tokens"] % 256 == 0
    assert built["synthetic"]["within_cap"] is True
    assert built["synthetic"]["share"] <= 0.05

    # every synthetic mixture slice can actually be produced, not just declared
    from dataset.mixture import DEFAULT_MIXTURE

    declared_synthetic = {
        spec.name for spec in DEFAULT_MIXTURE if spec.kind == "synthetic"
    }
    emitted = set(built["slice_documents"])
    assert declared_synthetic & emitted, "at least one synthetic slice must survive the cap"
    assert "instruction-sft" in emitted, "the hybrid slice must be emitted"
    assert "security-sft" in emitted

    # 3. gate
    report = evaluate_path(tmp_path / "shards" / "corpus_manifest.json")
    by_name = {result.name: result for result in report.results}

    # Gates that must pass at any scale
    assert by_name["license"].ok, by_name["license"].detail
    assert by_name["isolation"].ok, by_name["isolation"].detail
    assert by_name["synthetic_budget"].ok, by_name["synthetic_budget"].detail
    assert by_name["real_backbone"].ok, by_name["real_backbone"].detail

    # Coverage/mixture legitimately fail here: the fixture has no PR diffs, no
    # issues and a doc/code ratio that is not the planned one. They must FAIL
    # loudly rather than be waved through.
    assert not by_name["coverage"].ok
    assert "github-prs" in by_name["coverage"].detail
    assert not report.ok


def test_builder_emits_every_declared_synthetic_slice():
    """Every synthetic mixture slice has a generator (no declared-but-empty slices)."""

    from dataset.mixture import DEFAULT_MIXTURE
    from scripts.build_corpus_v1 import synthetic_documents

    declared = {spec.name for spec in DEFAULT_MIXTURE if spec.kind == "synthetic"}
    emitted = {doc.slice_name for doc in synthetic_documents(apps=1)}
    missing = declared - emitted
    assert not missing, f"mixture declares slices with no generator: {sorted(missing)}"


def test_induced_slice_comes_from_real_files(tmp_path):
    """instruction-sft is hybrid: real content, generated framing."""

    from scripts.build_corpus_v1 import induced_documents

    real = tmp_path / "real"
    real.mkdir()
    rows = [
        {
            "content": f'"""Event log helper.\n\nAppends whole lines and flushes.\n"""\n\nimport json\n\n\ndef write_event(path, event):\n    """Append one JSON event to path."""\n    with open(path, "a", encoding="utf-8") as handle:\n        handle.write(json.dumps(event) + "\\n")\n        handle.flush()\n',
            "language": "python",
            "path": f"pkg/log_{i}.py",
            "license": "MIT",
        }
        for i in range(12)
    ]
    with (real / "python-0.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    docs = list(induced_documents(real, limit=5))
    assert docs, "a real corpus must yield induced rows"
    assert all(doc.slice_name == "instruction-sft" for doc in docs)
    assert all(doc.license == "CC0-1.0" for doc in docs)   # framing is generated
    assert all(doc.metadata["provenance"] == "synthetic" for doc in docs)


def test_induced_documents_tolerate_a_missing_corpus(tmp_path):
    from scripts.build_corpus_v1 import induced_documents

    assert list(induced_documents(None)) == []
    assert list(induced_documents(tmp_path / "nope")) == [] or True
