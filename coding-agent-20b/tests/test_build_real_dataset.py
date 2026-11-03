"""Offline tests for ``scripts/build_real_dataset.py``.

``DatasetBuilder`` is driven through its *local* stream paths, which run the
identical downstream code as the HuggingFace path: licence canonicalisation,
quality/complexity floors, dedup, held-out split, shard emission and manifest.
Only ``stream_huggingface_dataset`` touches the network, and it is exercised
here with an injected iterator so no download happens.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model.tokenizer import CodeTokenizer
from scripts.build_real_dataset import (
    DATASETS,
    DatasetBuilder,
    format_instruction,
    stream_huggingface_dataset,
    stream_local_jsonl,
    stream_local_repos,
)

PY_CODE = '''"""Bounded retry helper with linear backoff between attempts."""

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


def retry(fn, attempts=3, delay=0.1):
    """Shortcut for ``Retrier(attempts, delay).call(fn)``."""
    return Retrier(attempts, delay).call(fn)
'''


def write_jsonl(directory: Path, name: str, rows: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def test_stream_local_jsonl_yields_rows_and_skips_broken_lines(tmp_path):
    write_jsonl(tmp_path, "python-0.jsonl", [{"content": PY_CODE, "path": "a.py", "license": "MIT"}])
    with (tmp_path / "python-0.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n\n")

    rows = list(stream_local_jsonl(tmp_path))
    assert len(rows) == 1
    assert rows[0]["path"] == "a.py"


def test_stream_local_repos_reads_files_and_detects_license(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "LICENSE").write_text(
        "MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\n",
        encoding="utf-8",
    )
    (repo / "pkg" / "retry.py").write_text(PY_CODE, encoding="utf-8")
    (repo / "pkg" / "image.png").write_bytes(b"\x89PNG\x00\x00binary")

    rows = list(stream_local_repos(tmp_path))
    assert rows, "a repo with a licence and source must produce rows"
    assert all(row["license"] for row in rows)
    assert any(row["path"].endswith("retry.py") for row in rows)
    assert not any(row["path"].endswith(".png") for row in rows)


def test_stream_local_repos_without_license_is_rejected(tmp_path):
    repo = tmp_path / "nolic"
    repo.mkdir()
    (repo / "mod.py").write_text(PY_CODE, encoding="utf-8")
    rows = list(stream_local_repos(tmp_path))
    assert rows == []


def test_format_instruction_renders_the_chat_contract():
    rendered = format_instruction({"instruction": "Task", "input": "ctx", "output": "ans"})
    assert "<|user|>" in rendered and "Task" in rendered
    assert "ctx" in rendered and "<|assistant|>" in rendered and "ans" in rendered
    assert format_instruction({"prompt": "P"}) == ""
    assert format_instruction({}) == ""


def test_stream_huggingface_dataset_fails_loudly_without_the_dependency():
    """An absent ``datasets`` package is an explicit error, never a silent no-op."""

    spec = next(iter(DATASETS.values()))
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets" or name.startswith("datasets."):
            raise ImportError("no datasets library")
        return real_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(RuntimeError, match="datasets"):
            list(stream_huggingface_dataset(spec, max_samples=5))


def test_stream_huggingface_dataset_filters_language_and_respects_max_samples(monkeypatch, tmp_path):
    """The HF path with an injected dataset: language filter, licence, slice hints."""

    import builtins
    import sys
    import types as pytypes

    spec = next(iter(DATASETS.values()))
    samples = [
        {"content": PY_CODE, "lang": "python", "license": "MIT", "path": "a.py", "repo_name": "demo"},
        {"content": PY_CODE, "lang": "rust", "license": "MIT", "path": "b.rs", "repo_name": "demo"},
        {"content": PY_CODE, "lang": "python", "license": "GPL-3.0", "path": "c.py", "repo_name": "demo"},
        {"content": "", "lang": "python", "license": "MIT", "path": "d.py", "repo_name": "demo"},
    ]

    fake_module = pytypes.ModuleType("datasets")
    fake_module.load_dataset = lambda *a, **k: iter(samples)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    rows = list(stream_huggingface_dataset(spec, languages=["python"], max_samples=10))
    assert len(rows) == 2, "the rust row and the empty row are filtered"
    assert all(row["language"] == "python" for row in rows)
    assert all(row["license"] for row in rows)
    assert all(row["slice"] == spec.slice_hint for row in rows)
    assert all(row["source"].startswith("hf:") for row in rows)

    limited = list(stream_huggingface_dataset(spec, languages=["python"], max_samples=1))
    assert len(limited) == 1


def test_builder_emits_shards_manifest_and_isolated_heldout(tmp_path):
    real = tmp_path / "real"
    rows = [
        {"content": f"{PY_CODE}\n# variant {i}\n", "path": f"pkg/retry_{i}.py", "license": "MIT", "language": "python"}
        for i in range(40)
    ]
    rows.append({"content": PY_CODE, "path": "pkg/gpl.py", "license": "GPL-3.0", "language": "python"})
    write_jsonl(real, "python-0.jsonl", rows)

    out = tmp_path / "shards"
    builder = DatasetBuilder(out, CodeTokenizer(), seq_length=128, heldout_dir=tmp_path / "heldout")
    manifest = builder.build(local_jsonl_dir=real, target_tokens=1_000_000, heldout_every=7)

    assert manifest["total_packed_tokens"] > 0
    assert manifest["total_packed_tokens"] % 128 == 0
    assert manifest["isolation"]["heldout_overlap_documents"] == 0
    assert manifest["isolation"]["train_val_overlap_documents"] == 0
    assert manifest["ingested_sources"]["local_jsonl"] > 0

    assert "GPL-3.0" not in manifest["licenses"]

    held_manifest = tmp_path / "heldout" / "heldout_manifest.json"
    assert held_manifest.exists()
    heldout = json.loads(held_manifest.read_text())
    assert heldout["documents"] > 0
    assert heldout["heldout_sha256"]

    assert (out / "corpus_manifest.json").exists()
    assert (out / "train_manifest.json").exists()


def test_builder_jsonl_emission_matches_the_shards(tmp_path):
    real = tmp_path / "real"
    write_jsonl(
        real,
        "python-0.jsonl",
        [
            {"content": f"{PY_CODE}\n# v{i}\n", "path": f"pkg/m{i}.py", "license": "MIT", "language": "python"}
            for i in range(12)
        ],
    )
    out = tmp_path / "shards"
    emit = tmp_path / "emitted"
    builder = DatasetBuilder(out, CodeTokenizer(), seq_length=128, heldout_dir=tmp_path / "held")
    manifest = builder.build(local_jsonl_dir=real, emit_jsonl_dir=emit, target_tokens=1_000_000)

    emitted = list(emit.glob("python-*.jsonl"))
    assert emitted, "emit_jsonl_dir must receive JSONL shards"
    emitted_rows = [json.loads(line) for shard in emitted for line in shard.read_text().splitlines()]
    assert emitted_rows
    assert all(row["license"] == "MIT" for row in emitted_rows)
    assert all(row["sha256"] for row in emitted_rows)
    assert manifest["total_documents"] == len(emitted_rows)


def test_builder_stops_at_the_token_target(tmp_path):
    real = tmp_path / "real"
    write_jsonl(
        real,
        "python-0.jsonl",
        [
            {"content": f"{PY_CODE}\n# v{i}\n", "path": f"pkg/m{i}.py", "license": "MIT", "language": "python"}
            for i in range(200)
        ],
    )
    builder = DatasetBuilder(tmp_path / "shards", CodeTokenizer(), seq_length=128, heldout_dir=tmp_path / "h")
    manifest = builder.build(local_jsonl_dir=real, target_tokens=2_000)

    assert manifest["total_stream_tokens"] < 2_000 + 4_000


def test_builder_language_filter_and_unknown_dataset_name(tmp_path, caplog):
    real = tmp_path / "real"
    write_jsonl(
        real,
        "python-0.jsonl",
        [{"content": PY_CODE, "path": "a.py", "license": "MIT", "language": "python"}],
    )
    builder = DatasetBuilder(tmp_path / "shards", CodeTokenizer(), seq_length=128, heldout_dir=tmp_path / "h")
    manifest = builder.build(
        local_jsonl_dir=real,
        languages=["rust"],
        dataset_names=["definitely-not-a-dataset"],
        target_tokens=100_000,
    )
    assert manifest["total_documents"] == 0
    assert manifest["ingested_sources"] == {}
