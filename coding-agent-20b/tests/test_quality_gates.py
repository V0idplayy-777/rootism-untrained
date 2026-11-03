"""Pre-training quality gates: each gate must fail for its own reason."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset.quality_gates import (
    check_coverage,
    check_diversity,
    check_isolation,
    check_licenses,
    check_mixture,
    check_real_backbone,
    check_synthetic_budget,
    evaluate_manifest,
    evaluate_path,
)
from model.tokenizer import CodeTokenizer
from scripts.build_corpus_v1 import build


def manifest(**overrides) -> dict:
    """A manifest that passes every gate, mutated per test."""

    base = {
        "corpus_version": "corpus-v1",
        "total_documents": 1000,
        "slice_tokens": {
            "the-stack-code": 480_000,
            "tests": 70_000,
            "github-prs": 70_000,
            "github-issues": 60_000,
            "docs-and-comments": 50_000,
            "instruction-sft": 60_000,
            "app-sft": 6_000,
            "repair-trace-sft": 5_000,
            "security-sft": 4_000,
            "app-repair-sft": 3_000,
            "app-preference-dpo": 2_000,
            "recovery-sft": 2_000,
            "project-scaffold-sft": 1_000,
        },
        "licenses": {"MIT": 600, "Apache-2.0": 300, "BSD-3-Clause": 100},
        "diversity": {
            "unique_files": 900,
            "unique_ast_patterns": 400,
            "multi_file_ratio": 0.15,
        },
        "isolation": {"heldout_overlap_documents": 0, "train_val_overlap_documents": 0},
        "synthetic": {"within_cap": True},
    }
    base.update(overrides)
    return base


def test_the_reference_manifest_passes_every_gate():
    report = evaluate_manifest(manifest())
    assert report.ok, report.render()
    assert {result.name for result in report.results} == {
        "coverage",
        "mixture",
        "license",
        "diversity",
        "isolation",
        "synthetic_budget",
        "real_backbone",
    }


def test_coverage_catches_a_missing_backbone_slice():
    broken = manifest()
    broken["slice_tokens"] = {k: v for k, v in broken["slice_tokens"].items() if k != "tests"}
    result = check_coverage(broken)
    assert not result.ok
    assert "tests" in result.detail


def test_coverage_catches_a_thin_stack_slice():
    broken = manifest()
    broken["slice_tokens"]["the-stack-code"] = 1_000
    result = check_coverage(broken)
    assert not result.ok
    assert "the-stack-code" in result.detail


def test_mixture_flags_share_drift():
    broken = manifest()
    broken["slice_tokens"]["github-prs"] = 900_000
    result = check_mixture(broken)
    assert not result.ok
    assert "github-prs" in result.detail


def test_mixture_ignores_synthetic_slices_which_are_capped_by_token_budget():
    capped = manifest()
    capped["slice_tokens"]["security-sft"] = 1
    assert check_mixture(capped).ok


def test_license_gate_is_fail_closed():
    for bad in ({"GPL-3.0": 1}, {"AGPL-3.0-only": 1}, {"proprietary": 1}):
        broken = manifest(licenses={"MIT": 10, **bad})
        assert not check_licenses(broken).ok, bad
    assert not check_licenses(manifest(licenses={})).ok


def test_diversity_gate_rejects_one_file_repeated():
    flat = manifest(diversity={"unique_files": 1, "unique_ast_patterns": 1, "multi_file_ratio": 0.0})
    result = check_diversity(flat)
    assert not result.ok
    assert "pattern ratio" in result.detail


def test_isolation_gate_rejects_any_overlap():
    leaked = manifest(isolation={"heldout_overlap_documents": 2, "train_val_overlap_documents": 0})
    assert not check_isolation(leaked).ok
    assert not check_isolation(
        manifest(isolation={"heldout_overlap_documents": 0, "train_val_overlap_documents": 3})
    ).ok


def test_synthetic_budget_gate_uses_tokens_not_the_flag_alone():
    over = manifest()
    over["slice_tokens"]["app-preference-dpo"] = 900_000
    result = check_synthetic_budget(over)
    assert not result.ok
    assert "cap" in result.detail

    flagged = manifest(synthetic={"within_cap": False})
    assert not check_synthetic_budget(flagged).ok


def test_real_backbone_floor():
    synthetic_heavy = manifest()
    synthetic_heavy["slice_tokens"] = {
        "the-stack-code": 100_000,
        "security-sft": 400_000,
        "app-sft": 500_000,
    }
    assert not check_real_backbone(synthetic_heavy).ok


def test_report_raises_with_the_failing_gate_names():
    broken = manifest(licenses={"GPL-3.0": 5})
    report = evaluate_manifest(broken)
    with pytest.raises(AssertionError, match="license"):
        report.assert_passing()


def test_evaluate_path_reads_a_written_manifest(tmp_path):
    path = tmp_path / "corpus_manifest.json"
    path.write_text(json.dumps(manifest()), encoding="utf-8")
    assert evaluate_path(path).ok
    assert evaluate_path(path).to_dict()["failures"] == []

    path.write_text(json.dumps(manifest(licenses={"SSPL-1.0": 1})), encoding="utf-8")
    assert not evaluate_path(path).ok


def build_corpus(tmp_path: Path) -> Path:
    """Build a real (small) corpus with enough backbone to fund synthetic rows."""

    code = (
        '"""Bounded retry with backoff."""\n\n'
        "import time\n\n\n"
        "def retry(fn, attempts=3, delay=0.1):\n"
        "    for attempt in range(attempts):\n"
        "        try:\n"
        "            return fn()\n"
        "        except OSError:\n"
        "            time.sleep(delay * (attempt + 1))\n"
        '    raise RuntimeError("retries exhausted")\n'
    )
    test = (
        "from pkg.retry import retry\n\n\n"
        "def test_retry_succeeds_second_time():\n"
        '    assert retry(lambda: "ok") == "ok"\n'
    )
    doc = "# Retry\n\nLinear backoff from a 100ms base.\n"
    real = tmp_path / "real"
    real.mkdir()
    rows = []
    for i in range(150):
        rows.append({"content": f"{code}\n# file {i}\n", "language": "python", "path": f"pkg/retry_{i}.py", "license": "MIT", "source": "gh"})
        rows.append({"content": test, "language": "python", "path": f"tests/test_retry_{i}.py", "license": "Apache-2.0", "source": "gh"})
        rows.append({"content": doc, "language": "markdown", "path": f"docs/retry_{i}.md", "license": "BSD-3-Clause", "source": "gh"})
    with (real / "shard-0.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    out = tmp_path / "shards"
    build(out, CodeTokenizer(), real_dir=real, apps=1, seq_length=256, shard_size_tokens=8192)
    return out / "corpus_manifest.json"


def test_built_corpus_passes_the_gates_we_can_meet_at_this_size(tmp_path):
    """A real build satisfies isolation/license/budget; coverage is size-bound."""

    path = build_corpus(tmp_path)
    report = evaluate_path(path)
    by_name = {result.name: result for result in report.results}

    assert by_name["license"].ok
    assert by_name["isolation"].ok
    assert by_name["synthetic_budget"].ok

    assert not by_name["diversity"].ok

    assert not by_name["coverage"].ok
    assert "github-issues" in by_name["coverage"].detail
    assert not by_name["mixture"].ok
    assert not report.ok


def test_trainer_refuses_a_manifest_that_fails_the_gates(tmp_path):
    """The 3B trainer must not start a run on a corpus that fails a gate."""

    from training.train_3b import verify_corpus_manifest_for_3b

    data_dir = tmp_path / "shards"
    data_dir.mkdir()
    (data_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest(licenses={"GPL-3.0": 3})), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="quality gates"):
        verify_corpus_manifest_for_3b(data_dir, 4096)

    (data_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest() | {"seq_length": 4096}), encoding="utf-8"
    )
    info = verify_corpus_manifest_for_3b(data_dir, 4096)
    assert info is not None
    assert info["quality_gates"]["ok"] is True


def test_trainer_still_rejects_isolation_breaches_before_the_gates(tmp_path):
    from training.train_3b import verify_corpus_manifest_for_3b

    data_dir = tmp_path / "shards-breach"
    data_dir.mkdir()
    (data_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest(isolation={"heldout_overlap_documents": 1, "train_val_overlap_documents": 0})),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="held-out contamination"):
        verify_corpus_manifest_for_3b(data_dir, 4096)


def test_no_manifest_means_no_gate_claim(tmp_path):
    from training.train_3b import verify_corpus_manifest_for_3b

    assert verify_corpus_manifest_for_3b(tmp_path, 4096) is None
