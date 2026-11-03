"""Tests for the ingestion capacity plan and the post-fetch acceptance audit.

The plan is arithmetic, so the tests pin the arithmetic (monotonicity, the
documented funnel, the raw->surviving relationship) rather than a magic number.
The audit is a gate, so every finding gets a case where it *fails* for its own
reason - a gate that cannot fail is decoration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset.ingestion_plan import (
    DEFAULT_FUNNEL,
    FunnelStage,
    IngestionPlan,
    audit_fetch_manifest,
    audit_path,
    funnel_yield,
)


def healthy_manifest(**overrides):
    base = {
        "corpus_version": "corpus-v1-real",
        "documents": 250_000,
        "chars": 108_000_000_000,
        "unique_sources": 640,
        "per_language": {
            language: {"documents": 10_000, "chars": 100_000_000}
            for language in (
                "python",
                "typescript",
                "javascript",
                "go",
                "rust",
                "java",
                "cpp",
                "c",
                "csharp",
                "ruby",
                "php",
                "kotlin",
            )
        },
        "per_license": {"MIT": 120_000, "Apache-2.0": 90_000, "BSD-3-Clause": 40_000},
        "dedup": {
            "checked": 900_000,
            "exact_duplicates": 60_000,
            "near_duplicates": 120_000,
            "dedup_rate": 0.2,
        },
        "heldout": {"documents": 12_000, "unique_hashes": 11_800},
        "sources_configured": {"github": 1077, "pypi": 278, "npm": 307},
        "budget_usage": {
            "python": {"chars": 300_000_000, "budget_chars": 80_000_000, "saturated": True}
        },
    }
    base.update(overrides)
    return base


def test_the_documented_funnel_multiplies_to_the_reported_yield():
    expected = 1.0
    for stage in DEFAULT_FUNNEL:
        expected *= stage.keep
    assert funnel_yield() == pytest.approx(expected)
    assert 0.10 < funnel_yield() < 0.50, "document filters should retain a plausible minority"


def test_the_plan_asks_for_more_raw_code_than_it_needs_to_train_on():
    plan = IngestionPlan(target_train_tokens=18_000_000_000)
    assert plan.required_surviving_tokens == pytest.approx(30_000_000_000, rel=1e-6)
    assert plan.required_raw_tokens > plan.required_surviving_tokens
    assert plan.required_raw_tokens == pytest.approx(
        plan.required_surviving_tokens / plan.yield_fraction, rel=1e-6
    )
    assert plan.required_raw_bytes == pytest.approx(plan.required_raw_tokens * 3.6, rel=1e-9)


def test_the_plan_is_scalar_in_its_target():
    small = IngestionPlan(target_train_tokens=1_000_000_000)
    large = IngestionPlan(target_train_tokens=5_000_000_000)
    assert large.required_raw_tokens == pytest.approx(small.required_raw_tokens * 5, rel=1e-6)


def test_extra_headroom_lowers_the_raw_requirement():
    lean = IngestionPlan(headroom=1.0)
    padded = IngestionPlan(headroom=2.0)
    assert padded.required_surviving_tokens == 2 * lean.required_surviving_tokens


def test_stage_table_is_monotone_and_ends_at_the_surviving_target():
    plan = IngestionPlan()
    stages = plan.stages()
    assert stages[0]["tokens_before"] == plan.required_raw_tokens
    for earlier, later in zip(stages, stages[1:]):
        assert later["tokens_before"] == earlier["tokens_after"]
        assert later["tokens_after"] <= later["tokens_before"]
    assert stages[-1]["tokens_after"] == pytest.approx(
        plan.required_surviving_tokens, rel=1e-6
    )


def test_repository_overhead_is_separate_from_the_token_funnel():
    plan = IngestionPlan()
    expected = plan.required_raw_bytes / (0.35 * 0.55)
    assert plan.required_repository_bytes == pytest.approx(expected, rel=1e-9)
    assert plan.required_repository_bytes > plan.required_raw_bytes
    assert {row["stage"] for row in plan.stages()}.isdisjoint(
        {"tarball_vs_repo_size", "source_share"}
    ), "byte procurement overhead must not be applied as a token filter"


def test_the_curated_inventory_is_not_discounted_for_licence_twice():
    plan = IngestionPlan(github_raw_tokens=1_000_000)
    post_license_yield = 0.45 * 0.82 * 0.95
    assert plan.github_surviving_tokens == int(1_000_000 * post_license_yield)


def test_the_curated_fetch_path_cannot_supply_the_whole_requirement():
    """The plan must say out loud that The Stack has to supply the bulk."""

    plan = IngestionPlan()
    roles = plan.source_roles()
    assert plan.github_surviving_tokens < plan.required_surviving_tokens
    assert plan.stack_required_surviving_tokens > 0
    assert (
        roles["the-stack-v2"]["share_of_requirement"]
        > roles["curated-github-pypi-npm"]["share_of_requirement"]
    )
    assert plan.stack_required_surviving_tokens + plan.github_surviving_tokens == pytest.approx(
        plan.required_surviving_tokens, rel=1e-6
    )


def test_per_language_targets_sum_to_one_and_survive_the_target():
    plan = IngestionPlan()
    per_language = plan.per_language()
    assert sum(row["weight"] for row in per_language.values()) == pytest.approx(1.0)
    assert sum(row["surviving_tokens"] for row in per_language.values()) == pytest.approx(
        plan.required_surviving_tokens, rel=0.01
    )
    assert per_language["python"]["weight"] == max(row["weight"] for row in per_language.values())


def test_plan_serialises_and_names_the_sources_it_depends_on():
    payload = IngestionPlan(tokens_per_repo=50_000).to_dict()
    assert payload["required_repositories"] == pytest.approx(
        payload["required_surviving_tokens"] / 50_000, rel=0.01
    )
    assert set(payload["sources"]) == {"the-stack-v2", "curated-github-pypi-npm"}
    assert "caveat" in payload
    json.dumps(payload)




def test_a_healthy_fetch_passes_every_audit_finding():
    report = audit_fetch_manifest(healthy_manifest())
    assert report.ok, report.render()
    assert {finding.name for finding in report.findings} == {
        "surviving_token_scale",
        "document_scale",
        "language_coverage",
        "language_saturation",
        "deduplication_ran",
        "deduplication_sane",
        "licence_gate",
        "heldout_split",
        "source_list_exercised",
    }


def test_a_small_fetch_fails_on_scale_not_on_anything_else():
    small = healthy_manifest(chars=2_000_000_000, documents=20_000)
    report = audit_fetch_manifest(small)
    failures = {finding.name for finding in report.findings if not finding.ok}
    assert failures == {"surviving_token_scale", "document_scale"}, report.render()
    detail = next(f.detail for f in report.findings if f.name == "surviving_token_scale")
    assert "% of plan" in detail


def test_language_poverty_fails_the_coverage_finding():
    poor = healthy_manifest(per_language={"python": {"documents": 1, "chars": 10_000}})
    report = audit_fetch_manifest(poor, min_languages=10)
    coverage = next(f for f in report.findings if f.name == "language_coverage")
    assert not coverage.ok
    assert "1 languages" in coverage.detail


def test_a_corpus_that_is_mostly_duplicates_fails_the_sane_rate_finding():
    duplicated = healthy_manifest(
        dedup={"checked": 100, "exact_duplicates": 70, "near_duplicates": 10, "dedup_rate": 0.8}
    )
    report = audit_fetch_manifest(duplicated)
    sane = next(f for f in report.findings if f.name == "deduplication_sane")
    assert not sane.ok
    ran = next(f for f in report.findings if f.name == "deduplication_ran")
    assert ran.ok


def test_no_dedup_at_all_fails_because_it_never_ran():
    report = audit_fetch_manifest(healthy_manifest(dedup={}))
    ran = next(f for f in report.findings if f.name == "deduplication_ran")
    assert not ran.ok


def test_missing_heldout_split_fails():
    report = audit_fetch_manifest(healthy_manifest(heldout={"documents": 0}))
    heldout = next(f for f in report.findings if f.name == "heldout_split")
    assert not heldout.ok


def test_licence_finding_rejects_an_empty_license_table():
    report = audit_fetch_manifest(healthy_manifest(per_license={}))
    licence = next(f for f in report.findings if f.name == "licence_gate")
    assert not licence.ok


def test_licence_finding_rejects_copyleft_even_though_the_id_is_not_empty():
    report = audit_fetch_manifest(healthy_manifest(per_license={"MIT": 10, "GPL-3.0": 2}))
    licence = next(f for f in report.findings if f.name == "licence_gate")
    assert not licence.ok
    assert "GPL-3.0" in licence.detail


def test_curated_scope_uses_the_curated_allocation_not_the_stack_residual():
    plan = IngestionPlan()
    chars = int(plan.github_surviving_tokens * 3.6)
    report = audit_fetch_manifest(healthy_manifest(chars=chars), plan=plan, scope="curated")
    scale = next(f for f in report.findings if f.name == "surviving_token_scale")
    assert scale.ok, scale.detail
    assert "for curated" in scale.detail

    combined = audit_fetch_manifest(healthy_manifest(chars=chars), plan=plan, scope="combined")
    combined_scale = next(f for f in combined.findings if f.name == "surviving_token_scale")
    assert not combined_scale.ok


def test_audit_rejects_an_unknown_scope():
    with pytest.raises(ValueError, match="scope"):
        audit_fetch_manifest(healthy_manifest(), scope="everything")


def test_audit_assert_passing_raises_with_the_failing_gate_names():
    report = audit_fetch_manifest(healthy_manifest(chars=1, documents=1))
    with pytest.raises(AssertionError, match="real-corpus audit failed"):
        report.assert_passing()


def test_audit_path_reads_a_manifest_from_disk(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(healthy_manifest()), encoding="utf-8")
    assert audit_path(path).ok

    path.write_text(json.dumps(healthy_manifest(documents=10)), encoding="utf-8")
    assert not audit_path(path).ok
