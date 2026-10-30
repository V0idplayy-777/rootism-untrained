"""The held-out routing suite is measured, not tuned after the fact."""

from __future__ import annotations

from evaluation.prompt_suite import (
    HOLDOUT_CASES,
    SPLITS,
    TUNING_CASES,
    evaluate_prompts,
    guards,
    main,
)
from scaffold.blueprints import match_blueprint, match_blueprint_scored


def test_tuning_split_passes_all_routing_guards():
    report = evaluate_prompts(SPLITS["tuning"])
    assert report["metrics"]["catalogue_accuracy"] == 1.0
    assert report["metrics"]["out_refusal_rate"] == 1.0
    assert guards(report) == []


def test_holdout_split_meets_its_recorded_regression_floor():
    report = evaluate_prompts(SPLITS["holdout"])
    metrics = report["metrics"]
    assert metrics["catalogue_accepted"] >= 15
    assert metrics["out_refusal_rate"] == 1.0
    assert metrics["false_claims"] == 0
    assert metrics["overall_accuracy"] >= 0.75


def test_out_of_catalogue_prompts_are_not_claimed():
    for prompt in (
        "a distributed GPU scheduler with Kubernetes operators",
        "a stock trading bot with live market data",
        "a compiler backend for WebAssembly",
        "a game engine editor",
    ):
        assert match_blueprint(prompt) is None


def test_a_match_carries_auditable_evidence():
    match = match_blueprint_scored("a kanban board for our sprint work")
    assert match is not None
    assert match.blueprint.id in {"task-board", "sprint-board"}
    assert match.strong is True
    assert match.score >= 4
    assert any(item.startswith(("hint:", "word:")) for item in match.evidence)


def test_cli_writes_both_splits(tmp_path):
    for split in ("tuning", "holdout"):
        out = tmp_path / f"{split}.json"
        assert main(["--split", split, "--json", str(out)]) == 0
        assert out.exists()


def test_case_counts_are_stable():
    assert len(TUNING_CASES) == 40
    assert len(HOLDOUT_CASES) == 34
