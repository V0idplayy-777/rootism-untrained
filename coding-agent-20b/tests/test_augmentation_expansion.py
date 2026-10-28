"""Distribution-shift generators: conflicts, dialogues, recovery, families."""

import ast

from dataset.augmentation import (
    DOMAIN_PAIRS,
    ClarificationDialogue,
    ConceptCombiner,
    ConstraintConflictGenerator,
    RecoveryInstructionGenerator,
    augment_rows,
)


def test_domain_pair_count_and_capabilities_covered():
    from dataset.augmentation import DOMAIN_CAPABILITIES

    assert len(DOMAIN_PAIRS) >= 18
    for primary, secondary, _task in DOMAIN_PAIRS:
        assert primary in DOMAIN_CAPABILITIES, primary
        assert secondary in DOMAIN_CAPABILITIES, secondary


def test_every_cross_domain_skeleton_compiles():
    tasks = ConceptCombiner(seed=7).combine(len(DOMAIN_PAIRS))
    assert len(tasks) == len(DOMAIN_PAIRS)
    for task in tasks:
        ast.parse(task["input_context"])
        assert task["category"] == "cross_domain_synthesis"


def test_constraint_conflict_rows_name_the_tension():
    rows = ConstraintConflictGenerator().generate()
    assert rows
    for row in rows:
        assert row["category"] == "constraint_conflict"
        assert len(row["metadata"]["conflicts"]) == 2
        lowered = row["output"].lower()
        assert any(w in lowered for w in ("conflict", "contradict", "tension"))
        assert row["output"].rstrip().endswith("?")


def test_clarification_dialogue_rows_are_multi_turn():
    rows = ClarificationDialogue().generate()
    assert rows
    for row in rows:
        assert row["category"] == "clarification_dialogue"
        assert row["metadata"]["turns"] >= 4
        assert row["output"].count("user:") >= 2
        assert row["output"].count("assistant:") >= 2
        first_assistant = row["output"].split("assistant:")[1]
        assert "?" in first_assistant.split("user:")[0]


def test_recovery_rows_preach_changed_approach():
    rows = RecoveryInstructionGenerator().generate()
    assert rows
    for row in rows:
        assert row["category"] == "error_recovery"
        assert row["metadata"]["failure_class"]
        assert row["metadata"]["anti_pattern"]
        assert row["metadata"]["anti_pattern"].split()[0].lower() not in (
            row["output"].split()[0].lower()
        )


def test_augment_rows_includes_new_categories():
    rows = augment_rows(
        [{"instruction": "write a function", "output": "def f(): pass"}],
        per_row=0,
        include_cross_domain=0,
        include_clarification=0,
    )
    cats = {r.get("category", "original") for r in rows}
    assert "constraint_conflict" in cats
    assert "clarification_dialogue" in cats
    assert "error_recovery" in cats


def test_generators_are_deterministic():
    a = ConstraintConflictGenerator(seed=3).generate()
    b = ConstraintConflictGenerator(seed=3).generate()
    assert a == b
    a = RecoveryInstructionGenerator(seed=3).generate()
    b = RecoveryInstructionGenerator(seed=3).generate()
    assert a == b
