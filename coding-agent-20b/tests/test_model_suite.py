"""The model-path harness must compile, test and report honestly."""

from __future__ import annotations

from evaluation.model_suite import TASKS, evaluate_tasks, main


def _slugify_double(prompt: str, max_tokens: int) -> str:
    if "decomposition plan" in prompt:
        return "1. slugify(text): normalise text into a url slug\nInvariants:\n- lowercase\n"
    if "Implement the module interface" in prompt:
        return (
            '"""Slug helpers."""\n\nimport re\n\n\n'
            "def slugify(text: str) -> str:\n"
            '    """Lowercase and hyphenate text."""\n'
            "    ...\n"
        )
    if "Write ONLY the body of" in prompt:
        return (
            'return "-".join('
            'part for part in re.sub(r"[^a-z0-9]+", "-", text.lower()).split("-") if part'
            ")"
        )
    return "return text"


def test_longform_harness_compiles_and_passes_held_out_tests():
    task = next(t for t in TASKS if t.id == "slugify")
    report = evaluate_tasks([task], _slugify_double, timeout=60)
    assert report["metrics"]["tasks"] == 1
    assert report["metrics"]["compiled"] == 1
    assert report["metrics"]["tests_passed"] == 1, report["results"][0]
    assert report["results"][0]["units"] >= 1


def test_harness_reports_a_failing_backend_without_crashing():
    task = next(t for t in TASKS if t.id == "slugify")

    def broken(prompt: str, max_tokens: int) -> str:
        raise RuntimeError("model offline")

    report = evaluate_tasks([task], broken)
    assert report["metrics"]["compiled"] == 0
    assert "model offline" in " ".join(report["results"][0]["errors"])


def test_cli_without_a_checkpoint_exits_two_and_writes_a_report(tmp_path):
    out = tmp_path / "model.json"
    code = main(["--backend", "none", "--json", str(out)])
    assert code == 2
    assert "No trained checkpoint" in out.read_text()
