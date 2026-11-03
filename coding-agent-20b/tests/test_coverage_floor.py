"""Coverage floors for the Corpus v1 pipeline modules.

A module that no test touches is not "done" — it is unverified code on the path
to training data. These floors encode that rule so the corpus pipeline cannot
silently rot back to 0% (which is exactly what happened to
``scripts/fetch_real_corpus.py`` before these tests existed).

Run interactively:

    python -m coverage run --source=dataset,scripts -m pytest tests/ -q
    python -m coverage report --include="scripts/fetch_real_corpus.py"

The test skips itself when no coverage data has been collected, so a plain
``pytest`` run stays fast; CI runs it after the coverage pass.
"""

from __future__ import annotations

import pytest

# Module -> minimum branch-relevant statement coverage.
FLOORS: dict[str, float] = {
    "scripts/fetch_real_corpus.py": 70.0,
    "scripts/build_real_dataset.py": 70.0,
    "scripts/sample_tokenizer_corpus.py": 65.0,
    "scripts/build_corpus_v1.py": 60.0,
    "dataset/quality_gates.py": 80.0,
    "dataset/production_processor.py": 65.0,
    "dataset/repair_traces.py": 70.0,
    "dataset/security_data.py": 70.0,
    "evaluation/security_eval.py": 70.0,
}


def _coverage_data():
    coverage = pytest.importorskip("coverage")
    try:
        data = coverage.CoverageData()
        data.read()
    except Exception:  # no data file collected
        return None
    return data if data.measured_files() else None


def test_no_corpus_pipeline_module_is_untested():
    data = _coverage_data()
    if data is None:
        pytest.skip("run under `coverage run` to check coverage floors")

    from coverage import Coverage

    cov = Coverage()
    cov.load()

    failures: list[str] = []
    for module, floor in FLOORS.items():
        try:
            analysis = cov.analysis2(module)
        except Exception:
            failures.append(f"{module}: no coverage data (module never imported)")
            continue
        statements = analysis[1]
        missing = analysis[3]
        measured = len(statements)
        if measured == 0:
            continue
        covered = 100.0 * (measured - len(missing)) / measured
        if covered < floor:
            failures.append(f"{module}: {covered:.0f}% < {floor:.0f}% floor")

    assert not failures, "coverage floors not met:\n  " + "\n  ".join(failures)
