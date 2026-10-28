import ast

import pytest

from dataset.augmentation import (
    AbstractionLadder,
    AmbiguityGenerator,
    ConceptCombiner,
    PhrasingAugmenter,
    augment_rows,
)
from dataset.security_data import (
    SECURITY_SAMPLES,
    SecurityDataGenerator,
    security_review_rubric,
)
from evaluation.security_eval import (
    REQUIRED_PATTERNS,
    SecurityBenchmark,
    SecurityScanner,
    cases_from_dataset,
    scan_generated_code,
)
from model.longform import (
    LongFormSynthesizer,
    ReasoningBudget,
    missing_definitions,
    parse_plan,
    python_syntax_errors,
)


# Security data + evaluation


def test_curated_security_samples_are_paired_and_valid_python():

    assert len(SECURITY_SAMPLES) >= 15

    cwes = {s.cwe for s in SECURITY_SAMPLES}

    assert {"CWE-89", "CWE-78", "CWE-22", "CWE-502", "CWE-295", "CWE-338"} <= cwes

    for sample in SECURITY_SAMPLES:

        assert sample.vulnerable.strip() != sample.safe.strip()

        assert sample.why.strip()

        if sample.language == "python":

            ast.parse(sample.safe)

            ast.parse(sample.vulnerable)

        assert sample.category in REQUIRED_PATTERNS, sample.category


# in SecurityBenchmark instead of by the static scanner.
STATIC_UNDETECTABLE = {"ssrf"}


def test_scanner_catches_vulnerable_and_passes_safe_samples():

    scanner = SecurityScanner()

    for sample in SECURITY_SAMPLES:

        good = scanner.scan(sample.safe, sample.language)

        assert scanner.is_safe(
            sample.safe, sample.language, severities=("critical", "high", "medium")
        ), (
            sample.id,
            [f.to_dict() for f in good],
        )

        if sample.id not in STATIC_UNDETECTABLE:

            assert not scanner.is_safe(
                sample.vulnerable,
                sample.language,
                severities=("critical", "high", "medium"),
            ), sample.id

    bench = SecurityBenchmark()

    for sample in SECURITY_SAMPLES:

        report = bench.evaluate(
            [
                {
                    "case_id": sample.id,
                    "code": sample.vulnerable,
                    "language": sample.language,
                    "category": sample.category,
                }
            ]
        )

        assert report.overall_vuln_rate == 1.0, sample.id


def test_security_benchmark_scores_both_directions():

    bench = SecurityBenchmark()

    safe = [
        {
            "case_id": s.id,
            "code": s.safe,
            "language": s.language,
            "category": s.category,
        }
        for s in SECURITY_SAMPLES
    ]

    bad = [
        {
            "case_id": s.id,
            "code": s.vulnerable,
            "language": s.language,
            "category": s.category,
        }
        for s in SECURITY_SAMPLES
    ]

    safe_report = bench.evaluate(safe)

    bad_report = bench.evaluate(bad)

    assert safe_report.overall_vuln_rate == 0.0

    assert safe_report.passed

    assert bad_report.overall_vuln_rate == 1.0

    assert not bad_report.passed

    assert bad_report.by_cwe

    bench.assert_threshold(safe_report)

    with pytest.raises(AssertionError):

        bench.assert_threshold(bad_report)

    import json

    json.dumps(bad_report.to_dict())


def test_scanner_ast_checks_and_required_pattern_absence():

    report = scan_generated_code(
        "import hashlib\n\ndef store(password):\n    return hashlib.md5(password.encode()).hexdigest()\n"
    )

    assert not report["safe"]

    assert any(f["rule"] == "ast-weak-hash" for f in report["findings"])

    timing = scan_generated_code(
        "def check(expected, provided):\n    return expected == provided\n"
    )

    assert any(f["cwe"] == "CWE-208" for f in timing["findings"])

    bench = SecurityBenchmark()

    missing = bench.evaluate(
        [
            {
                "case_id": "timing-compare",
                "code": "def check(expected, provided):\n    return expected.strip() == provided.strip()\n",
            }
        ]
    )

    assert missing.overall_vuln_rate == 1.0

    assert missing.missing_safe_pattern == 1

    cases = cases_from_dataset()

    all_cases = cases_from_dataset(heldout=False)

    from dataset.security_data import heldout_split, train_split


    assert 0 < len(cases) < len(all_cases)

    assert len(all_cases) == len(SECURITY_SAMPLES)

    assert all(c.required for c in all_cases)

    heldout_ids = {c.case_id for c in cases}

    train_ids = {s.id for s in train_split()}

    assert heldout_ids == {s.id for s in heldout_split()}

    assert not (heldout_ids & train_ids)


def test_security_generator_shapes_match_training_schema():

    gen = SecurityDataGenerator()

    reviews = gen.review_samples()

    repairs = gen.repair_samples()

    implements = gen.implement_samples()

    pairs = gen.dpo_pairs()


    from dataset.security_data import train_split

    assert len(reviews) == len(train_split()) == len(pairs)

    assert len(reviews) < len(SECURITY_SAMPLES)

    assert implements

    for row in reviews + repairs + implements:

        for key in ("instruction", "input_context", "output", "category", "language"):

            assert key in row

        assert row["metadata"]["cwe"].startswith("CWE-")

    for pair in pairs:

        assert pair["chosen"] != pair["rejected"]

        assert pair["category"] == "security_preference"

    batch = gen.generate_batch(10, seed=7)

    assert len(batch) == 10

    assert len({(r["instruction"], r["output"]) for r in batch}) == 10

    assert "AuthN/AuthZ" in security_review_rubric()

    stats = gen.stats()

    assert stats["samples"] == len(train_split())

    assert len(stats["cwes"]) >= 8


    assert len(SecurityDataGenerator(heldout=True).samples) == len(SECURITY_SAMPLES) - len(train_split())




def test_phrasing_variants_are_deterministic_and_preserve_technical_content():

    instruction = "Write a function that parses an ISO-8601 date and returns a datetime."

    a = PhrasingAugmenter(seed=1).variants(instruction)

    b = PhrasingAugmenter(seed=1).variants(instruction)

    assert [v["instruction"] for v in a] == [v["instruction"] for v in b]

    styles = {v["phrasing"] for v in a}

    assert {"terse", "casual", "constraint_first", "partial_repro"} <= styles

    for variant in a:

        assert "ISO-8601" in variant["instruction"]

        assert variant["source_instruction"] == instruction

        assert variant["instruction"].strip()


def test_cross_domain_tasks_are_compilable_and_have_acceptance_criteria():

    rows = ConceptCombiner(seed=3).combine(4)

    assert len(rows) == 4

    for row in rows:

        ast.parse(row["input_context"])

        assert row["category"] == "cross_domain_synthesis"

        assert row["metadata"]["primary_domain"] != row["metadata"]["secondary_domain"]

        assert len(row["metadata"]["requirements"]) >= 4

        assert len(row["metadata"]["acceptance"]) >= 3

        assert "Plan:" in row["output"]

    assert [r["family"] for r in rows] == [
        r["family"] for r in ConceptCombiner(seed=3).combine(4)
    ]


def test_ambiguity_tasks_teach_clarification():

    rows = AmbiguityGenerator(seed=2).generate(3)

    assert len(rows) == 3

    for row in rows:

        assert row["metadata"]["missing_information"]

        assert len(row["output"]) > 60

        assert row["category"] == "clarification"


def test_augment_rows_expands_and_keeps_schema():

    rows = [
        {
            "instruction": "Implement a retry policy with exponential backoff.",
            "input_context": "",
            "output": "def retry(): ...",
            "category": "code_gen",
            "language": "python",
            "difficulty": "intermediate",
        }
    ]

    out = augment_rows(rows, per_row=3, seed=5, include_cross_domain=2)

    assert len(out) > len(rows)

    assert any(":" in r["category"] for r in out)

    assert any(r["category"] == "cross_domain_synthesis" for r in out)

    assert any(r["category"] == "clarification" for r in out)

    assert rows[0]["category"] == "code_gen"

    for row in out:

        assert "instruction" in row and "output" in row

    ladder = AbstractionLadder().rungs("add caching to the API")

    assert [r["level"] for r in ladder] == ["spec", "design", "implement"]




def test_reasoning_budget_scales_with_window():

    small = ReasoningBudget(context_window=8192, prompt_tokens=7000, max_new_tokens=2048)

    large = ReasoningBudget(context_window=32768, prompt_tokens=7000, max_new_tokens=8192)

    small_alloc = small.allocate()

    large_alloc = large.allocate()

    assert small_alloc["available"] < large_alloc["available"]

    assert small_alloc["scratch"] >= 64

    assert large_alloc["answer"] > small_alloc["answer"]

    assert "window=8192" in small.describe()


def test_plan_parsing_handles_formats_and_invariants():

    plan = parse_plan(
        "1. `parse_config(path)`: read and validate\n"
        "2. merge_configs(base, override): deep merge\n"
        "Invariants:\n- unknown keys rejected\n- returns a new dict\n",
        "config loader",
    )

    assert plan.size == 2

    assert plan.units[0].name == "parse_config"

    assert plan.invariants == ["unknown keys rejected", "returns a new dict"]

    assert "1. parse_config" in plan.render()

    fallback = parse_plan("no list here", "do the thing")

    assert fallback.size == 1

    assert fallback.units[0].purpose == "do the thing"


class _ScriptedGenerator:
    """Deterministic stand-in for a served model."""

    def __init__(self, skeleton: str, bodies: dict[str, str]):

        self.skeleton = skeleton

        self.bodies = bodies

        self.calls: list[str] = []

    def __call__(self, prompt: str, max_new_tokens: int) -> str:

        self.calls.append(prompt[:40])

        if "Write a decomposition plan" in prompt:

            return (
                "1. `load(path)`: load the file\n"
                "2. total(items): sum the items\n"
                "Invariants:\n- returns ints\n"
            )

        if "Output ONLY Python" in prompt:

            return f"```python\n{self.skeleton}```"

        if "Write ONLY the body" in prompt:

            for name, body in self.bodies.items():

                if f"`{name}`" in prompt:

                    return body

        if "does not compile" in prompt:

            return f"```python\n{self.bodies.get('__fixed__', '')}```"

        return ""


def test_longform_synthesis_fills_units_and_validates():

    skeleton = (
        "def load(path: str) -> dict:\n"
        "    ...\n\n\n"
        "def total(items: list[int]) -> int:\n"
        "    ...\n"
    )

    gen = _ScriptedGenerator(
        skeleton,
        {
            "load": "    import json\n    with open(path) as f:\n        return json.load(f)",
            "total": "    return sum(items)",
        },
    )

    synth = LongFormSynthesizer(gen, budget=ReasoningBudget(context_window=8192, prompt_tokens=2000))

    result = synth.synthesise("load a json file and sum items")

    assert result.ok

    assert result.plan.size == 2

    assert "json.load" in result.code and "sum(items)" in result.code

    assert python_syntax_errors(result.code) == []

    assert missing_definitions(result.code) == []

    assert [t["step"] for t in result.trace][:3] == ["plan", "skeleton", "fill"]

    body_calls = [c for c in gen.calls if c]

    assert body_calls

    assert result.to_dict()["units"] == 2


def test_longform_repair_loop_recovers_from_bad_first_attempt():

    broken = "def f(x):\n    return x +\n"

    fixed = "def f(x):\n    return x + 1\n"

    class Gen:
        def __init__(self):

            self.count = 0

        def __call__(self, prompt: str, n: int) -> str:

            if "Write a decomposition plan" in prompt:

                return "1. `f(x)`: add one"

            if "Output ONLY Python" in prompt:

                return broken

            if "Write ONLY the body" in prompt:

                return "    return x + 1"

            if "does not compile" in prompt:

                self.count += 1

                return fixed

            return ""

    gen = Gen()

    synth = LongFormSynthesizer(gen, max_repairs=2)

    result = synth.synthesise("add one")

    assert result.repairs == 1

    assert python_syntax_errors(result.code) == []

    assert gen.count == 1


def test_self_consistency_prefers_execution_checked_candidate():

    answers = iter(["ANSWER: 4", "ANSWER: 5", "ANSWER: 5"])

    def gen(prompt: str, n: int) -> str:

        return f"<|scratchpad|> thinking\n{next(answers, 'ANSWER: 5')}"

    synth = LongFormSynthesizer(gen)

    majority = synth.reason("2+2?", samples=3)

    assert majority["strategy"] == "majority"

    assert majority["answer"] == "5"

    assert majority["agreement"] == pytest.approx(2 / 3)

    answers2 = iter(["ANSWER: 4", "ANSWER: 5", "ANSWER: 5"])

    def gen2(prompt: str, n: int) -> str:

        return f"<|scratchpad|> thinking\n{next(answers2, 'ANSWER: 5')}"

    checked = LongFormSynthesizer(gen2).reason(
        "2+2?", samples=3, checker=lambda raw: "ANSWER: 4" in raw
    )

    assert checked["strategy"] == "execution_check"

    assert checked["answer"] == "4"


def test_long_form_generate_fn_signature_is_respected():

    seen = []

    def gen(prompt: str, max_new_tokens: int) -> str:

        seen.append(max_new_tokens)

        return "1. `one(x)`: do one"

    syn = LongFormSynthesizer(gen)

    plan = syn.plan("task")

    assert plan.size == 1

    assert all(isinstance(v, int) and v > 0 for v in seen)
