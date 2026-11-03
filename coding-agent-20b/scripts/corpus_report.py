#!/usr/bin/env python3
"""Emit a machine-checkable report for every Corpus v1 slice.

The point of this script is to make the corpus *auditable*: each slice reports
what it contains, how it is grounded, and which invariant keeps it honest, so a
reviewer can reproduce the numbers with one command instead of trusting prose.

    python scripts/corpus_report.py            # human-readable
    python scripts/corpus_report.py --json     # for CI / manifest diffing
    python scripts/corpus_report.py --strict   # non-zero exit if an invariant fails
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.app_corpus import AppCorpus  # noqa: E402
from dataset.code_pattern_catalog import pattern_count, validate_catalog  # noqa: E402
from dataset.mixture import (  # noqa: E402
    DEFAULT_MIXTURE,
    hybrid_share,
    real_backbone_weight,
    synthetic_share,
    synthetic_token_budget,
    validate_mixture,
    validate_synthetic_budget,
)
from dataset.repair_traces import RepairTraceGenerator, generate_repair_rows  # noqa: E402
from dataset.security_data import SECURITY_SAMPLES, STATIC_UNDETECTABLE  # noqa: E402
from evaluation.security_eval import (  # noqa: E402
    REQUIRED_OVERRIDES,
    REQUIRED_PATTERNS,
    SecurityScanner,
    cases_from_dataset,
)
from scaffold.blueprints import all_blueprints, split, stats as blueprint_stats  # noqa: E402

MIN_TRACES = 60
MIN_LANGUAGES = 6
MIN_BLUEPRINTS = 12
MIN_SECURITY_PAIRS = 100


def security_report() -> dict:
    """Pair inventory, detector sweep and the contract the dataset enforces.

    Three verdicts are reported separately because they answer different
    questions:

    - **static recall / specificity** — does the scanner trip every vulnerable
      snippet, and stay silent on every safe one? ``STATIC_UNDETECTABLE`` records
      the classes a static rule cannot see (e.g. ``assert`` used as an
      authorization check); missing one there is expected, not a regression.
    - **case verdict under the training contract** — a case counts as passed
      when a safe snippet stays clean *and* a vulnerable one is flagged or is
      documented as statically undetectable.
    - **review contract** — the safe snippet additionally carries the required
      safe pattern for its category (what the SFT rubric asks for).
    """

    samples = list(SECURITY_SAMPLES)

    scanner = SecurityScanner()

    def _severe(code: str, language: str) -> list[str]:
        return [
            finding.rule
            for finding in scanner.scan(code, language)
            if finding.severity in {"critical", "high"}
        ]

    def _required(sample) -> tuple[str, ...]:
        return REQUIRED_OVERRIDES.get(sample.id) or REQUIRED_PATTERNS.get(
            sample.category, ()
        )

    def _has_required(sample, code: str) -> bool:
        required = _required(sample)
        if not required:
            return True
        return any(re.search(pattern, code, re.IGNORECASE) for pattern in required)

    false_positives: list[str] = []
    missed: list[str] = []
    missed_undocumented: list[str] = []
    review_failures: list[str] = []
    detected_now: list[str] = []

    for sample in samples:
        if _severe(sample.safe, sample.language):
            false_positives.append(sample.id)

        if not _severe(sample.vulnerable, sample.language):
            missed.append(sample.id)
            if sample.id not in STATIC_UNDETECTABLE:
                missed_undocumented.append(sample.id)
        elif sample.id in STATIC_UNDETECTABLE:
            # a documented gap that is now caught: the set must shrink
            detected_now.append(sample.id)

        if not _has_required(sample, sample.safe):
            review_failures.append(sample.id)

    detectable = [s for s in samples if s.id not in STATIC_UNDETECTABLE]
    caught = [s for s in detectable if _severe(s.vulnerable, s.language)]

    return {
        "pairs": len(samples),
        "categories": len({s.category for s in samples}),
        "cwes": len({s.cwe for s in samples if s.cwe}),
        "languages": len({s.language for s in samples}),
        "ids_unique": len({s.id for s in samples}) == len(samples),
        "every_pair_differs": all(
            s.vulnerable.strip() != s.safe.strip() for s in samples
        ),
        "heldout_benchmark_cases": len(cases_from_dataset()),
        "static_undetectable": len(STATIC_UNDETECTABLE),
        "detectable_pairs": len(detectable),
        "static_recall": round(len(caught) / len(detectable), 4) if detectable else 1.0,
        "static_missed": missed,
        "static_missed_undocumented": missed_undocumented,
        "static_undetectable_detected": detected_now,
        "static_undetectable_exact": (
            set(missed) == set(STATIC_UNDETECTABLE) and not missed_undocumented
        ),
        "static_false_positives": false_positives,
        "review_contract_failures": review_failures,
    }


def repair_report() -> dict:
    generator = RepairTraceGenerator()
    rows = generate_repair_rows()
    stats = generator.stats()
    languages = stats["by_language"]
    python_ok = 0
    python_bad: list[str] = []
    for example in generator.examples:
        if example.language != "python":
            continue
        try:
            ast.parse(example.fixed)
        except SyntaxError:
            python_bad.append(example.bug_id)
        else:
            python_ok += 1
    held_out = len(rows["sft_heldout"])
    total = len(rows["sft"]) + held_out
    return {
        "traces": stats["examples"],
        "languages": len(languages),
        "by_language": languages,
        "python_fixes_parse": python_ok,
        "python_fixes_broken": python_bad,
        "sft_rows": len(rows["sft"]),
        "held_out_rows": held_out,
        "held_out_share": round(held_out / total, 4) if total else 0.0,
        "dpo_rows": len(rows["dpo"]),
        "distinct_rejections": all(
            example.buggy.strip() != example.rejected.strip()
            and example.fixed.strip() != example.rejected.strip()
            for example in generator.examples
        ),
    }


def app_report() -> dict:
    train, held_out = split("app")
    sizes = [len(sample.files) for sample in AppCorpus(train_only=False, verify=False).samples()]
    return {
        **blueprint_stats(),
        "train_blueprints": len(train),
        "held_out_blueprints": len(held_out),
        "held_out_share": round(len(held_out) / (len(train) + len(held_out)), 4),
        "min_files_per_app": min(sizes) if sizes else 0,
        "max_files_per_app": max(sizes) if sizes else 0,
    }


def induction_report() -> dict:
    from dataset.instruction_induction import InstructionInduction

    shapes = [
        "docstring_to_code",
        "signature_to_body",
        "code_to_explanation",
        "test_to_contract",
        "docstring_example_to_implementation",
        "imports_to_module",
        "error_branches_to_tests",
        "docstring_to_contract_tests",
    ]
    return {
        "shapes": shapes,
        "shape_count": len(shapes),
        "all_present": all(hasattr(InstructionInduction, shape) for shape in shapes),
    }


def mixture_report(total_tokens: int = 18_000_000_000) -> dict:
    validate_mixture(DEFAULT_MIXTURE)
    audit = validate_synthetic_budget()
    budget = synthetic_token_budget(total_tokens, synthetic_share_of_slice=0.05)
    return {
        "sources": len(DEFAULT_MIXTURE),
        "real_backbone_share": real_backbone_weight(),
        "hybrid_share": hybrid_share(),
        "synthetic_document_share": synthetic_share(),
        "synthetic_token_cap": audit["max_share"],
        "synthetic_token_plan": budget["plan_share"],
        "within_token_cap": budget["within_cap"],
        "kinds": dict(Counter(spec.kind for spec in DEFAULT_MIXTURE)),
    }


def catalog_report() -> dict:
    problems = validate_catalog()
    return {"patterns": pattern_count(), "valid": problems == [], "problems": problems[:10]}


def build_report(total_tokens: int = 18_000_000_000) -> dict:
    sections = {
        "catalog": catalog_report(),
        "mixture": mixture_report(total_tokens),
        "security": security_report(),
        "repair": repair_report(),
        "apps": app_report(),
        "induction": induction_report(),
    }
    return {"kind": "corpus-v1-report", "total_tokens": total_tokens, "sections": sections}


def invariants(report: dict) -> list[dict]:
    sections = report["sections"]
    repair = sections["repair"]
    apps = sections["apps"]
    security = sections["security"]
    mixture = sections["mixture"]

    checks = [
        (
            "repair traces >= 60",
            repair["traces"] >= MIN_TRACES,
            f"{repair['traces']} traces",
        ),
        (
            "repair languages >= 6",
            repair["languages"] >= MIN_LANGUAGES,
            f"{repair['languages']} languages",
        ),
        (
            "python repairs parse",
            not repair["python_fixes_broken"],
            f"{repair['python_fixes_parse']} parse, broken={repair['python_fixes_broken']}",
        ),
        (
            "repair held-out >= 15%",
            repair["held_out_share"] >= 0.15,
            f"{repair['held_out_share']:.2%}",
        ),
        (
            "repair rejections distinct",
            repair["distinct_rejections"],
            "every rejected patch differs from buggy and fixed",
        ),
        (
            "blueprints >= 12",
            apps["blueprints"] >= MIN_BLUEPRINTS,
            f"{apps['blueprints']} blueprints",
        ),
        (
            "apps are multi-file",
            apps["min_files_per_app"] >= 12,
            f"min {apps['min_files_per_app']} files",
        ),
        (
            "app held-out >= 15%",
            apps["held_out_share"] >= 0.15,
            f"{apps['held_out_share']:.2%}",
        ),
        (
            "security pairs >= 100",
            security["pairs"] >= MIN_SECURITY_PAIRS,
            f"{security['pairs']} pairs",
        ),
        (
            "security pairs well-formed",
            security["every_pair_differs"] and security["ids_unique"],
            f"{security['categories']} categories / {security['languages']} languages",
        ),
        (
            "scanner catches every statically detectable pair",
            not security["static_missed_undocumented"],
            f"recall {security['static_recall']:.0%} over {security['detectable_pairs']} pairs, "
            f"{security['static_undetectable']} documented as unsupported",
        ),
        (
            "static-unsupported set matches reality",
            security["static_undetectable_exact"],
            f"{security['static_undetectable']} documented, "
            f"{len(security['static_missed'])} actually missed, "
            f"stale={security['static_undetectable_detected']}",
        ),
        (
            "scanner never flags a safe pair",
            not security["static_false_positives"],
            f"false positives={security['static_false_positives']}",
        ),
        (
            "safe snippets carry their required pattern",
            not security["review_contract_failures"],
            f"failures={security['review_contract_failures']}",
        ),
        (
            "induction task types >= 8",
            sections["induction"]["shape_count"] >= 8
            and sections["induction"]["all_present"],
            f"{sections['induction']['shape_count']} shapes",
        ),
        (
            "synthetic tokens <= 5%",
            mixture["within_token_cap"],
            f"plan {mixture['synthetic_token_plan']:.2%} vs cap {mixture['synthetic_token_cap']:.0%}",
        ),
        (
            "real backbone >= 70%",
            mixture["real_backbone_share"] >= 0.70,
            f"{mixture['real_backbone_share']:.2%}",
        ),
        (
            "catalog valid",
            sections["catalog"]["valid"],
            f"{sections['catalog']['patterns']} patterns",
        ),
    ]

    return [
        {"name": name, "ok": bool(ok), "detail": detail} for name, ok, detail in checks
    ]


def render(report: dict) -> str:
    lines = [f"Corpus v1 report ({report['total_tokens']:,} tokens)", ""]
    sections = report["sections"]
    for name, payload in sections.items():
        lines.append(f"[{name}]")
        for key, value in payload.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
    lines.append("[invariants]")
    for check in invariants(report):
        mark = "PASS" if check["ok"] else "FAIL"
        lines.append(f"  {mark}  {check['name']}: {check['detail']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any invariant fails")
    parser.add_argument("--total-tokens", type=int, default=18_000_000_000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = build_report(args.total_tokens)
    checks = invariants(report)
    report["invariants"] = checks

    if args.json:
        text = json.dumps(report, indent=2, sort_keys=False)
    else:
        text = render(report)

    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")

    print(text)

    if args.strict and not all(check["ok"] for check in checks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
