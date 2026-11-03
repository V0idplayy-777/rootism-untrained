"""Pre-training gates over a built Corpus v1 manifest.

`scripts/build_corpus_v1.py` writes a manifest; this module decides whether that
manifest is allowed to become a training run. The gates are deliberately
mechanical and fail closed:

- **coverage** — every expected slice is present and contributes at least a
  floor share, so a silently missing source cannot become "a model that can't
  do X".
- **mixture** — realized per-slice token shares are close to the planned
  weights; large drift means the fetch produced a skewed corpus.
- **license** — only permissive SPDX identifiers appear in the manifest.
- **diversity** — the corpus is not one file repeated: unique paths, AST
  patterns and multi-file documents all have floors.
- **isolation** — held-out content and train/val overlap are exactly zero.
- **synthetic budget** — model-generated tokens stay under the 5% cap.

    python -m dataset.quality_gates data/corpus_v1/shards_3b/corpus_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .mixture import (
    DEFAULT_MIXTURE,
    MAX_SYNTHETIC_TOKEN_SHARE,
    SYNTHETIC_TOKEN_SLICES,
    real_backbone_weight,
)
from .production_processor import NON_PERMISSIVE_MARKERS

# A slice may be absent only if the build explicitly reports it as missing for a
# documented reason (e.g. no app rows requested in a smoke build).
REQUIRED_SLICES: tuple[str, ...] = (
    "the-stack-code",
    "tests",
    "github-prs",
    "github-issues",
    "docs-and-comments",
    "instruction-sft",
)

MIN_SLICE_SHARE: dict[str, float] = {
    "the-stack-code": 0.20,
    "tests": 0.01,
    "docs-and-comments": 0.01,
}

MIN_REALIZED_RATIO = 0.5   # realized share >= 50% of the planned weight
MAX_REALIZED_RATIO = 2.5   # and <= 250% of it
# Near-duplicate files intentionally collapse to one structural pattern, so a
# healthy corpus sits well above this; the floor only catches "everything is the
# same file".
MIN_UNIQUE_PATTERN_RATIO = 0.01
MIN_MULTI_FILE_RATIO = 0.01


@dataclass
class GateResult:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class GateReport:
    results: list[GateResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.results)

    def failures(self) -> list[GateResult]:
        return [result for result in self.results if not result.ok]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "gates": [result.to_dict() for result in self.results],
            "failures": [result.name for result in self.failures()],
        }

    def render(self) -> str:
        lines = []
        for result in self.results:
            lines.append(f"[{'PASS' if result.ok else 'FAIL'}] {result.name}: {result.detail}")
        lines.append(f"RESULT: {'PASS' if self.ok else 'FAIL'}")
        return "\n".join(lines)

    def assert_passing(self) -> None:
        if not self.ok:
            names = ", ".join(result.name for result in self.failures())
            raise AssertionError(f"corpus quality gates failed: {names}")


def _slice_tokens(manifest: dict) -> dict[str, int]:
    tokens = {name: int(value) for name, value in (manifest.get("slice_tokens") or {}).items()}
    if tokens:
        return tokens
    # older manifests only carry per-source token counts
    return {
        name: int(payload.get("tokens", 0))
        for name, payload in (manifest.get("sources") or {}).items()
    }


def _shares(tokens: dict[str, int]) -> dict[str, float]:
    total = sum(tokens.values())
    if total <= 0:
        return {}
    return {name: value / total for name, value in tokens.items()}


def check_coverage(manifest: dict) -> GateResult:
    tokens = _slice_tokens(manifest)
    shares = _shares(tokens)
    missing = [name for name in REQUIRED_SLICES if tokens.get(name, 0) <= 0]
    thin = [
        f"{name} {shares.get(name, 0.0):.2%} < {floor:.0%}"
        for name, floor in MIN_SLICE_SHARE.items()
        if shares.get(name, 0.0) < floor
    ]
    ok = not missing and not thin
    detail = (
        f"{len(tokens)} slices, real backbone "
        f"{sum(tokens.get(spec.name, 0) for spec in DEFAULT_MIXTURE if spec.kind == 'real') / max(sum(tokens.values()), 1):.1%}"
    )
    if missing:
        detail += f"; missing={missing}"
    if thin:
        detail += f"; thin={thin}"
    return GateResult("coverage", ok, detail)


def check_mixture(manifest: dict) -> GateResult:
    """Realized shares vs planned weights, for the slices the mix controls.

    Pure synthetic slices are excluded: their planned weight is a *document*
    share, while their admitted *token* share is governed by
    `check_synthetic_budget` (5% of tokens). Comparing the two would fail every
    build that correctly enforces the cap.
    """

    tokens = _slice_tokens(manifest)
    shares = _shares(tokens)
    drift: list[str] = []
    for spec in DEFAULT_MIXTURE:
        if spec.name in SYNTHETIC_TOKEN_SLICES:
            continue
        planned = spec.weight
        realized = shares.get(spec.name, 0.0)
        if realized == 0.0:
            continue  # absence is coverage's job, not drift
        ratio = realized / planned if planned else 0.0
        if ratio < MIN_REALIZED_RATIO or ratio > MAX_REALIZED_RATIO:
            drift.append(f"{spec.name} {realized:.3f} vs planned {planned:.3f} (x{ratio:.2f})")
    return GateResult(
        "mixture",
        not drift,
        "realized shares within tolerance" if not drift else "; ".join(drift[:5]),
    )


def check_licenses(manifest: dict) -> GateResult:
    licenses = manifest.get("licenses") or {}
    if not licenses:
        return GateResult("license", False, "manifest records no licenses")
    bad = [
        name
        for name in licenses
        if any(marker in name.lower() for marker in NON_PERMISSIVE_MARKERS)
    ]
    return GateResult(
        "license",
        not bad,
        f"{len(licenses)} SPDX ids, all permissive" if not bad else f"non-permissive: {bad}",
    )


def check_diversity(manifest: dict) -> GateResult:
    diversity = manifest.get("diversity") or {}
    documents = int(manifest.get("total_documents") or 0)
    unique_files = int(diversity.get("unique_files") or 0)
    patterns = int(diversity.get("unique_ast_patterns") or 0)
    multi = float(diversity.get("multi_file_ratio") or 0.0)

    if documents <= 0:
        return GateResult("diversity", False, "no documents")

    pattern_ratio = patterns / documents
    problems = []
    if unique_files <= 0:
        problems.append("no unique paths")
    if pattern_ratio < MIN_UNIQUE_PATTERN_RATIO:
        problems.append(f"pattern ratio {pattern_ratio:.3f} < {MIN_UNIQUE_PATTERN_RATIO}")
    if multi < MIN_MULTI_FILE_RATIO:
        problems.append(f"multi-file ratio {multi:.3%} < {MIN_MULTI_FILE_RATIO:.1%}")
    detail = (
        f"{unique_files} paths, {patterns} AST patterns "
        f"({pattern_ratio:.1%} of docs), multi-file {multi:.1%}"
    )
    return GateResult("diversity", not problems, detail if not problems else "; ".join(problems))


def check_isolation(manifest: dict) -> GateResult:
    isolation = manifest.get("isolation") or {}
    heldout = int(isolation.get("heldout_overlap_documents") or 0)
    overlap = int(isolation.get("train_val_overlap_documents") or 0)
    ok = heldout == 0 and overlap == 0
    return GateResult(
        "isolation",
        ok,
        f"held-out overlap {heldout}, train/val overlap {overlap}",
    )


def check_synthetic_budget(manifest: dict, max_share: float = MAX_SYNTHETIC_TOKEN_SHARE) -> GateResult:
    tokens = _slice_tokens(manifest)
    total = sum(tokens.values())
    if total <= 0:
        return GateResult("synthetic_budget", False, "no tokens")
    synthetic = sum(value for name, value in tokens.items() if name in SYNTHETIC_TOKEN_SLICES)
    share = synthetic / total
    reported = manifest.get("synthetic") or {}
    ok = share <= max_share + 1e-9
    if "within_cap" in reported and not reported["within_cap"]:
        ok = False
    return GateResult(
        "synthetic_budget",
        ok,
        f"{share:.2%} synthetic vs {max_share:.0%} cap ({synthetic:,}/{total:,} tokens)",
    )


def check_real_backbone(manifest: dict, floor: float = 0.70) -> GateResult:
    tokens = _slice_tokens(manifest)
    total = sum(tokens.values())
    if total <= 0:
        return GateResult("real_backbone", False, "no tokens")
    real_names = {spec.name for spec in DEFAULT_MIXTURE if spec.kind == "real"}
    share = sum(value for name, value in tokens.items() if name in real_names) / total
    return GateResult(
        "real_backbone",
        share >= floor,
        f"{share:.2%} real (planned {real_backbone_weight():.0%}, floor {floor:.0%})",
    )


def evaluate_manifest(manifest: dict, max_synthetic_share: float = MAX_SYNTHETIC_TOKEN_SHARE) -> GateReport:
    """Run every gate against a corpus manifest."""

    return GateReport(
        [
            check_coverage(manifest),
            check_mixture(manifest),
            check_licenses(manifest),
            check_diversity(manifest),
            check_isolation(manifest),
            check_synthetic_budget(manifest, max_synthetic_share),
            check_real_backbone(manifest),
        ]
    )


def evaluate_path(path: str | Path, max_synthetic_share: float = MAX_SYNTHETIC_TOKEN_SHARE) -> GateReport:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return evaluate_manifest(manifest, max_synthetic_share)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--synthetic_cap", type=float, default=MAX_SYNTHETIC_TOKEN_SHARE)
    args = parser.parse_args(argv)

    report = evaluate_path(args.manifest, args.synthetic_cap)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
