"""Scale plan and acceptance audit for the real-code backbone ingestion.

Task 4 of Corpus v1 is a *capacity* question, not a code question: before a
multi-day fetch and a multi-GPU run, answer

- how many raw tokens does the fetch target have to yield so that, after the
  licence gate, the quality/complexity floors, the vendored/minified filters and
  exact + MinHash de-duplication, at least ``> 18B`` tokens survive?
- how many permissive repositories are needed for that, per language?
- and, once a fetch *has* run, does the measured yield meet the plan?

``IngestionPlan`` is the arithmetic (funnel-adjusted, language-weighted, with an
explicit order-of-magnitude error bar), and ``audit_fetch_manifest`` is the
post-hoc check that turns a ``manifest.json`` from ``fetch_real_corpus`` into a
pass/fail verdict with the specific shortfall named.

    python -m dataset.ingestion_plan --target-tokens 25000000000
    python -m dataset.ingestion_plan --audit data/corpus_v1/real_jsonl/manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataset.production_processor import PERMISSIVE_LICENSES

# --------------------------------------------------------------------------- #
# Funnel assumptions
# --------------------------------------------------------------------------- #
#
# Every number below is a *fraction of the previous stage*. Licence coverage and
# repository availability come from the checked-in inventory; quality and dedup
# survival are planning priors until a large fetch measures them. They are
# deliberately conservative and separately configurable.

BYTES_PER_TOKEN = 3.6
"""Real code tokenises at ~3.1-3.6 bytes/token with the in-repo BPE; 3.6 is the
pessimistic end, so token estimates are not optimistic."""

TARBALL_FRACTION_OF_REPO_SIZE = 0.35
"""GitHub reports repository size in KB of the *git history*. A codeload tarball
of the working tree is a fraction of it (no history, compressed). 0.35 is what
``scripts/source_inventory.py`` uses when it estimates availability."""

WORKING_TREE_IS_SOURCE = 0.55
"""Share of a working tree that is source in a language we ingest (the rest is
images, binaries, generated assets, vendored third-party code, configs)."""

LICENCE_GATE_PASS = 0.81
"""Share of curated repositories whose archive LICENSE classifies permissively.

The checked-in inventory measured 869/1075 (80.84%) on 2026-11-03; 0.81 is the
rounded planning value. The fetch still re-classifies every archive fail-closed.
"""

QUALITY_GATE_PASS = 0.45
"""Planning prior for files surviving the hard filters (line/byte bounds,
minified/vendored/generated rejection, complexity floor and syntax check).
Replace it with measured manifest yield when a representative fetch completes."""

DEDUP_PASS = 0.82
"""Planning prior for quality-passing documents that are not near duplicates.
Vendored copies, forked files and reformatted pastes are common in real
repositories. Replace 0.82 with the measured survival rate at threshold 0.85."""

HELDOUT_SHARE = 0.05
"""Documents reserved for the held-out split and never trained on."""

TRAIN_SUBSAMPLE = 0.60
"""The 18B-token budget is a *subsample* of what is available: task 4 asks for
enough real code that >>18B tokens survive filtering, then subsample to budget.
0.60 means the target is 30B tokens of unique, filtered, permissive code."""

# Language weights for the availability target. This is *not* the training
# mixture (`dataset.mixture.DEFAULT_MIXTURE`); it is how the raw corpus is
# expected to be distributed when it is fetched by budget-priority order.
LANGUAGE_TARGETS: dict[str, float] = {
    "python": 0.20,
    "typescript": 0.12,
    "javascript": 0.09,
    "go": 0.10,
    "rust": 0.08,
    "java": 0.09,
    "cpp": 0.08,
    "c": 0.06,
    "csharp": 0.04,
    "ruby": 0.03,
    "php": 0.03,
    "kotlin": 0.02,
    "swift": 0.02,
    "scala": 0.01,
    "shell": 0.01,
    "other": 0.02,
}

GITHUB_AVAILABILITY_RAW_TOKENS = 20_350_000_000
"""Estimated pre-filter source tokens in the permissive curated repositories.

Measured, not assumed: ``scripts/source_inventory.py`` sums the repository-size
estimate for the 869 permissively licensed candidates (~73.26 GB) and converts
at ``BYTES_PER_TOKEN``. This number is *already after* repository-size and
licence selection, so only quality, dedup and held-out survival apply to it.
After those filters it is ~7.1B tokens: useful for language coverage and
repository artefacts, but not enough for the 30B-token unique-code target.
The Stack v2 has to supply the bulk; the curated path supplies diversity,
issues/PRs/tests/docs, and a reproducible, licence-verified floor that does not
depend on a Hub mirror. PyPI and npm availability is additional and deliberately
not counted in this conservative floor."""


@dataclass(frozen=True)
class FunnelStage:
    name: str
    keep: float
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.name, "keep": self.keep, "note": self.note}


DEFAULT_FUNNEL: tuple[FunnelStage, ...] = (
    FunnelStage("licence_gate", LICENCE_GATE_PASS, "fail-closed SPDX permissive only"),
    FunnelStage("quality_gate", QUALITY_GATE_PASS, "line/byte/complexity/syntax/vendored/minified"),
    FunnelStage("dedup", DEDUP_PASS, "exact SHA-256 + MinHash >= 0.85"),
    FunnelStage("heldout", 1.0 - HELDOUT_SHARE, "reserved and never trained on"),
)

CURATED_FUNNEL: tuple[FunnelStage, ...] = DEFAULT_FUNNEL[1:]
"""The inventory is already permissive source, so do not charge licence twice.

Repository-size and non-source overhead are also excluded here: they affect how
many bytes must be downloaded, not how many candidate source tokens survive a
document filter. ``required_repository_bytes`` accounts for that overhead.
"""


def funnel_yield(funnel: tuple[FunnelStage, ...] = DEFAULT_FUNNEL) -> float:
    """Fraction of candidate source tokens that survive every stage."""

    keep = 1.0
    for stage in funnel:
        keep *= stage.keep
    return keep


@dataclass
class IngestionPlan:
    """How much raw material the 3B run needs, and how many sources that implies."""

    target_train_tokens: int = 18_000_000_000
    headroom: float = 1.0 / TRAIN_SUBSAMPLE
    tokens_per_repo: int = 90_000
    github_raw_tokens: int = GITHUB_AVAILABILITY_RAW_TOKENS
    funnel: tuple[FunnelStage, ...] = DEFAULT_FUNNEL
    language_targets: dict[str, float] = field(
        default_factory=lambda: dict(LANGUAGE_TARGETS)
    )

    # -- core arithmetic ---------------------------------------------------
    @property
    def required_surviving_tokens(self) -> int:
        """Filtered, unique, permissive tokens the fetch must produce."""

        return int(self.target_train_tokens * self.headroom)

    @property
    def yield_fraction(self) -> float:
        return funnel_yield(self.funnel)

    @property
    def required_raw_tokens(self) -> int:
        return int(self.required_surviving_tokens / max(self.yield_fraction, 1e-9))

    @property
    def required_raw_bytes(self) -> int:
        """Candidate source bytes to process, before document-level filters."""

        return int(round(self.required_raw_tokens * BYTES_PER_TOKEN))

    @property
    def required_repository_bytes(self) -> int:
        """Repository-size equivalent needed if GitHub supplied the whole plan.

        This is a storage/procurement estimate, not a token-filtering stage.
        GitHub's reported size includes history; a working tree also contains
        non-source assets. Keeping this separate avoids mixing bytes and tokens
        in the survival table.
        """

        source_fraction = TARBALL_FRACTION_OF_REPO_SIZE * WORKING_TREE_IS_SOURCE
        return int(round(self.required_raw_bytes / source_fraction))

    @property
    def required_repositories(self) -> int:
        """Curated repositories needed if each contributes ``tokens_per_repo``."""

        return math.ceil(self.required_surviving_tokens / max(self.tokens_per_repo, 1))

    # -- split across the two ingestion paths ------------------------------
    @property
    def github_surviving_tokens(self) -> int:
        """Surviving tokens the curated GitHub/PyPI/npm path can contribute.

        ``github_raw_tokens`` comes from the permissive-source inventory, so the
        licence gate and repository-size overhead have already been applied.
        """

        curated_funnel = tuple(stage for stage in self.funnel if stage.name != "licence_gate")
        return int(self.github_raw_tokens * funnel_yield(curated_funnel))

    @property
    def stack_required_surviving_tokens(self) -> int:
        """Residual that must come from The Stack v2 / StarCoderData streaming."""

        return max(0, self.required_surviving_tokens - self.github_surviving_tokens)

    @property
    def stack_required_raw_tokens(self) -> int:
        return int(self.stack_required_surviving_tokens / max(self.yield_fraction, 1e-9))

    def source_roles(self) -> dict[str, dict[str, Any]]:
        return {
            "the-stack-v2": {
                "role": "bulk real code",
                "script": "scripts/build_real_dataset.py",
                "raw_tokens_needed": self.stack_required_raw_tokens,
                "surviving_tokens": self.stack_required_surviving_tokens,
                "share_of_requirement": round(
                    self.stack_required_surviving_tokens
                    / max(self.required_surviving_tokens, 1),
                    4,
                ),
                "note": (
                    "streamed and filtered, never downloaded wholesale; the "
                    "licence gate still runs per document because the Hub mirror's "
                    "metadata is not the authority"
                ),
            },
            "curated-github-pypi-npm": {
                "role": "licence-verified diversity, issues/PRs/tests/docs, coverage floor",
                "script": "scripts/fetch_real_corpus.py",
                "raw_tokens_available": self.github_raw_tokens,
                "availability_basis": "permissive-source estimate; quality/dedup/heldout still apply",
                "surviving_tokens": self.github_surviving_tokens,
                "share_of_requirement": round(
                    self.github_surviving_tokens / max(self.required_surviving_tokens, 1),
                    4,
                ),
            },
        }

    def per_language(self) -> dict[str, dict[str, float]]:
        surviving = self.required_surviving_tokens
        return {
            language: {
                "weight": weight,
                "surviving_tokens": int(surviving * weight),
                "unique_documents": int(surviving * weight / 620),
                "repositories_at_90k_tokens": math.ceil(
                    surviving * weight / max(self.tokens_per_repo, 1)
                ),
            }
            for language, weight in sorted(
                self.language_targets.items(), key=lambda item: -item[1]
            )
        }

    def stages(self) -> list[dict[str, Any]]:
        """Per-stage survival table, useful in a PR comment or a docs table."""

        rows: list[dict[str, Any]] = []
        remaining = self.required_raw_tokens
        for stage in self.funnel:
            before = remaining
            remaining = int(remaining * stage.keep)
            rows.append(
                {
                    "stage": stage.name,
                    "keep": stage.keep,
                    "tokens_before": before,
                    "tokens_after": remaining,
                    "docs_before": int(before / 620),
                    "docs_after": int(remaining / 620),
                    "note": stage.note,
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_train_tokens": self.target_train_tokens,
            "headroom_factor": round(self.headroom, 4),
            "required_surviving_tokens": self.required_surviving_tokens,
            "funnel_yield_fraction": round(self.yield_fraction, 6),
            "required_raw_tokens": self.required_raw_tokens,
            "required_raw_bytes": self.required_raw_bytes,
            "required_raw_gigabytes": round(self.required_raw_bytes / 1e9, 1),
            "required_repository_bytes": self.required_repository_bytes,
            "required_repository_gigabytes": round(self.required_repository_bytes / 1e9, 1),
            "repository_size_assumptions": {
                "tarball_fraction": TARBALL_FRACTION_OF_REPO_SIZE,
                "working_tree_source_fraction": WORKING_TREE_IS_SOURCE,
            },
            "required_repositories": self.required_repositories,
            "github_surviving_tokens": self.github_surviving_tokens,
            "stack_required_surviving_tokens": self.stack_required_surviving_tokens,
            "stack_required_raw_tokens": self.stack_required_raw_tokens,
            "sources": self.source_roles(),
            "repositories_available": self.available_repositories(),
            "tokens_per_repo": self.tokens_per_repo,
            "bytes_per_token": BYTES_PER_TOKEN,
            "stages": self.stages(),
            "per_language": self.per_language(),
            "caveat": (
                "capacity plan: every funnel fraction is an estimate with roughly "
                "a factor-of-two error bar; the authoritative number is the audit "
                "of a measured fetch manifest"
            ),
        }

    def available_repositories(self) -> int:
        """Curated candidates known to exist (imported lazily to keep this pure)."""

        try:
            from scripts import fetch_real_corpus as frc
        except Exception:  # pragma: no cover - import cycle guard
            return 0
        return len(frc.merged_github_repos())


# --------------------------------------------------------------------------- #
# Post-fetch audit
# --------------------------------------------------------------------------- #


@dataclass
class AuditFinding:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class AuditReport:
    findings: list[AuditFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(finding.ok for finding in self.findings)

    def render(self) -> str:
        lines = [
            f"[{'PASS' if finding.ok else 'FAIL'}] {finding.name}: {finding.detail}"
            for finding in self.findings
        ]
        lines.append(f"RESULT: {'PASS' if self.ok else 'FAIL'}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "findings": [finding.to_dict() for finding in self.findings],
            "failures": [finding.name for finding in self.findings if not finding.ok],
        }

    def assert_passing(self) -> None:
        if not self.ok:
            names = ", ".join(f.name for f in self.findings if not f.ok)
            raise AssertionError(f"real-corpus audit failed: {names}")


def _estimate_tokens(chars: int) -> int:
    return int(chars / BYTES_PER_TOKEN)


def audit_fetch_manifest(
    manifest: dict[str, Any],
    *,
    plan: IngestionPlan | None = None,
    min_languages: int = 10,
    scope: str = "combined",
) -> AuditReport:
    """Decide whether a fetch manifest meets its surviving-token target.

    ``scope='combined'`` audits the whole 30B-token real-code requirement.
    ``scope='curated'`` audits only what the permissive GitHub inventory was
    expected to contribute; a healthy curated fetch is not supposed to satisfy
    The Stack's residual as well.
    """

    if scope not in {"combined", "curated"}:
        raise ValueError("scope must be 'combined' or 'curated'")
    plan = plan or IngestionPlan()
    report = AuditReport()

    chars = int(manifest.get("chars") or 0)
    documents = int(manifest.get("documents") or 0)
    tokens = _estimate_tokens(chars)
    required_tokens = (
        plan.required_surviving_tokens if scope == "combined" else plan.github_surviving_tokens
    )

    report.findings.append(
        AuditFinding(
            "surviving_token_scale",
            tokens >= required_tokens * 0.9,
            f"{tokens:,} surviving tokens vs {required_tokens:,} planned for {scope} "
            f"({tokens / max(required_tokens, 1):.0%} of plan)",
        )
    )
    report.findings.append(
        AuditFinding(
            "document_scale",
            documents >= 100_000,
            f"{documents:,} documents (>= 100k required for a 3B-class backbone)",
        )
    )

    per_language = manifest.get("per_language") or {}
    covered = {
        language
        for language, payload in per_language.items()
        if int(payload.get("chars") or 0) > 0
    }
    report.findings.append(
        AuditFinding(
            "language_coverage",
            len(covered) >= min_languages,
            f"{len(covered)} languages: {', '.join(sorted(covered)) or 'none'}",
        )
    )

    usage = manifest.get("budget_usage") or {}
    saturated = [
        language
        for language, payload in usage.items()
        if payload.get("saturated") and language != "diff"
    ]
    report.findings.append(
        AuditFinding(
            "language_saturation",
            bool(saturated) or not usage or tokens >= required_tokens,
            (
                f"availability-limited: {', '.join(sorted(saturated))}"
                if saturated
                else (
                    "no per-language budgets recorded"
                    if not usage
                    else (
                        "target met without exhausting a language budget"
                        if tokens >= required_tokens
                        else "no language hit its availability budget"
                    )
                )
            ),
        )
    )

    dedup = manifest.get("dedup") or {}
    rate = float(dedup.get("dedup_rate") or 0.0)
    report.findings.append(
        AuditFinding(
            "deduplication_ran",
            int(dedup.get("checked") or 0) > 0,
            f"{int(dedup.get('checked') or 0):,} documents checked, "
            f"{rate:.1%} removed "
            f"({int(dedup.get('exact_duplicates') or 0):,} exact, "
            f"{int(dedup.get('near_duplicates') or 0):,} near)",
        )
    )
    report.findings.append(
        AuditFinding(
            "deduplication_sane",
            rate <= 0.60,
            f"de-duplication rate {rate:.1%} (a rate above 60% means the source list "
            "is mostly copies, not that dedup is working)",
        )
    )

    licences = manifest.get("per_license") or {}
    rejected = sorted(str(name) for name in licences if name not in PERMISSIVE_LICENSES)
    report.findings.append(
        AuditFinding(
            "licence_gate",
            bool(licences) and not rejected,
            (
                f"{len(licences)} permissive SPDX ids: {', '.join(sorted(licences))}"
                if licences and not rejected
                else f"non-permissive or missing SPDX ids: {', '.join(rejected) or 'none'}"
            ),
        )
    )

    heldout = manifest.get("heldout") or {}
    report.findings.append(
        AuditFinding(
            "heldout_split",
            int(heldout.get("documents") or 0) > 0,
            f"{int(heldout.get('documents') or 0):,} held-out documents "
            f"({int(heldout.get('unique_hashes') or 0):,} unique hashes)",
        )
    )

    configured = manifest.get("sources_configured") or {}
    report.findings.append(
        AuditFinding(
            "source_list_exercised",
            (int(manifest.get("unique_sources") or 0) >= 50) or not configured,
            f"{int(manifest.get('unique_sources') or 0)} distinct sources contributed "
            f"(of {configured.get('github', '?')} GitHub candidates)",
        )
    )

    return report


def audit_path(path: str | Path, **kwargs: Any) -> AuditReport:
    return audit_fetch_manifest(json.loads(Path(path).read_text(encoding="utf-8")), **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-tokens", type=int, default=18_000_000_000)
    parser.add_argument("--headroom", type=float, default=1.0 / TRAIN_SUBSAMPLE)
    parser.add_argument("--tokens-per-repo", type=int, default=90_000)
    parser.add_argument("--audit", type=Path, default=None, help="fetch manifest.json to audit")
    parser.add_argument(
        "--scope",
        choices=("combined", "curated"),
        default="combined",
        help="audit the full real-code target or only the curated-path allocation",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    plan = IngestionPlan(
        target_train_tokens=args.target_tokens,
        headroom=args.headroom,
        tokens_per_repo=args.tokens_per_repo,
    )

    if args.audit:
        report = audit_path(args.audit, plan=plan, scope=args.scope)
        print(json.dumps(report.to_dict(), indent=2) if args.json else report.render())
        return 0 if report.ok else 1

    payload = plan.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"target train tokens:      {plan.target_train_tokens:,}")
    print(f"headroom factor:          x{plan.headroom:.2f} (subsample to budget)")
    print(f"must survive filtering:   {plan.required_surviving_tokens:,} tokens")
    print(f"funnel yield:             {plan.yield_fraction:.4%}")
    print(f"must process:             {plan.required_raw_tokens:,} candidate tokens "
          f"(~{plan.required_raw_bytes / 1e9:,.1f} GB of source)")
    print(f"repository-size estimate: ~{plan.required_repository_bytes / 1e9:,.1f} GB "
          "(history + non-source overhead; not a token filter)")
    print(f"curated repositories:     {plan.available_repositories():,} on the list "
          f"(~{plan.github_surviving_tokens:,} surviving tokens)")
    print(f"the-stack-v2 must supply: {plan.stack_required_surviving_tokens:,} surviving tokens "
          f"(~{plan.stack_required_raw_tokens:,} raw)")
    print()
    print(f"{'stage':24s} {'keep':>7s} {'tokens before':>16s} {'tokens after':>16s}")
    for row in plan.stages():
        print(
            f"{row['stage']:24s} {row['keep']:>7.2f} "
            f"{row['tokens_before']:>16,} {row['tokens_after']:>16,}"
        )
    print()
    print(f"{'language':12s} {'weight':>7s} {'surviving tokens':>17s} {'repos @90k':>11s}")
    for language, row in plan.per_language().items():
        print(
            f"{language:12s} {row['weight']:>7.0%} {row['surviving_tokens']:>17,} "
            f"{row['repositories_at_90k_tokens']:>11,}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
