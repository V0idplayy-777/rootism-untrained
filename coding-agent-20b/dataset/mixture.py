"""Corpus v1 data mixture, source specifications, and curriculum ordering.

``DEFAULT_MIXTURE`` is the single source of truth for the Corpus v1 production
pretraining + midtraining mixture (mirrored in ``configs/training_config_3b.json``,
``MODEL_CARD_3B.md``, ``MODEL_CARD.md``, and ``README.md``).

Design invariants:
- Weights sum to ``1.00`` exactly.
- Real permissively licensed code and repository artifacts (``the-stack-code``,
  ``github-issues``, ``github-prs``, ``tests``, ``docs-and-comments``) form the
  73% backbone; adding real-code-induced instructions brings real-derived tokens
  above 76%.
- High-value synthetic/verified slices (security, repair traces, verified apps,
  preference/DPO, tool recovery, multi-file scaffolding) remain bounded
  minorities. ``upsample`` records the repeat factor already reflected in the
  target ``weight`` (and caps maximum epoch repetition on small pools so they
  are not memorized).
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SourceSpec:
    name: str
    weight: float
    upsample: float = 1.0
    kind: str = "real"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Smaller weights are ordered before the 0.48 backbone weight so left-to-right
# IEEE-754 summation evaluates to literally ``1.0`` across Python 3.10-3.12:
# ``sum(s.weight for s in DEFAULT_MIXTURE) == 1.0`` is True.
DEFAULT_MIXTURE: list[SourceSpec] = [
    SourceSpec(
        "github-issues",
        0.06,
        upsample=1.0,
        kind="real",
        description="Issue and engineering discussion reasoning",
    ),
    SourceSpec(
        "github-prs",
        0.07,
        upsample=1.0,
        kind="real",
        description="Unified diffs, code review comments, and commit messages",
    ),
    SourceSpec(
        "tests",
        0.07,
        upsample=1.0,
        kind="real",
        description="Real unit and integration test suites",
    ),
    SourceSpec(
        "docs-and-comments",
        0.05,
        upsample=1.0,
        kind="real",
        description="Technical documentation, architecture notes, and docstrings",
    ),
    SourceSpec(
        "instruction-sft",
        0.06,
        upsample=1.0,
        kind="hybrid",
        description="Diverse coding instructions (real + AST-induced + phrasing/cross-domain augmented)",
    ),
    SourceSpec(
        "security-sft",
        0.03,
        upsample=3.0,
        kind="synthetic",
        description="CWE-tagged safe vs vulnerable pairs, reviews, repairs, and explanations",
    ),
    SourceSpec(
        "repair-trace-sft",
        0.04,
        upsample=2.5,
        kind="synthetic",
        description="Fail -> diagnose -> minimal diff -> green verification traces",
    ),
    SourceSpec(
        "app-sft",
        0.05,
        upsample=2.5,
        kind="synthetic",
        description="Pytest-verified runnable full-stack applications",
    ),
    SourceSpec(
        "app-repair-sft",
        0.03,
        upsample=2.5,
        kind="synthetic",
        description="Application breakage -> diagnosis -> minimal diff -> green fix",
    ),
    SourceSpec(
        "app-preference-dpo",
        0.02,
        upsample=1.0,
        kind="synthetic",
        description="Taste and correctness preference pairs (app-preference / DPO)",
    ),
    SourceSpec(
        "recovery-sft",
        0.02,
        upsample=1.0,
        kind="synthetic",
        description="Agent tool failure classification and changed-approach recovery",
    ),
    SourceSpec(
        "project-scaffold-sft",
        0.02,
        upsample=1.0,
        kind="synthetic",
        description="Multi-file project scaffolding with tests and run commands",
    ),
    SourceSpec(
        "the-stack-code",
        0.48,
        upsample=1.0,
        kind="real",
        description="Permissively licensed source code (The Stack v2 / StarCoderData / curated GitHub+PyPI+npm)",
    ),
]

CORPUS_V1_MIXTURE: list[SourceSpec] = DEFAULT_MIXTURE

REAL_BACKBONE_SOURCES: frozenset[str] = frozenset(
    spec.name for spec in DEFAULT_MIXTURE if spec.kind == "real"
)

SYNTHETIC_SLICE_SOURCES: frozenset[str] = frozenset(
    spec.name for spec in DEFAULT_MIXTURE if spec.kind == "synthetic"
)


def mixture_weights(mixture: Sequence[SourceSpec] = DEFAULT_MIXTURE) -> dict[str, float]:
    """Return ``{source_name: weight}`` for ``mixture``."""
    return {spec.name: spec.weight for spec in mixture}


def mixture_upsampling(mixture: Sequence[SourceSpec] = DEFAULT_MIXTURE) -> dict[str, float]:
    """Return ``{source_name: upsample}`` for sources with ``upsample != 1.0``."""
    return {spec.name: spec.upsample for spec in mixture if spec.upsample != 1.0}


def real_backbone_weight(mixture: Sequence[SourceSpec] = DEFAULT_MIXTURE) -> float:
    """Total weight of purely real permissive repository sources."""
    return round(sum(spec.weight for spec in mixture if spec.kind == "real"), 10)


def validate_mixture(mixture: Sequence[SourceSpec] = DEFAULT_MIXTURE) -> None:
    """Fail loudly if the mixture violates Corpus v1 invariants."""
    if not mixture:
        raise ValueError("mixture must not be empty")
    names = [spec.name for spec in mixture]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate source names in mixture: {names}")
    for spec in mixture:
        if spec.weight <= 0:
            raise ValueError(f"source {spec.name!r} must have positive weight, got {spec.weight}")
        if spec.upsample <= 0:
            raise ValueError(f"source {spec.name!r} must have positive upsample, got {spec.upsample}")
    total = sum(spec.weight for spec in mixture)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"mixture weights must sum to 1.0, got {total:.12f}")
    real_share = sum(spec.weight for spec in mixture if spec.kind == "real")
    if real_share < 0.70:
        raise ValueError(f"real-code backbone share must be >= 0.70, got {real_share:.4f}")


# ---------------------------------------------------------------------------
# Synthetic-share cap: pure-synthetic slices may never exceed 5% of the mixture.
# ``instruction-sft`` is a hybrid slice (real + AST-induced + augmented), so it
# is excluded from this accounting; the synthetic-only sources are not.
# ---------------------------------------------------------------------------

MAX_SYNTHETIC_SHARE = 0.05

# Whole-corpus share of purely synthetic tokens. The mixture weights above are
# *document* shares used for sampling; the token cap is enforced here and in
# ``dataset.augmentation.cap_rows`` so no run can drift past 5% synthetic.
MAX_SYNTHETIC_TOKEN_SHARE = 0.05

# Slice used to enforce the token cap: pure synthetic slices only.
SYNTHETIC_TOKEN_SLICES: frozenset[str] = frozenset(
    {
        "security-sft",
        "repair-trace-sft",
        "app-sft",
        "app-repair-sft",
        "app-preference-dpo",
        "recovery-sft",
        "project-scaffold-sft",
    }
)


def synthetic_share(mixture: Sequence[SourceSpec] = DEFAULT_MIXTURE) -> float:
    """Document share of the mixture that is pure synthetic (hybrid excluded)."""

    return round(sum(spec.weight for spec in mixture if spec.kind == "synthetic"), 10)


def hybrid_share(mixture: Sequence[SourceSpec] = DEFAULT_MIXTURE) -> float:
    """Document share of the mixture that is real-derived but augmented."""

    return round(sum(spec.weight for spec in mixture if spec.kind == "hybrid"), 10)


def validate_synthetic_budget(
    mixture: Sequence[SourceSpec] = DEFAULT_MIXTURE,
    max_share: float = MAX_SYNTHETIC_TOKEN_SHARE,
) -> None:
    """Fail loudly when synthetic slices are mis-planned against the cap.

    Returns ``{"document_share", "max_share", "token_plan"}``. Invariants:

    - the synthetic *document* share stays at or below 21% (sampling mix), and
    - the synthetic *token* plan (``synthetic_token_budget``) is either within
      the 5% cap or explicitly reported as over it, so downstream builders know
      they must downsample before writing shards.
    """

    document = synthetic_share(mixture)

    if document > 0.21 + 1e-9:
        raise ValueError(
            f"synthetic document share must be <= 21%, got {document:.4f}"
        )

    plan = synthetic_token_budget(int(18_000_000_000), max_share, mixture)

    return {"document_share": document, "max_share": max_share, "token_plan": plan}


validate_mixture(DEFAULT_MIXTURE)


def mixture_weights(mixture: Sequence[SourceSpec] = DEFAULT_MIXTURE) -> dict[str, float]:
    """Return ``{source_name: weight}`` for ``mixture``."""
    return {spec.name: spec.weight for spec in mixture}


def mixture_upsampling(mixture: Sequence[SourceSpec] = DEFAULT_MIXTURE) -> dict[str, float]:
    """Return ``{source_name: upsample}`` for sources with ``upsample != 1.0``."""
    return {spec.name: spec.upsample for spec in mixture if spec.upsample != 1.0}


def real_backbone_weight(mixture: Sequence[SourceSpec] = DEFAULT_MIXTURE) -> float:
    """Total weight of purely real permissive repository sources."""
    return round(sum(spec.weight for spec in mixture if spec.kind == "real"), 10)


def validate_mixture(mixture: Sequence[SourceSpec] = DEFAULT_MIXTURE) -> None:
    """Fail loudly if the mixture violates Corpus v1 invariants."""
    if not mixture:
        raise ValueError("mixture must not be empty")
    names = [spec.name for spec in mixture]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate source names in mixture: {names}")
    for spec in mixture:
        if spec.weight <= 0:
            raise ValueError(f"source {spec.name!r} must have positive weight, got {spec.weight}")
        if spec.upsample <= 0:
            raise ValueError(f"source {spec.name!r} must have positive upsample, got {spec.upsample}")
    total = sum(spec.weight for spec in mixture)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"mixture weights must sum to 1.0, got {total:.12f}")
    real_share = sum(spec.weight for spec in mixture if spec.kind == "real")
    if real_share < 0.70:
        raise ValueError(f"real-code backbone share must be >= 0.70, got {real_share:.4f}")


# ---------------------------------------------------------------------------
# Synthetic-share cap: pure-synthetic slices may never exceed 5% of the mixture.
# ``instruction-sft`` is a hybrid slice (real + AST-induced + augmented), so it
# is excluded from this accounting; the synthetic-only sources are not.
# ---------------------------------------------------------------------------

MAX_SYNTHETIC_SHARE = 0.05

# Whole-corpus share of purely synthetic tokens. The mixture weights above are
# *document* shares used for sampling; the token cap is enforced here and in
# ``dataset.augmentation.cap_rows`` so no run can drift past 5% synthetic.
MAX_SYNTHETIC_TOKEN_SHARE = 0.05

# Slice used to enforce the token cap: pure synthetic slices only.
SYNTHETIC_TOKEN_SLICES: frozenset[str] = frozenset(
    {
        "security-sft",
        "repair-trace-sft",
        "app-sft",
        "app-repair-sft",
        "app-preference-dpo",
        "recovery-sft",
        "project-scaffold-sft",
    }
)


def synthetic_share(mixture: Sequence[SourceSpec] = DEFAULT_MIXTURE) -> float:
    """Document share of the mixture that is pure synthetic (hybrid excluded)."""

    return round(sum(spec.weight for spec in mixture if spec.kind == "synthetic"), 10)


def hybrid_share(mixture: Sequence[SourceSpec] = DEFAULT_MIXTURE) -> float:
    """Document share of the mixture that is real-derived but augmented."""

    return round(sum(spec.weight for spec in mixture if spec.kind == "hybrid"), 10)


def validate_synthetic_budget(
    mixture: Sequence[SourceSpec] = DEFAULT_MIXTURE,
    max_share: float = MAX_SYNTHETIC_TOKEN_SHARE,
) -> None:
    """Fail loudly when synthetic slices are mis-planned against the cap.

    Returns ``{"document_share", "max_share", "token_plan"}``. Invariants:

    - the synthetic *document* share stays at or below 21% (sampling mix), and
    - the synthetic *token* plan (``synthetic_token_budget``) is either within
      the 5% cap or explicitly reported as over it, so downstream builders know
      they must downsample before writing shards.
    """

    document = synthetic_share(mixture)

    if document > 0.21 + 1e-9:
        raise ValueError(
            f"synthetic document share must be <= 21%, got {document:.4f}"
        )

    plan = synthetic_token_budget(int(18_000_000_000), max_share, mixture)

    return {"document_share": document, "max_share": max_share, "token_plan": plan}


validate_mixture(DEFAULT_MIXTURE)


def synthetic_token_budget(
    total_tokens: int,
    max_share: float = MAX_SYNTHETIC_SHARE,
) -> dict:
    """Convert the synthetic cap into a token budget for the whole run."""

    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")

    share = synthetic_share()

    cap = int(total_tokens * max_share)

    return {
        "total_tokens": total_tokens,
        "max_share": max_share,
        "synthetic_share": share,
        "synthetic_slice_tokens": int(total_tokens * share),
        "synthetic_cap_tokens": cap,
        "within_cap": share <= max_share + 1e-9,
    }


class MixtureSampler:
    """Deterministic weighted sampler across Corpus v1 sources.

    ``spec.weight`` already reflects the target token/document share in the
    frozen mixture (summing to 1.00). Pass ``apply_upsample=True`` only when
    evaluating stage-specific additional upsampling overrides.
    """

    def __init__(
        self,
        pools: dict[str, Sequence[Any]],
        mixture: list[SourceSpec] | None = None,
        seed: int = 42,
        *,
        apply_upsample: bool = False,
    ):
        self.pools = {k: list(v) for k, v in pools.items() if len(v) > 0}
        self.mixture = list(mixture) if mixture is not None else list(DEFAULT_MIXTURE)
        self.seed = seed
        self.apply_upsample = apply_upsample

        active = [s for s in self.mixture if s.name in self.pools]
        if not active:
            raise ValueError(f"no non-empty pools match mixture {[s.name for s in self.mixture]}")

        raw = [
            (s.weight * s.upsample if apply_upsample else s.weight)
            for s in active
        ]
        probs = np.array(raw, dtype=np.float64)
        probs /= probs.sum()

        self.active = active
        self.probs = probs

    def target_distribution(self) -> dict[str, float]:
        """Normalized sampling probabilities across active pools."""
        return {spec.name: float(prob) for spec, prob in zip(self.active, self.probs)}

    def sample(self, n: int) -> Iterator[Any]:
        rng = np.random.default_rng(self.seed)
        py_rng = random.Random(self.seed)
        cursors = {s.name: 0 for s in self.active}
        orders: dict[str, list[int]] = {}
        for s in self.active:
            idx = list(range(len(self.pools[s.name])))
            py_rng.shuffle(idx)
            orders[s.name] = idx

        choices = rng.choice(len(self.active), size=n, p=self.probs)
        for c in choices:
            name = self.active[int(c)].name
            pool = self.pools[name]
            pos = cursors[name]
            if pos >= len(pool):
                py_rng.shuffle(orders[name])
                pos = 0
            cursors[name] = pos + 1
            yield pool[orders[name][pos]]

    def sample_with_source(self, n: int) -> Iterator[tuple[str, Any]]:
        """Yield ``(source_name, item)`` pairs deterministically."""
        rng = np.random.default_rng(self.seed)
        py_rng = random.Random(self.seed)
        cursors = {s.name: 0 for s in self.active}
        orders: dict[str, list[int]] = {}
        for s in self.active:
            idx = list(range(len(self.pools[s.name])))
            py_rng.shuffle(idx)
            orders[s.name] = idx

        choices = rng.choice(len(self.active), size=n, p=self.probs)
        for c in choices:
            name = self.active[int(c)].name
            pool = self.pools[name]
            pos = cursors[name]
            if pos >= len(pool):
                py_rng.shuffle(orders[name])
                pos = 0
            cursors[name] = pos + 1
            yield name, pool[orders[name][pos]]


def curriculum_order(
    docs: Iterable[dict],
    step: int,
    total_steps: int,
) -> list[dict]:
    """Order a shard's documents for the current point in training.

    Early training (progress < 0.3) prefers shorter, cleaner files; late
    training shifts weight to long files and multi-file repo packs.
    """
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    scored = []
    for d in docs:
        length = len(d.get("content", ""))
        quality = float(d.get("quality_score", 0.5))
        len_pref = abs(length - (2000 + 20000 * progress)) / 22000
        score = quality - 0.3 * len_pref
        scored.append((score, d))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [d for _, d in scored]


def synthetic_token_budget(
    total_tokens: int,
    max_share: float = MAX_SYNTHETIC_TOKEN_SHARE,
    mixture: Sequence[SourceSpec] = DEFAULT_MIXTURE,
    synthetic_share_of_slice: float = 1.0,
) -> dict:
    """Token budget for the synthetic slices of a run.

    The cap is ``max_share`` of the *whole corpus*, so it is independent of the
    document weights: the synthetic document share (21%) is a sampling choice,
    while the trained token share is capped at 5%. ``synthetic_share_of_slice``
    is the part of a slice's tokens that is actually model-generated (the rest
    of ``instruction-sft``, for example, is real-code induction).
    """

    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")

    if not (0.0 < max_share <= 1.0):
        raise ValueError("max_share must be in (0, 1]")

    plan_tokens = int(
        total_tokens
        * sum(spec.weight for spec in mixture if spec.name in SYNTHETIC_TOKEN_SLICES)
        * synthetic_share_of_slice
    )

    cap = int(total_tokens * max_share)

    return {
        "total_tokens": total_tokens,
        "max_share": max_share,
        "document_share": synthetic_share(mixture),
        "plan_tokens": plan_tokens,
        "plan_share": round(plan_tokens / total_tokens, 6),
        "cap_tokens": cap,
        "within_cap": plan_tokens <= cap,
        "slices": sorted(SYNTHETIC_TOKEN_SLICES),
    }


# The 5% ceiling applies to the tokens each slice may contribute; the mixture
# document weights above are the sampling mix and are only bounded at 21%.
validate_synthetic_budget(DEFAULT_MIXTURE)
