"""Distribution-shift data: unusual phrasing and distant concept combinations.

Synthetic instruction data plus The Stack covers *common* patterns well. It is
weak exactly where users are unusual:

- terse, vague, frustrated, non-native or contradictory phrasing,
- requests that require combining distant ideas (compilers + billing, ML +
  databases) instead of recalling a template,
- underspecified requests where the right move is to ask, not to guess.

This module synthesises those deliberately. All generation is deterministic
(seeded) and dependency-free, and every generated task carries explicit
requirements and acceptance criteria so it can be verified rather than merely
"looked plausible".

- ``PhrasingAugmenter``: style-transfer of an existing instruction, preserving
  the technical content (so correctness labels stay valid).
- ``ConceptCombiner``: cross-domain task synthesis with a compilable interface
  skeleton and a testable acceptance list.
- ``AbstractionLadder``: the same task at spec level vs implementation level
  (teaches the model to move between them).
- ``AmbiguityTask``: underspecified prompts plus the clarification question a
  good assistant must ask before writing code.
"""

from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Phrasing
# ---------------------------------------------------------------------------

_FORMAL_TO_CASUAL = {
    "Please implement": "can you write",
    "implement": "write",
    "Create": "make",
    "Provide": "give me",
    "Ensure": "make sure",
    "must": "should",
    "function": "function",
    "returns": "gives back",
}

_POLITE_PREFIX = [
    "hey, ",
    "",
    "quick one — ",
    "sorry if this is obvious, but ",
]


def _terse(text: str) -> str:
    """Drop filler, keep the task. "Write a function that parses X and ..."."""

    text = re.sub(r"\b(?:Please|Kindly|I would like you to|Could you please)\b", "", text)
    text = re.sub(r"\bthat (?:takes|accepts|receives)\b", "taking", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 160:
        text = text.split(". ")[0].rstrip(".") + "."
    return text


def _casual(text: str, rng: random.Random) -> str:
    out = text
    for formal, casual in _FORMAL_TO_CASUAL.items():
        out = re.sub(rf"\b{formal}\b", casual, out)
    return rng.choice(_POLITE_PREFIX) + out[0].lower() + out[1:] if out else out


def _verbose(text: str, rng: random.Random) -> str:
    return (
        "Context: I'm working in a Python service that is deployed to a small "
        "team and needs to stay maintainable. "
        f"{text} Please include type hints, a short docstring, and handle the "
        "obvious edge cases. If you make an assumption, state it explicitly."
    )


def _constraint_first(text: str) -> str:
    return (
        "Constraints: Python 3.11, standard library only, no global state, "
        "must be unit-testable, and must not exceed 40 lines.\n"
        f"Task: {text}"
    )


def _non_native(text: str) -> str:
    out = text
    out = re.sub(r"\bthe function\b", "the function", out)
    out = out.replace("returns", "return").replace("creates", "create")
    return f"Hello, I need help with this: {out} Please explain what you do."


def _typo(text: str, rng: random.Random) -> str:
    swaps = [
        ("function", "funciton"),
        ("returns", "retsuns"),
        ("implement", "implment"),
        ("string", "stirng"),
        ("error", "errror"),
        ("database", "databse"),
    ]
    out = text
    for correct, wrong in rng.sample(swaps, k=2):
        out = out.replace(correct, wrong, 1)
    return out


def _frustrated(text: str) -> str:
    return (
        "This is the third time I'm asking — the previous answers didn't work. "
        f"{text} Please actually check it runs, and tell me if my request is "
        "impossible instead of inventing APIs."
    )


def _partial_repro(text: str) -> str:
    return (
        "Repro (fails):\n"
        "```python\n>>> result = solve(input_data)\n"
        "TypeError: argument of type 'NoneType' is not iterable\n```\n"
        f"Expected behaviour: {text}"
    )


PHRASING_STYLES = {
    "terse": _terse,
    "casual": _casual,
    "verbose": _verbose,
    "constraint_first": _constraint_first,
    "non_native": _non_native,
    "typo": _typo,
    "frustrated": _frustrated,
    "partial_repro": _partial_repro,
}


@dataclass
class PhrasingAugmenter:
    """Rewrite an instruction in a different register, preserving content."""

    seed: int = 42

    styles: tuple[str, ...] = tuple(PHRASING_STYLES)

    def variants(self, instruction: str, limit: int | None = None) -> list[dict]:
        rng = random.Random(f"{self.seed}:{instruction[:64]}")

        chosen = list(self.styles) if limit is None else rng.sample(
            list(self.styles), min(limit, len(self.styles))
        )

        out = []

        for style in chosen:

            fn = PHRASING_STYLES[style]

            try:

                text = fn(instruction, rng) if fn.__code__.co_argcount > 1 else fn(instruction)

            except TypeError:  # pragma: no cover - defensive

                text = instruction

            out.append(
                {
                    "instruction": text,
                    "phrasing": style,
                    "source_instruction": instruction,
                }
            )

        return out


# ---------------------------------------------------------------------------
# Distant concept combinations
# ---------------------------------------------------------------------------

DOMAIN_CAPABILITIES = {
    "compilers": "lexing, parsing, AST transformation, type checking, IR lowering",
    "databases": "query planning, indexes, transactions, MVCC, WAL, b-tree pages",
    "distributed_systems": "leases, quorums, idempotency, retries, clock skew, partitions",
    "machine_learning": "embeddings, vector similarity, online/offline skew, drift, batching",
    "billing": "idempotent charges, proration, rounding, tax, currency, reconciliation",
    "networking": "backpressure, framing, TLS, connection pools, timeouts, retries",
    "security": "authentication, authorisation, constant-time comparison, secret rotation",
    "observability": "structured logs, traces, sampling, cardinality, SLOs, alert fatigue",
    "storage": "chunking, checksums, compaction, tiering, durability, garbage collection",
    "cli_design": "subcommands, streaming output, exit codes, config discovery, idempotency",
    "streaming": "windowing, watermarks, exactly-once, replay, late data, operator state",
    "graphics": "transforms, scene graphs, tessellation, hit testing, dirty regions",
}

DOMAIN_PAIRS = [
    (
        "compilers",
        "billing",
        "Compile a pricing rule DSL into an executable price calculator, then reconcile "
        "invoices produced by two versions of the DSL.",
    ),
    (
        "databases",
        "machine_learning",
        "Build a vector index with crash-safe persistence and evaluate recall/latency "
        "under concurrent inserts.",
    ),
    (
        "distributed_systems",
        "cli_design",
        "A CLI that acquires a lease, runs a migration idempotently, and resumes after "
        "an interrupted run.",
    ),
    (
        "observability",
        "streaming",
        "Compute per-tenant p99 latency over an event stream with bounded cardinality "
        "and late-arriving events.",
    ),
    (
        "storage",
        "security",
        "A content-addressed blob store with deduplication, authenticated chunk hashes, "
        "and safe garbage collection.",
    ),
    (
        "networking",
        "compilers",
        "A packet framing protocol with a schema definition language and a generated "
        "encoder/decoder that rejects malformed input.",
    ),
    (
        "graphics",
        "databases",
        "Spatial queries over a scene graph persisted in a file format that supports "
        "incremental updates.",
    ),
    (
        "machine_learning",
        "billing",
        "An inference gateway that meters token usage per customer with retry-safe "
        "billing and budget enforcement.",
    ),
    (
        "compilers",
        "distributed_systems",
        "A distributed rule engine where each worker compiles the same expression "
        "language and results are deterministic across versions.",
    ),
    (
        "streaming",
        "storage",
        "A log-structured event store with segment compaction and exactly-once replay.",
    ),
]


def _slug(text: str) -> str:

    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:24]


@dataclass
class ConceptCombiner:
    """Cross-domain tasks with a skeleton that must compile and a test plan."""

    seed: int = 42

    pairs: list[tuple[str, str, str]] = field(default_factory=lambda: list(DOMAIN_PAIRS))

    def combine(self, count: int = 4, seed: int | None = None) -> list[dict]:
        rng = random.Random(self.seed if seed is None else seed)

        chosen = rng.sample(self.pairs, min(count, len(self.pairs)))

        out = []

        for primary, secondary, task in chosen:

            module = _slug(f"{primary}_{secondary}")

            interface = [
                "from __future__ import annotations",
                "",
                "from dataclasses import dataclass, field",
                "",
                "",
                "@dataclass",
                "class Config:",
                '    """Runtime knobs; defaults must be safe for production."""',
                "",
                "    max_retries: int = 3",
                "    timeout_s: float = 5.0",
                "    extra: dict[str, object] = field(default_factory=dict)",
                "",
                "",
                f"class {self._class_name(module)}:",
                f'    """{task.split(",")[0]}."""',
                "",
                "    def __init__(self, config: Config | None = None) -> None:",
                "        self.config = config or Config()",
                "",
                "    def run(self, payload: dict[str, object]) -> dict[str, object]:",
                "        raise NotImplementedError",
                "",
            ]

            skeleton = "\n".join(interface)

            try:

                ast.parse(skeleton)

            except SyntaxError:  # pragma: no cover - generator bug guard

                raise

            requirements = [
                f"Integrate {primary} concerns: {DOMAIN_CAPABILITIES[primary]}",
                f"Integrate {secondary} concerns: {DOMAIN_CAPABILITIES[secondary]}",
                "State the invariants that must hold across failures/retries.",
                "Handle concurrent or partial execution explicitly.",
                "No network or third-party dependencies; standard library only.",
            ]

            acceptance = [
                "Unit tests cover the happy path and at least two failure modes.",
                "Re-running the same operation twice is safe (idempotent).",
                "The error path fails closed and does not leak secrets.",
                "A short design note explains the trade-off you chose.",
            ]

            out.append(
                {
                    "family": f"{primary}+{secondary}",
                    "instruction": (
                        f"Design and implement: {task} "
                        "Start by writing the plan and the invariants, then the code."
                    ),
                    "input_context": skeleton,
                    "output": self._reference_answer(module, requirements, acceptance),
                    "category": "cross_domain_synthesis",
                    "language": "python",
                    "difficulty": "expert",
                    "metadata": {
                        "primary_domain": primary,
                        "secondary_domain": secondary,
                        "requirements": requirements,
                        "acceptance": acceptance,
                    },
                }
            )

        return out

    @staticmethod
    def _class_name(module: str) -> str:

        return "".join(part.capitalize() for part in module.split("_")) or "Engine"

    @staticmethod
    def _reference_answer(module: str, requirements: list[str], acceptance: list[str]) -> str:
        """A plan-shaped answer (what the model should produce first)."""

        lines = ["Plan:", ""]
        lines.extend(f"{i + 1}. {r}" for i, r in enumerate(requirements))
        lines.extend(["", "Acceptance criteria:"])
        lines.extend(f"- {a}" for a in acceptance)
        lines.extend(
            [
                "",
                "Invariants:",
                "- Ordering is preserved per key; operations are idempotent on retry.",
                "- Any partially applied state is detected and repaired on restart.",
                "",
                "Next: implement the dataclasses above, then the failure-mode tests.",
            ]
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abstraction ladder + ambiguity
# ---------------------------------------------------------------------------


@dataclass
class AbstractionLadder:
    """Same task as a spec, a design, and an implementation brief."""

    def rungs(self, instruction: str) -> list[dict]:
        return [
            {
                "level": "spec",
                "instruction": (
                    "Write a precise specification (inputs, outputs, invariants, "
                    f"failure modes) for: {instruction}"
                ),
            },
            {
                "level": "design",
                "instruction": (
                    "Propose two designs for the following, then pick one and "
                    f"justify it against the constraints: {instruction}"
                ),
            },
            {
                "level": "implement",
                "instruction": (
                    f"Implement it. {instruction} Include tests for the invariants "
                    "you claimed."
                ),
            },
        ]


AMBIGUITY_TASKS: list[dict] = [
    {
        "instruction": "Make the sync faster.",
        "missing": ["which system", "what latency target", "measured baseline", "consistency requirement"],
        "clarification": (
            "Before changing code I need to know: which sync path (initial backfill "
            "or incremental?), the current p50/p99 latency, the target, and whether "
            "strong consistency is required. Meanwhile, here is a measurement plan: ..."
        ),
    },
    {
        "instruction": "Fix the login bug.",
        "missing": ["repro steps", "error text", "affected users/environment"],
        "clarification": (
            "I can't reproduce this yet. Please share the failing request, the error "
            "and the auth provider. If you can't, I'll start by adding structured "
            "logging around token validation and issuing a narrower repro."
        ),
    },
    {
        "instruction": "Add caching.",
        "missing": ["what to cache", "invalidation policy", "consistency tolerance"],
        "clarification": (
            "What data is hot, how stale may it be, and what invalidates it? I'll "
            "assume read-through with TTL only after you confirm staleness tolerance."
        ),
    },
    {
        "instruction": "Refactor the service.",
        "missing": ["goal (readability/perf/a testability)", "scope", "deadline"],
        "clarification": (
            "Which property should improve, and what must not change? I'll propose "
            "a scoped plan with tests first so the refactor is verifiable."
        ),
    },
    {
        "instruction": "Support multi-tenancy.",
        "missing": ["isolation level", "migration strategy", "billing model"],
        "clarification": (
            "Tenant isolation can mean separate schemas, row-level filters, or "
            "separate deployments. Which is required, and is cross-tenant leakage "
            "regulatory or convenience?"
        ),
    },
    {
        "instruction": "Write tests for the API.",
        "missing": ["which endpoints matter", "contract source", "fixtures available"],
        "clarification": (
            "I'll start with characterisation tests for the endpoints in openapi.yaml "
            "using the fixtures in tests/. Tell me if a different contract is authoritative."
        ),
    },
]


@dataclass
class AmbiguityGenerator:
    """Tasks where the correct first action is a targeted question."""

    seed: int = 42

    def generate(self, count: int = 3) -> list[dict]:
        rng = random.Random(self.seed)

        chosen = rng.sample(AMBIGUITY_TASKS, min(count, len(AMBIGUITY_TASKS)))

        return [
            {
                "instruction": item["instruction"],
                "input_context": "",
                "output": item["clarification"],
                "category": "clarification",
                "difficulty": "intermediate",
                "metadata": {"missing_information": item["missing"]},
            }
            for item in chosen
        ]


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------


def augment_rows(
    rows: list[dict],
    per_row: int = 2,
    seed: int = 42,
    include_cross_domain: int = 4,
    include_clarification: int = 3,
) -> list[dict]:
    """Expand instruction rows into a distribution-shift batch.

    Keeps the original rows, adds phrasing variants (same content, same answer)
    and appends cross-domain + clarification tasks. Output rows are plain dicts
    matching ``dataset.instruction_data.InstructionSample`` fields.
    """

    rng = random.Random(seed)

    augmenter = PhrasingAugmenter(seed=seed)

    out: list[dict] = [dict(r) for r in rows]

    for row in rows:

        if "output" not in row:

            continue

        for variant in augmenter.variants(row.get("instruction", ""), limit=per_row):

            new_row = dict(row)

            new_row["instruction"] = variant["instruction"]

            new_row["category"] = f"{row.get('category', 'task')}:{variant['phrasing']}"

            new_row["metadata"] = {
                **(row.get("metadata") or {}),
                "phrasing": variant["phrasing"],
                "source_instruction": variant["source_instruction"],
            }

            out.append(new_row)

    out.extend(ConceptCombiner(seed=seed).combine(include_cross_domain))

    out.extend(AmbiguityGenerator(seed=seed).generate(include_clarification))

    rng.shuffle(out)

    return out


def ladders_for(rows: list[dict], limit: int = 3) -> list[dict]:
    """Spec/design/implement triplets for the first few rows."""

    ladder = AbstractionLadder()

    out = []

    for row in rows[:limit]:

        for rung in ladder.rungs(row.get("instruction", "")):

            out.append(
                {
                    "instruction": rung["instruction"],
                    "input_context": row.get("input_context", ""),
                    "output": row.get("output", ""),
                    "category": f"abstraction:{rung['level']}",
                    "language": row.get("language", "python"),
                    "difficulty": row.get("difficulty", "intermediate"),
                    "metadata": {"abstraction_level": rung["level"]},
                }
            )

    return out
