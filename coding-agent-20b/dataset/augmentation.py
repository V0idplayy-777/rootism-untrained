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
    "parsing": "grammars, tokenizers, error recovery, streaming lexing, ambiguity",
    "privacy": "PII classification, redaction, retention, k-anonymity, purpose binding",
    "caching": "eviction, invalidation, stampedes, TTL vs version tokens, hit ratios",
    "consistency": "read-your-writes, monotonic reads, staleness bounds, quorum reads",
    "queues": "backpressure, leases, dead letters, ordering, redelivery storms",
    "reliability": "crash safety, at-least/at-most once, idempotency keys, poison messages",
    "search": "inverted indexes, fuzzy matching, ranking, recall/precision, analyzers",
    "localization": "unicode normalization, collation, locale formatting, transliteration",
    "auth": "sessions, tokens, scopes, rotation, least privilege, device flows",
    "scheduling": "priority, fairness, preemption, bin packing, deadline awareness",
    "finops": "budgets, metering, allocation, unit economics, cost ceilings",
    "protocols": "framing, versioning, negotiation, heartbeats, checksums, backpressure",
    "embedded": "bounded memory, watchdogs, power modes, deterministic timing, bitpacking",
    "games": "tick loops, determinism, interpolation, rollback, anti-cheat commitments",
    "pipelines": "staging, lineage, backfills, schema evolution, replayability",
    "compliance": "data residency, purpose limitation, audit evidence, deletion requests",
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
    (
        "parsing",
        "privacy",
        "A log scrubber with a regex-based PII detector and per-retention-class "
        "redaction that never reorders log lines.",
    ),
    (
        "caching",
        "consistency",
        "A read-through cache with explicit invalidation tokens that stays coherent "
        "when the backing store is briefly unavailable.",
    ),
    (
        "queues",
        "reliability",
        "A durable work queue with lease deadlines, poison-message quarantine, and "
        "replay after a consumer crash.",
    ),
    (
        "search",
        "localization",
        "A typo-tolerant index over multilingual product names with per-locale "
        "collation and accent-insensitive fallback.",
    ),
    (
        "auth",
        "observability",
        "An audit trail for authorization decisions that is append-only, queryable "
        "by principal, and free of credential material.",
    ),
    (
        "scheduling",
        "finops",
        "A job scheduler that bin-packs batch work against spot capacity with "
        "per-team budget ceilings and preemption fairness.",
    ),
    (
        "protocols",
        "embedded",
        "A framing layer for a constrained device protocol with bounded buffers, "
        "checksums, and a watchdog-friendly heartbeat.",
    ),
    (
        "games",
        "networking",
        "A deterministic lockstep simulation with rollback replay and a "
        "cheat-resistant input commitment scheme.",
    ),
    (
        "pipelines",
        "compliance",
        "A data pipeline that propagates data-classification tags through every "
        "transform and blocks sinks without an approved purpose.",
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




CONFLICT_TASKS = [
    {
        "instruction": (
            "Make the payments webhook handler process events in strict arrival "
            "order, and also handle them concurrently so throughput stays above "
            "500 events/s."
        ),
        "conflicts": [
            "strict arrival order serialises processing",
            "concurrent processing reorders by completion time",
        ],
        "resolution": (
            "These constraints conflict: strict ordering requires serialising per "
            "key, which caps throughput. Two reconciliations are possible: "
            "(a) per-account ordering with bounded concurrency between accounts, "
            "or (b) sequence numbers with out-of-order buffering. Which invariant "
            "is the real requirement - per-account order or global order?"
        ),
    },
    {
        "instruction": (
            "Store the audit log append-only and immutable, but add an endpoint "
            "to let support staff redact customer PII from past entries."
        ),
        "conflicts": [
            "append-only/immutable history",
            "in-place redaction of past entries",
        ],
        "resolution": (
            "Redaction contradicts immutability. The standard reconciliation is "
            "crypto-shredding: keep entries, redact the encryption key material "
            "for the affected fields, and record the redaction itself as a new "
            "append-only event. Is that acceptable, or must entries be physically "
            "removed (which changes the compliance story)?"
        ),
    },
    {
        "instruction": (
            "Our load balancer must never drop a request, and it must apply "
            "backpressure by rejecting requests when the queue exceeds 100."
        ),
        "conflicts": [
            "never drop a request",
            "reject when queue > 100",
        ],
        "resolution": (
            "Rejecting under backpressure is dropping by another name. If 'never "
            "drop' means 'never silently lose', rejecting with 503 + Retry-After "
            "is consistent; if it means 'always accept', the queue must be "
            "unbounded and we need a different overload story. Which is it?"
        ),
    },
    {
        "instruction": (
            "Add caching so the dashboard loads in under 200 ms, but the data "
            "shown must always be at most one second stale and every write must "
            "be visible to the reader immediately."
        ),
        "conflicts": [
            "cache serves data up to 1 s old",
            "writes visible immediately (read-your-writes)",
        ],
        "resolution": (
            "Read-your-writes plus a 1 s staleness window conflict for the writer "
            "themselves. Reconciliation: version tokens per session - writers "
            "bypass or invalidate their own cache key, everyone else accepts the "
            "1 s window. Should the strong guarantee apply to all readers or "
            "just the writer's session?"
        ),
    },
]


@dataclass
class ConstraintConflictGenerator:
    """Requests whose correct answer surfaces the conflict instead of
    silently picking a side.

    Models trained only on compliant data pick whichever constraint they read
    last and ship it. The trained behaviour here is: name both constraints,
    explain the tension, propose the standard reconciliations, ask the one
    question that decides it.
    """

    seed: int = 42

    def generate(self, count: int = 3) -> list[dict]:
        rng = random.Random(self.seed)

        chosen = rng.sample(CONFLICT_TASKS, min(count, len(CONFLICT_TASKS)))

        return [
            {
                "instruction": item["instruction"],
                "input_context": "",
                "output": item["resolution"],
                "category": "constraint_conflict",
                "difficulty": "expert",
                "metadata": {"conflicts": item["conflicts"]},
            }
            for item in chosen
        ]


@dataclass
class ClarificationDialogue:
    """Multi-turn SFT: ambiguous ask -> targeted question -> answer -> build.

    Single-turn clarification data teaches the model to ask; it does not teach
    it to *use the answer*. These rows are full dialogues where the assistant
    asks exactly one decisive question, receives the missing fact, and only
    then implements.
    """

    seed: int = 42

    DIALOGUES = [
        {
            "turns": [
                ("user", "Add pagination to the orders endpoint."),
                (
                    "assistant",
                    "Before I change the API: should pagination be cursor-based "
                    "(stable under inserts, opaque tokens) or offset-based "
                    "(simpler, but pages can skip/duplicate rows when orders "
                    "arrive mid-scroll)? And is this a breaking change or can I "
                    "version the endpoint?",
                ),
                ("user", "Cursor-based. It is a new v2 endpoint, breaking is fine."),
                (
                    "assistant",
                    "Cursor-based v2 it is. The cursor will be the base64 of "
                    "(created_at, id) - the id breaks ties for equal timestamps "
                    "so the ordering is total and stable. Default page size 50, "
                    "max 200, `next_cursor` null on the last page. Here is the "
                    "implementation with an index on (created_at, id) to keep "
                    "the keyset scan O(page_size).",
                ),
            ],
            "missing": "pagination style + versioning policy",
        },
        {
            "turns": [
                ("user", "Backfill the user_timezones column for all 40M users."),
                (
                    "assistant",
                    "One thing decides the design: is the source of truth the "
                    "last-login IP geolocation, or the account's country field? "
                    "Geolocation needs an offline database and is approximate at "
                    "borders; the country field is exact but often un-set. Which "
                    "should win when they disagree?",
                ),
                ("user", "Country field wins; fall back to geolocation only when it is empty."),
                (
                    "assistant",
                    "Then the backfill is two passes: (1) copy country -> "
                    "user_timezones where country is set, resolving the country "
                    "code to an IANA zone via the embedded mapping; (2) for the "
                    "remainder, batch-resolve last-login IPs against the offline "
                    "GeoLite DB. I will run it in 10k-row id-ordered chunks with "
                    "a resume checkpoint, so a restart never reprocesses a chunk.",
                ),
            ],
            "missing": "source-of-truth precedence",
        },
        {
            "turns": [
                ("user", "Make the export job faster; it takes six hours."),
                (
                    "assistant",
                    "Faster how - wall-clock or cost? And can I change the "
                    "output contract? If the rows must stay globally ordered, "
                    "parallelising breaks the contract unless we pay for a merge "
                    "step; if consumers accept per-part ordering, we can shard "
                    "by key and finish in minutes. Which constraint holds?",
                ),
                ("user", "Wall-clock. Per-part ordering is fine, consumers merge anyway."),
                (
                    "assistant",
                    "Sharding by hash(key) it is: 16 workers, each writing its "
                    "own part file plus a manifest with row counts and a per-part "
                    "checksum. Consumers that merge get the manifest to verify "
                    "completeness, and a failed worker only rewrites its own "
                    "part. Expected wall-clock: ~25 minutes including verification.",
                ),
            ],
            "missing": "ordering contract vs wall-clock goal",
        },
    ]

    def generate(self, count: int = 3) -> list[dict]:
        rng = random.Random(self.seed)

        chosen = rng.sample(self.DIALOGUES, min(count, len(self.DIALOGUES)))

        return [
            {
                "instruction": "Multi-turn conversation (clarify, then implement).",
                "input_context": "",
                "output": dialogue_text,
                "category": "clarification_dialogue",
                "difficulty": "advanced",
                "metadata": {
                    "turns": len(d["turns"]),
                    "missing_information": d["missing"],
                },
            }
            for d in chosen
            for dialogue_text in ["\n\n".join(f"{role}: {text}" for role, text in d["turns"])]
        ]


RECOVERY_SCENARIOS = [
    {
        "tool": "run_tests",
        "error": "ModuleNotFoundError: No module named 'requests'",
        "wrong_response": "run_tests again with the same test path",
        "right_response": (
            "The environment lacks the package, so re-running tests cannot fix "
            "it: install 'requests' first (execute_code with pip install), then "
            "re-run the same test selector."
        ),
        "failure_class": "environment",
    },
    {
        "tool": "apply_patch",
        "error": "AssertionError: hunk failed to apply at src/auth.py:42",
        "wrong_response": "apply the identical patch again",
        "right_response": (
            "The file drifted from what the patch expects. Re-read the current "
            "src/auth.py around line 42, regenerate the hunk with fresh context "
            "lines, and only then re-apply - or fall back to edit_function for "
            "the single symbol involved."
        ),
        "failure_class": "code",
    },
    {
        "tool": "search_code",
        "error": "Timeout: regex compilation exceeded 5s",
        "wrong_response": "search_code with the same pattern",
        "right_response": (
            "The pattern is pathologically backtracking. Rewrite it without "
            "nested quantifiers, or narrow file_pattern to the likely "
            "directories first - a bounded scope beats a faster regex."
        ),
        "failure_class": "transient",
    },
    {
        "tool": "execute_code",
        "error": "TypeError: dump() got an unexpected keyword argument 'ensure_ascii'",
        "wrong_response": "execute_code with the same call",
        "right_response": (
            "The installed serialiser is not the one the snippet assumed. Check "
            "the actual signature (execute_code with inspect.signature), then "
            "adapt the call - do not wrap it in try/except and move on."
        ),
        "failure_class": "arguments",
    },
    {
        "tool": "run_tests",
        "error": "FAILED tests/test_expiry.py::test_expired_token_rejected - assert 'revoked' == 'expired'",
        "wrong_response": "re-run the failing test unchanged",
        "right_response": (
            "The test encodes the requirement: expired tokens must report "
            "'expired', the code reports 'revoked'. Read the failing assertion, "
            "locate the status mapping, fix the code (not the test), and re-run "
            "exactly this test before anything else."
        ),
        "failure_class": "code",
    },
]


@dataclass
class RecoveryInstructionGenerator:
    """SFT data for the behaviour the recovery policy enforces at runtime.

    Given a failed tool call, the trained response is a *changed* approach
    that names the failure class and the next verifiable step - never the
    same call again. This is the offline twin of `agent/recovery.py`.
    """

    seed: int = 42

    def generate(self, count: int = 5) -> list[dict]:
        rng = random.Random(self.seed)

        chosen = rng.sample(RECOVERY_SCENARIOS, min(count, len(RECOVERY_SCENARIOS)))

        return [
            {
                "instruction": (
                    "You are working in a coding agent session. Your previous "
                    f"tool call `{s['tool']}` failed with:\n\n{s['error']}\n\n"
                    "What is your next action? Answer with the single next tool "
                    "call you would make and why."
                ),
                "input_context": "",
                "output": s["right_response"],
                "category": "error_recovery",
                "difficulty": "advanced",
                "metadata": {
                    "failure_class": s["failure_class"],
                    "tool": s["tool"],
                    "anti_pattern": s["wrong_response"],
                },
            }
            for s in chosen
        ]


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

    out.extend(ConstraintConflictGenerator(seed=seed).generate(2))

    out.extend(ClarificationDialogue(seed=seed).generate(2))

    out.extend(RecoveryInstructionGenerator(seed=seed).generate(3))

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
