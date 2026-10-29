"""Recovery policy for long-horizon tool use.

Long sessions fail in characteristically boring ways: the same malformed call
is retried forever, a failing test is re-run with the same command, or the
model keeps editing a file it has already broken. Bigger models degrade here
too; the fix is an explicit policy with bounded, escalating interventions.

Escalation ladder (cheap -> expensive):

1. ``retry``      — transient error, same call (bounded attempts).
2. ``repair``     — malformed call: feed the parser/schema error back.
3. ``change``     — same failure signature twice: force different arguments
                    (different file, different test selector, narrower scope).
4. ``switch``     — the tool itself is wrong for the job: fall back along a
                    known substitution chain (apply_patch -> edit_function ->
                    write_file).
5. ``reflect``    — 3+ consecutive failures: produce a post-mortem and a new
                    plan, pin the failing repro command as a fact.
6. ``backtrack``  — progress stalled: discard the untested edits made since the
                    last known-good checkpoint (git stash/checkout) and retry
                    from evidence, not from the broken state.
7. ``escalate``   — budget exhausted: stop, report blockers and the exact
                    unresolved failure rather than hallucinating success.

The policy is pure logic (no I/O), so it is unit-testable and deterministic; the
agent supplies the environment callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .session import SessionState, failure_signature


class RecoveryAction(str, Enum):
    PROCEED = "proceed"
    RETRY = "retry"
    REPAIR = "repair"
    CHANGE = "change"
    SWITCH = "switch"
    REFLECT = "reflect"
    BACKTRACK = "backtrack"
    WRAP_UP = "wrap_up"
    ESCALATE = "escalate"


TOOL_FALLBACKS: dict[str, list[str]] = {
    "apply_patch": ["edit_function", "write_file"],
    "edit_function": ["write_file", "apply_patch"],
    "write_file": ["apply_patch"],
    "run_tests": ["execute_code"],
    "execute_code": ["run_tests"],
    "search_code": ["list_files", "read_file"],
    "read_file": ["search_code"],
    "git_diff": ["git_status"],
    "git_status": ["git_diff"],
}


_FAILURE_PATTERNS = (
    ("environment", (
        "modulenotfounderror", "importerror", "command not found",
        "enoent", "no such file or directory", "eacces", "permission denied",
        "docker", "container", "sandbox", "connection refused", "enameknown",
        "not installed", "unsupported python",
    )),
    ("transient", (
        "timeout", "timed out", "temporarily unavailable", "rate limit",
        "econnreset", "connection reset", "503", "502", "deadlock",
        "resource temporarily unavailable",
    )),
    ("arguments", (
        "unexpected keyword argument", "positional argument",
        "invalid argument", "schema", "validation", "jsondecodeerror",
        "expected", "malformed", "bad request", "400",
    )),
    ("code", (
        "assertionerror", "assert", "test failed", "tests failed",
        "syntaxerror", "nameerror", "undefined", "typeerror", "valueerror",
        "exception:", "traceback",
    )),
)


def classify_failure(error: str | None) -> str:
    """environment | transient | arguments | code | unknown.

    Environment problems are not fixable by retrying the same call; code
    problems are not fixable by changing arguments; the ladder entry point
    depends on getting this roughly right.
    """

    err = (error or "").lower()

    if not err:
        return "unknown"

    for klass, needles in _FAILURE_PATTERNS:

        if any(n in err for n in needles):

            return klass

    return "unknown"


@dataclass
class RecoveryDecision:
    action: RecoveryAction
    reason: str
    hint: str = ""
    attempt: int = 0
    signature: str = ""
    next_tool: str | None = None

    def to_dict(self) -> dict:

        return {
            "action": self.action.value,
            "reason": self.reason,
            "hint": self.hint,
            "attempt": self.attempt,
            "signature": self.signature,
            "next_tool": self.next_tool,
        }


@dataclass
class RecoveryPolicy:
    """Decide what to do after a failure or a stalled step.

    ``max_retries`` bounds identical retries; ``max_consecutive_failures``
    bounds the ladder before escalation; ``stall_window`` is how many steps
    without progress count as "stuck".
    """

    max_retries: int = 1
    max_transient_retries: int = 2
    max_consecutive_failures: int = 6
    stall_window: int = 6
    max_backtracks: int = 1
    wrap_up_reserve: int = 3

    _history: list[str] = field(default_factory=list)
    _retries: dict[str, int] = field(default_factory=dict)
    _switches: dict[str, int] = field(default_factory=dict)
    _tool_streak: dict[str, int] = field(default_factory=dict)
    _backtracks: int = 0

    def reset(self) -> None:

        self._history.clear()

        self._retries.clear()

        self._switches.clear()

        self._tool_streak.clear()

        self._backtracks = 0
        self.metrics = RecoveryMetrics()

    def decide(
        self,
        state: SessionState,
        tool: str | None = None,
        arguments: dict | None = None,
        error: str | None = None,
        parse_error: bool = False,
        stalled: bool = False,
    ) -> RecoveryDecision:
        """Return the next action. Pure: does not mutate the environment."""

        if error and not parse_error:

            signature = failure_signature(tool, arguments, error)

        elif parse_error:

            signature = failure_signature(tool, arguments, error or "parse error")

        else:

            signature = ""

        if signature:

            self._history.append(signature)

            self._retries[signature] = self._retries.get(signature, 0) + 1

            attempt = self._retries[signature]

            repeats = state.repeats(signature) + attempt - 1

            consecutive = state.consecutive_failures() + 1

            failure_class = classify_failure(error)

            if tool:

                self._tool_streak[tool] = self._tool_streak.get(tool, 0) + 1

            else:

                self._tool_streak.clear()

            if consecutive >= self.max_consecutive_failures:

                return RecoveryDecision(
                    RecoveryAction.ESCALATE,
                    f"{consecutive} consecutive failures; stopping instead of looping",
                    hint=(
                        "Report the exact unresolved error, the files changed, and what "
                        "you tried — do not claim the task is complete."
                    ),
                    attempt=attempt,
                    signature=signature,
                )

            if parse_error:

                return RecoveryDecision(
                    RecoveryAction.REPAIR,
                    "output was not a valid tool call",
                    hint=(
                        "Re-emit exactly one call as <|tool_call|>{\"tool\": ..., "
                        "\"arguments\": {...}} with no prose inside the JSON."
                    ),
                    attempt=attempt,
                    signature=signature,
                )


            if failure_class == "environment" and tool in TOOL_FALLBACKS:

                used = self._switches.get(tool, 0)

                chain = TOOL_FALLBACKS[tool]

                if used < len(chain):

                    self._switches[tool] = used + 1

                    return RecoveryDecision(
                        RecoveryAction.SWITCH,
                        f"environment failure ({failure_class}) for {tool}",
                        hint=(
                            f"`{tool}` cannot work in this environment "
                            f"({(error or '')[:120]}). Use `{chain[used]}` instead."
                        ),
                        attempt=attempt,
                        signature=signature,
                        next_tool=chain[used],
                    )


            tool_hot = tool is not None and self._tool_streak.get(tool, 0) >= 3

            transient_ok = failure_class == "transient" and attempt <= self.max_transient_retries

            if repeats >= 2 or attempt > self.max_retries or tool_hot and not transient_ok:

                if tool and tool in TOOL_FALLBACKS and not (failure_class == "code"):

                    used = self._switches.get(tool, 0)

                    chain = TOOL_FALLBACKS[tool]

                    if used < len(chain):

                        self._switches[tool] = used + 1

                        return RecoveryDecision(
                            RecoveryAction.SWITCH,
                            f"{signature} repeated {repeats + 1}x; tool {tool} is not working",
                            hint=(
                                f"Use `{chain[used]}` instead of `{tool}` and explain the "
                                "smaller step you are taking."
                            ),
                            attempt=attempt,
                            signature=signature,
                            next_tool=chain[used],
                        )


                if (failure_class == "code" and repeats >= 3) or consecutive >= 4:

                    if self._backtracks < self.max_backtracks:

                        self._backtracks += 1

                        return RecoveryDecision(
                            RecoveryAction.BACKTRACK,
                            "the change itself is wrong; roll back unverified edits",
                            hint=(
                                "Restore the last verified milestone (revert / git "
                                "checkout -- <files>), re-read the failing test, and "
                                "form a new hypothesis before editing again."
                            ),
                            attempt=attempt,
                            signature=signature,
                        )

                    return RecoveryDecision(
                        RecoveryAction.REFLECT,
                        "repeated code failures; the hypothesis is wrong",
                        hint=(
                            "Stop editing. Write a post-mortem: what you believed, "
                            "what the evidence says, the smallest next verifiable "
                            "step. Update the plan before the next tool call."
                        ),
                        attempt=attempt,
                        signature=signature,
                    )

                return RecoveryDecision(
                    RecoveryAction.CHANGE,
                    f"same failure again ({signature}); arguments must change",
                    hint=(
                        "Do not repeat the previous call. Change the path, the test "
                        "selector, or the scope, and state what you expect to observe."
                    ),
                    attempt=attempt,
                    signature=signature,
                )

            return RecoveryDecision(
                RecoveryAction.RETRY,
                f"failure class {failure_class!r} for {signature}",
                hint=(
                    "Transient-looking failure: retry once"
                    if transient_ok
                    else "Retry once; if it fails again you must change approach."
                ),
                attempt=attempt,
                signature=signature,
            )

        if stalled:

            consecutive = state.consecutive_failures()

            if consecutive >= 2:

                if self._backtracks < self.max_backtracks:

                    self._backtracks += 1

                    return RecoveryDecision(
                        RecoveryAction.BACKTRACK,
                        "stalled with recent failures",
                        hint="Roll back the last unverified edits and re-read the failure.",
                    )

                return RecoveryDecision(
                    RecoveryAction.ESCALATE,
                    "stalled and out of recovery budget",
                    hint="Summarise blockers and stop.",
                )

            return RecoveryDecision(
                RecoveryAction.REFLECT,
                f"no measurable progress in ~{self.stall_window} steps",
                hint=(
                    "Write a post-mortem: what you believed, what the evidence says, "
                    "what the smallest next verifiable step is. Update the plan."
                ),
            )

        dec = RecoveryDecision(RecoveryAction.PROCEED, "no intervention needed")
        self.metrics.record(dec)
        return dec

    def note_success(self, tool: str | None) -> None:
        """A successful call clears the tool cooldown (call after every success)."""

        if tool:

            self._tool_streak.pop(tool, None)

    def wrap_up_needed(self, state: SessionState) -> RecoveryDecision | None:
        """Near the step budget with an incomplete plan: switch to verifying.

        Without this, agents spend their last steps mid-edit and never run
        the final test suite. Three steps of reserve is enough to run tests
        and write an honest summary — not enough to start anything new.
        """

        remaining = state.max_steps - state.step

        if remaining > max(self.wrap_up_reserve, 1):

            return None

        if state.todos.completion >= 1.0:

            return None

        return RecoveryDecision(
            RecoveryAction.WRAP_UP,
            f"only {remaining} steps left and the plan is "
            f"{state.todos.completion:.0%} complete",
            hint=(
                "Stop starting new work. Run the test suite once, mark what is "
                "verified with evidence, and write the final answer stating "
                "exactly what is done and what is not."
            ),
        )

    @staticmethod
    def reflect_prompt(state: SessionState, decision: RecoveryDecision) -> str:
        """Prompt injected for REPAIR/CHANGE/REFLECT/BACKTRACK."""

        lines = [
            "<|tool_error|>",
            f"Recovery required: {decision.reason}",
        ]

        if decision.hint:

            lines.append(decision.hint)

        if state.recent_failures(3):

            lines.append("Recent failures:")

            for record in state.recent_failures(3):

                first = (record.error or "").splitlines()

                lines.append(
                    f"- step {record.step} {record.tool}: "
                    f"{(first[0] if first else 'error')[:200]}"
                )

        if state.plan:

            lines.append("Current plan:")

            lines.append(state.todos.render())

        lines.append(
            "Respond with the updated plan if it changed, then exactly one tool call."
        )

        lines.append("<|endoftext|>")

        return "\n".join(lines)

    @staticmethod
    def backtrack_command(files: list[str] | None = None) -> list[str]:
        """git command that returns the workspace to a verified state."""

        if files:

            return ["git", "checkout", "--", *files]

        return ["git", "stash", "--include-untracked"]


class StallDetector:
    """Detect "spinning": many steps, no new evidence, repeated actions."""

    def __init__(self, window: int = 6, min_progress: float = 0.5):

        self.window = window

        self.min_progress = min_progress

        self._scores: list[float] = []

        self._actions: list[str] = []

    def observe(self, state: SessionState, action_key: str) -> bool:
        """Record a step; True when the session is stalled."""

        self._scores.append(state.progress_score())

        self._actions.append(action_key)

        if len(self._scores) <= self.window:

            return False

        gained = self._scores[-1] - self._scores[-1 - self.window]

        recent = self._actions[-self.window :]

        repeated = len(set(recent)) <= 2

        return gained < self.min_progress or repeated

    def reset(self) -> None:

        self._scores.clear()

        self._actions.clear()

@dataclass
class RecoveryMetrics:
    """Measurable policy: track loop_rate / escalation_rate on held-out hard trajectories (row 5)."""
    total_steps: int = 0
    loops: int = 0
    escalations: int = 0
    switches: int = 0
    backtracks: int = 0
    history: list[dict] = field(default_factory=list)

    def record(self, decision) -> None:
        self.total_steps += 1
        if decision.action == RecoveryAction.PROCEED:
            return
        self.history.append({"step": self.total_steps, "action": str(decision.action), "reason": decision.reason})
        if decision.action in (RecoveryAction.RETRY, RecoveryAction.CHANGE):
            # loop proxy: repeated same signature
            self.loops += 1
        if decision.action == RecoveryAction.ESCALATE:
            self.escalations += 1
        if decision.action == RecoveryAction.SWITCH:
            self.switches += 1
        if decision.action == RecoveryAction.BACKTRACK:
            self.backtracks += 1

    @property
    def loop_rate(self) -> float:
        return self.loops / max(1, self.total_steps)

    @property
    def escalation_rate(self) -> float:
        return self.escalations / max(1, self.total_steps)

    def to_dict(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "loops": self.loops,
            "escalations": self.escalations,
            "switches": self.switches,
            "backtracks": self.backtracks,
            "loop_rate": round(self.loop_rate, 4),
            "escalation_rate": round(self.escalation_rate, 4),
            "history": self.history[-20:],
        }

    def minimise_report(self) -> str:
        return f"loop_rate {self.loop_rate:.2%} escalation_rate {self.escalation_rate:.2%} (goal: minimise both)"

