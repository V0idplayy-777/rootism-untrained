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
    max_consecutive_failures: int = 6
    stall_window: int = 6
    max_backtracks: int = 1

    _history: list[str] = field(default_factory=list)
    _retries: dict[str, int] = field(default_factory=dict)
    _switches: dict[str, int] = field(default_factory=dict)
    _backtracks: int = 0

    def reset(self) -> None:

        self._history.clear()

        self._retries.clear()

        self._switches.clear()

        self._backtracks = 0

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

            if repeats >= 2 or attempt > self.max_retries:

                if tool and tool in TOOL_FALLBACKS:

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

                if consecutive >= 3 and self._backtracks < self.max_backtracks:

                    self._backtracks += 1

                    return RecoveryDecision(
                        RecoveryAction.BACKTRACK,
                        "no progress after repeated failures; roll back unverified edits",
                        hint=(
                            "Restore the last known-good state (git stash / "
                            "git checkout -- <files>), re-read the failing test, "
                            "and form a new hypothesis before editing again."
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
                f"first failure for {signature}",
                hint="Retry once; if it fails again you must change approach.",
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

        return RecoveryDecision(RecoveryAction.PROCEED, "no intervention needed")

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
