"""Long-session state, planning and hierarchical context compaction.

The failure mode this addresses: after ~15 tool calls the transcript is mostly
raw tool output, the model loses the plan, repeats a failed action, and either
loops or forgets the constraint it was given. That is not a sampling problem,
it is a state-management problem.

Design:

- ``SessionState`` is the durable truth: plan, todos, pinned facts, failures,
  files touched, scratchpad. It serialises to JSON, so a session can be resumed
  and so evaluation can measure state coherence instead of eyeballing it.
- ``ObservationCompactor`` turns each tool result into a *small* structured
  observation (failing test names, error line, diff stat) instead of dumping
  thousands of characters back into the prompt.
- ``ContextWindow`` assembles the prompt in three layers: pinned (system/task/
  state digest), summarised older steps (level 2), raw recent steps (level 1).
  It compacts to a hard token budget, so 8k stays usable and 32k degrades
  gracefully instead of truncating the middle of the task.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .safety import sanitize_observation


def estimate_tokens(text: str, tokenizer=None) -> int:
    """Token count via the real tokenizer when available, else chars/3.5."""

    if not text:

        return 0

    if tokenizer is not None:

        return len(tokenizer.encode(text, add_special_tokens=False))

    return max(1, int(len(text) / 3.5))




@dataclass
class Todo:
    id: int
    text: str
    status: str = "pending"
    evidence: str = ""

    def to_dict(self) -> dict:

        return {
            "id": self.id,
            "text": self.text,
            "status": self.status,
            "evidence": self.evidence,
        }


class TodoList:
    """Ordered plan with explicit completion evidence.

    Evidence matters: a model that can mark a todo "done" without naming the
    test it ran or the file it changed is a model that will claim success.
    """

    def __init__(self, items: list[str] | None = None):

        self.todos: list[Todo] = [
            Todo(i + 1, text) for i, text in enumerate(items or [])
        ]

    def add(self, text: str) -> Todo:

        todo = Todo(len(self.todos) + 1, text)

        self.todos.append(todo)

        return todo

    def replace_plan(self, items: list[str]) -> None:

        done = {t.text for t in self.todos if t.status == "done"}

        self.todos = [
            Todo(i + 1, text, "done" if text in done else "pending")
            for i, text in enumerate(items)
        ]

    def update(self, todo_id: int, status: str, evidence: str = "") -> Todo:

        assert status in {"pending", "in_progress", "done", "blocked"}

        for todo in self.todos:

            if todo.id == todo_id:

                todo.status = status

                if evidence:

                    todo.evidence = evidence

                return todo

        raise KeyError(f"unknown todo id {todo_id}")

    def next_pending(self) -> Todo | None:

        for todo in self.todos:

            if todo.status in {"pending", "in_progress"}:

                return todo

        return None

    def mark_in_progress(self) -> Todo | None:

        todo = self.next_pending()

        if todo is not None and todo.status == "pending":

            todo.status = "in_progress"

        return todo

    @property
    def completion(self) -> float:

        if not self.todos:

            return 0.0

        return sum(1 for t in self.todos if t.status == "done") / len(self.todos)

    def blockers(self) -> list[Todo]:

        return [t for t in self.todos if t.status == "blocked"]

    def render(self) -> str:

        if not self.todos:

            return "(no plan yet)"

        mark = {
            "pending": "[ ]",
            "in_progress": "[~]",
            "done": "[x]",
            "blocked": "[!]",
        }

        lines = []

        for todo in self.todos:

            line = f"{mark.get(todo.status, '[ ]')} {todo.id}. {todo.text}"

            if todo.evidence:

                line += f"  (evidence: {todo.evidence})"

            lines.append(line)

        return "\n".join(lines)

    def to_dict(self) -> dict:

        return {
            "todos": [t.to_dict() for t in self.todos],
            "completion": self.completion,
        }




@dataclass
class FailureRecord:
    step: int
    tool: str | None
    signature: str
    error: str
    recovery: str = ""

    def to_dict(self) -> dict:

        return {
            "step": self.step,
            "tool": self.tool,
            "signature": self.signature,
            "error": self.error,
            "recovery": self.recovery,
        }


_ERROR_LINE = re.compile(
    r"(?:Error|Exception|FAILED|failed|error|Traceback)[^\n]*", re.IGNORECASE
)


def failure_signature(tool: str | None, arguments: dict | None, error: str) -> str:
    """Stable key for "this same failure again".

    Normalises volatile parts (line numbers, temp paths, durations, hex ids) so
    the same underlying mistake maps to the same signature across attempts.
    """

    err = error or ""

    err = re.sub(r"\b0x[0-9a-fA-F]+\b", "0xADDR", err)

    err = re.sub(r"\b\d+\.\d+(?:ms|s)\b", "TIME", err)

    err = re.sub(r"/tmp/[^\s:]+", "/tmp/PATH", err)

    err = re.sub(r"\bline \d+\b", "line N", err)

    err = re.sub(r"\b\d+\b", "N", err)

    key = _ERROR_LINE.search(err)

    summary = (key.group(0) if key else err)[:160].strip().lower()

    args_key = ""

    if isinstance(arguments, dict):

        for k in ("path", "test_path", "file_pattern", "function"):

            if k in arguments:

                args_key += f"{k}={arguments[k]};"

    return f"{tool or 'parse'}|{args_key}|{summary}"




@dataclass
class SessionState:
    """Serialisable state for one task session."""

    task: str

    max_steps: int = 20

    step: int = 0

    plan: list[str] = field(default_factory=list)

    todos: TodoList = field(default_factory=TodoList)

    facts: list[str] = field(default_factory=list)

    failures: list[FailureRecord] = field(default_factory=list)

    scratchpad: list[str] = field(default_factory=list)

    files_touched: list[str] = field(default_factory=list)
    working_set: WorkingSet = field(default_factory=lambda: WorkingSet())

    tests_passed: list[str] = field(default_factory=list)

    tests_failed: list[str] = field(default_factory=list)

    replans: int = 0

    def __post_init__(self):

        if self.plan and not self.todos.todos:

            self.todos = TodoList(self.plan)

    def plan_as(self, items: list[str]) -> None:

        self.plan = list(items)

        self.todos.replace_plan(items)

        self.replans += 1

    def note_fact(self, fact: str, dedupe: bool = True) -> None:
        """Pin a durable fact (constraint, API shape, repro command)."""

        fact = fact.strip()

        if not fact:

            return

        if dedupe and fact in self.facts:

            return

        self.facts.append(fact)

        if len(self.facts) > 40:

            self.facts = self.facts[-40:]

    def note_scratch(self, text: str) -> None:

        text = text.strip()

        if text:

            self.scratchpad.append(text)

            self.scratchpad = self.scratchpad[-20:]

    def record_failure(
        self, tool: str | None, arguments: dict | None, error: str, recovery: str = ""
    ) -> FailureRecord:

        record = FailureRecord(
            step=self.step,
            tool=tool,
            signature=failure_signature(tool, arguments, error),
            error=(error or "")[:400],
            recovery=recovery,
        )

        self.failures.append(record)

        return record

    def record_test_result(self, passed: list[str], failed: list[str]) -> None:

        for name in passed:

            if name not in self.tests_passed:

                self.tests_passed.append(name)

        for name in failed:

            if name not in self.tests_failed:

                self.tests_failed.append(name)

        self.tests_failed = [t for t in self.tests_failed if t not in self.tests_passed]

    def note_file(self, path: str) -> None:

        if path and path not in self.files_touched:

            self.files_touched.append(path)

    def consecutive_failures(self, window: int = 5) -> int:

        count = 0

        for record in reversed(self.failures[-window:]):

            if record.error:

                count += 1

            else:

                break

        return count

    def repeats(self, signature: str) -> int:

        return sum(1 for f in self.failures if f.signature == signature)

    def recent_failures(self, n: int = 3) -> list[FailureRecord]:

        return self.failures[-n:]

    def progress_score(self) -> float:
        """Composite progress signal used by the recovery policy."""

        return (
            len(self.tests_passed)
            + len(self.files_touched)
            + 2 * self.todos.completion
            + 0.5 * len(self.facts)
        )

    def render(self, max_facts: int = 12, max_failures: int = 3) -> str:
        """Compact state digest injected into every prompt."""

        parts = ["<|plan|>", self.todos.render()]

        if self.facts:

            parts.append("Pinned facts:")

            parts.extend(f"- {f}" for f in self.facts[-max_facts:])

        if self.failures:

            recent = self.recent_failures(max_failures)

            parts.append("Recent failures (do not repeat these):")

            for f in recent:

                parts.append(
                    f"- step {f.step} {f.tool}: {f.error.splitlines()[0][:160] if f.error else 'error'}"
                )

        if self.tests_failed:

            parts.append("Failing tests: " + ", ".join(self.tests_failed[:8]))

        elif self.tests_passed:

            parts.append(f"Passing tests: {len(self.tests_passed)}")

        if self.files_touched:

            parts.append("Files touched: " + ", ".join(self.files_touched[-8:]))

        parts.append("<|endoftext|>")

        return "\n".join(parts)

    def to_dict(self) -> dict:

        return {
            "task": self.task,
            "max_steps": self.max_steps,
            "step": self.step,
            "plan": self.plan,
            "todos": self.todos.to_dict(),
            "facts": self.facts,
            "failures": [f.to_dict() for f in self.failures],
            "scratchpad": self.scratchpad,
            "files_touched": self.files_touched,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "replans": self.replans,
        }

    def save(self, path: str) -> None:
        """Atomic JSON write (a crashed agent must not truncate its own state)."""

        p = Path(path)

        p.parent.mkdir(parents=True, exist_ok=True)

        tmp = p.with_suffix(p.suffix + ".tmp")

        with open(tmp, "w") as f:

            json.dump(self.to_dict(), f, indent=2)

        tmp.replace(p)

    @classmethod
    def load(cls, path: str) -> "SessionState":
        """Resume a session; todos and failures are rebuilt, not re-derived."""

        with open(path) as f:

            data = json.load(f)

        state = cls(task=data["task"], max_steps=data.get("max_steps", 20))

        state.step = data.get("step", 0)

        state.plan = list(data.get("plan", []))

        state.todos = TodoList()

        for row in data.get("todos", {}).get("todos", []):

            state.todos.todos.append(
                Todo(
                    row["id"],
                    row["text"],
                    row.get("status", "pending"),
                    row.get("evidence", ""),
                )
            )

        state.facts = list(data.get("facts", []))

        state.failures = [
            FailureRecord(**row) for row in data.get("failures", [])
        ]

        state.scratchpad = list(data.get("scratchpad", []))

        state.files_touched = list(data.get("files_touched", []))

        state.tests_passed = list(data.get("tests_passed", []))

        state.tests_failed = list(data.get("tests_failed", []))

        state.replans = data.get("replans", 0)

        return state




class ObservationCompactor:
    """Turn raw tool output into a small, high-signal observation.

    Each summariser keeps the *decision-relevant* part: which test failed and
    why, which file changed, what the error was. Raw output is deliberately
    dropped — it is re-readable with a tool call if the model needs it.
    """

    def __init__(self, max_lines: int = 12, max_chars: int = 1600):

        self.max_lines = max_lines

        self.max_chars = max_chars

    @staticmethod
    def _clip(text: str, max_lines: int) -> str:

        lines = str(text).splitlines()

        if len(lines) <= max_lines:

            return "\n".join(lines)

        head = lines[: max_lines // 2]

        tail = lines[-(max_lines - len(head)) :]

        return "\n".join(head + [f"... ({len(lines) - max_lines} lines omitted) ..."] + tail)

    def summarise_tests(self, result: dict) -> str:
        """pytest output -> counts + failing test ids + first assertion error."""

        out = result.get("stdout", "") or result.get("output", "")

        failed = sorted(set(re.findall(r"FAILED\s+([^\s]+)", out)))

        passed = len(re.findall(r"\bPASSED\b", out)) or None

        summary_line = ""

        for line in reversed(out.splitlines()):

            if re.search(r"\d+ (?:passed|failed)", line):

                summary_line = line.strip()

                break

        root_cause = ""

        for i, line in enumerate(out.splitlines()):

            if line.startswith("E   ") or "AssertionError" in line or "Error" in line:

                root_cause = "\n".join(out.splitlines()[i : i + 4])

                break

        parts = [f"run_tests rc={result.get('returncode', '?')}"]

        if summary_line:

            parts.append(summary_line)

        if passed:

            parts.append(f"passed={passed}")

        if failed:

            parts.append("failed: " + ", ".join(failed[:8]))

        if root_cause:

            parts.append("first error:\n" + self._clip(root_cause, 6))

        if not failed and not root_cause:

            parts.append(self._clip(out, 6))

        return "\n".join(parts)

    def summarise(self, tool: str, arguments: dict, result: Any) -> str:
        """Dispatch to a per-tool summariser; always returns bounded text."""

        if not isinstance(result, dict):

            return self._clip(str(result), self.max_lines)[: self.max_chars]

        if result.get("error") and not result.get("success", True):

            return f"{tool} error: {str(result['error'])[:400]}"

        if tool == "run_tests":

            text = self.summarise_tests(result)

        elif tool == "read_file":

            content = result.get("content", "")

            lines = content.splitlines()

            outline = [
                line.strip()
                for line in lines
                if re.match(r"^\s*(?:def |class |async def |func |fn |export )", line)
            ][:10]

            text = f"read_file {arguments.get('path')} ({len(lines)} lines, {len(content)} chars)"

            if outline:

                text += "\n" + "\n".join(outline)

        elif tool == "search_code":

            matches = result.get("matches") or result.get("results") or []

            paths = []

            for m in matches if isinstance(matches, list) else []:

                path = m.get("path") if isinstance(m, dict) else str(m)

                if path and path not in paths:

                    paths.append(path)

            text = f"search_code {arguments.get('pattern')!r}: {len(paths)} files\n" + "\n".join(
                paths[:10]
            )

        elif tool in {"git_diff", "apply_patch", "edit_function", "write_file"}:

            text = f"{tool}: " + self._clip(
                json.dumps(
                    {
                        k: result.get(k)
                        for k in ("files", "path", "success", "insertions", "deletions")
                        if k in result
                    }
                ),
                4,
            )

            diff = result.get("diff") or ""

            if diff:

                stats = [
                    line
                    for line in diff.splitlines()
                    if line.startswith(("+++", "---", "@@"))
                ][:6]

                text += "\n" + "\n".join(stats)

        elif tool == "execute_code":

            text = (
                f"execute_code rc={result.get('returncode', '?')}\n"
                + self._clip(result.get("stdout", ""), 6)
            )

            if result.get("stderr"):

                text += "\nstderr: " + self._clip(result["stderr"], 4)

        elif tool == "list_files":

            files = result.get("files") or []

            text = f"list_files: {len(files)} entries\n" + ", ".join(
                str(f) for f in files[:15]
            )

        else:

            text = f"{tool}: " + self._clip(json.dumps(result, default=str), self.max_lines)

        return sanitize_observation(text, max_chars=self.max_chars)




@dataclass
class StepRecord:
    step: int
    role: str
    content: str
    tool: str | None = None
    summary: str | None = None
    failed: bool = False

    def to_dict(self) -> dict:

        return {
            "step": self.step,
            "role": self.role,
            "tool": self.tool,
            "failed": self.failed,
            "summary": self.summary,
            "content": self.content,
        }


class ContextWindow:
    """Budgeted prompt assembly with two-level compaction.

    Level 1: recent steps, verbatim but observation-compacted.
    Level 2: older steps collapse into one line each ("step 3 run_tests failed:
    test_auth_expiry"), which keeps the causal thread at ~2% of the tokens.
    Pinned layer (system + task + state digest) is never dropped.
    """

    def __init__(
        self,
        system_prompt: str,
        task: str,
        state: SessionState,
        tokenizer=None,
        budget_tokens: int = 6000,
        keep_recent_steps: int = 6,
        max_step_tokens: int = 500,
    ):

        self.system_prompt = system_prompt

        self.task = task

        self.state = state

        self.tokenizer = tokenizer

        self.budget_tokens = budget_tokens

        self.keep_recent_steps = keep_recent_steps

        self.max_step_tokens = max_step_tokens

        self.steps: list[StepRecord] = []

        self.compactor = ObservationCompactor()

    def add_user(self, content: str) -> StepRecord:

        record = StepRecord(self.state.step, "user", content)

        self.steps.append(record)

        return record

    def add_assistant(self, content: str) -> StepRecord:

        record = StepRecord(self.state.step, "assistant", content[: self.max_step_tokens * 4])

        self.steps.append(record)

        return record

    def add_observation(
        self, tool: str, arguments: dict, result: Any
    ) -> StepRecord:
        """Record a tool observation, compacted, and fold it into state."""

        summary = self.compactor.summarise(tool, arguments, result)

        failed = bool(isinstance(result, dict) and result.get("error"))

        if failed:

            self.state.record_failure(tool, arguments, str(result.get("error")))

        if isinstance(result, dict):

            if result.get("tests_passed") or result.get("tests_failed"):

                self.state.record_test_result(
                    list(result.get("tests_passed") or []),
                    list(result.get("tests_failed") or []),
                )

            path = result.get("path") or (arguments or {}).get("path")

            if path and tool in {"write_file", "apply_patch", "edit_function"}:

                self.state.note_file(str(path))

        record = StepRecord(
            self.state.step, "tool", summary, tool=tool, summary=summary, failed=failed
        )

        self.steps.append(record)

        return record

    def _count(self, text: str) -> int:

        return estimate_tokens(text, self.tokenizer)

    def _render_step(self, record: StepRecord) -> str:
        """Render one step at level 1 (recent) or level 2 (compacted)."""

        if record.role == "tool":

            tag = "<|tool_error|>" if record.failed else "<|observation|>"

            return f"{tag} {record.tool}\n{record.content}\n<|endoftext|>"

        return f"<|{record.role}|>\n{record.content}\n<|endoftext|>"

    def _digest_step(self, record: StepRecord) -> str:
        """One-line level-2 summary of an old step."""

        if record.role == "tool":

            status = "FAILED" if record.failed else "ok"

            first = (record.content or "").splitlines()[0][:120] if record.content else ""

            return f"- step {record.step} {record.tool} [{status}] {first}"

        text = (record.content or "").replace("\n", " ")[:120]

        return f"- step {record.step} {record.role}: {text}"

    def assemble(self, budget_tokens: int | None = None) -> tuple[str, dict]:
        """Return (prompt, stats). Deterministic given the same state/steps."""

        budget = budget_tokens or self.budget_tokens

        pinned = f"<|system|>\n{self.system_prompt}\n<|endoftext|>\n"
        pinned += f"<|user|>\n{self.task}\n<|endoftext|>\n"
        digest = self.state.render()
        spent = self._count(pinned) + self._count(digest)

        recent: list[str] = []

        for record in reversed(self.steps[-self.keep_recent_steps :]):

            rendered = self._render_step(record)

            cost = self._count(rendered)

            if spent + cost > budget:

                break

            recent.append(rendered)

            spent += cost

        recent.reverse()

        older = self.steps[: max(0, len(self.steps) - len(recent))]

        summarized = [self._digest_step(r) for r in older]

        allowed = int(budget * 0.25)

        kept: list[str] = []

        used = 0

        for line in reversed(summarized):

            cost = self._count(line) + 1

            if used + cost > allowed:

                break

            kept.append(line)

            used += cost

        kept.reverse()

        if len(kept) < len(summarized):

            kept.insert(0, f"- ({len(summarized) - len(kept)} earlier steps omitted)")

        parts = [pinned, "<|plan|>", digest, "<|endoftext|>"]

        if kept:

            parts.append("Earlier steps (compacted):")

            parts.extend(kept)

        parts.extend(recent)

        text = "\n".join(parts)

        stats = {
            "prompt_tokens": self._count(text),
            "budget_tokens": budget,
            "steps_total": len(self.steps),
            "steps_recent": len(recent),
            "steps_compacted": len(kept),
            "digest_tokens": self._count(digest),
            "truncated": bool(older) and len(kept) < len(summarized),
        }

        return text, stats

    def to_dict(self) -> dict:

        return {
            "steps": [s.to_dict() for s in self.steps],
            "state": self.state.to_dict(),
        }

class WorkingSet:
    """A pin-protected set of files that always survive compaction.

    The working set contains files the agent is currently working on,
    recently modified files, or files the user has explicitly pinned.
    These files are preserved across observation compaction so the model
    never loses track of the files it is actively modifying.
    """

    def __init__(self, max_files: int = 20):
        self.max_files = max_files
        self.pinned: set[str] = set()
        self.recent: collections.deque = collections.deque(maxlen=max_files // 2)
        self._file_access_counts: dict[str, int] = defaultdict(int)

    def pin(self, path: str) -> None:
        """Pin a file so it always survives compaction."""
        self.pinned.add(path)
        if path not in self._file_access_counts:
            self._file_access_counts[path] = 0

    def access(self, path: str) -> None:
        """Record that a file was accessed."""
        self._file_access_counts[path] += 1
        if path not in self.pinned:
            if path in self.recent:
                self.recent.remove(path)
            self.recent.append(path)
            # Evict if over max
            if len(self.recent) > self.max_files // 2:
                # Remove least recently used non-pinned file
                for p in self.recent:
                    if p not in self.pinned:
                        self.recent.remove(p)
                        break

    def get_surviving_files(self) -> list[str]:
        """Get the list of files that should survive compaction."""
        surviving = list(self.pinned)
        surviving.extend(list(self.recent))
        return surviving[:self.max_files]

    def to_dict(self) -> dict:
        return {
            "pinned": list(self.pinned),
            "recent": list(self.recent),
            "access_counts": dict(self._file_access_counts),
        }

    @classmethod
    def from_dict(cls, data: dict, max_files: int = 20) -> "WorkingSet":
        ws = cls(max_files=max_files)
        ws.pinned = set(data.get("pinned", []))
        ws.recent = collections.deque(data.get("recent", []), maxlen=max_files // 2)
        ws._file_access_counts = defaultdict(int, data.get("access_counts", {}))
        return ws
