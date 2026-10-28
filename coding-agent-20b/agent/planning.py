"""Dependency-aware planning, milestone checkpoints and assumption tracking.

The recovery ladder in ``agent/recovery.py`` bounds damage *after* a failure.
This module attacks the other half of the long-horizon problem: preventing
avoidable failures and keeping progress that has already been verified.

- ``PlanGraph`` is a DAG of steps with explicit dependencies. Execution walks
  ``ready()`` (all dependencies done); when a step fails, only its
  *downstream subgraph* is replanned — a failed test fix should not throw
  away three verified features.
- ``MilestoneCheckpointer`` snapshots the workspace whenever a step is
  verified (test/symbol evidence). Backtracking therefore returns to the last
  *verified* state, not to zero; and "progress" is a measured quantity, not a
  feeling.
- ``AssumptionLedger`` makes hidden assumptions explicit and tracks whether
  each was confirmed or refuted by evidence. Most long-session disasters are
  an unverified assumption that silently stopped being true five steps ago.

All three are pure state machines over JSON-serialisable data, so they are
unit-testable without a model and resumable across sessions.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

_DEP_MARKER = re.compile(r"^\[after\s*([0-9,\s]*)\]\s*")


@dataclass
class PlanStep:
    id: int
    description: str
    depends_on: list[int] = field(default_factory=list)
    status: str = "pending"
    evidence: str = ""
    attempts: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "depends_on": list(self.depends_on),
            "status": self.status,
            "evidence": self.evidence,
            "attempts": self.attempts,
        }


class PlanGraph:
    """Ordered DAG of steps; sequential when no dependencies are given.

    Steps may be plain strings (implicit chain) or carry an explicit
    dependency marker: ``"[after 1,3] run the auth tests"``.
    """

    def __init__(self, steps: list[PlanStep] | None = None):
        self.steps: list[PlanStep] = steps or []

    @classmethod
    def parse(cls, items: list) -> "PlanGraph":
        steps: list[PlanStep] = []

        for i, item in enumerate(items, start=1):

            if isinstance(item, dict):

                description = str(item.get("description", "")).strip()

                deps = [int(d) for d in (item.get("depends_on") or []) if int(d) < i]

                step = PlanStep(i, description, deps)

                step.status = str(item.get("status", "pending"))

                step.evidence = str(item.get("evidence", ""))

                step.attempts = int(item.get("attempts", 0))

            else:

                text = str(item).strip()

                deps: list[int] = []

                m = _DEP_MARKER.match(text)

                if m:

                    deps = [
                        int(n) for n in m.group(1).split(",") if n.strip().isdigit()
                    ]

                    deps = [d for d in deps if 0 < d < i]

                    text = text[m.end():].strip()


                elif i > 1 and steps:

                    deps = [steps[-1].id]

                step = PlanStep(i, text, deps)

            steps.append(step)

        return cls(steps)

    def get(self, step_id: int) -> PlanStep | None:
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def ready(self) -> list[PlanStep]:
        """Pending steps whose dependencies are all done."""

        done = {s.id for s in self.steps if s.status == "done"}

        out = []

        for step in self.steps:

            if step.status not in {"pending", "in_progress"}:

                continue

            if all(d in done for d in step.depends_on):

                out.append(step)

        return out

    def mark(self, step_id: int, status: str, evidence: str = "") -> PlanStep:
        step = self.get(step_id)

        if step is None:

            raise KeyError(f"unknown step {step_id}")

        step.status = status

        if evidence:

            step.evidence = evidence

        if status in {"failed", "in_progress"}:

            step.attempts += 1

        return step

    def downstream(self, step_id: int) -> set[int]:
        """Transitive dependents of a step (the local-replan subgraph)."""

        dependents: dict[int, set[int]] = {s.id: set() for s in self.steps}

        for step in self.steps:

            for dep in step.depends_on:

                if dep in dependents:

                    dependents[dep].add(step.id)

        seen: set[int] = set()

        frontier = [step_id]

        while frontier:

            current = frontier.pop()

            for nxt in dependents.get(current, ()):

                if nxt not in seen:

                    seen.add(nxt)

                    frontier.append(nxt)

        return seen

    def reset_subtree(self, step_id: int) -> set[int]:
        """Return failed step + downstream steps to pending (local replan)."""

        affected = {step_id} | self.downstream(step_id)

        for step in self.steps:

            if step.id in affected and step.status not in {"pending", "in_progress"}:

                step.status = "pending"

                step.evidence = ""

        return affected

    @property
    def completion(self) -> float:
        if not self.steps:
            return 0.0
        done = sum(1 for s in self.steps if s.status == "done")
        return done / len(self.steps)

    def blocked(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == "blocked"]

    def render(self) -> str:
        if not self.steps:
            return "(no plan)"
        mark = {
            "pending": "[ ]",
            "in_progress": "[~]",
            "done": "[x]",
            "blocked": "[!]",
            "failed": "[✗]",
        }
        lines = []
        for step in self.steps:
            deps = f" (after {','.join(str(d) for d in step.depends_on)})" if step.depends_on else ""
            line = f"{mark.get(step.status, '[ ]')} {step.id}. {step.description}{deps}"
            if step.evidence:
                line += f"  (evidence: {step.evidence})"
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"steps": [s.to_dict() for s in self.steps], "completion": self.completion}

    @classmethod
    def from_dict(cls, data: dict) -> "PlanGraph":
        return cls.parse(data.get("steps", []))


@dataclass
class Milestone:
    label: str
    step: int
    timestamp: float
    files: list[str]
    tests_passed: list[str]
    kind: str = "git"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "step": self.step,
            "timestamp": self.timestamp,
            "files": list(self.files),
            "tests_passed": list(self.tests_passed),
            "kind": self.kind,
        }


class MilestoneCheckpointer:
    """Snapshot verified progress; roll back to the last verified milestone.

    Two backends:

    - ``git`` (preferred): commit the workspace at each milestone; rollback is
      ``git checkout`` of files changed since then (nothing is destroyed —
      the broken state stays on a stash/branch).
    - ``snapshot``: copy the files listed in ``SessionState.files_touched``
      into ``.agent/checkpoints/<n>/`` (works without git; only tracks files
      the agent itself recorded touching).
    """

    def __init__(self, workspace: str, checkpoint_dir: str | None = None):
        self.workspace = Path(workspace)
        self.checkpoint_dir = Path(checkpoint_dir or self.workspace / ".agent" / "checkpoints")
        self.milestones: list[Milestone] = []

    def _files_since(self, milestone: Milestone | None) -> list[str]:
        if milestone is None:
            return []
        return list(milestone.files)

    def checkpoint(
        self,
        label: str,
        step: int,
        files_touched: list[str],
        tests_passed: list[str],
        executor=None,
    ) -> Milestone:
        kind = "snapshot"

        if executor is not None:

            try:

                result = executor.execute_tool(
                    "git_commit", {"message": f"milestone: {label}", "add_all": True}
                )

                kind = "git" if isinstance(result, dict) and not result.get("error") else "snapshot"

            except Exception:

                kind = "snapshot"

        milestone = Milestone(
            label=label,
            step=step,
            timestamp=time.time(),
            files=list(files_touched),
            tests_passed=list(tests_passed),
            kind=kind,
        )

        if kind == "snapshot":

            target = self.checkpoint_dir / f"{len(self.milestones):03d}-{_slug(label)}"

            for rel in files_touched:

                src = self.workspace / rel

                if src.is_file():

                    dst = target / rel

                    dst.parent.mkdir(parents=True, exist_ok=True)

                    shutil.copy2(src, dst)

        self.milestones.append(milestone)

        return milestone

    def last_verified(self) -> Milestone | None:
        return self.milestones[-1] if self.milestones else None

    def rollback_files(self, files_touched_now: list[str]) -> tuple[list[str], Milestone | None]:
        """Files to revert to return to the last verified milestone.

        Returns (files, milestone). With no milestone the caller decides
        (typically full stash).
        """

        milestone = self.last_verified()

        if milestone is None:

            return list(files_touched_now), None

        files = [f for f in files_touched_now if f not in set(milestone.files)]

        return files, milestone

    def restore_snapshot(self, milestone: Milestone, workspace: str | None = None) -> list[str]:
        """Copy snapshot files back (snapshot backend only)."""

        restored: list[str] = []

        root = Path(workspace or self.workspace)

        src_dir = self.checkpoint_dir / f"{self.milestones.index(milestone):03d}-{_slug(milestone.label)}"

        if not src_dir.is_dir():

            return restored

        for path in sorted(src_dir.rglob("*")):

            if path.is_file():

                rel = path.relative_to(src_dir)

                dst = root / rel

                dst.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(path, dst)

                restored.append(str(rel))

        return restored

    def to_dict(self) -> dict:
        return {"milestones": [m.to_dict() for m in self.milestones]}

    @classmethod
    def from_dict(cls, data: dict, workspace: str) -> "MilestoneCheckpointer":
        cp = cls(workspace)
        for row in data.get("milestones", []):
            cp.milestones.append(
                Milestone(
                    label=row["label"],
                    step=row.get("step", 0),
                    timestamp=row.get("timestamp", 0.0),
                    files=list(row.get("files", [])),
                    tests_passed=list(row.get("tests_passed", [])),
                    kind=row.get("kind", "snapshot"),
                )
            )
        return cp


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "milestone"


ASSUMPTION_STATUSES = ("unverified", "confirmed", "refuted")


@dataclass
class Assumption:
    id: int
    text: str
    status: str = "unverified"
    how: str = ""
    step: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "status": self.status,
            "how": self.how,
            "step": self.step,
        }


class AssumptionLedger:
    """Explicit assumptions with verification state.

    The prompt contract: before acting on an assumption, record it; when
    evidence confirms or refutes it, record that too. ``render()`` injects
    the unverified ones into every prompt so they cannot be compacted away.
    """

    def __init__(self):
        self.assumptions: list[Assumption] = []

    def add(self, text: str, step: int = 0) -> Assumption:
        text = text.strip()
        for existing in self.assumptions:
            if existing.text.lower() == text.lower():
                return existing
        assumption = Assumption(len(self.assumptions) + 1, text, step=step)
        self.assumptions.append(assumption)
        return assumption

    def record(self, assumption_id: int, status: str, how: str = "") -> Assumption:
        if status not in ASSUMPTION_STATUSES:
            raise ValueError(f"status must be one of {ASSUMPTION_STATUSES}")
        for assumption in self.assumptions:
            if assumption.id == assumption_id:
                assumption.status = status
                if how:
                    assumption.how = how
                return assumption
        raise KeyError(f"unknown assumption {assumption_id}")

    def unverified(self) -> list[Assumption]:
        return [a for a in self.assumptions if a.status == "unverified"]

    def refuted(self) -> list[Assumption]:
        return [a for a in self.assumptions if a.status == "refuted"]

    def render(self, max_items: int = 6) -> str:
        if not self.assumptions:
            return ""
        lines = []
        pending = self.unverified()
        if pending:
            lines.append("Unverified assumptions (verify or refute before relying on them):")
            for a in pending[:max_items]:
                lines.append(f"- A{a.id}: {a.text}")
        refuted = self.refuted()
        if refuted:
            lines.append("Refuted assumptions (their conclusions are wrong — replan if used):")
            for a in refuted[:max_items]:
                lines.append(f"- A{a.id}: {a.text} (refuted: {a.how or 'evidence'})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"assumptions": [a.to_dict() for a in self.assumptions]}

    @classmethod
    def from_dict(cls, data: dict) -> "AssumptionLedger":
        ledger = cls()
        for row in data.get("assumptions", []):
            ledger.assumptions.append(
                Assumption(
                    id=int(row["id"]),
                    text=row["text"],
                    status=row.get("status", "unverified"),
                    how=row.get("how", ""),
                    step=row.get("step", 0),
                )
            )
        return ledger
