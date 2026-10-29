"""Structured project plans that a human approves *before* files are written.

``PlanGraph`` inside an agent run is about dependencies between steps. This
module is about consent: for anything bigger than a one-file edit, the agent
should say what it is going to build (files, steps, risks, open questions) and
wait for a yes.

* :class:`PlanDrafter` produces a plan either from a model (``generate_fn``
  returning JSON) or deterministically from the scaffold registry - so planning
  works without a model and is testable.
* :class:`PlanStore` persists plans as JSON under ``.agent/plans`` with an
  explicit status machine (``pending -> approved | rejected | revised``).
* ``render_markdown`` is what the CLI and the UI show; ``approve``/``reject``
  are one call each, and the plan id is short enough to type.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PLAN_STATUSES = {"pending", "approved", "rejected", "revised"}


@dataclass
class PlanItem:
    action: str
    files: list[str] = field(default_factory=list)
    why: str = ""
    risk: str = "low"

    def to_dict(self) -> dict:
        return {"action": self.action, "files": list(self.files), "why": self.why, "risk": self.risk}

    @classmethod
    def from_dict(cls, data: dict) -> "PlanItem":
        return cls(
            action=str(data.get("action", "")),
            files=[str(f) for f in (data.get("files") or [])],
            why=str(data.get("why", "")),
            risk=str(data.get("risk", "low")),
        )


@dataclass
class ProjectPlan:
    goal: str
    items: list[PlanItem] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    status: str = "pending"
    id: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    template_id: str | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _plan_id(self.goal)
        if not self.created_at:
            self.created_at = time.time()
        self.updated_at = self.updated_at or self.created_at

    @property
    def files(self) -> list[str]:
        seen: list[str] = []
        for item in self.items:
            for path in item.files:
                if path not in seen:
                    seen.append(path)
        return seen

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_at)),
            "template_id": self.template_id,
            "items": [item.to_dict() for item in self.items],
            "risks": list(self.risks),
            "questions": list(self.questions),
            "decisions": list(self.decisions),
            "notes": list(self.notes),
            "file_count": len(self.files),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectPlan":
        return cls(
            goal=data.get("goal", ""),
            items=[PlanItem.from_dict(item) for item in data.get("items") or []],
            risks=list(data.get("risks") or []),
            questions=list(data.get("questions") or []),
            decisions=list(data.get("decisions") or []),
            status=data.get("status", "pending"),
            id=data.get("id", ""),
            created_at=float(data.get("created_at", 0)),
            updated_at=float(data.get("updated_at", 0)),
            template_id=data.get("template_id"),
            notes=list(data.get("notes") or []),
        )

    def render_markdown(self) -> str:
        badge = {"pending": "awaiting approval", "approved": "approved", "rejected": "rejected", "revised": "revised"}.get(
            self.status, self.status
        )
        lines = [f"# Plan {self.id} - {badge}", "", f"**Goal:** {self.goal}", ""]
        if self.template_id:
            lines.append(f"**Starter:** `{self.template_id}`")
            lines.append("")
        lines.append("## Steps")
        for index, item in enumerate(self.items, 1):
            lines.append(f"{index}. **{item.action}**" + (f" - _{item.why}_" if item.why else ""))
            for path in item.files:
                lines.append(f"   - `{path}`")
            if item.risk and item.risk != "low":
                lines.append(f"   - risk: {item.risk}")
        if self.risks:
            lines += ["", "## Risks"] + [f"- {risk}" for risk in self.risks]
        if self.questions:
            lines += ["", "## Open questions"] + [f"- {question}" for question in self.questions]
        if self.decisions:
            lines += ["", "## Decisions"] + [f"- {decision}" for decision in self.decisions]
        if self.files:
            lines += ["", f"_{len(self.files)} files touched_"]
        return "\n".join(lines)


def _plan_id(goal: str) -> str:
    digest = re.sub(r"[^a-z0-9]+", "-", (goal or "plan").lower()).strip("-")[:24]
    return f"{digest or 'plan'}-{int(time.time()) % 100000:05d}"


class PlanStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.dir = self.root / ".agent" / "plans"

    def save(self, plan: ProjectPlan) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        plan.updated_at = time.time()
        target = self.dir / f"{plan.id}.json"
        target.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    def list(self) -> list[ProjectPlan]:
        if not self.dir.exists():
            return []
        plans = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                plans.append(ProjectPlan.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(plans, key=lambda plan: plan.created_at)

    def get(self, plan_id: str) -> ProjectPlan:
        matches = [plan for plan in self.list() if plan.id == plan_id or plan.id.startswith(plan_id)]
        if not matches:
            raise KeyError(f"no plan matching {plan_id!r}")
        return matches[-1]

    def pending(self) -> list[ProjectPlan]:
        return [plan for plan in self.list() if plan.status == "pending"]

    def set_status(self, plan_id: str, status: str, note: str = "") -> ProjectPlan:
        if status not in PLAN_STATUSES:
            raise ValueError(f"status must be one of {sorted(PLAN_STATUSES)}")
        plan = self.get(plan_id)
        plan.status = status
        plan.updated_at = time.time()
        if note:
            plan.notes.append(note)
        self.save(plan)
        return plan

    def approve(self, plan_id: str, note: str = "") -> ProjectPlan:
        return self.set_status(plan_id, "approved", note or "approved")

    def reject(self, plan_id: str, note: str = "") -> ProjectPlan:
        return self.set_status(plan_id, "rejected", note or "rejected")

    def revise(self, plan_id: str, changes: list[str], note: str = "") -> ProjectPlan:
        plan = self.get(plan_id)
        plan.items.extend(PlanItem(action=change) for change in changes)
        plan.status = "revised"
        if note:
            plan.notes.append(note)
        self.save(plan)
        return plan


class PlanDrafter:
    """Draft a plan deterministically, or with a model when one is available."""

    def __init__(self, root: str | Path, generate_fn: Callable[[str], str] | None = None):
        self.root = Path(root)
        self.generate_fn = generate_fn

    def _template_plan(self, goal: str) -> ProjectPlan | None:
        try:
            from scaffold.generator import plan_for
        except ImportError:
            return None
        try:
            draft = plan_for(goal)
        except Exception:  # noqa: BLE001 - planning must not crash on odd input
            return None
        items = [
            PlanItem(
                action=f"Scaffold {draft['template']['name']}",
                files=draft["files"][:40],
                why="; ".join(draft["reasons"][:2]) or draft["template"]["tagline"],
                risk="low",
            ),
            PlanItem(
                action="Write agentdeploy.toml and run the generated test suite",
                files=["agentdeploy.toml", "tests/"],
                why="the project should be deployable and green before you touch it",
            ),
            PlanItem(
                action="Review with the critic and fix blocking findings",
                files=[],
                why="static review catches secrets, swallowed errors and missing tests",
            ),
        ]
        plan = ProjectPlan(
            goal=goal,
            items=items,
            questions=list(draft["questions"]),
            template_id=draft["template"]["id"],
            decisions=[f"starter: {draft['template']['id']}"],
            risks=["Generated starters are a starting point, not a finished product."],
        )
        return plan

    def _generic_plan(self, goal: str) -> ProjectPlan:
        return ProjectPlan(
            goal=goal,
            items=[
                PlanItem(action="Map the repository", files=[], why="read structure, entry points and key modules first"),
                PlanItem(action="Write or update failing tests", files=["tests/"], why="tests define done"),
                PlanItem(action="Implement the change", files=[], why="smallest change that satisfies the tests"),
                PlanItem(action="Run tests and the critic until green", files=[], why="no 'done' without evidence"),
                PlanItem(action="Summarise the diff and how to verify it", files=[], why="the user needs a review path"),
            ],
            risks=["The affected modules are unknown until the map is built."],
        )

    def _model_plan(self, goal: str) -> ProjectPlan | None:
        if self.generate_fn is None:
            return None
        prompt = (
            "Draft a project plan as JSON with keys: items (list of {action, files, why, risk}), "
            "risks, questions. No prose outside the JSON.\n\nTask: " + goal
        )
        try:
            raw = self.generate_fn(prompt)
        except Exception:  # noqa: BLE001
            return None
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        items = [PlanItem.from_dict(item) for item in data.get("items") or [] if isinstance(item, dict)]
        if not items:
            return None
        return ProjectPlan(
            goal=goal,
            items=items,
            risks=[str(r) for r in data.get("risks") or []],
            questions=[str(q) for q in data.get("questions") or []],
            notes=["drafted by the model"],
        )

    def draft(self, goal: str, *, prefer: str = "auto") -> ProjectPlan:
        if prefer in {"model", "auto"}:
            plan = self._model_plan(goal)
            if plan is not None:
                return plan
            if prefer == "model":
                raise RuntimeError("no model plan available (generate_fn missing or unusable)")
        created_empty = not (self.root.exists() and any(self.root.iterdir()))
        if created_empty or prefer == "scaffold":
            plan = self._template_plan(goal)
            if plan is not None:
                return plan
        return self._generic_plan(goal)

    def draft_and_store(self, goal: str, **kwargs) -> tuple[ProjectPlan, Path]:
        plan = self.draft(goal, **kwargs)
        return plan, PlanStore(self.root).save(plan)


def draft_plan(root: str | Path, goal: str, generate_fn: Callable[[str], str] | None = None) -> ProjectPlan:
    return PlanDrafter(root, generate_fn).draft(goal)

HIGH_RISK_KEYWORDS = {"deploy", "delete", "remove", "production", "database", "migrate", "drop", "secrets", "credentials", "overwrite"}
HIGH_RISK_FILE_PATTERNS = {".env", "production", "migrate", "alembic", "Dockerfile"}

def risk_score(item: PlanItem) -> str:
    """Heuristic risk: low/medium/high based on action/files."""
    text = (item.action + " " + " ".join(item.files)).lower()
    if any(kw in text for kw in HIGH_RISK_KEYWORDS):
        return "high"
    if len(item.files) > 10:
        return "high"
    if any(pat in f for f in item.files for pat in HIGH_RISK_FILE_PATTERNS):
        return "high"
    if len(item.files) > 3:
        return "medium"
    return item.risk or "low"

def needs_approval(plan: ProjectPlan) -> bool:
    """True if any item is high risk; low-risk can proceed with logging."""
    for item in plan.items:
        if risk_score(item) == "high":
            return True
    return False

def auto_approve_low_risk(plan: ProjectPlan) -> bool:
    """Auto-approve if no high-risk items; log instead of blocking."""
    if not needs_approval(plan):
        plan.status = "approved"
        plan.updated_at = __import__("time").time()
        plan.notes.append("auto-approved (low-risk)")
        return True
    return False
