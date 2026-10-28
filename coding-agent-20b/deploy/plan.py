"""Deploy plan and result value objects (shared by engine, providers, CLI, UI)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeployStep:
    """One unit of deploy work, rendered in plans and step outcomes."""

    key: str
    title: str
    detail: str = ""
    kind: str = "step"  # preflight | build | push | release | health | report

    def to_dict(self) -> dict:
        return {"key": self.key, "title": self.title, "detail": self.detail, "kind": self.kind}


@dataclass
class StepOutcome:
    step: DeployStep
    status: str  # ok | failed | skipped
    detail: str = ""
    elapsed: float = 0.0

    @property
    def icon(self) -> str:
        return {
            "ok": "ok",
            "failed": "FAILED",
            "skipped": "skipped",
            "unknown": "?",
        }.get(self.status, self.status)

    def to_dict(self) -> dict:
        return {
            "key": self.step.key,
            "title": self.step.title,
            "kind": self.step.kind,
            "status": self.status,
            "detail": self.detail,
            "elapsed": round(self.elapsed, 2),
        }


@dataclass
class DeployPlan:
    provider: str
    provider_label: str
    steps: list[DeployStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    health_url: str | None = None
    image: str | None = None
    dry_run: bool = True

    def render(self) -> str:
        lines = [
            f"Provider: {self.provider_label} ({self.provider})",
            f"Mode:     {'dry run - nothing will be created' if self.dry_run else 'apply'}",
        ]
        if self.image:
            lines.append(f"Image:    {self.image}")
        if self.health_url:
            lines.append(f"Health:   {self.health_url}")
        lines.append("")
        for index, step in enumerate(self.steps, 1):
            lines.append(f"  {index}. [{step.kind}] {step.title}")
            if step.detail:
                lines.append(f"     {step.detail}")
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "provider_label": self.provider_label,
            "steps": [step.to_dict() for step in self.steps],
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "health_url": self.health_url,
            "image": self.image,
            "dry_run": self.dry_run,
        }


@dataclass
class ProviderResult:
    """What a provider achieved."""

    ok: bool
    url: str | None = None
    health_url: str | None = None
    public: bool = False
    detail: str = ""
    error: str = ""
    logs: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    steps: list[DeployStep] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeployResult:
    ok: bool
    provider: str
    url: str | None = None
    health_url: str | None = None
    public: bool = False
    dry_run: bool = False
    plan: DeployPlan | None = None
    outcomes: list[StepOutcome] = field(default_factory=list)
    health: dict[str, Any] | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    human_error: dict[str, Any] | None = None
    logs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    landing: str | None = None
    readme_updated: bool = False
    report_path: str | None = None
    started_at: float = field(default_factory=time.time)
    elapsed: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "provider": self.provider,
            "url": self.url,
            "health_url": self.health_url,
            "public": self.public,
            "dry_run": self.dry_run,
            "plan": self.plan.to_dict() if self.plan else None,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "health": self.health,
            "artifacts": self.artifacts,
            "error": self.error,
            "human_error": self.human_error,
            "notes": self.notes,
            "landing": self.landing,
            "readme_updated": self.readme_updated,
            "report_path": self.report_path,
            "elapsed": round(self.elapsed, 2),
            "meta": self.meta,
        }

    def save(self, path: str) -> str:
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.report_path = str(target)
        return str(target)

    def render(self) -> str:
        if self.dry_run:
            head = "Deploy plan (dry run)"
        elif self.ok and self.url and self.public:
            head = f"Deployed to {self.url}"
        elif self.ok and self.url:
            head = f"Running at {self.url} (not public)"
        elif self.ok:
            head = f"Deploy artifact ready ({self.provider})"
        else:
            head = f"Deploy failed ({self.provider})"

        lines = [head, "=" * len(head)]
        for outcome in self.outcomes:
            suffix = f" - {outcome.detail}" if outcome.detail else ""
            lines.append(f"[{outcome.icon}] {outcome.step.title}{suffix}")
        if self.health:
            lines.append("")
            lines.append(
                f"health: {'ok' if self.health.get('ok') else 'FAILED'} "
                f"({self.health.get('status')}, {self.health.get('detail')})"
            )
        if self.artifacts.get("written"):
            lines.append(f"files:  {', '.join(self.artifacts['written'])}")
        if self.landing:
            lines.append(f"landing: {self.landing}")
        if self.readme_updated:
            lines.append(
                "README updated with the live link" if self.url else "README updated with the deploy status"
            )
        if self.error:
            lines.append("")
            if self.human_error:
                lines.append(f"{self.human_error.get('title')}: {self.human_error.get('plain')}")
                for step in self.human_error.get("next_steps", []):
                    lines.append(f"  - {step}")
            else:
                lines.append(f"error: {self.error}")
        for note in self.notes:
            lines.append(f"note: {note}")
        lines.append(f"({self.elapsed:.1f}s)")
        return "\n".join(lines)
