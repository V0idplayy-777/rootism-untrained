"""Build a named product from a sentence, then prove it runs.

This is the product surface behind ``agent app`` and the blueprint path of
``agent ship``:

* resolve the request to a blueprint (or say clearly that nothing fits),
* render the whole app (schema, auth, store, API, UI, tests, deploy files),
* run its tests, the code critic and the taste gate,
* produce a deploy plan and the exact command for a local preview,
* and **fail loudly** with the failing assertion or missing design rule instead
  of printing a success banner.

:func:`repair_generated` is the deterministic half of the repair loop: it
restores generated modules and web assets from the blueprint, backs up whatever
it replaced, and re-runs the gates. It never touches files the generator does
not own, and it reports when a failure is not one it can fix.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from quality.critic import CodeCritic
from quality.taste import TasteReport, taste_of
from quality.verify import TestRun, run_tests

from .blueprint import AppProject, render_blueprint
from .blueprints import AppBlueprint, all_blueprints, match_blueprint, split

GENERATED_MODULES = (
    "config.py",
    "errors.py",
    "db.py",
    "schemas.py",
    "store.py",
    "api.py",
    "main.py",
    "auth.py",
    "auth_routes.py",
    "schema.sql",
    "migrations/0001_init.sql",
)
WEB_ASSETS = ("web/index.html", "web/app.css", "web/app.js", "web/config.js")


class NoBlueprint(ValueError):
    """The request did not match any product blueprint."""

    def __init__(self, request: str, suggestions: list[str]):
        super().__init__(
            "no blueprint matches " + repr(request) + "; try one of: " + ", ".join(suggestions)
        )
        self.request = request
        self.suggestions = suggestions


@dataclass
class BuildReport:
    """Everything the caller needs to decide whether to show a success banner."""

    blueprint_id: str = ""
    title: str = ""
    prompt: str = ""
    root: str = ""
    files: list[str] = field(default_factory=list)
    verify: TestRun | None = None
    critic_score: int = 0
    critic_ok: bool = False
    critic_findings: list[str] = field(default_factory=list)
    taste: TasteReport | None = None
    deploy_provider: str = ""
    deploy_steps: int = 0
    deploy_warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration: float = 0.0

    @property
    def ok(self) -> bool:
        """True only when every gate passed; no partial credit."""
        return not self.failures

    @property
    def module(self) -> str:
        """Importable package name for the local preview command."""
        for path in self.files:
            if path.endswith("/main.py"):
                return path.split("/")[0]
        return "app"

    @property
    def preview_command(self) -> str:
        """The command a user runs to see the app on their machine."""
        return f"python -m uvicorn {self.module}.main:app --reload --port 8000"

    def to_dict(self) -> dict:
        """Serialise for ``--json`` output."""
        return {
            "ok": self.ok,
            "blueprint": self.blueprint_id,
            "title": self.title,
            "prompt": self.prompt,
            "root": self.root,
            "files": self.files,
            "module": self.module,
            "preview": self.preview_command,
            "verify": self.verify.to_dict() if self.verify is not None else None,
            "critic": {"score": self.critic_score, "ok": self.critic_ok, "findings": self.critic_findings},
            "taste": self.taste.to_dict() if self.taste is not None else None,
            "deploy": {
                "provider": self.deploy_provider,
                "steps": self.deploy_steps,
                "warnings": self.deploy_warnings,
            },
            "failures": self.failures,
            "warnings": self.warnings,
            "duration": round(self.duration, 2),
        }

    def summary(self) -> str:
        """Human-readable, compact, honest."""
        if self.ok:
            return (
                f"built {self.title} ({self.blueprint_id}) in {self.root}: "
                f"{len(self.files)} files, tests green, critic {self.critic_score}/100, "
                f"taste {self.taste.score if self.taste else 0}/100, deploy {self.deploy_provider}"
            )
        return "build blocked: " + "; ".join(self.failures)


def resolve_blueprint(request: str, blueprint_id: str | None = None) -> AppBlueprint:
    """Explicit id, exact id in the request, or the best keyword match."""
    if blueprint_id:
        for item in all_blueprints():
            if item.id == blueprint_id:
                return item
        raise NoBlueprint(request, sorted(item.id for item in all_blueprints())[:12])
    match = match_blueprint(request)
    if match is None:
        raise NoBlueprint(
            request, [item.id for item in split("app")[0][:8]] or [item.id for item in all_blueprints()[:8]]
        )
    return match


def _gate_tests(report: BuildReport, root: Path) -> None:
    run = run_tests(root, timeout=180)
    report.verify = run
    if not run.ok:
        report.failures.append("tests: " + run.failure_signal(1200))


def _gate_critic(report: BuildReport, root: Path, threshold: int) -> None:
    review = CodeCritic(threshold=threshold).review(root)
    report.critic_score = review.score
    report.critic_ok = review.ok
    report.critic_findings = [item.rule + " @ " + item.path for item in review.findings[:10]]
    if not review.ok:
        report.failures.append(
            "critic: " + ", ".join(report.critic_findings[:4] or [f"{review.score}/100"])
        )


def _gate_taste(report: BuildReport, root: Path, threshold: int) -> None:
    taste = taste_of(root, threshold=threshold)
    report.taste = taste
    if not taste.ok:
        report.failures.append(
            "taste: "
            + f"{taste.score}/100 "
            + ", ".join(item.rule for item in taste.findings[:5])
        )


def _gate_deploy(report: BuildReport, root: Path) -> None:
    from deploy.engine import DeployEngine

    try:
        plan, _context, warnings = DeployEngine(root).plan()
    except Exception as exc:  # noqa: BLE001 - a missing plan is a build failure
        report.failures.append("deploy: " + type(exc).__name__ + ": " + str(exc))
        return
    report.deploy_provider = plan.provider
    report.deploy_steps = len(plan.steps)
    report.deploy_warnings = list(plan.warnings)
    if not plan.provider or not plan.steps:
        report.failures.append("deploy: no provider could plan this project")


def build_app(
    root: str | Path,
    request: str,
    *,
    blueprint_id: str | None = None,
    force: bool = False,
    taste_threshold: int = 70,
    critic_threshold: int = 80,
    include_deploy: bool = True,
    write_project: bool = True,
) -> BuildReport:
    """Render the app into ``root`` and run every gate before calling it success."""
    started = time.time()
    root = Path(root).resolve()
    blueprint = resolve_blueprint(request, blueprint_id)
    project = render_blueprint(blueprint)
    if write_project:
        project.write(root, force=force)
    written = sorted(project.files)
    report = BuildReport(
        blueprint_id=blueprint.id,
        title=blueprint.title,
        prompt=request,
        root=str(root),
        files=written,
    )
    _gate_tests(report, root)
    _gate_critic(report, root, critic_threshold)
    _gate_taste(report, root, taste_threshold)
    if include_deploy:
        _gate_deploy(report, root)
    report.duration = time.time() - started
    return report


@dataclass
class RepairReport:
    """Before/after evidence for restoring generated files."""

    ok: bool = False
    before: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)
    restored: list[str] = field(default_factory=list)
    backup: str = ""
    unfixable: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise for ``--json`` output."""
        return {
            "ok": self.ok,
            "before": self.before,
            "after": self.after,
            "restored": self.restored,
            "backup": self.backup,
            "unfixable": self.unfixable,
        }

    def summary(self) -> str:
        """One line for the CLI."""
        if self.ok:
            return (
                "repaired: restored "
                + (", ".join(self.restored) if self.restored else "nothing")
                + " (backup in "
                + self.backup
                + ")"
            )
        if not self.restored:
            return "repair made no change; the failure is not a generated-module regression"
        return "repair did not fix it: " + "; ".join(self.after[:3])


def _gate_names(report: BuildReport) -> list[str]:
    names: list[str] = []
    if report.verify is not None and not report.verify.ok:
        names.append("tests")
    if not report.critic_ok:
        names.append("critic")
    if report.taste is not None and not report.taste.ok:
        names.append("taste")
    return names


def _restore(
    root: Path,
    project: AppProject,
    module: str,
    names: tuple[str, ...],
    backup: Path,
    repaired: list[str],
) -> None:
    for relative in names:
        candidates = (module + "/" + relative, relative)
        key = next((item for item in candidates if item in project.files), None)
        if key is None:
            continue
        source = project.files[key]
        target = root / key
        if target.exists() and target.read_text(encoding="utf-8", errors="replace") == source:
            continue
        if target.exists():
            saved = backup / target.relative_to(root)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
        repaired.append(str(target.relative_to(root)))


def repair_generated(
    root: str | Path,
    blueprint: AppBlueprint,
    *,
    taste_threshold: int = 70,
    critic_threshold: int = 80,
) -> RepairReport:
    """Restore generated modules/assets when tests or taste regressed."""
    root = Path(root).resolve()
    report = RepairReport()
    before = build_app(
        root,
        blueprint.prompt,
        blueprint_id=blueprint.id,
        force=True,
        taste_threshold=taste_threshold,
        critic_threshold=critic_threshold,
        write_project=False,
    )
    if before.ok:
        report.ok = True
        report.before = []
        report.after = []
        return report
    report.before = list(before.failures)
    gates = _gate_names(before)
    project = render_blueprint(blueprint)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = root / ".repair-backup" / stamp
    module = before.module
    restored: list[str] = []
    if "tests" in gates:
        _restore(root, project, module, GENERATED_MODULES, backup, restored)
    if "taste" in gates:
        _restore(root, project, module, WEB_ASSETS, backup, restored)
    report.restored = restored
    report.backup = str(backup.relative_to(root)) if backup.exists() else ""
    if "critic" in gates:
        report.unfixable = [item for item in before.failures if item.startswith("critic:")]
    after = build_app(
        root,
        blueprint.prompt,
        blueprint_id=blueprint.id,
        force=True,
        taste_threshold=taste_threshold,
        critic_threshold=critic_threshold,
        write_project=False,
    )
    report.after = list(after.failures)
    report.ok = after.ok
    return report


def preview_command(root: str | Path) -> str:
    """Best local preview command for an already-built project."""
    root = Path(root)
    for candidate in root.glob("*/main.py"):
        return f"python -m uvicorn {candidate.parent.name}.main:app --reload --port 8000"
    return "python -m http.server 8000"


def temporary_build(request: str, blueprint_id: str | None = None) -> BuildReport:
    """Build in a throwaway directory (used by the agent tool and tests)."""
    with tempfile.TemporaryDirectory(prefix="agent-app-") as tmp:
        report = build_app(tmp, request, blueprint_id=blueprint_id)
        report.files = [str(Path(report.root) / item) for item in report.files]
        return report
