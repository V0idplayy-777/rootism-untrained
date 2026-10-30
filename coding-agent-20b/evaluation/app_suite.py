"""The scoreboard: can this stack build the apps people actually ask for?

One fixed suite over the blueprint catalogue. Each prompt is run through the
whole product path, in order, and every gate has to pass before the prompt counts
as a success:

1. **render** - a complete multi-file project with no placeholders.
2. **tests** - the project's own pytest suite, executed.
3. **code critic** - no blocking maintainability/security finding.
4. **taste** - the interface clears the design bar (tokens, dark mode, phone
   layout, states, focus, microcopy).
5. **deploy plan** - the deploy engine can plan a real artifact.
6. **running app** - uvicorn actually starts, ``/healthz`` returns ok and ``/``
   serves the UI.

Failures are reported per prompt, with the exact failing assertion or missing
taste rule. There is no partial credit and no "looks good" pass: a prompt either
produces a running app or it does not.

Run ``python -m evaluation.app_suite --split eval``. The held-out split is stable
by id hash, so a rising number means the stack generalises.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from deploy.engine import DeployEngine
from quality.critic import CodeCritic
from quality.taste import taste_of
from quality.verify import run_tests
from scaffold.blueprint import render_blueprint
from scaffold.blueprint_variants import (
    apply_broken_owner_scope,
    apply_incomplete_entities,
    apply_legacy_ui,
)
from scaffold.blueprints import AppBlueprint, all_blueprints, split


@dataclass
class Check:
    """One gate, its verdict and the reason."""

    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False
    duration: float = 0.0

    def to_dict(self) -> dict:
        """Serialise for the JSON report."""
        return {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "skipped": self.skipped,
            "duration": round(self.duration, 3),
        }


@dataclass
class AppResult:
    """Everything that happened for one prompt."""

    blueprint_id: str
    title: str
    prompt: str
    checks: list[Check] = field(default_factory=list)
    files: int = 0
    taste_score: int = 0
    critic_score: int = 0
    duration: float = 0.0

    @property
    def ok(self) -> bool:
        """Success means every gate that ran passed and none were skipped."""
        required = [check for check in self.checks if not check.skipped]
        return bool(required) and all(check.ok for check in required) and not any(
            check.skipped for check in self.checks
        )

    @property
    def failures(self) -> list[str]:
        """Human-readable reasons this prompt did not succeed."""
        return [check.name + ": " + (check.detail or "failed") for check in self.checks if not check.ok]

    def to_dict(self) -> dict:
        """Serialise for the JSON report."""
        return {
            "id": self.blueprint_id,
            "title": self.title,
            "prompt": self.prompt,
            "ok": self.ok,
            "files": self.files,
            "critic_score": self.critic_score,
            "taste_score": self.taste_score,
            "duration": round(self.duration, 2),
            "checks": [check.to_dict() for check in self.checks],
            "failures": self.failures,
        }


@dataclass
class SuiteReport:
    """Aggregate success rate plus the per-prompt evidence."""

    results: list[AppResult]
    split: str
    threshold: int = 70
    started: float = field(default_factory=time.time)
    duration: float = 0.0

    @property
    def passed(self) -> int:
        """Prompts where every gate passed."""
        return sum(1 for result in self.results if result.ok)

    @property
    def total(self) -> int:
        """Prompts attempted."""
        return len(self.results)

    @property
    def success_rate(self) -> float:
        """Primary metric: running apps / prompts."""
        return round(self.passed / self.total, 3) if self.total else 0.0

    def by_check(self) -> dict[str, dict]:
        """Pass counts per gate, to show where failures cluster."""
        names = {check.name for result in self.results for check in result.checks}
        counts: dict[str, dict] = {}
        for name in sorted(names):
            relevant = [check for result in self.results for check in result.checks if check.name == name]
            counts[name] = {
                "passed": sum(1 for check in relevant if check.ok and not check.skipped),
                "failed": sum(1 for check in relevant if not check.ok),
                "skipped": sum(1 for check in relevant if check.skipped),
                "total": len(relevant),
            }
        return counts

    def to_dict(self) -> dict:
        """Serialise for CI or the docs table."""
        return {
            "split": self.split,
            "total": self.total,
            "passed": self.passed,
            "success_rate": self.success_rate,
            "threshold": self.threshold,
            "duration": round(self.duration, 2),
            "by_check": self.by_check(),
            "failures": [result.to_dict() for result in self.results if not result.ok],
            "results": [result.to_dict() for result in self.results],
        }

    def render(self) -> str:
        """Markdown: the table a reviewer reads first, then failure detail."""
        lines = [
            "# Idea -> running app",
            "",
            f"- split: `{self.split}`",
            f"- prompts: **{self.total}**",
            f"- running apps: **{self.passed}**",
            f"- success rate: **{self.success_rate:.1%}**",
            f"- wall clock: {self.duration:.1f}s",
            "",
            "| gate | passed | failed | skipped |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, row in self.by_check().items():
            lines.append(f"| {name} | {row['passed']} | {row['failed']} | {row['skipped']} |")
        failed = [result for result in self.results if not result.ok]
        if failed:
            lines.extend(["", "## Failures", ""])
            for result in failed:
                lines.append(f"### {result.blueprint_id} - {result.title}")
                lines.append("")
                for reason in result.failures:
                    lines.append("- " + reason.replace("\n", " ")[:700])
                lines.append("")
        else:
            lines.extend(["", "Every prompt produced a running, tested, deployable app."])
        return "\n".join(lines).rstrip() + "\n"


def write_tree(files: dict[str, str], root: Path) -> None:
    """Materialise an in-memory project so real tools can run on it."""
    for path, content in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get(url: str, timeout: float = 1.5) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - localhost only
        return response.status, response.read(2000).decode("utf-8", "replace")


def health_check(root: Path, module: str, timeout: float = 20.0) -> Check:
    """Start the real ASGI server, hit health and the page, then stop it."""
    started = time.time()
    port = _free_port()
    env = os.environ.copy()
    env["APP_ENV"] = "test"
    env["SQLITE_PATH"] = str(root / "data" / "health.db")
    env["DATA_DIR"] = str(root / "data")
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        module + ".main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        return Check("server", False, "could not start uvicorn: " + str(exc), duration=time.time() - started)
    deadline = time.time() + timeout
    health: tuple[int, str] | None = None
    try:
        while time.time() < deadline:
            if process.poll() is not None:
                break
            try:
                health = _get(f"http://127.0.0.1:{port}/healthz")
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                time.sleep(0.15)
        if health is None:
            tail = ""
            if process.poll() is not None and process.stdout is not None:
                tail = process.stdout.read()[-900:]
            return Check(
                "server",
                False,
                "the app did not answer /healthz within "
                + str(int(timeout))
                + "s"
                + ("; output: " + tail.replace("\n", " ") if tail else ""),
                duration=time.time() - started,
            )
        status, body = health
        if status != 200 or '"ok"' not in body:
            return Check("server", False, f"/healthz returned {status}: {body[:200]}", duration=time.time() - started)
        page_status, page = _get(f"http://127.0.0.1:{port}/")
        if page_status != 200 or "<html" not in page.lower():
            return Check("server", False, f"/ returned {page_status} without HTML", duration=time.time() - started)
        return Check("server", True, f"/healthz ok, / served {len(page)} bytes", duration=time.time() - started)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def module_name(files: dict[str, str]) -> str:
    """The importable package name, taken from the rendered tree."""
    for path in files:
        if path.endswith("/main.py"):
            return path.split("/")[0]
    return "app"


class AppSuite:
    """Run the fixed prompt suite end to end."""

    def __init__(
        self,
        blueprints: list[AppBlueprint] | None = None,
        *,
        which: str = "eval",
        limit: int | None = None,
        taste_threshold: int = 70,
        start_servers: bool = True,
    ):
        if blueprints is not None:
            selected = blueprints
        elif which == "all":
            selected = all_blueprints()
        elif which == "train":
            selected = split("app")[0]
        else:
            selected = split("app")[1]
        self.blueprints = selected[:limit] if limit else selected
        self.which = which
        self.taste_threshold = taste_threshold
        self.start_servers = start_servers

    def run_one(self, blueprint: AppBlueprint) -> AppResult:
        """Render, test, review, plan and boot one prompt."""
        started = time.time()
        result = AppResult(blueprint_id=blueprint.id, title=blueprint.title, prompt=blueprint.prompt)
        try:
            project = render_blueprint(blueprint)
        except Exception as exc:  # noqa: BLE001 - a render crash is a suite failure, not a suite crash
            result.checks.append(Check("render", False, f"{type(exc).__name__}: {exc}"))
            result.duration = time.time() - started
            return result
        result.files = len(project.files)
        result.checks.append(Check("render", True, f"{len(project.files)} files"))
        with tempfile.TemporaryDirectory(prefix="suite-") as tmp:
            root = Path(tmp)
            write_tree(project.files, root)
            self._check_tests(root, result)
            self._check_critic(root, result)
            self._check_taste(root, result)
            self._check_deploy(root, result)
            if self.start_servers:
                result.checks.append(health_check(root, module_name(project.files)))
            else:
                result.checks.append(Check("server", True, "disabled", skipped=True))
        result.duration = time.time() - started
        return result

    def _check_tests(self, root: Path, result: AppResult) -> None:
        started = time.time()
        run = run_tests(root, timeout=180)
        result.checks.append(
            Check(
                "tests",
                run.ok,
                f"{run.command}: {len(run.passed)} passed, {len(run.failed)} failed"
                if run.ok
                else run.failure_signal(1200),
                duration=time.time() - started,
            )
        )

    def _check_critic(self, root: Path, result: AppResult) -> None:
        started = time.time()
        review = CodeCritic(threshold=80).review(root)
        result.critic_score = review.score
        detail = (
            f"{review.score}/100, {len(review.findings)} findings"
            if review.ok
            else "; ".join(item.rule + " @ " + item.path for item in review.blocking[:4]) or f"{review.score}/100"
        )
        result.checks.append(Check("critic", review.ok, detail, duration=time.time() - started))

    def _check_taste(self, root: Path, result: AppResult) -> None:
        started = time.time()
        report = taste_of(root, threshold=self.taste_threshold)
        result.taste_score = report.score
        detail = (
            f"{report.score}/100"
            if report.ok
            else f"{report.score}/100: " + ", ".join(item.rule for item in report.findings[:6])
        )
        result.checks.append(Check("taste", report.ok, detail, duration=time.time() - started))

    def _check_deploy(self, root: Path, result: AppResult) -> None:
        started = time.time()
        try:
            plan, _context, warnings = DeployEngine(root).plan()
        except Exception as exc:  # noqa: BLE001 - the plan is a gate
            result.checks.append(Check("deploy", False, f"{type(exc).__name__}: {exc}", duration=time.time() - started))
            return
        kinds = {step.kind for step in plan.steps}
        ok = bool(plan.provider) and {"preflight", "report"} <= kinds
        detail = plan.provider + f" ({len(plan.steps)} steps, {len(warnings)} warnings)"
        if not ok:
            detail = "no usable deploy plan: " + "; ".join(warnings[:3] or ["provider missing"])
        result.checks.append(Check("deploy", ok, detail, duration=time.time() - started))

    def run(self) -> SuiteReport:
        """Run every selected prompt and aggregate honestly."""
        started = time.time()
        results = [self.run_one(blueprint) for blueprint in self.blueprints]
        return SuiteReport(
            results=results,
            split=self.which,
            threshold=self.taste_threshold,
            started=started,
            duration=time.time() - started,
        )


def variant_guards(blueprint: AppBlueprint | None = None) -> list[dict]:
    """Prove the rejected variants are rejected for the right reason.

    A preference pair is only meaningful when the default passes and each
    downgrade fails the specific gate it is supposed to fail.
    """
    target = blueprint or next(
        (
            item
            for item in split("app")[0]
            if item.auth and any(entity.relations for entity in item.entities)
        ),
        split("app")[0][0],
    )
    project = render_blueprint(target)
    with tempfile.TemporaryDirectory(prefix="guard-") as tmp:
        root = Path(tmp)
        write_tree(project.files, root)
        good_tests = run_tests(root, timeout=120)
        good_taste = taste_of(root)
        guards = [
            {"variant": "default", "gate": "tests", "ok": good_tests.ok, "detail": good_tests.failure_signal(400)},
            {"variant": "default", "gate": "taste", "ok": good_taste.ok, "detail": good_taste.summary()},
        ]
    variants = {
        "legacy-ui": apply_legacy_ui(project.files, target.title),
        "incomplete-entities": apply_incomplete_entities(project.files, target),
    }
    if target.auth:
        variants["broken-auth"] = apply_broken_owner_scope(project.files, target, auth=True)
    for name, files in variants.items():
        with tempfile.TemporaryDirectory(prefix="guard-") as tmp:
            root = Path(tmp)
            write_tree(files, root)
            gate = "taste" if name == "legacy-ui" else "tests"
            if gate == "taste":
                report = taste_of(root)
                guards.append({"variant": name, "gate": gate, "ok": not report.ok, "detail": report.summary()})
            else:
                run = run_tests(root, timeout=120)
                guards.append(
                    {"variant": name, "gate": gate, "ok": not run.ok, "detail": run.failure_signal(400)}
                )
    return guards


def main(argv: list[str] | None = None) -> int:
    """Run the suite; exit non-zero when any selected prompt fails."""
    parser = argparse.ArgumentParser(description="Idea -> running app success rate")
    parser.add_argument("--split", default="eval", choices=["eval", "train", "all"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", default=None, help="write the full report here")
    parser.add_argument("--no-server", action="store_true", help="skip booting the app (not a success)")
    parser.add_argument("--guards", action="store_true", help="check the preference/repair variants instead")
    args = parser.parse_args(argv)

    if args.guards:
        guards = variant_guards()
        print(json.dumps(guards, indent=2, ensure_ascii=False))
        return 0 if all(item["ok"] for item in guards) else 1
    suite = AppSuite(which=args.split, limit=args.limit, start_servers=not args.no_server)
    report = suite.run()
    print(report.render())
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"report written to {path}")
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
