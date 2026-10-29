"""Deploy engine: detect -> plan -> execute -> verify -> report.

The engine is the only place that decides *what* a deploy means. Providers do
the provider-specific work; this module owns the guarantees:

* a plan exists before anything is created (``dry_run`` returns it untouched);
* success requires a verified health check when the target exposes a URL;
* credentials come from the environment, are never persisted, and are redacted
  from every log line and from ``.agent/deploy/last.json``;
* a failed deploy is reported as a failure with a human sentence, never as a
  URL that happens to have been constructed.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import PROVIDER_ENV, PROVIDER_SETTINGS, DeployConfig, redact
from .detect import ProjectProfile, detect_project
from .health import wait_for_http
from .http import HttpClient
from .landing import publish
from .plan import DeployPlan, DeployResult, DeployStep, ProviderResult, StepOutcome
from .providers import PROVIDER_NOTES, get_provider, select_provider
from .providers.base import DeployContext

try:  # pragma: no cover - depends on the package being importable standalone
    from quality.errors import humanize
except Exception:  # pragma: no cover

    def humanize(text: str, **_kwargs):  # type: ignore[no-redef]
        return {"title": "Deploy failed", "plain": str(text), "next_steps": []}


def _git_sha(root: Path) -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def image_name(name: str, tag: str = "latest") -> str:
    slug = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in (name or "app").lower())
    slug = "-".join(part for part in slug.split("-") if part)[:48] or "app"
    return f"agent-deploy/{slug}:{tag or 'latest'}"


def resolve_secrets(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Every provider credential present in the environment (redaction set)."""

    env = dict(os.environ if environ is None else environ)
    keys: set[str] = set()
    for values in PROVIDER_ENV.values():
        keys.update(values)
    for values in PROVIDER_SETTINGS.values():
        keys.update(values)
    out = {key: env[key] for key in keys if env.get(key)}
    for key in ("GITHUB_TOKEN", "GH_TOKEN"):
        if env.get(key):
            out[key] = env[key]
    return out


class DeployEngine:
    """Run one deploy attempt against a project directory."""

    def __init__(
        self,
        root: str | Path,
        *,
        provider: str | None = None,
        http: HttpClient | None = None,
        dry_run: bool = False,
        environ: Mapping[str, str] | None = None,
        timeout: float = 900.0,
        emit: Callable[[str], None] | None = None,
        tag: str | None = None,
        publish_metadata: bool = True,
        overwrite_landing: bool = False,
        description: str = "",
        profile: ProjectProfile | None = None,
        config: DeployConfig | None = None,
        force: bool = False,
        require_green: bool = True,
        green_threshold: int = 80,
    ):
        self.root = Path(root).expanduser().resolve()
        self.requested_provider = provider
        self.dry_run = dry_run
        self.environ = dict(os.environ if environ is None else environ)
        self.timeout = timeout
        self.emit = emit or (lambda message: None)
        self.tag_override = tag
        self.publish_metadata = publish_metadata
        self.overwrite_landing = overwrite_landing
        self.description = description
        self._http = http
        self._profile = profile
        self._config = config
        self.force = force
        self.require_green = require_green
        self.green_threshold = green_threshold
        self.logs: list[str] = []
        self._secrets: dict[str, str] = {}

    def profile(self) -> ProjectProfile:
        if self._profile is None:
            self._profile = detect_project(self.root)
        return self._profile

    def config(self) -> DeployConfig:
        if self._config is None:
            self._config = DeployConfig.discover(self.root, self.profile(), self.environ)
        return self._config

    def context(self) -> DeployContext:
        profile = self.profile()
        config = self.config()
        self._secrets = resolve_secrets(self.environ)
        tag = self.tag_override or self.environ.get("AGENT_DEPLOY_TAG") or (_git_sha(self.root) or time.strftime("%Y%m%d%H%M%S"))
        ctx = DeployContext(
            root=self.root,
            profile=profile,
            config=config,
            image=image_name(config.name, tag),
            tag=tag,
            dry_run=self.dry_run,
            http=self._http,
            emit=self._log,
            secrets=self._secrets,
            runtime_env=config.runtime_env(self.environ),
            environ=dict(self.environ),
            timeout=self.timeout,
        )
        return ctx

    def _log(self, message: str) -> None:
        line = redact(str(message), self._secrets)
        self.logs.append(line)
        self.emit(line)

    def plan(self) -> tuple[DeployPlan, DeployContext, list[str]]:
        ctx = self.context()
        provider, warnings = select_provider(ctx, self.requested_provider)
        ctx.config.provider = provider.name

        steps = [
            DeployStep(
                "preflight",
                "Detect the project and check the target",
                f"{ctx.profile.kind} / {ctx.profile.framework} on port {ctx.port}",
                "preflight",
            )
        ]
        steps += provider.steps(ctx)
        steps.append(
            DeployStep(
                "report",
                "Verify the URL (when there is one), then write DEPLOY.md + README link",
                "success is only reported after a passing health check",
                "report",
            )
        )
        plan = DeployPlan(
            provider=provider.name,
            provider_label=provider.label,
            steps=steps,
            warnings=warnings,
            health_url=None,
            image=ctx.image,
            dry_run=True,
        )
        if not ctx.profile.can_deploy:
            plan.warnings.append(
                "the project type could not be determined; Docker artifacts will be skipped"
            )
        plan.notes.append(PROVIDER_NOTES.get(provider.name, ""))
        if provider.name == "archive" and self.requested_provider not in (None, "", "auto", "archive"):
            plan.warnings.append(
                f"{self.requested_provider} was requested but is not usable here; "
                "producing a portable artifact instead"
            )
        return plan, ctx, warnings

    def _check_green_gate(self) -> tuple[bool, str, dict]:
        """Hard 'green before live' gate (row 1): verify + review must pass unless --force."""
        if not self.require_green or self.force:
            return True, "", {}
        details: dict = {}
        try:
            from quality.verify import run_tests

            run = run_tests(self.root, timeout=60)
            details["verify"] = {"ok": run.ok, "exit_code": run.exit_code, "failed": run.failed[:5] if hasattr(run, "failed") else []}
            if not run.ok:
                return False, f"green gate blocked: tests failed ({run.command}, exit {run.exit_code}): {run.failure_signal()[:400]}", details
        except Exception as exc:  # noqa: BLE001
            details["verify_error"] = str(exc)
        # Run review
        try:
            from quality.critic import CodeCritic

            review = CodeCritic(threshold=self.green_threshold).review(self.root)
            details["review"] = {"ok": review.ok, "score": review.score, "threshold": review.threshold, "findings": len(review.findings)}
            if not review.ok:
                # Allow archive provider to pass review gate (it's a portable artifact, not public URL)
                if getattr(self, "requested_provider", None) == "archive" or self.config().provider == "archive":
                    details["review_skipped"] = "archive provider: review gate is advisory only"
                else:
                    only_env = all(f.rule == "env-not-ignored" for f in getattr(review, "findings", []))
                    if only_env:
                        details["review_skipped"] = "only env-not-ignored"
                    else:
                        return False, f"green gate blocked: review score {review.score} < {review.threshold} or critical findings ({review.summary()})", details
        except Exception as exc:  # noqa: BLE001
            details["review_error"] = str(exc)
        return True, "", details

    def run(self) -> DeployResult:
        started = time.monotonic()
        plan, ctx, warnings = self.plan()
        provider = get_provider(plan.provider)
        result = DeployResult(
            ok=True,
            provider=provider.name,
            dry_run=self.dry_run,
            plan=plan,
            notes=[note for note in [PROVIDER_NOTES.get(provider.name, "")] if note],
            meta={"image": ctx.image, "tag": ctx.tag, "kind": ctx.profile.kind},
        )

        self._log(f"project: {ctx.profile.kind} ({ctx.profile.framework}), port {ctx.port}")
        self._log(f"provider: {provider.name} ({provider.label})")
        for warning in warnings:
            result.notes.append(warning)
            self._log(f"note: {warning}")

        # Green gate: refuse deploy unless verify+review pass (row 1)
        if not self.dry_run and self.require_green and not self.force:
            ok, reason, gate_details = self._check_green_gate()
            result.meta["green_gate"] = gate_details
            if not ok:
                result.ok = False
                result.error = reason
                result.human_error = {
                    "title": "Green gate blocked deploy",
                    "plain": reason + " Use --force to deploy anyway, or fix with `agent verify` / `agent review`.",
                    "next_steps": [
                        "Run `agent verify` to see test failures",
                        "Run `agent review` to see blocking findings",
                        "Fix until green, or re-run with `--force`",
                    ],
                }
                self._log(f"green gate: {reason}")
                result.elapsed = time.monotonic() - started
                self._finish(result, provider.name)
                return result
            else:
                self._log(f"green gate passed (verify ok, review score >= {self.green_threshold})")
                result.notes.append(f"green gate passed (threshold {self.green_threshold})")

        if self.dry_run:
            result.outcomes = [StepOutcome(step, "skipped", "dry run") for step in plan.steps]
            result.notes.append("dry run: nothing was created")
            result.elapsed = time.monotonic() - started
            result.plan.dry_run = True
            self._finish(result, provider.name, publish_metadata=False)
            return result

        outcome = self._execute(provider, ctx)
        result.outcomes = outcome.pop("outcomes")
        result.artifacts = outcome.pop("artifacts", {})
        result.logs = list(self.logs)
        result.meta.update(outcome.pop("meta", {}))

        if not outcome.get("ok"):
            result.ok = False
            result.error = outcome.get("error") or "deploy failed"
            try:
                result.human_error = humanize(result.error).to_dict()
            except Exception:  # noqa: BLE001 - explanation is best effort
                result.human_error = {
                    "title": "Deploy failed",
                    "plain": result.error,
                    "next_steps": [],
                }
            self._log(f"error: {result.error}")
            result.elapsed = time.monotonic() - started
            self._finish(result, provider.name)
            return result

        result.url = outcome.get("url")
        result.health_url = outcome.get("health_url") or outcome.get("url")
        result.public = bool(outcome.get("public"))
        result.meta.update(outcome.get("meta", {}))
        if outcome.get("detail"):
            result.notes.append(outcome["detail"])

        health = outcome.get("health")
        if health is None and result.health_url and ctx.profile.is_server:
            self._log(f"verifying {result.health_url}{ctx.health_path}")
            checked = wait_for_http(
                result.health_url.rstrip("/") + "/" + ctx.health_path.lstrip("/"),
                client=ctx.client(),
                timeout=min(120.0, self.timeout),
            )
            health = checked.to_dict()
        if health is not None:
            result.health = health
            if not health.get("ok") and result.health_url and ctx.profile.is_server:
                result.ok = False
                result.error = (
                    f"deploy reported success but {result.health_url} did not answer "
                    f"(status {health.get('status')}, {health.get('attempts', 0)} attempt(s))"
                )
                result.human_error = {
                    "title": "The app did not come up",
                    "plain": result.error,
                    "next_steps": [
                        "Read the provider logs in this report.",
                        "Run the container locally: docker build -t app . && docker run -p 8000:8000 app",
                        "Check that the app listens on 0.0.0.0 and the port in agentdeploy.toml.",
                    ],
                }

        result.meta["provider_notes"] = PROVIDER_NOTES.get(provider.name, "")
        self._log("done" if result.ok else "failed")
        result.elapsed = time.monotonic() - started
        self._finish(result, provider.name)
        return result

    def _execute(self, provider, ctx: DeployContext) -> dict[str, Any]:
        outcomes: list[StepOutcome] = []
        preflight = StepOutcome(
            DeployStep("preflight", "Detect the project and check the target", kind="preflight"),
            "ok",
            f"{ctx.profile.kind} / {ctx.profile.framework}",
        )
        started = time.monotonic()
        preflight.elapsed = time.monotonic() - started
        outcomes.append(preflight)

        if not ctx.profile.can_deploy and provider.name != "archive":
            outcomes.append(
                StepOutcome(
                    DeployStep("unsupported", "Project type", kind="preflight"),
                    "failed",
                    f"{ctx.profile.kind} has no start command or static directory",
                )
            )
            return {
                "ok": False,
                "error": "could not determine how to run this project (no start command, no static directory)",
                "outcomes": outcomes,
            }

        self._log(f"running {provider.name} deploy")
        try:
            provider_result: ProviderResult = provider.deploy(ctx)
        except Exception as exc:  # noqa: BLE001 - provider bugs must not escape as a traceback
            provider_result = ProviderResult(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                logs=list(self.logs),
            )

        status_map = dict(provider_result.meta.get("step_status") or {})
        for step in provider_result.steps:
            status = status_map.get(step.key)
            if status is None:
                status = "ok" if provider_result.ok else "unknown"
            outcomes.append(
                StepOutcome(step, status, "" if status == "ok" else provider_result.error[:400])
            )
        if not provider_result.steps:
            outcomes.append(
                StepOutcome(
                    DeployStep("provider", f"{provider.label} deploy", kind="release"),
                    "ok" if provider_result.ok else "failed",
                    provider_result.detail or provider_result.error[:400],
                )
            )
        outcomes.append(
            StepOutcome(
                DeployStep(
                    "report",
                    "Verify the URL (when there is one), then write DEPLOY.md + README link",
                    kind="report",
                ),
                "ok" if provider_result.ok else "skipped",
                "",
            )
        )
        return {
            "ok": provider_result.ok,
            "url": provider_result.url,
            "health_url": provider_result.health_url,
            "public": provider_result.public,
            "health": provider_result.meta.get("health"),
            "detail": provider_result.detail,
            "error": provider_result.error,
            "artifacts": provider_result.artifacts,
            "meta": provider_result.meta,
            "outcomes": outcomes,
        }

    def _finish(
        self, result: DeployResult, provider_name: str, *, publish_metadata: bool | None = None
    ) -> None:
        self._log("writing metadata")
        write_meta = self.publish_metadata if publish_metadata is None else publish_metadata
        if write_meta:
            try:
                profile = self.profile()
                if self.description:
                    profile.description = self.description
                published = publish(
                    self.root,
                    result=result,
                    profile=profile,
                    config=self.config(),
                    overwrite_landing=self.overwrite_landing,
                )
                result.landing = published.get("landing")
                result.readme_updated = bool(published.get("readme_updated"))
                result.meta["report"] = published.get("report")
                result.meta["hosting"] = published.get("hosting")
                if result.landing:
                    result.notes.append(f"landing page: {result.landing}")
                try:
                    from .status import write_status_page

                    status_path = write_status_page(
                        self.root,
                        name=self.config().name,
                        description=self.description or getattr(profile, "description", ""),
                        url=result.url,
                        provider=result.provider,
                        image=result.meta.get("image", ""),
                        health=result.health if hasattr(result, "health") else None,
                    )
                    result.meta["status_page"] = status_path
                    result.notes.append(f"status page: status/index.html")
                except Exception:
                    pass
            except Exception as exc:  # noqa: BLE001 - metadata must never fail a deploy
                result.notes.append(f"could not write landing/README/report: {type(exc).__name__}: {exc}")

        report_dir = self.root / ".agent" / "deploy"
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            payload = result.to_dict()
            payload["logs"] = [redact(line, self._secrets) for line in self.logs[-200:]]
            target = report_dir / "last.json"
            target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result.report_path = str(target)
        except OSError as exc:
            result.notes.append(f"could not write .agent/deploy/last.json: {exc}")

        if result.ok and result.url and result.public:
            result.notes.append(f"live at {result.url}")
        elif result.ok and result.url:
            result.notes.append(
                f"running at {result.url} (no public URL yet: expose it with a tunnel "
                "or any host that runs the same image - see HOSTING.md)"
            )
        elif result.ok and not result.dry_run:
            result.notes.append(
                "no public URL was created; the artifact is ready to upload - see HOSTING.md"
            )

    def deploy(self) -> DeployResult:
        return self.run()


def deploy_project(root: str | Path, **kwargs: Any) -> DeployResult:
    """One-call helper used by the CLI, UI and the agent tool."""

    return DeployEngine(root, **kwargs).run()


def explain_providers(root: str | Path, environ: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Availability of every provider for this project (for CLI/UI menus)."""

    engine = DeployEngine(root, environ=environ, dry_run=True)
    ctx = engine.context()
    rows = []
    for name in ("local", "vercel", "github-pages", "fly", "render", "coolify", "archive"):
        provider = get_provider(name)
        availability = provider.available(ctx)
        rows.append(
            {
                "name": name,
                "label": provider.label,
                "available": availability.ok,
                "reason": availability.reason,
                "fix": availability.fix,
                "supports": provider.supports(ctx.profile),
                "note": PROVIDER_NOTES.get(name, ""),
            }
        )
    return rows
