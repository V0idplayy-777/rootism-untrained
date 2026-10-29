"""Write a starter into a directory, safely and idempotently.

Safety rules:

* Refuses to write into a non-empty directory unless ``force=True`` (so it can
  never trample an existing project).
* Never overwrites an existing file during a forced run; it reports the files it
  skipped instead.
* Validates that no ``{{ placeholder }}`` survives rendering, and compiles every
  generated Python file before declaring success - a broken scaffold is worse
  than no scaffold.
* Records what it created in ``.agent/scaffold.json`` and writes
  ``agentdeploy.toml`` so `agent deploy` already knows the port and health path.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from deploy.config import DeployConfig

from .intent import intent_from_request
from .registry import get_template, template_ids
from .spec import ProjectOptions, find_placeholders, module_name, slugify, titleize


@dataclass
class ScaffoldResult:
    ok: bool
    template_id: str = ""
    root: str = ""
    files: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    options: dict = field(default_factory=dict)
    next_steps: list[str] = field(default_factory=list)
    port: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    plan: str = ""
    dry_run: bool = False
    reasons: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)

    def render(self) -> str:
        if not self.ok:
            return f"Scaffold failed: {self.error}"
        head = f"{'Would create' if self.dry_run else 'Created'} {self.template_id} in {self.root}"
        lines = [head]
        if self.reasons:
            lines += [f"  why: {reason}" for reason in self.reasons]
        lines += [f"  + {path}" for path in self.files[:25]]
        if len(self.files) > 25:
            lines.append(f"  + ... {len(self.files) - 25} more files")
        if self.skipped:
            lines.append(f"  kept existing: {', '.join(self.skipped[:10])}")
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        if self.next_steps:
            lines.append("")
            lines.append("Next:")
            lines += [f"  {index}. {step}" for index, step in enumerate(self.next_steps, 1)]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "template": self.template_id,
            "root": self.root,
            "files": list(self.files),
            "skipped": list(self.skipped),
            "options": self.options,
            "next_steps": list(self.next_steps),
            "port": self.port,
            "warnings": list(self.warnings),
            "error": self.error,
            "dry_run": self.dry_run,
            "reasons": list(self.reasons),
            "questions": list(self.questions),
        }


MERGE_SAFE_NAMES = frozenset(
    {".agent", ".git", ".gitignore", ".DS_Store", ".venv", "venv", "node_modules"}
)


class ScaffoldGenerator:
    def __init__(self, root: str | Path, *, force: bool = False, dry_run: bool = False):
        self.root = Path(root)
        self.force = force
        self.dry_run = dry_run

    def create(
        self,
        template_id: str,
        name: str | None = None,
        *,
        description: str = "",
        use_auth: bool | None = None,
        database: str | None = None,
        extra_options: dict | None = None,
        auth_provider: str | None = None,
        add_jobs: bool | None = None,
        verify_after: bool = False,
        run_critic: bool = False,
    ) -> ScaffoldResult:
        try:
            spec = get_template(template_id)
        except KeyError as exc:
            return ScaffoldResult(ok=False, error=str(exc), root=str(self.root))

        project_name = slugify(name or self.root.name or "app")
        extra = dict(extra_options or {})
        if "auth_provider" in extra:
            _popped_auth = extra.pop("auth_provider")
            auth_provider = auth_provider or _popped_auth
        if "add_jobs" in extra:
            _popped_jobs = extra.pop("add_jobs")
            if add_jobs is None:
                add_jobs = _popped_jobs
        options = ProjectOptions(
            name=project_name,
            display_name=titleize(name or self.root.name or "app"),
            module=module_name(project_name),
            description=description,
            port=spec.default_port,
            use_auth=bool(spec.supports_auth and (use_auth if use_auth is not None else False)),
            database=(database or extra.pop("database", None) or ("sqlite" if spec.supports_database else "none")),
            warnings=[],
            **extra,
        )
        if options.database == "postgres" and not spec.supports_database:
            options.database = "none"
        chosen_auth_provider = (auth_provider or "").strip().lower() or None

        result = ScaffoldResult(
            ok=True,
            template_id=spec.id,
            root=str(self.root),
            options=options.to_dict(),
            port=options.port,
            warnings=list(options.warnings),
            dry_run=self.dry_run,
        )

        project_files = []
        if self.root.exists():
            project_files = [
                path
                for path in self.root.iterdir()
                if path.name not in MERGE_SAFE_NAMES
            ]
        if project_files and not self.force:
            result.ok = False
            result.error = (
                f"{self.root} is not empty; pass --force to add the starter alongside existing files"
            )
            return result

        try:
            files = spec.render(options)
        except Exception as exc:  # noqa: BLE001 - template bug should be visible, not silent
            result.ok = False
            result.error = f"template {spec.id} failed to render: {type(exc).__name__}: {exc}"
            return result

        if chosen_auth_provider:
            try:
                from .auth_providers import auth_provider_files, get_auth_provider

                provider = get_auth_provider(chosen_auth_provider)
                extra_files = auth_provider_files(chosen_auth_provider, module=options.module)
                for rel, content in extra_files.items():
                    if rel not in files:
                        files[rel] = content
                env_key = provider["env"][0] if provider["env"] else ""
                if env_key and ".env.example" in files:
                    files[".env.example"] += f"\n# {provider['label']} auth\n{env_key}=\n"
                result.warnings.append(f"auth provider {chosen_auth_provider} added ({provider['label']})")
                result.options["auth_provider"] = chosen_auth_provider
            except Exception as exc:  # noqa: BLE001
                result.ok = False
                result.error = f"auth provider {chosen_auth_provider} failed: {exc}"
                return result

        should_add_jobs = add_jobs if add_jobs is not None else True
        if should_add_jobs:
            try:
                from .jobs import flask_jobs, node_jobs, python_jobs

                if spec.id == "flask-htmx":
                    job_files = flask_jobs()
                elif spec.kind in ("python-web", "cli", "worker") or spec.id in ("fastapi-react", "cli-tool", "discord-bot"):
                    job_files = python_jobs(options.module, options.display_name)
                else:
                    job_files = node_jobs()
                for rel, content in job_files.items():
                    if rel not in files:
                        files[rel] = content
            except Exception:
                pass

        placeholders = find_placeholders(files)
        if placeholders:
            result.ok = False
            result.error = "template left placeholders: " + ", ".join(placeholders[:5])
            return result

        for rel, content in sorted(files.items()):
            target = self.root / rel
            if target.exists() and not self.dry_run:
                result.skipped.append(rel)
                continue
            result.files.append(rel)
            if self.dry_run:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        if not self.dry_run:
            self._merge_gitignore(files.get(".gitignore"))
            self._write_config(spec, options, result)
            self._record(spec, options, result)
            self._check_python(result)
            if result.ok:
                self._run_post_generation_critic(result, options)
            if result.ok and verify_after:
                self._verify_generated(result)

        result.next_steps = [
            step.replace("{module}", options.module).replace("{name}", options.name)
            for step in spec.next_steps
        ] or [
            f"cd {self.root}",
            f"Read README.md for the run/test commands (port {options.port})",
        ]
        if result.skipped:
            result.warnings.append(
                f"{len(result.skipped)} existing file(s) were left untouched (no overwrites)"
            )
        result.plan = "\n".join(sorted(result.files))
        return result

    def create_from_request(self, prose: str, name: str | None = None) -> ScaffoldResult:
        """Detect intent from plain English, then scaffold the matching starter."""

        intent, choice = intent_from_request(prose, template_ids())
        result = self.create(
            choice.template_id,
            name=name,
            description=prose.strip()[:300],
            use_auth=bool(choice.options.get("use_auth")),
            database=choice.options.get("database"),
        )
        result.reasons = list(choice.reasons)
        result.questions = list(choice.questions) + list(result.questions)
        result.options["intent"] = intent.to_dict()
        return result

    def _merge_gitignore(self, wanted: str | None) -> None:
        if not wanted:
            return
        path = self.root / ".gitignore"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        additions = [
            line
            for line in wanted.splitlines()
            if line.strip() and line.strip() not in {row.strip() for row in existing.splitlines()}
        ]
        if additions:
            suffix = "" if existing.endswith("\n") or not existing else "\n"
            path.write_text(existing + suffix + "\n".join(additions) + "\n", encoding="utf-8")

    def _write_config(self, spec, options: ProjectOptions, result: ScaffoldResult) -> None:
        config_path = self.root / "agentdeploy.toml"
        if config_path.exists():
            result.warnings.append("agentdeploy.toml already existed; left as is")
            return
        config = DeployConfig(
            name=options.name,
            provider="auto",
            port=options.port or None,
            health_path="/healthz" if spec.kind != "static" else "/",
            static_dir="." if spec.kind == "static" else None,
        )
        config.save(self.root)

    def _record(self, spec, options: ProjectOptions, result: ScaffoldResult) -> None:
        record = {
            "template": spec.id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "options": options.to_dict(),
            "files": list(result.files),
            "skipped": list(result.skipped),
        }
        path = self.root / ".agent" / "scaffold.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    def _check_python(self, result: ScaffoldResult) -> None:
        """Compile generated Python files; a scaffold that does not parse is a failure."""

        import py_compile

        for rel in result.files:
            if not rel.endswith(".py"):
                continue
            try:
                py_compile.compile(str(self.root / rel), doraise=True)
            except py_compile.PyCompileError as exc:
                result.ok = False
                result.error = f"generated {rel} does not compile: {exc.msg}"
                return

    def _run_post_generation_critic(self, result: ScaffoldResult, options: ProjectOptions) -> None:
        """Deterministic post-generation critic that can force rewrite (row 3)."""

        try:
            from quality.critic import CodeCritic
        except Exception:
            return
        try:
            critic = CodeCritic()
            review = critic.review(self.root)
            if not review.ok:
                result.warnings.append(f"critic score {review.score}/100: {review.summary()}")
                fixed = self._auto_fix_critic_findings(review)
                if fixed:
                    result.warnings.append(f"auto-fixed {len(fixed)} critic findings")
                    # re-review
                    review2 = critic.review(self.root)
                    if review2.ok:
                        result.warnings.append(f"critic now passing: {review2.score}/100")
                    result.options["critic"] = {"initial_score": review.score, "final_score": review2.score, "findings": len(review.findings)}
                else:
                    result.options["critic"] = {"score": review.score, "findings": len(review.findings)}
        except Exception:
            pass

    def _auto_fix_critic_findings(self, review) -> list[str]:
        fixed: list[str] = []
        for finding in getattr(review, "findings", []):
            if finding.rule == "no-readme" and not (self.root / "README.md").exists():
                try:
                    (self.root / "README.md").write_text(f"# {self.root.name}\n\nGenerated starter.\n", encoding="utf-8")
                    fixed.append("no-readme")
                except Exception:
                    pass
            if finding.rule == "env-not-ignored":
                try:
                    path = self.root / ".gitignore"
                    text = path.read_text(encoding="utf-8") if path.exists() else ""
                    if ".env" not in text:
                        path.write_text(text + ("\n" if text and not text.endswith("\n") else "") + ".env\n.env.*\n!.env.example\n", encoding="utf-8")
                        fixed.append("env-not-ignored")
                except Exception:
                    pass
        return fixed

    def _verify_generated(self, result: ScaffoldResult) -> None:
        """Close the loop: every generated project must run its own tests (row 3)."""

        try:
            from quality.verify import run_tests
        except Exception:
            return
        try:
            run = run_tests(self.root, timeout=120)
            result.options["verify"] = {"ok": run.ok, "exit_code": run.exit_code, "passed": run.passed, "failed": run.failed}
            if not run.ok:
                result.warnings.append(f"generated tests failed: {run.failure_signal()[:200]}")
            else:
                result.warnings.append(f"generated tests passed: {len(run.passed)} passed")
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(f"verify error: {exc}")


def scaffold_project(root: str | Path, template_id: str, name: str | None = None, **kwargs) -> ScaffoldResult:
    return ScaffoldGenerator(root, force=kwargs.pop("force", False), dry_run=kwargs.pop("dry_run", False)).create(
        template_id, name, **kwargs
    )


def plan_for(prose: str, name: str | None = None) -> dict:
    """Dry-run plan for a request: template, files, steps and open questions."""

    intent, choice = intent_from_request(prose, template_ids())
    spec = get_template(choice.template_id)
    options = ProjectOptions(
        name=slugify(name or "app"),
        display_name=titleize(name or "app"),
        module=module_name(name or "app"),
        port=spec.default_port,
        use_auth=bool(choice.options.get("use_auth")),
        database=choice.options.get("database") or ("sqlite" if spec.supports_database else "none"),
    )
    files = sorted(spec.render(options))
    return {
        "intent": intent.to_dict(),
        "template": spec.to_dict(),
        "options": options.to_dict(),
        "files": files,
        "reasons": choice.reasons,
        "questions": choice.questions,
        "steps": [
            f"Create {len(files)} files for {spec.name}",
            "Write agentdeploy.toml (deploy already knows the port and health path)",
            "Compile-check generated Python and record the scaffold in .agent/scaffold.json",
            "Run the tests, then `agent deploy --dry-run` to see the deploy plan",
        ],
    }
