"""GitHub Pages provider for static sites (free, no extra account).

Works with whatever git credentials the repository already uses. It builds the
site, stages the output in a throwaway repository and force-pushes a single
``gh-pages`` commit. Nothing is stored outside the project and ``.git`` is never
touched.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from ..plan import DeployStep, ProviderResult
from .base import Availability, DeployContext, DeployProvider


def page_url(remote: str) -> str | None:
    """Derive the Pages URL from a GitHub remote (ssh or https form)."""

    match = re.search(r"github\.com[:/]+([^/]+)/([^/\s]+?)(?:\.git)?$", remote.strip())
    if not match:
        return None
    owner, repo = match.group(1), match.group(2)
    if repo.lower() == f"{owner.lower()}.github.io":
        return f"https://{owner.lower()}.github.io"
    return f"https://{owner.lower()}.github.io/{repo}"


class GitHubPagesProvider(DeployProvider):
    name = "github-pages"
    label = "GitHub Pages (static sites, free)"
    supports_kinds = ("static",)

    def _remote(self, ctx: DeployContext) -> str | None:
        if ctx.config.repo_url:
            return ctx.config.repo_url
        code, output = self.run(["git", "remote", "get-url", "origin"], ctx.root, timeout=20)
        return output.strip() if code == 0 else None

    def available(self, ctx: DeployContext) -> Availability:
        if not self.which("git"):
            return Availability(ok=False, reason="git is not installed")
        if ctx.profile.kind != "static":
            return Availability(
                ok=False,
                reason=f"GitHub Pages serves static files, project is {ctx.profile.kind}",
                fix="Use `--provider local` (Docker) or a host that runs containers.",
            )
        remote = self._remote(ctx)
        if not remote or "github.com" not in remote:
            return Availability(
                ok=False,
                reason="no GitHub remote found",
                fix="git remote add origin git@github.com:<you>/<repo>.git",
            )
        return Availability(ok=True, reason=f"pushes to {remote}")

    def steps(self, ctx: DeployContext) -> list[DeployStep]:
        steps = []
        if ctx.profile.build_cmd:
            steps.append(DeployStep("build", "Build the static site", ctx.profile.build_cmd, "build"))
        steps += [
            DeployStep("stage", "Stage the built output in a clean tree", "temporary git repo", "build"),
            DeployStep("push", "Push a single commit to the gh-pages branch", "git push --force", "release"),
            DeployStep("health", "Check the published URL", "GitHub Pages can take 30-60s", "health"),
        ]
        return steps

    def deploy(self, ctx: DeployContext) -> ProviderResult:
        logs: list[str] = []
        static_dir = Path(ctx.root) / (ctx.profile.static_dir or ".")
        if ctx.profile.build_cmd:
            code, output = self.run(["sh", "-c", ctx.profile.build_cmd], ctx.root, timeout=ctx.timeout)
            logs.append(output[-3000:])
            if code != 0:
                return ProviderResult(ok=False, error=f"build failed (exit {code})", logs=logs, steps=self.steps(ctx))
            static_dir = Path(ctx.root) / (ctx.profile.static_dir or "dist")
        if not (static_dir / "index.html").exists():
            return ProviderResult(
                ok=False,
                error=f"no index.html in {static_dir}",
                logs=logs,
                steps=self.steps(ctx),
            )

        remote = self._remote(ctx)
        url = page_url(remote or "")
        with tempfile.TemporaryDirectory(prefix="gh-pages-") as tmp:
            stage = Path(tmp)
            for item in static_dir.iterdir():
                target = stage / item.name
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
            (stage / ".nojekyll").write_text("", encoding="utf-8")
            commands = [
                ["git", "init", "-q", "-b", "gh-pages"],
                ["git", "add", "-A"],
                ["git", "-c", "user.name=agent deploy", "-c", "user.email=agent@localhost", "commit", "-qm", "deploy"],
                ["git", "push", "--force", remote or "", "HEAD:gh-pages"],
            ]
            for command in commands:
                code, output = self.run(command, stage, timeout=300)
                logs.append(output[-1500:])
                if code != 0:
                    return ProviderResult(
                        ok=False,
                        error=f"{' '.join(command[:2])} failed: {output.strip()[:400]}",
                        logs=logs,
                        steps=self.steps(ctx),
                        url=url,
                    )

        return ProviderResult(
            ok=True,
            url=url,
            health_url=url,
            public=True,
            detail="pushed gh-pages; GitHub Pages serves it after the first build",
            logs=logs,
            steps=self.steps(ctx),
            meta={"branch": "gh-pages", "remote": remote},
        )
