"""Vercel provider.

Two real paths:

* **CLI** - when the `vercel` binary is installed, `vercel deploy --prod` is the
  supported, complete path (it uploads a full project, builds it remotely and
  prints the production URL).
* **REST API** - when only a token is present, small static projects are
  uploaded inline (`POST /v13/deployments`), which is the documented API shape.
  Larger or framework projects get an honest refusal pointing at the CLI rather
  than a half-upload that would fail remotely.

A token comes from ``VERCEL_TOKEN`` (or ``DEPLOY_TOKEN``); it is never written to
the project or to the deploy report.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..health import wait_for_http
from ..plan import DeployStep, ProviderResult
from .base import Availability, DeployContext, DeployProvider

IGNORE = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".agent",
    "deploy",
    ".next",
    ".pytest_cache",
}
MAX_INLINE_FILES = 80
MAX_INLINE_BYTES = 3_500_000


def collect_inline_files(root: Path) -> tuple[list[dict[str, str]], list[str]]:
    files: list[dict[str, str]] = []
    skipped: list[str] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if any(part in IGNORE for part in parts[:-1]):
            continue
        if path.name in {"Dockerfile", ".dockerignore", ".env"} or path.name.startswith(".env."):
            continue
        if path.suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".zip", ".pdf", ".woff2", ".mp4"}:
            skipped.append(str(path.relative_to(root)) + " (binary)")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            skipped.append(str(path.relative_to(root)) + " (not text)")
            continue
        size = len(text.encode("utf-8"))
        if len(files) >= MAX_INLINE_FILES or total + size > MAX_INLINE_BYTES:
            skipped.append(str(path.relative_to(root)) + " (over inline budget)")
            continue
        total += size
        files.append({"file": str(path.relative_to(root)), "data": text})
    return files, skipped


class VercelProvider(DeployProvider):
    name = "vercel"
    label = "Vercel"
    requires_token = True
    supports_kinds = ("static", "nextjs", "node-web")

    def _token(self, ctx: DeployContext) -> str:
        return ctx.token("vercel")

    def available(self, ctx: DeployContext) -> Availability:
        if not self._token(ctx):
            return Availability(
                ok=False,
                reason="no Vercel token (set VERCEL_TOKEN)",
                fix="Create a token at https://vercel.com/account/tokens and export VERCEL_TOKEN=...",
            )
        if self.which("vercel"):
            return Availability(ok=True, reason="vercel CLI + token")
        if ctx.profile.kind == "static" or ctx.profile.build_cmd is None:
            return Availability(ok=True, reason="token (inline static upload)")
        return Availability(
            ok=False,
            reason="API path only supports small static uploads; this project needs a remote build",
            fix="npm i -g vercel && vercel login (then re-run with VERCEL_TOKEN set), or use --provider local",
        )

    def steps(self, ctx: DeployContext) -> list[DeployStep]:
        mode = "vercel CLI" if self.which("vercel") else "REST API"
        return [
            DeployStep("upload", f"Upload the project ({mode})", "", "push"),
            DeployStep("build", "Remote build on Vercel", "", "build"),
            DeployStep("release", "Promote the deployment to production", "", "release"),
            DeployStep("health", "Check the production URL", "", "health"),
        ]

    def deploy(self, ctx: DeployContext) -> ProviderResult:
        if self.which("vercel"):
            return self._deploy_cli(ctx)
        return self._deploy_api(ctx)

    # ------------------------------------------------------------------- CLI
    def _deploy_cli(self, ctx: DeployContext) -> ProviderResult:
        code, output = self.run(
            ["vercel", "deploy", "--prod", "--yes", "--token", self._token(ctx)],
            ctx.root,
            timeout=max(ctx.timeout, 900),
        )
        url = None
        for line in reversed(output.splitlines()):
            match = re.search(r"https://[a-zA-Z0-9.\-]+\.vercel\.app", line)
            if match:
                url = match.group(0)
                break
        if code != 0 or not url:
            return ProviderResult(
                ok=False,
                error=f"vercel CLI failed (exit {code}): {output.strip()[-500:]}",
                logs=[output[-4000:]],
                steps=self.steps(ctx),
                url=url,
            )
        health = wait_for_http(url, client=ctx.client(), timeout=min(ctx.timeout, 120), interval=4)
        return ProviderResult(
            ok=health.ok,
            url=url,
            health_url=url,
            public=True,
            detail="deployed with the vercel CLI",
            error="" if health.ok else health.detail,
            logs=[output[-2000:]],
            steps=self.steps(ctx),
            meta={"health": health.to_dict()},
        )

    # ------------------------------------------------------------------- API
    def _deploy_api(self, ctx: DeployContext) -> ProviderResult:
        files, skipped = collect_inline_files(Path(ctx.root))
        if not files:
            return ProviderResult(ok=False, error="no deployable text files found", steps=self.steps(ctx))
        if ctx.dry_run:
            return ProviderResult(
                ok=True,
                public=True,
                detail=f"dry run: would upload {len(files)} file(s)",
                steps=self.steps(ctx),
            )

        settings = ctx.config.settings
        query = ""
        if settings.get("VERCEL_TEAM_ID"):
            query = f"?teamId={settings['VERCEL_TEAM_ID']}"
        payload = {
            "name": ctx.config.name,
            "files": files,
            "target": "production",
            "projectSettings": {"framework": None},
        }
        headers = {"Authorization": f"Bearer {self._token(ctx)}"}
        response = ctx.client().post(
            f"https://api.vercel.com/v13/deployments{query}",
            json_body=payload,
            headers=headers,
            timeout=120,
        )
        if not response.ok:
            return ProviderResult(
                ok=False,
                error=f"Vercel API {response.status}: {response.body[:400]}",
                steps=self.steps(ctx),
            )
        data = response.json()
        deployment_id = data.get("id")
        url = data.get("url")
        if url and not url.startswith("http"):
            url = f"https://{url}"

        # Poll until the deployment leaves the BUILDING state (best effort).
        for _ in range(30):
            if not deployment_id:
                break
            status = ctx.client().get(
                f"https://api.vercel.com/v13/deployments/{deployment_id}{query}",
                headers=headers,
                timeout=60,
            )
            state = status.json().get("readyState", "")
            if state in {"READY", "ERROR", "CANCELED"}:
                if state != "READY":
                    return ProviderResult(
                        ok=False,
                        url=url,
                        public=True,
                        error=f"Vercel build ended in {state}",
                        steps=self.steps(ctx),
                        meta={"skipped": skipped},
                    )
                break

        if not url:
            return ProviderResult(ok=False, error="Vercel returned no URL", steps=self.steps(ctx))
        health = wait_for_http(url, client=ctx.client(), timeout=min(ctx.timeout, 120), interval=5)
        return ProviderResult(
            ok=health.ok,
            url=url,
            health_url=url,
            public=True,
            detail=f"uploaded {len(files)} file(s) via the REST API",
            error="" if health.ok else health.detail,
            steps=self.steps(ctx),
            meta={"health": health.to_dict(), "skipped": skipped, "deployment_id": deployment_id},
        )
