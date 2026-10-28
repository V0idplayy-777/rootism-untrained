"""Coolify provider (self-hosted PaaS, a good no-lock-in default for teams).

Coolify exposes a small REST API: trigger a deployment for an existing
application UUID and poll the deployment status. The application itself
(git or docker-compose source) is created once in the Coolify UI, so credentials
here are only a token + base URL + UUID, all read from the environment/config.
"""

from __future__ import annotations

import time

from ..plan import DeployStep, ProviderResult
from .base import Availability, DeployContext, DeployProvider


class CoolifyProvider(DeployProvider):
    name = "coolify"
    label = "Coolify (self-hosted)"
    requires_token = True
    supports_kinds = ("python-web", "node-web", "nextjs", "static", "worker")

    def _base(self, ctx: DeployContext) -> str:
        return (ctx.config.settings.get("COOLIFY_URL") or "").rstrip("/")

    def _uuid(self, ctx: DeployContext) -> str:
        return ctx.config.settings.get("COOLIFY_APP_UUID", "")

    def available(self, ctx: DeployContext) -> Availability:
        if not ctx.token("coolify"):
            return Availability(ok=False, reason="no Coolify token (set COOLIFY_TOKEN)")
        if not self._base(ctx):
            return Availability(
                ok=False,
                reason="COOLIFY_URL is not set",
                fix="export COOLIFY_URL=https://coolify.example.com",
            )
        if not self._uuid(ctx):
            return Availability(
                ok=False,
                reason="COOLIFY_APP_UUID is not set (create the app in Coolify once)",
                fix="export COOLIFY_APP_UUID=<application uuid>",
            )
        return Availability(ok=True, reason=self._base(ctx))

    def steps(self, ctx: DeployContext) -> list[DeployStep]:
        return [
            DeployStep("trigger", "Trigger a Coolify deployment", "POST /api/v1/deploy", "release"),
            DeployStep("wait", "Wait for the deployment to finish", "GET /api/v1/deployments/{uuid}", "release"),
            DeployStep("health", "Check the application URL", ctx.health_path, "health"),
        ]

    def deploy(self, ctx: DeployContext) -> ProviderResult:
        headers = {"Authorization": f"Bearer {ctx.token('coolify')}", "Accept": "application/json"}
        if ctx.dry_run:
            return ProviderResult(
                ok=True,
                public=True,
                detail=f"dry run: would deploy app {self._uuid(ctx)} on {self._base(ctx)}",
                steps=self.steps(ctx),
            )
        response = ctx.client().post(
            f"{self._base(ctx)}/api/v1/deploy?uuid={self._uuid(ctx)}",
            json_body={},
            headers=headers,
            timeout=90,
        )
        if not response.ok:
            return ProviderResult(
                ok=False,
                error=f"Coolify API {response.status}: {response.body[:300]}",
                steps=self.steps(ctx),
            )
        deployment_uuid = response.json().get("deployments", [{}])[0].get("deployment_uuid") or response.json().get(
            "deployment_uuid"
        )
        status = "queued"
        for _ in range(60):
            if not deployment_uuid:
                break
            poll = ctx.client().get(
                f"{self._base(ctx)}/api/v1/deployments/{deployment_uuid}", headers=headers, timeout=60
            )
            status = str(poll.json().get("status", status)).lower()
            if status in {"finished", "failed", "cancelled", "cancelled-by-user"}:
                break
            time.sleep(5)
        url = ctx.config.settings.get("COOLIFY_APP_URL")
        ok = status in {"finished", "running", "queued", "in_progress"}
        return ProviderResult(
            ok=ok,
            url=url,
            health_url=(url or "") + ctx.health_path if url else None,
            public=True,
            detail=f"Coolify deployment {deployment_uuid or '?'} status={status}",
            error="" if ok else f"Coolify deployment ended in {status}",
            steps=self.steps(ctx),
            meta={"deployment_uuid": deployment_uuid, "status": status},
        )
