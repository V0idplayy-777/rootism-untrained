"""Render provider (Docker web service from a git repository).

Render builds from a git remote, so this path requires the project to be pushed.
The provider creates (or reuses) a Docker web service, triggers a deploy and
returns the service URL. It is explicitly marked experimental because Render's
service payload has changed over time; failures surface the API response instead
of pretending success.
"""

from __future__ import annotations

from ..plan import DeployStep, ProviderResult
from .base import Availability, DeployContext, DeployProvider

API = "https://api.render.com/v1"


class RenderProvider(DeployProvider):
    name = "render"
    label = "Render (Docker web service)"
    requires_token = True
    experimental = True
    supports_kinds = ("python-web", "node-web", "nextjs", "worker")

    def _token(self, ctx: DeployContext) -> str:
        return ctx.token("render")

    def available(self, ctx: DeployContext) -> Availability:
        if not self._token(ctx):
            return Availability(
                ok=False,
                reason="no Render API key (set RENDER_API_KEY)",
                fix="Create a key at https://dashboard.render.com/u/settings#api-keys",
            )
        if not ctx.config.repo_url:
            return Availability(
                ok=False,
                reason="Render deploys from a git remote and none is configured",
                fix="git remote add origin <repo> && git push -u origin HEAD, then re-run "
                "(or use --provider local to deploy the image from this machine)",
            )
        return Availability(ok=True, reason=f"repo {ctx.config.repo_url}")

    def steps(self, ctx: DeployContext) -> list[DeployStep]:
        return [
            DeployStep("service", "Create or reuse the Render web service", "POST /v1/services", "release"),
            DeployStep("deploy", "Trigger a deploy of the current commit", "POST /v1/services/{id}/deploys", "release"),
            DeployStep("health", "Check the service URL", ctx.health_path, "health"),
        ]

    def _headers(self, ctx: DeployContext) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token(ctx)}", "Accept": "application/json"}

    def deploy(self, ctx: DeployContext) -> ProviderResult:
        client = ctx.client()
        owner = ctx.config.settings.get("RENDER_OWNER_ID", "")
        payload = {
            "type": "web_service",
            "name": ctx.config.name,
            "ownerId": owner,
            "repo": ctx.config.repo_url,
            "branch": "main",
            "autoDeploy": "yes",
            "serviceDetails": {
                "env": "docker",
                "plan": "free",
                "region": ctx.config.region or "oregon",
                "healthCheckPath": ctx.health_path,
                "envSpecificDetails": {
                    "dockerfilePath": "./Dockerfile",
                    "dockerContext": "./",
                },
            },
        }
        if ctx.dry_run:
            return ProviderResult(
                ok=True,
                public=True,
                detail="dry run: would create a Render Docker web service and deploy",
                steps=self.steps(ctx),
            )
        response = client.post(f"{API}/services", json_body=payload, headers=self._headers(ctx), timeout=90)
        if response.status == 402:
            return ProviderResult(
                ok=False,
                error="Render rejected the service (payment required / plan limits)",
                steps=self.steps(ctx),
            )
        if not response.ok:
            return ProviderResult(
                ok=False,
                error=f"Render API {response.status}: {response.body[:400]}",
                steps=self.steps(ctx),
            )
        service = response.json().get("service", response.json())
        service_id = service.get("id")
        url = service.get("serviceDetails", {}).get("url") or service.get("url")
        if not service_id:
            return ProviderResult(ok=False, error="Render returned no service id", steps=self.steps(ctx))

        deploy = client.post(
            f"{API}/services/{service_id}/deploys",
            json_body={"clearCache": "do_not_clear"},
            headers=self._headers(ctx),
            timeout=90,
        )
        if not deploy.ok:
            return ProviderResult(
                ok=False,
                url=url,
                error=f"Render deploy trigger failed {deploy.status}: {deploy.body[:300]}",
                steps=self.steps(ctx),
            )
        return ProviderResult(
            ok=bool(url),
            url=url,
            health_url=(url or "") + ctx.health_path if url else None,
            public=True,
            detail="Render is building the service; the first build takes a few minutes",
            error="" if url else "service created but no URL was returned",
            steps=self.steps(ctx),
            meta={"service_id": service_id, "owner_id": owner, "experimental": True},
        )
