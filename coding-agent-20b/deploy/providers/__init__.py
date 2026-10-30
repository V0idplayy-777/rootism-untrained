"""Provider registry and automatic selection.

``select_provider`` prefers the provider named in ``agentdeploy.toml``, then a
sensible free default for the project kind, and always falls back to the
``archive`` provider so a deploy attempt produces a portable artifact instead of
nothing. Selection returns the reasons it chose what it chose - the CLI shows
them, so "why did it pick local?" is never a mystery.
"""

from __future__ import annotations

from .archive import ArchiveProvider
from .base import Availability, DeployContext, DeployProvider
from .coolify import CoolifyProvider
from .fly import FlyProvider
from .github_pages import GitHubPagesProvider
from .local import LocalDockerProvider
from .process import LocalProcessProvider
from .render import RenderProvider
from .vercel import VercelProvider

PROVIDERS: dict[str, type[DeployProvider]] = {
    "local": LocalDockerProvider,
    "process": LocalProcessProvider,
    "vercel": VercelProvider,
    "fly": FlyProvider,
    "render": RenderProvider,
    "coolify": CoolifyProvider,
    "github-pages": GitHubPagesProvider,
    "archive": ArchiveProvider,
}

PROVIDER_ORDER = ["local", "vercel", "github-pages", "fly", "render", "coolify", "archive"]

PROVIDER_NOTES = {
    "local": "no account needed; container runs on this machine (localhost URL)",
    "process": "no Docker needed; runs the detected start command on this machine (localhost URL)",
    "vercel": "free tier; best for static/Next.js; needs VERCEL_TOKEN",
    "github-pages": "free public URL for static sites; uses the repo's git remote",
    "fly": "free allowance; containers; needs flyctl + a token",
    "render": "free Docker web service from a git remote; needs RENDER_API_KEY",
    "coolify": "self-hosted PaaS; set COOLIFY_URL/COOLIFY_TOKEN/COOLIFY_APP_UUID",
    "archive": "always works; portable .tar.gz + Dockerfile, no live URL",
}


def get_provider(name: str) -> DeployProvider:
    try:
        return PROVIDERS[name]()
    except KeyError as exc:
        raise ValueError(f"unknown provider {name!r}; choose from {', '.join(PROVIDERS)}") from exc


def available_providers(ctx: DeployContext) -> dict[str, Availability]:
    return {name: get_provider(name).available(ctx) for name in PROVIDER_ORDER}


def select_provider(ctx: DeployContext, requested: str | None = None) -> tuple[DeployProvider, list[str]]:
    """Choose a provider; return ``(provider, warnings)``."""

    warnings: list[str] = []
    name = (requested or ctx.config.provider or "auto").lower()
    if name not in {"", "auto"}:
        provider = get_provider(name)
        availability = provider.available(ctx)
        if availability.ok:
            return provider, warnings
        warnings.append(
            f"{name} is not usable here ({availability.reason})"
            + (f"; fix: {availability.fix}" if availability.fix else "")
        )

    kind = ctx.profile.kind
    if kind == "static":
        order = ["vercel", "github-pages", "local", "fly", "coolify", "render", "archive"]
    elif kind == "worker":
        order = ["fly", "render", "local", "coolify", "archive"]
    else:
        order = ["fly", "render", "coolify", "vercel", "local", "archive"]

    for candidate in order:
        provider = get_provider(candidate)
        if not provider.supports(ctx.profile):
            continue
        availability = provider.available(ctx)
        if availability.ok:
            return provider, warnings
        warnings.append(f"skipped {candidate}: {availability.reason}")

    return get_provider("archive"), warnings


__all__ = [
    "PROVIDERS",
    "PROVIDER_ORDER",
    "PROVIDER_NOTES",
    "Availability",
    "DeployContext",
    "DeployProvider",
    "available_providers",
    "get_provider",
    "select_provider",
]
