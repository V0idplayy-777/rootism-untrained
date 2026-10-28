"""Provider interface shared by every deploy target.

A provider must be honest about three things:

1. ``available()`` - can it run *here* (binary present, token present, repo
   configured)? If not, say exactly what is missing and how to fix it.
2. ``steps()`` - what will be created, in order, before anything happens. The
   same list becomes the dry-run plan and the live step log.
3. ``deploy()`` - what actually happened. ``ok=False`` with a concrete ``error``
   is always better than a fake URL.

Providers receive a :class:`DeployContext` and never read ``os.environ`` for
secrets themselves: the engine resolves credentials once and hands them over,
so redaction and testing stay in one place.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import PROVIDER_ENV, DeployConfig
from ..detect import ProjectProfile
from ..http import HttpClient
from ..plan import DeployStep, ProviderResult


@dataclass
class Availability:
    ok: bool
    reason: str = ""
    fix: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reason": self.reason, "fix": self.fix}


@dataclass
class DeployContext:
    root: Path
    profile: ProjectProfile
    config: DeployConfig
    image: str
    tag: str = "latest"
    dry_run: bool = False
    http: HttpClient | None = None
    emit: Callable[[str], None] = lambda message: None
    secrets: dict[str, str] = field(default_factory=dict)
    runtime_env: dict[str, str] = field(default_factory=dict)
    environ: dict[str, str] = field(default_factory=dict)
    timeout: float = 900.0

    @property
    def port(self) -> int:
        return int(self.config.port or self.profile.port or 8000)

    @property
    def health_path(self) -> str:
        return (self.config.health_path or self.profile.health_path or "/") or "/"

    def client(self) -> HttpClient:
        return self.http or HttpClient(secrets=self.secrets)

    def setting(self, key: str, default: str = "") -> str:
        """A non-secret provider setting (config first, then environment)."""

        return str(
            self.config.settings.get(key)
            or self.environ.get(key)
            or default
        )

    def token(self, provider_name: str | None = None) -> str:
        """Resolve a provider token from the resolved secret mapping."""

        for key in PROVIDER_ENV.get(provider_name or self.config.provider, ()):
            value = self.secrets.get(key)
            if value:
                return str(value)
        return ""


class DeployProvider(ABC):
    name: str = ""
    label: str = ""
    requires_token: bool = False
    supports_kinds: tuple[str, ...] = ("python-web", "node-web", "nextjs", "static", "worker", "cli")

    @abstractmethod
    def available(self, ctx: DeployContext) -> Availability:
        ...

    @abstractmethod
    def steps(self, ctx: DeployContext) -> list[DeployStep]:
        ...

    @abstractmethod
    def deploy(self, ctx: DeployContext) -> ProviderResult:
        ...

    # ------------------------------------------------------------------ utils
    def supports(self, profile: ProjectProfile) -> bool:
        return profile.kind in self.supports_kinds

    def secret_values(self, ctx: DeployContext) -> dict[str, str]:
        return dict(ctx.secrets)

    @staticmethod
    def run(
        command: list[str],
        cwd: Path | str,
        timeout: float = 900.0,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        """Run a subprocess, returning (returncode, combined output)."""

        import os
        import subprocess

        merged = dict(os.environ)
        merged.update(env or {})
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=merged,
            )
            return completed.returncode, (completed.stdout or "") + (completed.stderr or "")
        except FileNotFoundError:
            return 127, f"command not found: {command[0]}"
        except subprocess.TimeoutExpired:
            return 124, f"timed out after {timeout:.0f}s: {' '.join(command)}"

    @staticmethod
    def which(binary: str) -> str | None:
        return shutil.which(binary)
