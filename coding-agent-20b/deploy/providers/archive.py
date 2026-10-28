"""Fallback provider: produce a portable build artifact (and never pretend).

When no host is configured, ``agent deploy`` still does something useful: it
generates the Docker artifacts and a reproducible tarball plus a one-page
portable ``.tar.gz`` that works with any host. ``ok=True`` because the artifact *is*
the deliverable, but ``url=None`` and ``public=False`` so nothing downstream
claims a live site.
"""

from __future__ import annotations

import hashlib
import tarfile
import time
from pathlib import Path

from ..dockerfile import write_artifacts
from ..plan import DeployStep, ProviderResult
from .base import Availability, DeployContext, DeployProvider

EXCLUDE = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".agent",
    "dist",
    "build",
    "coverage",
    ".next",
}


def build_archive(root: Path, name: str, out_dir: Path | None = None) -> tuple[Path, str, int]:
    root = Path(root)
    out_dir = out_dir or (root / "deploy")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = out_dir / f"{name}-{stamp}.tar.gz"

    def filt(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if any(part in EXCLUDE for part in parts):
            return None
        if info.name.endswith((".pyc", ".log")):
            return None
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        return info

    with tarfile.open(target, "w:gz") as tar:
        tar.add(root, arcname=name, filter=filt)

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return target, digest, target.stat().st_size


class ArchiveProvider(DeployProvider):
    name = "archive"
    label = "Portable build artifact (.tar.gz + Dockerfile)"
    supports_kinds = ("python-web", "node-web", "nextjs", "static", "worker", "cli", "unknown")

    def available(self, ctx: DeployContext) -> Availability:
        return Availability(ok=True, reason="always available (no account, no network)")

    def steps(self, ctx: DeployContext) -> list[DeployStep]:
        return [
            DeployStep("artifacts", "Generate Dockerfile, .dockerignore and docker-compose.yml", "", "build"),
            DeployStep("archive", "Pack the project into a reproducible .tar.gz", "deploy/", "build"),
            DeployStep("instructions", "Record the artifact digest for DEPLOY.md/HOSTING.md", "", "report"),
        ]

    def deploy(self, ctx: DeployContext) -> ProviderResult:
        notes: list[str] = []
        if ctx.profile.kind == "unknown":
            written = {"written": [], "skipped": [], "image": ctx.image, "notes": []}
            notes.append("no Dockerfile: project type is unknown (set [deploy] start_cmd to get one)")
        else:
            try:
                written = write_artifacts(ctx.root, ctx.profile, ctx.config, tag=ctx.tag)
            except ValueError as exc:
                written = {"written": [], "skipped": [], "image": ctx.image, "notes": []}
                notes.append(f"no Dockerfile: {exc}")
        path, digest, size = build_archive(Path(ctx.root), ctx.config.name)
        rel = str(path.relative_to(Path(ctx.root)))
        return ProviderResult(
            ok=True,
            url=None,
            public=False,
            detail=f"artifact ready: {rel} ({size / 1024:.0f} KiB, sha256 {digest[:16]}...)",
            artifacts={
                **written,
                "archive": rel,
                "sha256": digest,
                "size_bytes": size,
            },
            steps=self.steps(ctx),
            meta={"archive": rel, "sha256": digest},
            logs=notes,
        )
