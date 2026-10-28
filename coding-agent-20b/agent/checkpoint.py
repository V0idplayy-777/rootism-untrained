"""Workspace checkpoints: snapshot before a risky change, roll back if it sinks.

``MilestoneCheckpointer`` (in ``planning.py``) commits *verified* progress to git.
This module covers the other case: the working tree is dirty, git history is not
something you want to rewrite, and you need "undo the last 20 minutes" without
losing unrelated edits.

* Content-addressed blobs (sha256) under ``.agent/checkpoints/blobs`` mean
  repeated snapshots cost almost nothing.
* ``create`` records the current tree; ``restore`` rewrites files, deletes files
  that were added since, and leaves ignored/vendored trees (``.git``,
  ``node_modules``, ``.agent``, caches) alone.
* ``diff`` reports what changed since a snapshot so the agent can decide whether
  a rollback is the right move before taking it.
* ``prune`` keeps disk usage bounded (default: 10 snapshots).

Pure stdlib, deterministic, and safe to call from a tool or the CLI.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".agent",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    ".nuxt",
    "dist",
    "build",
    "coverage",
    ".turbo",
    ".idea",
}
IGNORE_SUFFIX = (".pyc", ".pyo", ".log", ".tmp", ".swp")
MAX_FILE_BYTES = 8 * 1024 * 1024


@dataclass
class Snapshot:
    id: str
    label: str
    created_at: float
    files: dict[str, str] = field(default_factory=dict)  # rel path -> blob sha
    total_bytes: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "created_at": self.created_at,
            "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_at)),
            "files": dict(self.files),
            "file_count": len(self.files),
            "total_bytes": self.total_bytes,
        }

    @property
    def name(self) -> str:
        return f"{time.strftime('%H:%M:%S', time.localtime(self.created_at))} {self.label}".strip()


class WorkspaceCheckpoints:
    def __init__(self, root: str | Path, max_snapshots: int = 10):
        self.root = Path(root).resolve()
        self.store = self.root / ".agent" / "checkpoints"
        self.blobs = self.store / "blobs"
        self.max_snapshots = max_snapshots

    # ------------------------------------------------------------------ create
    def create(self, label: str = "") -> Snapshot:
        self.blobs.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        total = 0
        for rel, path in self._walk():
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if len(data) > MAX_FILE_BYTES:
                continue
            digest = hashlib.sha256(data).hexdigest()
            blob = self.blobs / digest
            if not blob.exists():
                blob.write_bytes(data)
            files[rel] = digest
            total += len(data)
        snapshot = Snapshot(
            id=hashlib.sha256(
                (str(time.time()) + label + json.dumps(sorted(files.items()))).encode()
            ).hexdigest()[:12],
            label=label or "auto",
            created_at=time.time(),
            files=files,
            total_bytes=total,
        )
        self.store.mkdir(parents=True, exist_ok=True)
        (self.store / f"{snapshot.id}.json").write_text(
            json.dumps(snapshot.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        self.prune()
        return snapshot

    # -------------------------------------------------------------------- read
    def list(self) -> list[Snapshot]:
        if not self.store.exists():
            return []
        snapshots: list[Snapshot] = []
        for path in sorted(self.store.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            snapshots.append(
                Snapshot(
                    id=data["id"],
                    label=data.get("label", ""),
                    created_at=float(data.get("created_at", 0)),
                    files=dict(data.get("files") or {}),
                    total_bytes=int(data.get("total_bytes") or 0),
                )
            )
        return sorted(snapshots, key=lambda s: s.created_at)

    def latest(self) -> Snapshot | None:
        snapshots = self.list()
        return snapshots[-1] if snapshots else None

    def get(self, snapshot_id: str) -> Snapshot:
        matches = [snap for snap in self.list() if snap.id.startswith(snapshot_id)]
        if not matches:
            raise KeyError(f"no checkpoint matching {snapshot_id!r}")
        if len(matches) > 1:
            raise KeyError(f"ambiguous checkpoint prefix {snapshot_id!r}: {[s.id for s in matches]}")
        return matches[0]

    # -------------------------------------------------------------------- diff
    def diff(self, snapshot_id: str | None = None) -> dict:
        snapshot = self.get(snapshot_id) if snapshot_id else self.latest()
        if snapshot is None:
            return {"added": [], "removed": [], "modified": [], "snapshot": None}
        current = {rel: hashlib.sha256(path.read_bytes()).hexdigest() for rel, path in self._walk() if _readable(path)}
        added = sorted(set(current) - set(snapshot.files))
        removed = sorted(set(snapshot.files) - set(current))
        modified = sorted(
            rel for rel in set(current) & set(snapshot.files) if current[rel] != snapshot.files[rel]
        )
        return {
            "snapshot": snapshot.to_dict(),
            "added": added,
            "removed": removed,
            "modified": modified,
            "changed": len(added) + len(removed) + len(modified),
        }

    # ----------------------------------------------------------------- restore
    def restore(self, snapshot_id: str | None = None, delete_extra: bool = True) -> dict:
        """Rewrite the workspace to a snapshot. Returns what changed."""

        snapshot = self.get(snapshot_id) if snapshot_id else self.latest()
        if snapshot is None:
            raise KeyError("no checkpoints to restore")
        before = self.diff(snapshot.id)
        restored: list[str] = []
        for rel, digest in snapshot.files.items():
            blob = self.blobs / digest
            if not blob.exists():
                continue
            target = self.root / rel
            try:
                if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
                    continue
            except OSError:
                pass
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob.read_bytes())
            restored.append(rel)
        deleted: list[str] = []
        if delete_extra:
            for rel in before["added"]:
                target = self.root / rel
                try:
                    if target.is_file():
                        target.unlink()
                        deleted.append(rel)
                except OSError:
                    continue
        return {
            "snapshot": snapshot.to_dict(),
            "restored": restored,
            "deleted": deleted,
            "was": {key: before[key] for key in ("added", "removed", "modified")},
        }

    # ------------------------------------------------------------------ upkeep
    def prune(self, keep: int | None = None) -> int:
        keep = keep if keep is not None else self.max_snapshots
        snapshots = self.list()
        removed = 0
        for snapshot in snapshots[:-keep] if keep > 0 else snapshots:
            try:
                (self.store / f"{snapshot.id}.json").unlink()
                removed += 1
            except OSError:
                continue
        self._gc_blobs()
        return removed

    def _gc_blobs(self) -> None:
        if not self.blobs.exists():
            return
        referenced: set[str] = set()
        for snapshot in self.list():
            referenced.update(snapshot.files.values())
        for blob in self.blobs.iterdir():
            if blob.is_file() and blob.name not in referenced:
                try:
                    blob.unlink()
                except OSError:
                    continue

    # ------------------------------------------------------------------ helpers
    def _walk(self):
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root)
            if any(part in IGNORE_DIRS for part in rel.parts[:-1]):
                continue
            if path.name.endswith(IGNORE_SUFFIX):
                continue
            yield str(rel), path

    def stats(self) -> dict:
        snapshots = self.list()
        blob_count = len(list(self.blobs.iterdir())) if self.blobs.exists() else 0
        blob_bytes = sum(p.stat().st_size for p in self.blobs.iterdir() if p.is_file()) if self.blobs.exists() else 0
        return {
            "snapshots": len(snapshots),
            "blobs": blob_count,
            "blob_bytes": blob_bytes,
            "latest": snapshots[-1].to_dict() if snapshots else None,
        }


def _readable(path: Path) -> bool:
    try:
        return path.stat().st_size <= MAX_FILE_BYTES
    except OSError:
        return False


def create_checkpoint(root: str | Path, label: str = "") -> Snapshot:
    return WorkspaceCheckpoints(root).create(label)


def rollback(root: str | Path, snapshot_id: str | None = None, delete_extra: bool = True) -> dict:
    return WorkspaceCheckpoints(root).restore(snapshot_id, delete_extra=delete_extra)
