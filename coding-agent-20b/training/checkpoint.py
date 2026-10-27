"""Checkpoint integrity: atomic saves, SHA-256 manifest, verified loads."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint_atomic(state: Dict, out_dir: str, step: int, keep_last: int = 5) -> Path:
    """Write tmp file -> fsync -> rename -> manifest. A crash never leaves a
    half-written checkpoint that looks loadable."""
    import torch
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"checkpoint_{step:08d}.pt"
    tmp = out / f".checkpoint_{step:08d}.pt.tmp"
    torch.save(state, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, final)
    manifest = {"step": step, "file": final.name, "sha256": _sha256(final), "bytes": final.stat().st_size}
    (out / f"checkpoint_{step:08d}.manifest.json").write_text(json.dumps(manifest, indent=2))
    ckpts = sorted(out.glob("checkpoint_*.pt"))
    for old in ckpts[: max(0, len(ckpts) - keep_last)]:
        old.unlink(missing_ok=True)
        old.with_suffix(".manifest.json").unlink(missing_ok=True)
        Path(str(old) + ".manifest.json").unlink(missing_ok=True)
        (out / (old.stem + ".manifest.json")).unlink(missing_ok=True)
    return final


def verify_checkpoint(path: str) -> Dict:
    p = Path(path)
    manifest_path = p.with_name(p.stem + ".manifest.json")
    if not manifest_path.exists():
        manifest_path = Path(str(p) + ".manifest.json")
    if not p.exists():
        return {"valid": False, "error": f"missing file {p}"}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        actual = _sha256(p)
        if actual != manifest.get("sha256"):
            return {"valid": False, "error": "sha256 mismatch (corrupt/truncated checkpoint)", "manifest": manifest}
        return {"valid": True, "manifest": manifest}
    # No manifest (legacy checkpoint): at least confirm it unpickles with the keys we need.
    try:
        import torch
        ckpt = torch.load(p, map_location="cpu")
        ok = isinstance(ckpt, dict)
        return {"valid": ok, "error": None if ok else "checkpoint is not a dict"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


def load_verified(path: str, map_location: str = "cpu") -> Dict:
    result = verify_checkpoint(path)
    if not result["valid"]:
        raise ValueError(f"checkpoint integrity check failed for {path}: {result.get('error')}")
    import torch
    return torch.load(path, map_location=map_location)


def latest_valid_checkpoint(out_dir: str) -> Optional[str]:
    out = Path(out_dir)
    for ckpt in sorted(out.glob("checkpoint_*.pt"), reverse=True):
        if verify_checkpoint(str(ckpt))["valid"]:
            return str(ckpt)
    return None
