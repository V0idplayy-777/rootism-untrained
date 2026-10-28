"""Live preview for the workspace, proxied through the UI server.

The browser talks only to the UI (same origin); the UI talks to the project:

* **static** projects: files are served from the detected static directory;
* **server** projects: the detected start command is launched on a free
  localhost port and HTTP requests are proxied to it. The UI never exposes the
  child process port directly, and stops it on request or at exit.

If the app cannot start, ``start()`` returns the captured log tail instead of
pretending the preview works.
"""

from __future__ import annotations

import os
import signal
import socket
import sys
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

IGNORE = {".git", ".agent", "node_modules", ".venv", "__pycache__", ".pytest_cache", "data"}
MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
    ".woff2": "font/woff2",
}


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class PreviewState:
    running: bool = False
    kind: str = "none"
    port: int | None = None
    pid: int | None = None
    detail: str = ""
    logs: list[str] = field(default_factory=list)
    started_at: float = 0.0
    static_dir: str | None = None

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "kind": self.kind,
            "port": self.port,
            "pid": self.pid,
            "detail": self.detail,
            "logs": self.logs[-40:],
            "started_at": self.started_at,
            "static_dir": self.static_dir,
        }


def _prepend_interpreter(path: str) -> str:
    """Make the running interpreter's ``bin`` dir visible to child processes.

    ``uvicorn``/``pytest``/``flask`` are usually installed into the same venv as
    this process, so spawning ``sh -c "uvicorn ..."`` fails unless that ``bin``
    directory is on ``PATH`` (a bare system PATH does not include it).
    """

    bindir = str(Path(sys.executable).parent)
    parts = [part for part in path.split(os.pathsep) if part]
    if bindir in parts:
        return path
    return os.pathsep.join([bindir] + parts)


class PreviewManager:
    def __init__(self, root: str | Path, wait_seconds: float = 20.0):
        self.root = Path(root).resolve()
        self.wait_seconds = wait_seconds
        self.state = PreviewState()
        self._process: subprocess.Popen | None = None
        self._log_path = self.root / ".agent" / "preview.log"

    def start(self) -> PreviewState:
        from deploy.detect import detect_project

        self.stop()
        profile = detect_project(self.root)
        if profile.kind == "static":
            static_dir = self.root / (profile.static_dir or ".")
            if not (static_dir / "index.html").exists():
                self.state = PreviewState(detail="no index.html to preview")
                return self.state
            self.state = PreviewState(
                running=True, kind="static", detail=f"serving {static_dir.relative_to(self.root)}", static_dir=str(static_dir)
            )
            return self.state

        if not profile.start_cmd:
            self.state = PreviewState(detail="no start command detected; preview is unavailable")
            return self.state

        port = profile.port or free_port()
        env = dict(os.environ)
        env["PATH"] = _prepend_interpreter(env.get("PATH", ""))
        env.setdefault("PORT", str(port))
        env.setdefault("HOST", "0.0.0.0")
        env.setdefault("FLASK_RUN_PORT", str(port))
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self._log_path, "ab")  # noqa: SIM115 - handed to the child process
        try:
            self._process = subprocess.Popen(
                ["sh", "-c", profile.start_cmd],
                cwd=str(self.root),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            self.state = PreviewState(detail=f"could not start: {exc}")
            return self.state
        started = time.time()
        healthy = self._wait_for(port)
        self.state = PreviewState(
            running=healthy,
            kind="process",
            port=port,
            pid=self._process.pid,
            detail=("running " + profile.start_cmd) if healthy else "process started but the port never opened",
            started_at=started,
            logs=self._logs(),
        )
        if not healthy:
            self.stop()
            self.state.running = False
            self.state.detail = "process started but did not answer on its port"
        return self.state

    def _wait_for(self, port: int) -> bool:
        deadline = time.monotonic() + self.wait_seconds
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                return False
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    return True
            except OSError:
                time.sleep(0.4)
        return False

    def _logs(self) -> list[str]:
        if not self._log_path.exists():
            return []
        try:
            return self._log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        except OSError:
            return []

    def stop(self) -> PreviewState:
        if self._process is not None and self._process.poll() is None:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except (OSError, AttributeError):
                try:
                    self._process.terminate()
                except OSError:
                    pass
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                except OSError:
                    pass
        self._process = None
        if self.state.running:
            self.state = PreviewState(detail="stopped", logs=self.state.logs)
        return self.state

    def status(self) -> dict:
        if self._process is not None and self._process.poll() is not None:
            self.state.running = False
            self.state.detail = "the preview process exited; check the logs"
        return self.state.to_dict()

    def static_file(self, path: str) -> tuple[int, str, bytes] | None:
        """Return (status, mime, body) for a static preview request."""

        if not self.state.static_dir:
            return None
        base = Path(self.state.static_dir).resolve()
        relative = path.lstrip("/") or "index.html"
        target = (base / relative).resolve()
        if not str(target).startswith(str(base)):
            return 404, "text/plain; charset=utf-8", b"not found"
        if target.is_dir():
            target = target / "index.html"
        if not target.exists():
            index = base / "index.html"
            if index.exists():
                return 200, "text/html; charset=utf-8", index.read_bytes()
            return 404, "text/plain; charset=utf-8", b"not found"
        return 200, MIME.get(target.suffix.lower(), "application/octet-stream"), target.read_bytes()

    def proxy(self, path: str, method: str = "GET", body: bytes | None = None, headers: dict | None = None) -> tuple[int, str, bytes]:
        """Forward one request to the preview process (localhost only)."""

        if not self.state.running or not self.state.port:
            return 503, "text/plain; charset=utf-8", b"preview is not running"
        url = f"http://127.0.0.1:{self.state.port}/{path.lstrip('/')}"
        outgoing = {k: v for k, v in (headers or {}).items() if k.lower() in {"content-type", "accept", "authorization"}}
        request = urllib.request.Request(url, data=body, headers=outgoing, method=method)
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return (
                    response.status,
                    response.headers.get("content-type", "application/octet-stream"),
                    response.read(),
                )
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers.get("content-type", "text/plain"), exc.read()
        except Exception as exc:  # noqa: BLE001 - preview errors are shown, not raised
            return 502, "text/plain; charset=utf-8", f"preview proxy error: {exc}".encode()


def list_workspace_files(root: str | Path, max_files: int = 400) -> list[dict]:
    root = Path(root)
    out: list[dict] = []
    for path in sorted(root.rglob("*")):
        if len(out) >= max_files:
            break
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in IGNORE for part in rel.parts[:-1]):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        out.append({"path": str(rel), "size": size})
    return out


def read_workspace_file(root: str | Path, path: str, max_bytes: int = 200_000) -> dict:
    root = Path(root).resolve()
    target = (root / path).resolve()
    if not str(target).startswith(str(root)):
        return {"ok": False, "error": "path escapes the workspace"}
    if not target.is_file():
        return {"ok": False, "error": "not a file"}
    data = target.read_bytes()[:max_bytes]
    try:
        text = data.decode("utf-8")
        binary = False
    except UnicodeDecodeError:
        text = ""
        binary = True
    return {"ok": True, "path": path, "content": text, "binary": binary, "size": target.stat().st_size}
