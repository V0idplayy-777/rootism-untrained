"""Run a python-web project as a local process (no Docker required).

This is the honest fallback when the environment has no container runtime but
does have the project's dependencies installed: it starts the detected start
command on a free port, watches ``/health`` (or the configured path), and
reports the URL only after the health check passes. It does *not* claim to be a
public deployment - the URL is localhost - but it is a real running app, which
is what makes the idea -> running-app loop demonstrable without credentials.
"""

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path

from ..health import wait_for_http
from ..plan import DeployStep, ProviderResult
from .base import Availability, DeployContext, DeployProvider


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class LocalProcessProvider(DeployProvider):
    name = "process"
    label = "Local process (no container)"
    supports_kinds = ("python-web",)

    def available(self, ctx: DeployContext) -> Availability:
        if ctx.profile.kind != "python-web":
            return Availability(
                ok=False,
                reason=f"only python-web projects are supported here (detected {ctx.profile.kind})",
                fix="Use --provider local (Docker) or a hosted provider for this project.",
            )
        if not ctx.profile.start_cmd:
            return Availability(
                ok=False,
                reason="the project has no detected start command",
                fix="Add start_cmd to agentdeploy.toml, or add an ASGI app.py / main.py.",
            )
        return Availability(ok=True, reason="python available; starts the detected start command")

    def steps(self, ctx: DeployContext) -> list[DeployStep]:
        return [
            DeployStep("start", "Start the app on a free local port", ctx.profile.start_cmd or "", "release"),
            DeployStep("health", "Wait for the health endpoint", ctx.health_path, "health"),
        ]

    def stop(self, ctx: DeployContext) -> tuple[bool, str]:
        pid_file = ctx.root / ".deploy" / "process.pid"
        if not pid_file.exists():
            return False, "no recorded process"
        try:
            pid = int(pid_file.read_text().strip())
        except ValueError:
            return False, "invalid pid file"
        if _alive(pid):
            try:
                group = os.getpgid(pid)
                os.killpg(group, signal.SIGTERM)
                time.sleep(0.5)
                if _alive(pid):
                    os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        pid_file.unlink(missing_ok=True)
        return True, f"stopped pid {pid}"

    def deploy(self, ctx: DeployContext) -> ProviderResult:
        logs: list[str] = []
        state = ctx.root / ".deploy"
        state.mkdir(parents=True, exist_ok=True)
        pid_file = state / "process.pid"
        log_file = state / "process.log"

        if pid_file.exists():
            self.stop(ctx)
            logs.append("stopped the previous process")

        if ctx.dry_run:
            return ProviderResult(
                ok=True,
                detail="dry run: would start " + (ctx.profile.start_cmd or ""),
                logs=logs,
                steps=self.steps(ctx),
            )

        port = _free_port()
        start = (ctx.profile.start_cmd or "").replace("${PORT}", str(port)).replace("$PORT", str(port))
        # Detected commands often hard-code the project port; move it to ours.
        start = re.sub(r"--port\s+\d+", f"--port {port}", start)
        env = dict(os.environ)
        env.update(ctx.runtime_env)
        env["PORT"] = str(port)
        handle = log_file.open("wb")
        try:
            proc = subprocess.Popen(
                ["bash", "-o", "pipefail", "-lc", start],
                cwd=str(ctx.root),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            handle.close()
            return ProviderResult(
                ok=False,
                error=f"could not start process: {exc}",
                logs=logs,
                steps=self.steps(ctx),
            )
        pid_file.write_text(str(proc.pid), encoding="utf-8")
        logs.append(f"pid {proc.pid}: {start}")

        url = f"http://127.0.0.1:{port}"
        health = wait_for_http(
            url + ctx.health_path,
            client=ctx.client(),
            timeout=min(ctx.timeout, 60),
            interval=1,
        )
        if not health.ok:
            tail = ""
            try:
                tail = log_file.read_text(errors="replace")[-3000:]
            except OSError:
                pass
            logs.append(tail)
            self.stop(ctx)
            return ProviderResult(
                ok=False,
                url=url,
                health_url=url + ctx.health_path,
                error=f"process started but never became healthy: {health.detail}",
                logs=logs,
                steps=self.steps(ctx),
                meta={"health": health.to_dict()},
            )

        return ProviderResult(
            ok=True,
            url=url,
            health_url=url + ctx.health_path,
            public=False,
            detail="process healthy on localhost; stop it with `.deploy/process.pid`",
            logs=logs,
            steps=self.steps(ctx),
            meta={"health": health.to_dict(), "pid": proc.pid, "port": port},
        )
