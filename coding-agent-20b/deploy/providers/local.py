"""Local Docker provider: build the image, run it, prove it is healthy.

This is the zero-account path. It produces a running container bound to a free
localhost port and verifies ``/health`` before reporting success. It is *not* a
public URL, and the result says so (``public=False``) plus how to expose it
(a tunnel or any host that runs the same image).
"""

from __future__ import annotations

import socket

from ..dockerfile import write_artifacts
from ..health import wait_for_http
from ..plan import DeployStep, ProviderResult
from .base import Availability, DeployContext, DeployProvider


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LocalDockerProvider(DeployProvider):
    name = "local"
    label = "Local Docker container"
    supports_kinds = ("python-web", "node-web", "nextjs", "static", "worker")

    def available(self, ctx: DeployContext) -> Availability:
        if not self.which("docker"):
            return Availability(
                ok=False,
                reason="docker CLI is not installed",
                fix="Install Docker (https://docs.docker.com/get-docker/) or deploy to a hosted provider.",
            )
        code, output = self.run(["docker", "version", "--format", "{{.Server.Version}}"], ".", timeout=20)
        if code != 0:
            return Availability(
                ok=False,
                reason="docker daemon is not reachable",
                fix="Start Docker Desktop / `sudo systemctl start docker`, then re-run.",
            )
        return Availability(ok=True, reason=f"docker {output.strip()[:40]}")

    def steps(self, ctx: DeployContext) -> list[DeployStep]:
        return [
            DeployStep("artifacts", "Generate Dockerfile, .dockerignore and docker-compose.yml", "skipped when files already exist", "build"),
            DeployStep("build", f"Build image {ctx.image}", "docker build", "build"),
            DeployStep("run", "Start container on a free local port", "docker run -d", "release"),
            DeployStep("health", "Wait for the health endpoint", ctx.health_path, "health"),
        ]

    def deploy(self, ctx: DeployContext) -> ProviderResult:
        logs: list[str] = []
        written = write_artifacts(ctx.root, ctx.profile, ctx.config, tag=ctx.tag)
        logs.append(f"wrote: {', '.join(written['written']) or 'nothing (files already present)'}")

        code, output = self.run(
            ["docker", "build", "-t", ctx.image, "-f", "Dockerfile", "."],
            ctx.root,
            timeout=ctx.timeout,
        )
        logs.append(output[-4000:])
        if code != 0:
            return ProviderResult(
                ok=False,
                error=f"docker build failed (exit {code})",
                logs=logs,
                artifacts=written,
                steps=self.steps(ctx),
            )

        container = ctx.config.name or ctx.profile.name
        self.run(["docker", "rm", "-f", container], ctx.root, timeout=60)
        host_port = free_port()
        env_args: list[str] = ["-e", f"PORT={ctx.port}"]
        for key, value in ctx.runtime_env.items():
            env_args += ["-e", f"{key}={value}"]
        code, output = self.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container,
                "-p",
                f"127.0.0.1:{host_port}:{ctx.port}",
                *env_args,
                ctx.image,
            ],
            ctx.root,
            timeout=120,
        )
        logs.append(output.strip())
        if code != 0:
            return ProviderResult(
                ok=False,
                error=f"docker run failed: {output.strip()[:300]}",
                logs=logs,
                artifacts=written,
                steps=self.steps(ctx),
            )

        url = f"http://127.0.0.1:{host_port}"
        health = wait_for_http(
            url + ctx.health_path,
            client=ctx.client(),
            timeout=min(ctx.timeout, 90),
            interval=2,
        )
        if not health.ok:
            _, container_logs = self.run(["docker", "logs", "--tail", "60", container], ctx.root, timeout=30)
            logs.append(container_logs[-3000:])
            return ProviderResult(
                ok=False,
                url=url,
                health_url=url + ctx.health_path,
                error=f"container started but never became healthy: {health.detail}",
                logs=logs,
                artifacts=written,
                steps=self.steps(ctx),
                meta={"health": health.to_dict(), "container": container},
            )

        return ProviderResult(
            ok=True,
            url=url,
            health_url=url + ctx.health_path,
            public=False,
            detail="container healthy on localhost (use a tunnel or hosted provider for a public URL)",
            logs=logs,
            artifacts=written,
            steps=self.steps(ctx),
            meta={"health": health.to_dict(), "container": container, "host_port": host_port},
        )
