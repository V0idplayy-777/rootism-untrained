"""The no-Docker local provider genuinely starts and health-checks an app."""

from __future__ import annotations

import sys

from deploy.config import DeployConfig
from deploy.detect import detect_project
from deploy.http import HttpClient
from deploy.providers.base import DeployContext
from deploy.providers.process import LocalProcessProvider

APP = '''
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}
'''


def test_local_process_provider_starts_health_checks_and_stops(tmp_path):
    (tmp_path / "app.py").write_text(APP, encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("fastapi\nuvicorn\n", encoding="utf-8")
    profile = detect_project(tmp_path)
    profile.kind = "python-web"
    profile.start_cmd = f"{sys.executable} -m uvicorn app:app --host 0.0.0.0 --port 8000"
    config = DeployConfig(name="proc-app", health_path="/health")
    provider = LocalProcessProvider()
    ctx = DeployContext(
        root=tmp_path,
        profile=profile,
        config=config,
        image="proc-app:latest",
        http=HttpClient(timeout=2),
        timeout=20,
    )
    assert provider.available(ctx).ok
    try:
        result = provider.deploy(ctx)
        assert result.ok, result.error + "\n" + "\n".join(result.logs[-5:])
        assert result.url and result.url.startswith("http://127.0.0.1:")
        assert result.health_url and result.health_url.endswith("/health")
        assert ctx.client().get(result.health_url).status == 200
    finally:
        stopped, _ = provider.stop(ctx)
    assert stopped
