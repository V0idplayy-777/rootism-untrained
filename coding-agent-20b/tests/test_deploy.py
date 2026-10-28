"""Deploy engine: detection, artifacts, providers, health, landing, secrets."""

from __future__ import annotations

import json
from pathlib import Path

from deploy.config import DeployConfig, dump_toml, looks_secret, redact
from deploy.detect import detect_project
from deploy.dockerfile import generate_compose, generate_dockerfile, write_artifacts
from deploy.engine import DeployEngine
from deploy.health import check_http, wait_for_http
from deploy.http import FakeHttpClient, HttpResponse
from deploy.landing import README_END, README_START, generate_landing_page, publish_metadata, update_readme
from deploy.plan import DeployStep, ProviderResult
from deploy.providers import get_provider, select_provider
from deploy.providers.base import DeployContext
from deploy.providers.github_pages import page_url
from deploy.providers.vercel import collect_inline_files
from quality.errors import humanize


def fastapi_project(root: Path) -> Path:
    (root / "app").mkdir(parents=True)
    (root / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/healthz')\ndef health():\n    return {'ok': True}\n"
    )
    (root / "requirements.txt").write_text("fastapi==0.111.0\nuvicorn==0.30.1\n")
    (root / ".env.example").write_text("DATABASE_URL=postgres://localhost/app\n")
    return root


def static_project(root: Path) -> Path:
    (root / "index.html").write_text("<!doctype html><title>hi</title><h1>hello</h1>")
    return root


def node_project(root: Path) -> Path:
    (root / "package.json").write_text(
        json.dumps({"name": "x", "scripts": {"start": "node server.js", "test": "node --test"}, "dependencies": {"express": "^4"}})
    )
    (root / "server.js").write_text("const express=require('express');const app=express();app.get('/health',(q,r)=>r.json({ok:1}));app.listen(3000)\n")
    return root


def test_detect_fastapi(tmp_path):
    profile = detect_project(fastapi_project(tmp_path))
    assert profile.kind == "python-web"
    assert profile.framework == "fastapi"
    assert profile.entrypoint == "app/main.py"
    assert profile.app_module == "app.main:app"
    assert profile.port == 8000
    assert profile.health_path == "/healthz"
    assert profile.env_required == ["DATABASE_URL"]
    assert profile.can_deploy
    assert "uvicorn" in (profile.start_cmd or "")


def test_detect_static(tmp_path):
    profile = detect_project(static_project(tmp_path))
    assert profile.kind == "static"
    assert profile.static_dir == "."
    assert profile.port == 80


def test_detect_node_express(tmp_path):
    profile = detect_project(node_project(tmp_path))
    assert profile.kind == "node-web"
    assert profile.framework == "express"
    assert profile.start_cmd == "npm run start"
    assert profile.test_cmd == "npm run test"


def test_detect_unknown_is_honest(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    profile = detect_project(tmp_path)
    assert profile.kind == "unknown"
    assert not profile.can_deploy
    assert any("Could not identify" in note for note in profile.notes)


def test_dockerfile_for_python_has_expected_hardening(tmp_path):
    profile = detect_project(fastapi_project(tmp_path))
    dockerfile = generate_dockerfile(profile)
    assert "USER appuser" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert '"uvicorn"' in dockerfile and "app.main:app" in dockerfile
    assert "pip install --no-cache-dir -r requirements.txt" in dockerfile


def test_dockerfile_static_uses_nginx(tmp_path):
    profile = detect_project(static_project(tmp_path))
    dockerfile = generate_dockerfile(profile)
    assert "nginx" in dockerfile
    assert "USER nginx" in dockerfile


def test_compose_and_ignore(tmp_path):
    profile = detect_project(fastapi_project(tmp_path))
    compose = generate_compose(profile, "img:tag")
    assert "docker compose up" in compose or "build:" in compose
    assert "DATABASE_URL" in compose
    artifacts = write_artifacts(tmp_path, profile)
    ignore = (tmp_path / ".dockerignore").read_text()
    assert ".env" in ignore and "!.env.example" in ignore
    assert (tmp_path / "docker-compose.yml").exists()
    assert "Dockerfile" in artifacts["written"]


def test_artifacts_never_clobber(tmp_path):
    profile = detect_project(fastapi_project(tmp_path))
    (tmp_path / "Dockerfile").write_text("FROM scratch\n# mine\n")
    artifacts = write_artifacts(tmp_path, profile)
    assert (tmp_path / "Dockerfile").read_text().startswith("FROM scratch")
    assert "Dockerfile" in artifacts["skipped"]


def test_config_roundtrip_never_stores_secrets(tmp_path):
    config = DeployConfig(name="demo", env={"APP_ENV": "prod", "API_TOKEN": "super-secret-value"})
    config.save(tmp_path)
    text = (tmp_path / "agentdeploy.toml").read_text()
    assert "API_TOKEN" not in text
    assert "APP_ENV" in text
    loaded = DeployConfig.load(tmp_path)
    assert loaded.name == "demo"
    assert loaded.env == {"APP_ENV": "prod"}


def test_redaction_and_secret_detection():
    assert looks_secret("DATABASE_URL")
    assert looks_secret("VERCEL_TOKEN")
    assert not looks_secret("APP_ENV")
    text = redact("token=abcdef123456 and Bearer abcdefghijklmnop and super-secret-value", {"X": "super-secret-value"})
    assert "super-secret-value" not in text
    assert "abcdef123456" not in text


def test_toml_writer_roundtrips(tmp_path):
    from tomllib import loads

    payload = {"deploy": {"name": "a-b", "port": 8000, "health_path": "/healthz", "env": {"A": "1"}}}
    assert loads(dump_toml(payload)) == payload


def test_check_http_success_and_failure():
    ok = check_http("http://x/health", client=FakeHttpClient({("GET", "http://x/health"): HttpResponse(200, "ok")}))
    assert ok.ok and ok.status == 200
    bad = check_http("http://x/health", client=FakeHttpClient({("GET", "http://x/health"): HttpResponse(503, "down")}))
    assert not bad.ok and "unexpected status 503" in bad.detail


def test_wait_for_http_retries_until_ready():
    responses = [HttpResponse(503), HttpResponse(500), HttpResponse(200)]

    def handler(method, url, body, headers):
        return responses.pop(0)

    client = FakeHttpClient()
    client.handler = handler
    result = wait_for_http("http://x/", client=client, timeout=30, interval=0, sleep=lambda _: None, clock=lambda: 0)
    assert result.ok
    assert result.attempts == 3
    assert result.history == [503, 500, 200]


def test_wait_for_http_reports_timeout():
    client = FakeHttpClient(default=HttpResponse(503))
    ticks = iter([0, 1, 2, 3, 4, 5, 6, 7])
    result = wait_for_http("http://x/", client=client, timeout=2, interval=0, sleep=lambda _: None, clock=lambda: next(ticks))
    assert not result.ok
    assert "not healthy" in result.detail


def test_provider_availability_without_tools(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    profile = detect_project(static_project(tmp_path))
    config = DeployConfig(name="site")
    ctx = DeployContext(root=tmp_path, profile=profile, config=config, image="site:latest")
    assert not get_provider("local").available(ctx).ok
    assert get_provider("archive").available(ctx).ok


def test_select_provider_falls_back_to_archive(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    profile = detect_project(static_project(tmp_path))
    ctx = DeployContext(root=tmp_path, profile=profile, config=DeployConfig(name="site"), image="site:latest")
    provider, warnings = select_provider(ctx)
    assert provider.name == "archive"
    assert any("skipped" in warning or "not usable" in warning for warning in warnings)


def test_vercel_dry_run_uses_no_network(tmp_path):
    profile = detect_project(static_project(tmp_path))
    client = FakeHttpClient()
    ctx = DeployContext(
        root=tmp_path,
        profile=profile,
        config=DeployConfig(name="site"),
        image="site:latest",
        dry_run=True,
        http=client,
        secrets={"VERCEL_TOKEN": "t"},
    )
    provider = get_provider("vercel")
    assert provider.available(ctx).ok
    result = provider.deploy(ctx)
    assert result.ok and result.public
    assert client.calls == []


def test_vercel_upload_payload_and_poll(tmp_path):
    profile = detect_project(static_project(tmp_path))
    calls = []

    def handler(method, url, body, headers):
        calls.append((method, url, body, headers))
        if url.startswith("https://api.vercel.com/v13/deployments") and method == "POST":
            return HttpResponse(200, json.dumps({"id": "dpl_1", "url": "demo.vercel.app"}))
        if "/v13/deployments/dpl_1" in url:
            return HttpResponse(200, json.dumps({"readyState": "READY"}))
        if url.startswith("https://demo.vercel.app"):
            return HttpResponse(200, "ok")
        return HttpResponse(404, "{}")

    client = FakeHttpClient()
    client.handler = handler
    ctx = DeployContext(
        root=tmp_path,
        profile=profile,
        config=DeployConfig(name="site"),
        image="site:latest",
        http=client,
        secrets={"VERCEL_TOKEN": "secret-token"},
    )
    result = get_provider("vercel").deploy(ctx)
    assert result.ok
    assert result.url == "https://demo.vercel.app"
    post = next(call for call in calls if call[0] == "POST")
    assert post[3]["Authorization"] == "Bearer secret-token"
    files = {item["file"] for item in post[2]["files"]}
    assert "index.html" in files
    assert ".env" not in files and "Dockerfile" not in files


def test_collect_inline_files_skips_binaries(tmp_path):
    (tmp_path / "index.html").write_text("<h1>hi</h1>")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n")
    files, skipped = collect_inline_files(tmp_path)
    assert [item["file"] for item in files] == ["index.html"]
    assert any("logo.png" in item for item in skipped)


def test_pages_url_parsing():
    assert page_url("git@github.com:alice/site.git") == "https://alice.github.io/site"
    assert page_url("https://github.com/alice/alice.github.io") == "https://alice.github.io"
    assert page_url("https://gitlab.com/alice/site") is None


def test_github_pages_requires_static_and_remote(tmp_path):
    profile = detect_project(fastapi_project(tmp_path))
    ctx = DeployContext(root=tmp_path, profile=profile, config=DeployConfig(name="app"), image="app:latest")
    availability = get_provider("github-pages").available(ctx)
    assert not availability.ok
    assert "static" in availability.reason


def test_landing_page_contains_link_and_no_external_assets():
    page = generate_landing_page("Demo", "A demo", url="https://demo.example", provider="vercel", health={"ok": True, "detail": "healthy"})
    assert "https://demo.example" in page
    assert "http://" not in page.replace("https://demo.example", "")
    assert "<script" not in page


def test_readme_update_is_idempotent(tmp_path):
    assert update_readme(tmp_path, "Demo", "https://demo.example", "vercel")
    first = (tmp_path / "README.md").read_text()
    assert README_START in first and README_END in first
    assert not update_readme(tmp_path, "Demo", "https://demo.example", "vercel")
    text = (tmp_path / "README.md").read_text()
    assert text.count("## Live") == 1


def test_publish_metadata_writes_all_three(tmp_path):
    profile = detect_project(fastapi_project(tmp_path))
    meta = publish_metadata(
        tmp_path,
        name="demo",
        url="https://demo.example",
        provider="local",
        image="demo:latest",
        profile=profile,
        result={"outcomes": [{"title": "build", "status": "ok", "detail": "ok"}]},
        description="a demo",
        health={"ok": True, "detail": "healthy"},
    )
    assert (tmp_path / "landing" / "index.html").exists()
    assert (tmp_path / "HOSTING.md").exists()
    assert meta["readme_updated"] is True
    guide = (tmp_path / "HOSTING.md").read_text()
    assert "docker compose" in guide and "Eject" in guide


def test_engine_dry_run_plan(tmp_path):
    fastapi_project(tmp_path)
    result = DeployEngine(tmp_path, dry_run=True).deploy()
    assert result.ok and result.dry_run
    assert result.plan is not None and len(result.plan.steps) >= 3
    assert all(outcome.status == "skipped" for outcome in result.outcomes)
    assert (tmp_path / ".agent" / "deploy" / "last.json").exists()


def test_engine_archive_artifact_without_hosts(tmp_path):
    static_project(tmp_path)
    result = DeployEngine(tmp_path, provider="archive", environ={}).deploy()
    assert result.ok
    assert result.url is None
    assert result.artifacts.get("archive", "").startswith("deploy/")
    assert (tmp_path / result.artifacts["archive"]).exists()
    assert (tmp_path / "HOSTING.md").exists()
    assert any("no public URL" in note for note in result.notes)


def test_engine_reports_unusable_requested_provider(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    static_project(tmp_path)
    result = DeployEngine(tmp_path, provider="fly", environ={}).deploy()
    assert result.provider == "archive"
    assert any("fly is not usable" in note for note in result.notes)


def test_engine_refuses_unknown_project(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr("deploy.providers.local.LocalDockerProvider.available", lambda self, ctx: __import__("deploy.providers.base", fromlist=["Availability"]).Availability(ok=True, reason="stub"))
    (tmp_path / "notes.txt").write_text("hi")
    result = DeployEngine(tmp_path, provider="local", environ={}).deploy()
    assert not result.ok
    assert result.human_error is not None
    assert "not recognised" in result.human_error["title"]


def test_engine_archive_still_works_for_unknown_project(tmp_path):
    (tmp_path / "notes.txt").write_text("hi")
    result = DeployEngine(tmp_path, provider="archive", environ={}).deploy()
    assert result.ok
    assert result.url is None
    assert any("Dockerfile" in note for note in result.notes)


def test_engine_redacts_secrets_from_logs(tmp_path, monkeypatch):
    static_project(tmp_path)
    result = DeployEngine(tmp_path, provider="archive", environ={"VERCEL_TOKEN": "supersecret-token"}).deploy()
    payload = json.dumps(result.to_dict())
    assert "supersecret-token" not in payload


def test_humanize_known_failures():
    assert "Docker" in humanize("Cannot connect to the Docker daemon").title
    assert "port" in humanize("Error: EADDRINUSE: address already in use").title.lower()
    assert "test" in humanize("FAILED tests/test_x.py::test_y").title.lower()
    assert humanize("segfault at 0x0").matched is False


def test_deploy_step_serialises():
    step = DeployStep("build", "Build image", "docker build", "build")
    assert step.to_dict() == {"key": "build", "title": "Build image", "detail": "docker build", "kind": "build"}


def test_provider_result_is_plain_data():
    result = ProviderResult(ok=True, url="http://x", public=True)
    assert result.ok and result.public
