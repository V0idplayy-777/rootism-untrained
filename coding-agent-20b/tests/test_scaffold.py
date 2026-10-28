"""Scaffold: templates render and run, intent detection, generator safety, backends."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from deploy.http import FakeHttpClient, HttpResponse
from scaffold import (
    BackendProvisioner,
    ScaffoldGenerator,
    available_backends,
    catalog,
    get_template,
    intent_from_request,
    mask_url,
    parse_intent,
    plan_for,
    validate_postgres_url,
)
from scaffold.spec import find_placeholders, module_name, options_for, slugify, titleize

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None and importlib.util.find_spec("httpx") is not None
HAS_FLASK = importlib.util.find_spec("flask") is not None
HAS_NODE = shutil.which("node") is not None


def write_project(root: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def run_pytest(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=240,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


# -------------------------------------------------------------------- catalogue
def test_catalogue_has_five_plus_templates_with_plain_english():
    templates = catalog()
    assert len(templates) >= 5
    ids = {template["id"] for template in templates}
    assert {"fastapi-react", "nextjs-fullstack", "flask-htmx", "cli-tool", "discord-bot"} <= ids
    for template in templates:
        assert template["description"] and template["tagline"]
        assert template["good_for"], template["id"]
        assert template["run_command"] and template["test_command"]


@pytest.mark.parametrize("template_id", ["fastapi-react", "nextjs-fullstack", "flask-htmx", "cli-tool", "discord-bot", "static-site"])
def test_every_template_renders_cleanly(template_id):
    spec = get_template(template_id)
    options = options_for(spec, "Sample Project", use_auth=spec.supports_auth)
    files = spec.render(options)
    assert files, template_id
    assert find_placeholders(files) == [], template_id
    assert "README.md" in files
    for path, content in files.items():
        if path.endswith(".py"):
            compile(content, path, "exec")
    if "package.json" in files:
        json.loads(files["package.json"])


def test_module_names_are_safe():
    assert slugify("My Cool App!") == "my-cool-app"
    assert module_name("123 app") == "app_123_app"
    assert module_name("My Cool App") == "my_cool_app"
    assert titleize("my-cool_app") == "My Cool App"


# ---------------------------------------------------------------------- intent
def test_intent_web_app_with_users_and_database():
    intent = parse_intent("build me a web app where users sign up and save tasks")
    assert intent.wants_web and intent.wants_auth and intent.wants_db
    intent, choice = intent_from_request("build me a web app where users sign up and save tasks")
    assert choice.template_id == "fastapi-react"
    assert choice.options.get("use_auth") is True
    assert any("auth" in reason.lower() for reason in choice.reasons)


@pytest.mark.parametrize(
    ("prose", "expected"),
    [
        ("a discord bot that moderates my server", "discord-bot"),
        ("a command line tool to rename files", "cli-tool"),
        ("a next.js app with an api route", "nextjs-fullstack"),
        ("an internal dashboard with forms", "flask-htmx"),
    ],
)
def test_intent_routes_to_expected_starter(prose, expected):
    _, choice = intent_from_request(prose)
    assert choice.template_id == expected


def test_ambiguous_intent_asks_a_question():
    intent = parse_intent("make me something cool")
    assert intent.ambiguous
    _, choice = intent_from_request("make me something cool")
    assert choice.questions


def test_plan_for_is_dry_and_complete():
    plan = plan_for("an internal tool to track tickets", "tickets")
    assert plan["template"]["id"] == "fastapi-react"
    assert plan["intent"]["wants_db"] is True
    assert plan["files"] and "README.md" in plan["files"]
    assert plan["steps"] and plan["reasons"]


# ------------------------------------------------------------------- generator
def test_generator_refuses_non_empty_directory(tmp_path):
    (tmp_path / "keep.txt").write_text("x")
    result = ScaffoldGenerator(tmp_path).create("cli-tool", "thing")
    assert not result.ok
    assert "not empty" in result.error


def test_generator_ignores_agent_metadata_when_checking_empty(tmp_path):
    (tmp_path / ".agent").mkdir()
    (tmp_path / ".agent" / "scaffold.json").write_text("{}")
    result = ScaffoldGenerator(tmp_path).create("cli-tool", "thing")
    assert result.ok


def test_generator_dry_run_writes_nothing(tmp_path):
    result = ScaffoldGenerator(tmp_path, dry_run=True).create("static-site", "landing")
    assert result.ok and result.files
    assert not any(tmp_path.iterdir())


def test_generator_writes_config_metadata_and_gitignore(tmp_path):
    result = ScaffoldGenerator(tmp_path).create("cli-tool", "My Tool")
    assert result.ok
    assert (tmp_path / "agentdeploy.toml").exists()
    record = json.loads((tmp_path / ".agent" / "scaffold.json").read_text())
    assert record["template"] == "cli-tool"
    ignored = (tmp_path / ".gitignore").read_text()
    assert ".env" in ignored
    config = (tmp_path / "agentdeploy.toml").read_text()
    assert 'name = "my-tool"' in config


def test_generator_preserves_existing_files_with_force(tmp_path):
    (tmp_path / "README.md").write_text("# mine\n")
    result = ScaffoldGenerator(tmp_path, force=True).create("static-site", "landing")
    assert result.ok
    assert "README.md" in result.skipped
    assert (tmp_path / "README.md").read_text() == "# mine\n"
    assert (tmp_path / "index.html").exists()


def test_generator_merges_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("custom-entry/\n")
    ScaffoldGenerator(tmp_path).create("cli-tool", "tool")
    text = (tmp_path / ".gitignore").read_text()
    assert "custom-entry/" in text and "node_modules/" in text


def test_create_from_request_records_intent(tmp_path):
    result = ScaffoldGenerator(tmp_path).create_from_request("a discord bot that rolls dice", name="dice-bot")
    assert result.ok
    assert result.template_id == "discord-bot"
    assert result.options["intent"]["wants_bot"] is True
    assert result.reasons


# ------------------------------------------------- generated projects actually run
@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi/httpx not installed")
@pytest.mark.parametrize("use_auth", [False, True])
def test_generated_fastapi_project_passes_its_tests(tmp_path, use_auth):
    files = get_template("fastapi-react").render(options_for(get_template("fastapi-react"), "Demo App", use_auth=use_auth))
    root = write_project(tmp_path / "app", files)
    completed = run_pytest(root)
    assert completed.returncode == 0, completed.stdout[-3000:]


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi/httpx not installed")
def test_fastapi_index_is_useful_before_the_frontend_is_built(tmp_path):
    files = get_template("fastapi-react").render(
        options_for(get_template("fastapi-react"), "Demo App", use_auth=True)
    )
    root = write_project(tmp_path / "app", files)
    script = (
        "from fastapi.testclient import TestClient\n"
        "from demo_app.main import app\n"
        "response = TestClient(app).get('/')\n"
        "assert response.status_code == 200, response.status_code\n"
        "assert 'Demo App' in response.text\n"
        "assert '__' not in response.text\n"
        "print('ok')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=root, capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ok" in completed.stdout


@pytest.mark.skipif(not HAS_FLASK, reason="flask not installed")
def test_generated_flask_project_passes_its_tests(tmp_path):
    files = get_template("flask-htmx").render(options_for(get_template("flask-htmx"), "Task Board"))
    root = write_project(tmp_path / "board", files)
    completed = run_pytest(root)
    assert completed.returncode == 0, completed.stdout[-3000:]


def test_generated_cli_project_passes_its_tests(tmp_path):
    files = get_template("cli-tool").render(options_for(get_template("cli-tool"), "Word Count"))
    root = write_project(tmp_path / "wc", files)
    completed = run_pytest(root)
    assert completed.returncode == 0, completed.stdout[-3000:]


def test_generated_discord_logic_tests_pass_without_discord_installed(tmp_path):
    files = get_template("discord-bot").render(options_for(get_template("discord-bot"), "Dice Bot"))
    root = write_project(tmp_path / "bot", files)
    completed = run_pytest(root)
    assert completed.returncode == 0, completed.stdout[-3000:]


def test_generated_static_site_passes_its_tests(tmp_path):
    files = get_template("static-site").render(options_for(get_template("static-site"), "Landing Page"))
    root = write_project(tmp_path / "site", files)
    completed = run_pytest(root)
    assert completed.returncode == 0, completed.stdout[-3000:]


@pytest.mark.skipif(not HAS_NODE, reason="node not installed")
def test_generated_nextjs_tests_pass_without_npm_install(tmp_path):
    files = get_template("nextjs-fullstack").render(options_for(get_template("nextjs-fullstack"), "Notes App"))
    root = write_project(tmp_path / "notes", files)
    completed = subprocess.run(["node", "--test"], cwd=str(root), capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-3000:]


# ------------------------------------------------------------------- auth addon
def test_generated_auth_rejects_tampering(tmp_path):
    files = get_template("fastapi-react").render(options_for(get_template("fastapi-react"), "Auth App", use_auth=True))
    root = write_project(tmp_path / "auth", files)
    spec = importlib.util.spec_from_file_location("generated_auth", root / "auth_app" / "auth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    hashed = module.hash_password("correct horse battery staple")
    assert module.verify_password("correct horse battery staple", hashed)
    assert not module.verify_password("nope", hashed)
    assert module.hash_password("same pass") != module.hash_password("same pass")

    token = module.create_token("user-1")
    assert module.decode_token(token)["sub"] == "user-1"
    header, payload, signature = token.split(".")
    with pytest.raises(module.AuthError):
        module.decode_token(f"{header}.{payload}.{signature[:-2]}aa")
    with pytest.raises(module.AuthError):
        module.decode_token(module.create_token("user-1", ttl=-5))


def test_production_requires_auth_secret(tmp_path, monkeypatch):
    files = get_template("fastapi-react").render(options_for(get_template("fastapi-react"), "Auth App", use_auth=True))
    root = write_project(tmp_path / "auth", files)
    spec = importlib.util.spec_from_file_location("generated_auth2", root / "auth_app" / "auth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("AUTH_SECRET", raising=False)
    with pytest.raises(module.AuthError):
        module.create_token("user-1")


# --------------------------------------------------------------------- backends
def test_mask_url_hides_password():
    assert mask_url("postgresql://user:secret@db.example.com:5432/app") == "postgresql://user:***@db.example.com:5432/app"
    assert mask_url("") == ""
    assert "secret" not in mask_url("postgres://u:p@h/db")


def test_validate_postgres_url():
    assert validate_postgres_url("postgresql://u:p@h:5432/db")[0]
    assert not validate_postgres_url("mysql://u:p@h/db")[0]
    assert not validate_postgres_url("postgresql://h")[0]


def test_available_backends_reports_sqlite(tmp_path):
    backends = available_backends({})
    assert backends["sqlite"][0]
    assert not backends["neon"][0]


def test_provisioner_defaults_to_sqlite(tmp_path):
    result = BackendProvisioner(tmp_path, env={}).ensure(name="demo")
    assert result.ok and result.engine == "sqlite"
    assert not result.created
    gitignore = (tmp_path / ".gitignore").read_text()
    assert ".env" in gitignore


def test_provisioner_uses_existing_database_url(tmp_path):
    url = "postgresql://user:pass@db.internal:5432/app"
    result = BackendProvisioner(tmp_path, env={"DATABASE_URL": url}).ensure(name="demo")
    assert result.ok and result.provider == "existing"
    env_file = tmp_path / ".env"
    assert env_file.exists()
    assert url in env_file.read_text()
    mode = stat.S_IMODE(env_file.stat().st_mode)
    assert mode == 0o600
    example = (tmp_path / ".env.example").read_text()
    assert url not in example
    assert ".env" in (tmp_path / ".gitignore").read_text()


def test_provisioner_neon_with_fake_api(tmp_path):
    client = FakeHttpClient(
        {
            ("POST", "https://console.neon.tech/api/v2/projects"): HttpResponse(
                200,
                json.dumps(
                    {
                        "project": {"id": "proj_1"},
                        "connection_uris": [{"connection_uri": "postgresql://neon:pw@ep-x.neon.tech/neondb"}],
                    }
                ),
            )
        }
    )
    env = {"NEON_API_KEY": "key-123"}
    result = BackendProvisioner(tmp_path, http=client, env=env).ensure(name="demo", backend="neon")
    assert result.ok and result.created and result.provider == "neon"
    assert "pw@" not in result.to_dict()["url"]
    assert "neon" in (tmp_path / ".env").read_text()
    post = client.calls[0]
    assert post["headers"]["Authorization"] == "Bearer key-123"


def test_provisioner_neon_failure_is_reported(tmp_path):
    client = FakeHttpClient(default=HttpResponse(401, json.dumps({"message": "bad key"})))
    result = BackendProvisioner(tmp_path, http=client, env={"NEON_API_KEY": "nope"}).ensure(name="x", backend="neon")
    assert not result.ok
    assert "401" in result.error
    assert not (tmp_path / ".env").exists()


def test_provisioner_dry_run_writes_nothing(tmp_path):
    result = BackendProvisioner(tmp_path, env={"DATABASE_URL": "postgresql://u:p@h/db"}).ensure(dry_run=True)
    assert result.ok and "dry run" in result.detail
    assert not (tmp_path / ".env").exists()


def test_provisioner_rejects_unknown_backend(tmp_path):
    result = BackendProvisioner(tmp_path, env={}).ensure(backend="oracle")
    assert not result.ok
    assert "unknown backend" in result.error

