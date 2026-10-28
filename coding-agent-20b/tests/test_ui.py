"""UI surface: runtime events, preview serving/proxying, HTTP endpoints.

The UI is the non-engineer surface, so these tests check the promises that make
it usable rather than the CSS:

* simple mode never shows raw tool/file noise, power mode does;
* a first request scaffolds a runnable project, writes tests, runs them and
  plans a deploy - with no model configured;
* failures arrive as human sentences, not stack traces;
* the preview is served from one origin (static files, plus traversal blocked);
* the HTTP API returns JSON the page can render, and history persists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ui.preview import PreviewManager, list_workspace_files, read_workspace_file
from ui.runtime import AgentRuntime


def static_site(root: Path) -> Path:
    (root / "index.html").write_text(
        "<!doctype html><html><body><h1>hello</h1></body></html>", encoding="utf-8"
    )
    (root / "assets").mkdir()
    (root / "assets" / "app.css").write_text("body{color:red}", encoding="utf-8")
    return root


def event_types(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


# --------------------------------------------------------------------- runtime
def test_runtime_local_flow_scaffolds_tests_and_plans_a_deploy(tmp_path):
    runtime = AgentRuntime(tmp_path)
    info = runtime.info()
    assert info.mode == "local"

    events = list(runtime.run("a web app where people sign up and save tasks", mode="power"))
    types = event_types(events)
    assert "plan" in types and "scaffold" in types
    assert "tests" in types and "review" in types and "deploy_plan" in types
    assert "error" not in types

    assert (tmp_path / "requirements.txt").exists()
    assert (tmp_path / "tests").is_dir()
    assert (tmp_path / ".agent" / "ui_sessions.json").exists()

    tests_event = [e for e in events if e["type"] == "tests"][-1]
    assert tests_event["data"]["ok"] is True

    review_event = [e for e in events if e["type"] == "review"][-1]
    assert review_event["data"]["score"] >= 60


def test_runtime_simple_mode_hides_file_noise(tmp_path):
    simple_root = tmp_path / "simple"
    simple_root.mkdir()
    simple = list(AgentRuntime(simple_root).run("a landing page for my bakery", mode="simple"))
    assert "file" not in event_types(simple)
    assert "scaffold" in event_types(simple)
    assert event_types(simple)[-1] in {"deploy_plan", "done", "end"}
    assert "deploy_plan" in event_types(simple)

    power_root = tmp_path / "power"
    power_root.mkdir()
    power = list(AgentRuntime(power_root).run("a command line tool to rename files", mode="power"))
    assert "file" in event_types(power)


def test_runtime_keeps_history_and_resets(tmp_path):
    runtime = AgentRuntime(tmp_path)
    list(runtime.run("a landing page for my bakery", mode="simple"))
    assert runtime.history
    runtime.remember("assistant", "done")
    reloaded = AgentRuntime(tmp_path)
    assert any(turn.content == "done" for turn in reloaded.history)
    runtime.reset()
    assert runtime.history == []


def test_runtime_rejects_empty_message(tmp_path):
    events = list(AgentRuntime(tmp_path).run("   "))
    assert event_types(events) == ["error"]


def test_runtime_does_not_scaffold_over_existing_project(tmp_path):
    (tmp_path / "existing.py").write_text("VALUE = 1\n", encoding="utf-8")
    events = list(AgentRuntime(tmp_path).run("a web app with users", mode="power"))
    assert "scaffold" not in event_types(events)
    assert (tmp_path / "existing.py").read_text() == "VALUE = 1\n"


# --------------------------------------------------------------------- preview
def test_preview_serves_static_site(tmp_path):
    static_site(tmp_path)
    manager = PreviewManager(tmp_path)
    state = manager.start()
    assert state.running and state.kind == "static"

    status, mime, body = manager.static_file("index.html")
    assert status == 200 and b"hello" in body
    assert manager.static_file("assets/app.css")[1].startswith("text/css")

    missing = manager.static_file("nope.txt")
    assert missing[0] == 200 and b"hello" in missing[2]

    traversal = manager.static_file("../etc/passwd")
    assert traversal[0] in {403, 404}
    manager.stop()
    assert not manager.state.running


def test_preview_status_is_json_ready(tmp_path):
    manager = PreviewManager(tmp_path)
    data = manager.status()
    assert {"running", "kind", "detail"} <= set(data)
    json.dumps(data)


def test_workspace_file_helpers_block_traversal(tmp_path):
    (tmp_path / "ok.txt").write_text("hi", encoding="utf-8")
    assert read_workspace_file(tmp_path, "ok.txt")["content"] == "hi"
    assert read_workspace_file(tmp_path, "../etc/passwd")["ok"] is False
    assert "ok.txt" in [row["path"] for row in list_workspace_files(tmp_path)]


# ------------------------------------------------------------------------- API
def make_client(root: Path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from ui.app import create_app

    return TestClient(create_app(root))


@pytest.fixture()
def client(tmp_path):
    with make_client(static_site(tmp_path)) as test_client:
        yield test_client


@pytest.fixture()
def empty_client(tmp_path):
    root = tmp_path / "fresh"
    root.mkdir()
    with make_client(root) as test_client:
        yield test_client


def parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                continue
    return events


def test_ui_index_and_assets(client):
    index = client.get("/")
    assert index.status_code == 200
    assert "text/html" in index.headers["content-type"]
    assert client.get("/assets/app.css").status_code == 200
    assert client.get("/assets/../app.py").status_code == 404


def test_ui_info_reports_runtime_and_providers(client):
    data = client.get("/api/info").json()
    assert data["runtime"]["mode"] in {"local", "model"}
    assert "providers" in data and "archive" in data["providers"]
    assert data["files"] >= 1
    assert client.get("/api/healthz").json()["ok"] is True


def test_ui_templates_are_plain_english(client):
    templates = client.get("/api/templates").json()["templates"]
    assert len(templates) >= 5
    for template in templates:
        assert template["description"] and len(template["description"]) > 40
        assert template["good_for"]


def test_ui_chat_streams_and_hides_tool_noise_in_simple_mode(empty_client):
    response = empty_client.post("/api/chat", json={"message": "a flask app to track tickets", "mode": "simple"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    events = parse_sse(response.text)
    types = [event["type"] for event in events]
    assert "plan" in types and types[-1] in {"end", "done"}
    assert "file" not in types
    assert "error" not in types


def test_ui_chat_power_mode_includes_files(empty_client):
    response = empty_client.post("/api/chat", json={"message": "a command line tool to rename files", "mode": "power"})
    types = [event["type"] for event in parse_sse(response.text)]
    assert "file" in types


def test_ui_review_and_verify_endpoints(client):
    review = client.post("/api/review").json()
    assert "score" in review
    verified = client.post("/api/verify").json()
    assert "ok" in verified


def test_ui_deploy_dry_run_returns_plan(client):
    data = client.post("/api/deploy", json={"dry_run": True}).json()
    assert data["ok"] is True
    assert data["dry_run"] is True
    assert data["plan"]["steps"]


def test_ui_preview_route_serves_static_file(client):
    response = client.get("/preview/index.html")
    assert response.status_code == 200
    assert b"hello" in response.content


def test_ui_file_endpoint_blocks_traversal(client):
    data = client.get("/api/file", params={"path": "../../etc/passwd"}).json()
    assert data["ok"] is False
