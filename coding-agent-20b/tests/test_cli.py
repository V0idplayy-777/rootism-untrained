"""The ``agent`` CLI: one command per step of idea -> tests -> live URL.

Every command must (a) do real work without a model, (b) support ``--json`` so a
script or the UI can consume it, and (c) exit non-zero when it failed. These
tests drive the CLI in-process so they stay fast and assert exactly that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import cli


def run(argv: list[str], capsys=None) -> tuple[int, str]:
    code = cli.main(argv)
    out = capsys.readouterr().out if capsys is not None else ""
    return code, out


def empty(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def simple_project(path: Path) -> Path:
    empty(path)
    (path / "pkg").mkdir()
    (path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (path / "pkg" / "core.py").write_text(
        '"""Core."""\n\nfrom __future__ import annotations\n\n\ndef add(a: int, b: int) -> int:\n    """Add."""\n\n    return a + b\n',
        encoding="utf-8",
    )
    (path / "tests").mkdir()
    (path / "tests" / "test_core.py").write_text(
        "from pkg.core import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n", encoding="utf-8"
    )
    (path / ".gitignore").write_text(".env\n__pycache__/\n", encoding="utf-8")
    return path


def test_init_list_and_template_json(capsys):
    code, out = run(["init", "--list"], capsys)
    assert code == 0
    assert "fastapi-react" in out and "static-site" in out

    code, out = run(["init", "--list", "--json"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert len(payload["templates"]) >= 5
    assert payload["templates"][0]["description"]


def test_init_creates_a_runnable_project(tmp_path, capsys):
    target = tmp_path / "tools"
    code, out = run(["init", "cli-tool", str(target), "--name", "renamer", "--json"], capsys)
    assert code == 0, out
    payload = json.loads(out)
    assert payload["ok"] is True
    assert (target / "pyproject.toml").exists()
    assert (target / "agentdeploy.toml").exists()
    assert (target / "tests").is_dir()


def test_init_from_request_and_refuses_busy_directory(tmp_path, capsys):
    target = empty(tmp_path / "busy")
    (target / "keep.txt").write_text("mine", encoding="utf-8")
    code, out = run(["init", "--request", "a landing page for my bakery", str(target), "--json"], capsys)
    assert code == 1
    payload = json.loads(out)
    assert payload["ok"] is False
    assert "not empty" in payload["error"]


def test_init_dry_run_writes_nothing(tmp_path, capsys):
    target = tmp_path / "dry"
    code, out = run(["init", "fastapi-react", str(target), "--dry-run", "--json"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert payload["dry_run"] is True
    assert not target.exists() or not any(target.iterdir())


def test_plan_map_search_checkpoint_flow(tmp_path, capsys):
    workspace = simple_project(tmp_path / "proj")

    code, out = run(["plan", "add a refund helper", "--workspace", str(workspace)], capsys)
    assert code == 0
    assert "Plan" in out or "plan" in out

    code, out = run(["plan", "--list", "--json", "--workspace", str(workspace)], capsys)
    assert code == 0
    plans = json.loads(out)["plans"]
    assert plans and plans[0]["status"] == "pending"

    code, out = run(["map", "--workspace", str(workspace), "--json"], capsys)
    assert code == 0
    assert json.loads(out)["stats"]["files"]

    code, out = run(["search", "add numbers", "--workspace", str(workspace), "--json"], capsys)
    assert code == 0
    assert json.loads(out)["hits"]

    code, out = run(["checkpoint", "create", "--label", "start", "--workspace", str(workspace), "--json"], capsys)
    assert code == 0
    snapshot = json.loads(out)["snapshot"]["id"]
    (workspace / "pkg" / "core.py").write_text("BROKEN = True\n", encoding="utf-8")

    code, out = run(["checkpoint", "diff", "--id", snapshot, "--workspace", str(workspace), "--json"], capsys)
    assert code == 0
    assert "pkg/core.py" in json.loads(out)["modified"]

    code, _ = run(["checkpoint", "restore", "--id", snapshot, "--workspace", str(workspace)], capsys)
    assert code == 0
    assert "def add" in (workspace / "pkg" / "core.py").read_text()


def test_review_and_verify_exit_codes(tmp_path, capsys):
    clean = simple_project(tmp_path / "clean")
    code, out = run(["review", "--workspace", str(clean), "--json"], capsys)
    assert code in {0, 1}
    review = json.loads(out)
    assert "score" in review and "findings" in review

    (clean / "pkg" / "bad.py").write_text(
        'API_KEY = "sk-live-abcdef1234567890abcdef"\n\n\ndef f(x=[]):\n    try:\n        return x\n    except:\n        pass\n',
        encoding="utf-8",
    )
    code, out = run(["review", "--workspace", str(clean), "--json"], capsys)
    assert code == 1
    assert json.loads(out)["score"] < 100

    code, out = run(["verify", "--workspace", str(clean), "--json"], capsys)
    assert code in {0, 1}
    assert "ok" in json.loads(out)


def test_testgen_writes_tests_that_pass(tmp_path, capsys):
    workspace = empty(tmp_path / "gen")
    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "pkg" / "calc.py").write_text(
        '"""Calc."""\n\nfrom __future__ import annotations\n\n\ndef double(value: int) -> int:\n    """Double."""\n\n    return value * 2\n',
        encoding="utf-8",
    )
    code, out = run(["testgen", "--workspace", str(workspace), "--json"], capsys)
    assert code == 0
    assert json.loads(out)["written"]

    code, out = run(["verify", "--workspace", str(workspace), "--json"], capsys)
    assert json.loads(out)["ok"] is True


def test_backend_defaults_to_sqlite_and_masks_secrets(tmp_path, capsys):
    workspace = simple_project(tmp_path / "db")
    code, out = run(["backend", "--workspace", str(workspace), "--json"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert payload["provider"] == "sqlite"
    assert (workspace / ".env").exists()
    assert "SECRET" not in out.upper() or "***" in out

    code, out = run(["backend", "--list", "--workspace", str(workspace)], capsys)
    assert code == 0
    assert "neon" in out.lower() or "sqlite" in out.lower()


def test_deploy_dry_run_and_archive(tmp_path, capsys):
    workspace = simple_project(tmp_path / "ship")
    code, out = run(["deploy", "--dry-run", "--json", "--workspace", str(workspace)], capsys)
    assert code == 0
    payload = json.loads(out)
    assert payload["dry_run"] is True and payload["plan"]["steps"]

    code, out = run(["deploy", "--provider", "archive", "--json", "--workspace", str(workspace)], capsys)
    assert code == 0, out
    payload = json.loads(out)
    assert payload["ok"] is True
    assert (workspace / "DEPLOY.md").exists()
    assert (workspace / "HOSTING.md").exists()


def test_run_task_without_a_model(tmp_path, capsys):
    workspace = empty(tmp_path / "run")
    code, out = run(["run", "a landing page for my bakery", "--workspace", str(workspace)], capsys)
    assert code == 0, out
    assert (workspace / "index.html").exists()


def test_doctor_reports_toolchain(tmp_path, capsys):
    code, out = run(["doctor", "--workspace", str(simple_project(tmp_path / "doc")), "--json"], capsys)
    assert code == 0
    payload = json.loads(out)
    names = {row["name"] for row in payload["checks"]}
    assert {"python", "git", "docker", "templates"} <= names


def test_bad_command_raises_systemexit():
    with pytest.raises(SystemExit):
        cli.main(["does-not-exist"])
