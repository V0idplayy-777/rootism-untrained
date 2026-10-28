"""Retrieval ablation harness: gates, feature recording, determinism."""

import pytest

from evaluation.retrieval_ablation import make_tasks, run

FIXTURE = {
    "auth/session.py": (
        "from storage.db import lookup\n"
        "\n"
        "def create_session(user_id: str) -> str:\n"
        "    \"\"\"Create a signed session token.\"\"\"\n"
        "    return _sign(user_id)\n"
        "\n"
        "def _sign(payload: str) -> str:\n"
        "    return \"sig:\" + payload\n"
        "\n"
        "def verify_session(token: str) -> bool:\n"
        "    \"\"\"Check the signature.\"\"\"\n"
        "    return token.startswith(\"sig:\")\n"
    ),
    "api/login.py": (
        "from auth.session import create_session\n"
        "\n"
        "def login(user_id: str) -> dict:\n"
        "    \"\"\"Login handler.\"\"\"\n"
        "    return {\"session\": create_session(user_id)}\n"
    ),
    "storage/db.py": (
        "def lookup(key: str):\n"
        "    \"\"\"Fetch by primary key.\"\"\"\n"
        "    return {\"key\": key}\n"
    ),
    "billing/charge.py": (
        "from storage.db import lookup\n"
        "\n"
        "def charge(customer_id: str, amount_cents: int) -> bool:\n"
        "    \"\"\"Idempotent charge.\"\"\"\n"
        "    return bool(lookup(customer_id)) and amount_cents > 0\n"
    ),
}


@pytest.fixture()
def repo(tmp_path):
    for rel, text in FIXTURE.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return str(tmp_path)


def test_make_tasks_are_gold_anchored(repo):
    from agent.repo_context import RepoIndex

    index = RepoIndex.build(repo)
    tasks = make_tasks(index)
    assert tasks
    golds = {t["gold"] for t in tasks}
    assert golds <= set(FIXTURE)
    for t in tasks:
        assert t["symbol"] in t["task"]


def test_run_records_features_and_gates(repo):
    report = run(repo, budget=800)
    assert report["tasks"] >= 3
    assert report["flat_recall"] >= 0.0
    for row in report["rows"]:
        assert isinstance(row["hierarchical"]["features"], dict)
        assert row["hierarchical"]["features"].get("hybrid") in (True, False)
    assert "hierarchical_features" in report
    assert report["gate"]["hierarchical_not_worse"] is True
    assert report["passed"]


def test_run_is_deterministic(repo):
    a = run(repo, budget=800)
    b = run(repo, budget=800)
    assert a["flat_recall"] == b["flat_recall"]
    assert a["hierarchical_recall"] == b["hierarchical_recall"]
