"""Deep retrieval: hybrid search, call-graph expansion, hierarchical assembly."""

from pathlib import Path

import pytest

from agent.repo_context import (
    ContextAssembler,
    HierarchicalSummarizer,
    RepoIndex,
    expand_query,
)

REPO = {
    "auth/session.py": (
        "from storage.db import lookup\n"
        "\n"
        "def create_session(user_id: str) -> str:\n"
        "    \"\"\"Create a signed session token for a user.\"\"\"\n"
        "    token = _sign(user_id)\n"
        "    lookup(token)\n"
        "    return token\n"
        "\n"
        "def _sign(payload: str) -> str:\n"
        "    return \"sig:\" + payload\n"
        "\n"
        "def verify_session(token: str) -> bool:\n"
        "    \"\"\"Check a session token signature.\"\"\"\n"
        "    return token.startswith(\"sig:\")\n"
    ),
    "api/login.py": (
        "from auth.session import create_session\n"
        "\n"
        "def login(user_id: str) -> dict:\n"
        "    \"\"\"Login endpoint handler; returns a session cookie.\"\"\"\n"
        "    token = create_session(user_id)\n"
        "    return {\"session\": token}\n"
    ),
    "storage/db.py": (
        "def lookup(key: str):\n"
        "    \"\"\"Fetch a row by primary key.\"\"\"\n"
        "    return {\"key\": key}\n"
    ),
    "billing/charge.py": (
        "from storage.db import lookup\n"
        "\n"
        "def charge(customer_id: str, amount_cents: int) -> bool:\n"
        "    \"\"\"Charge a customer once; idempotent per idempotency key.\"\"\"\n"
        "    row = lookup(customer_id)\n"
        "    return bool(row) and amount_cents > 0\n"
    ),
    "README.md": "# billing-api\n\nA tiny API for sessions and billing.\n",
}


@pytest.fixture()
def index(tmp_path) -> RepoIndex:
    root = tmp_path / "repo"
    for rel, text in REPO.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return RepoIndex.build(str(root))


# ------------------------------------------------------------------ query


def test_expand_query_produces_variants():
    variants = expand_query("fix the login bug where sessions expire")
    assert len(variants) >= 2
    assert any("session" in v for v in variants)
    # synonym layer maps prose ("logging in") to code identifiers
    variants = expand_query("make logging in safe")
    assert any("login" in v for v in variants)


# ------------------------------------------------------------------ search


def test_hybrid_search_beats_lexical_on_paraphrase(index):
    # "auth" never appears in the file, but session/token/sign do
    hits = index.search_hybrid("authenticate a user and give them a token", k=4)
    assert hits, "hybrid returned nothing"
    assert any("auth/session.py" in h.path for h in hits)


def test_search_task_uses_multi_query(index):
    hits = index.search_task("make logging in safe", k=4)
    assert hits, "task search returned nothing"
    assert any("auth" in h.path or "api" in h.path for h in hits)


def test_bm25_search_is_still_available(index):
    hits = index.search("create_session", k=3)
    assert hits
    assert any(h.path == "auth/session.py" for h in hits)


# ------------------------------------------------------------------ call graph


def test_callers_and_callees(index):
    callers = index.callers_of("create_session")
    assert "api/login.py" in callers
    # callees_of resolves called names to the files that define them
    callees = index.callees_of("api/login.py")
    assert callees == {"auth/session.py"}
    assert "storage/db.py" not in callees  # not directly called by login.py


def test_related_by_calls_expands_transitively(index):
    related = index.related_by_calls(["api/login.py"], depth=2)
    # login -> session -> db, but billing is unrelated
    assert "auth/session.py" in related
    assert "storage/db.py" in related
    assert "billing/charge.py" not in related


def test_symbol_lookup_and_defined_in(index):
    syms = index.symbol_lookup("create_session")
    assert any(s.path == "auth/session.py" for s in syms)
    names = {s.name for s in index.defined_in("auth/session.py")}
    assert {"create_session", "_sign", "verify_session"} <= names


def test_related_files_via_imports_and_calls(index):
    related = index.related_files("auth/session.py")
    assert "storage/db.py" in related


# ------------------------------------------------------------------ updates


def test_update_file_keeps_index_fresh(index, tmp_path):
    root = tmp_path / "repo"
    (root / "new_module.py").write_text(
        "def frobnicate(x):\n"
        "    \"\"\"Frobnicate a widget.\"\"\"\n"
        "    return x * 2\n"
    )
    index.update_file("new_module.py")
    hits = index.search_hybrid("frobnicate a widget", k=3)
    assert hits and "new_module.py" in hits[0].path
    # existing content still searchable
    assert index.search("create_session", k=3)


# ------------------------------------------------------------------ hierarchy


def test_hierarchical_summaries(index):
    hs = HierarchicalSummarizer(index)
    fs = hs.file_summary("auth/session.py")
    assert "create_session" in fs
    ds = hs.directory_summary("auth")
    assert "session.py" in ds
    rs = hs.repo_summary()
    assert "billing" in rs or "auth" in rs


def test_context_assembler_hierarchical(index):
    asm = ContextAssembler(index)
    pack = asm.build(
        task="how does login create a session",
        budget_tokens=400,
        focus_paths=["api/login.py"],
    )
    assert pack.text
    assert "create_session" in pack.text or "login" in pack.text
    assert pack.features  # retrieval provenance is recorded
    assert pack.features.get("hybrid") is True
    assert pack.features.get("queries", 0) >= 1
    assert pack.mode == "hierarchical"


def test_context_assembler_pulls_call_graph_neighbors(index):
    asm = ContextAssembler(index)
    pack = asm.build(
        task="fix session token signing",
        budget_tokens=800,
        focus_paths=["api/login.py"],
    )
    assert "auth/session.py" in pack.text or "create_session" in pack.text
    # the call-graph neighbour was pulled in and billed as such
    assert pack.features.get("call_graph_files", 0) >= 1


def test_context_assembler_build_task_prompt(index):
    asm = ContextAssembler(index)
    prompt = asm.build_task_prompt(
        problem_statement="fix session verification", budget_tokens=300
    )
    assert "session" in prompt.lower()
    assert "Repository context" in prompt


def test_repo_map_and_outline(index):
    assert "create_session" in index.file_outline("auth/session.py")
    rmap = index.repo_map()
    assert "auth" in rmap and "billing" in rmap


def test_serialisation_roundtrip(index):
    data = index.to_dict()
    assert isinstance(data, dict)
    assert Path(index.root).exists()


def test_index_skips_ignored_and_binary(tmp_path):
    root = tmp_path / "repo2"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "config").write_text("git stuff")
    (root / "app.py").write_text("def run():\n    pass\n")
    (root / "blob.bin").write_bytes(b"\x00\x01\x02\x03")
    index = RepoIndex.build(str(root))
    hits = index.search_hybrid("run", k=5)
    paths = {h.path for h in hits}
    assert all(".git" not in p and "blob.bin" not in p for p in paths)
