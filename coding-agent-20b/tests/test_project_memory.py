"""Persistent project work: checkpoints, project map, retrieval, plan approval.

The behaviours under test are the ones the roadmap promises:

* a *persistent* project map (tree + per-file summary + key symbols) that can be
  refreshed incrementally, saved, reloaded and rendered under a char budget;
* local retrieval (keyword + hashed embeddings) that finds code by meaning
  without a network call, and can be replaced by a real embedder via a seam;
* a plan the user must approve before mass file writes happen;
* checkpoint/rollback that restores exact bytes and deletes files added since.
"""

from __future__ import annotations

import json
import time
import textwrap
from pathlib import Path

from agent.checkpoint import WorkspaceCheckpoints
from agent.plan_approval import PlanDrafter, PlanStore, ProjectPlan
from agent.project_map import ProjectMap
from agent.semantic_index import Document, HashingEmbedder, SemanticIndex


def seed(root: Path) -> Path:
    (root / "app").mkdir(parents=True, exist_ok=True)
    (root / "app" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "main.py").write_text(
        textwrap.dedent(
            '''
            """HTTP entry point for the billing service."""

            from __future__ import annotations


            def create_invoice(customer_id: str, cents: int) -> dict:
                """Create an invoice for a customer."""

                return {"customer": customer_id, "cents": cents}


            def refund(invoice_id: str) -> bool:
                """Refund an invoice."""

                return True
            '''
        ),
        encoding="utf-8",
    )
    (root / "app" / "storage.py").write_text(
        textwrap.dedent(
            '''
            """SQLite persistence for invoices."""

            from __future__ import annotations


            def connect(path: str) -> str:
                """Open the database."""

                return path
            '''
        ),
        encoding="utf-8",
    )
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test_main.py").write_text(
        "from app.main import create_invoice\n\n\ndef test_create():\n    assert create_invoice('c1', 100)\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Billing\n\nInvoices and refunds.\n", encoding="utf-8")
    return root


# ------------------------------------------------------------------ checkpoints
def test_checkpoint_restores_modified_files(tmp_path):
    seed(tmp_path)
    store = WorkspaceCheckpoints(tmp_path)
    snapshot = store.create("before-refactor")
    (tmp_path / "app" / "main.py").write_text("# broken\n", encoding="utf-8")

    changed = store.diff(snapshot.id)
    assert "app/main.py" in changed["modified"]

    outcome = store.restore(snapshot.id)
    assert "app/main.py" in outcome["restored"]
    assert "create_invoice" in (tmp_path / "app" / "main.py").read_text()


def test_checkpoint_deletes_files_added_after_snapshot(tmp_path):
    seed(tmp_path)
    store = WorkspaceCheckpoints(tmp_path)
    snapshot = store.create("clean")
    (tmp_path / "app" / "extra.py").write_text("X = 1\n", encoding="utf-8")

    assert store.diff(snapshot.id)["added"] == ["app/extra.py"]
    outcome = store.restore(snapshot.id, delete_extra=True)
    assert outcome["deleted"] == ["app/extra.py"]
    assert not (tmp_path / "app" / "extra.py").exists()

    (tmp_path / "app" / "keep.py").write_text("Y = 2\n", encoding="utf-8")
    store.restore(snapshot.id, delete_extra=False)
    assert (tmp_path / "app" / "keep.py").exists()


def test_checkpoint_skips_agent_state_and_prunes(tmp_path):
    seed(tmp_path)
    store = WorkspaceCheckpoints(tmp_path, max_snapshots=2)
    (tmp_path / ".agent").mkdir()
    (tmp_path / ".agent" / "session.json").write_text("{}", encoding="utf-8")

    for index in range(4):
        store.create(f"step-{index}")
        time.sleep(0.01)
    snapshots = store.list()
    assert len(snapshots) == 2
    assert snapshots[-1].label == "step-3"
    assert ".agent/session.json" not in snapshots[-1].files
    assert store.stats()["snapshots"] == 2


def test_checkpoint_round_trips_snapshot_json(tmp_path):
    seed(tmp_path)
    store = WorkspaceCheckpoints(tmp_path)
    snapshot = store.create("json")
    raw = json.loads((tmp_path / ".agent" / "checkpoints" / f"{snapshot.id}.json").read_text())
    assert raw["id"] == snapshot.id
    assert "app/main.py" in raw["files"]


# ---------------------------------------------------------------- project map
def test_project_map_summarises_and_indexes_symbols(tmp_path):
    seed(tmp_path)
    project = ProjectMap.build(tmp_path)
    assert set(project.files) >= {"app/main.py", "app/storage.py", "tests/test_main.py", "README.md"}
    main = project.files["app/main.py"]
    assert main.language == "python"
    assert any("create_invoice" in symbol for symbol in main.symbols)
    assert any("refund" in symbol for symbol in main.symbols)
    assert "billing" in main.summary.lower() or "invoice" in main.summary.lower()
    assert project.layout["languages"]["python"] >= 2

    hits = project.find_symbol("create_invoice")
    assert hits and hits[0].path == "app/main.py"


def test_project_map_persists_and_refreshes_incrementally(tmp_path):
    seed(tmp_path)
    path = ProjectMap.build(tmp_path).save()
    assert Path(path).exists()

    loaded = ProjectMap.load(path)
    assert loaded.files.keys() == ProjectMap.build(tmp_path).files.keys()

    (tmp_path / "app" / "new.py").write_text("def fresh() -> int:\n    return 1\n", encoding="utf-8")
    changed = loaded.refresh()
    assert "app/new.py" in changed
    assert loaded.find_symbol("fresh")

    (tmp_path / "app" / "new.py").unlink()
    assert "app/new.py" in loaded.refresh()


def test_project_map_render_respects_budget(tmp_path):
    seed(tmp_path)
    project = ProjectMap.build(tmp_path)
    text = project.render(max_chars=200)
    assert len(text) <= 240
    assert "app/main.py" in text
    focused = project.render(max_chars=2000, focus=["app/main.py"])
    assert "focused files:" in focused


def test_project_map_ignores_noise_and_flags_monorepo(tmp_path):
    (tmp_path / "packages" / "api").mkdir(parents=True)
    (tmp_path / "packages" / "web").mkdir(parents=True)
    (tmp_path / "packages" / "api" / "package.json").write_text('{"name":"api"}', encoding="utf-8")
    (tmp_path / "packages" / "web" / "package.json").write_text('{"name":"web"}', encoding="utf-8")
    (tmp_path / "node_modules" / "left-pad").mkdir(parents=True)
    (tmp_path / "node_modules" / "left-pad" / "index.js").write_text("module.exports=1", encoding="utf-8")

    project = ProjectMap.build(tmp_path)
    assert "node_modules/left-pad/index.js" not in project.files
    assert project.layout["monorepo"] is True
    assert project.stats()["files"] >= 2


# --------------------------------------------------------------- retrieval
def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder(dim=512)
    first = embedder.embed("refund an invoice")
    second = HashingEmbedder(dim=512).embed("refund an invoice")
    assert first == second
    assert abs(sum(value * value for value in first.values()) - 1.0) < 1e-6


def test_semantic_index_finds_code_by_meaning(tmp_path):
    seed(tmp_path)
    index = SemanticIndex.from_directory(tmp_path)
    hits = index.search("give money back for an invoice", k=3)
    assert hits, "expected at least one semantic hit"
    assert any("main.py" in hit.document.path for hit in hits)

    keyword = index.keyword_search("create_invoice", k=3)
    assert keyword and "create_invoice" in json.dumps(keyword[0].to_dict())

    hybrid = index.hybrid_search("refund invoice", k=3)
    assert hybrid and hybrid[0].score >= 0


def test_semantic_index_persists_and_accepts_external_embedder(tmp_path):
    seed(tmp_path)
    index = SemanticIndex.from_directory(tmp_path)
    path = index.save(tmp_path / ".agent" / "index.json")
    loaded = SemanticIndex.load(path)
    assert len(loaded.documents) == len(index.documents)
    assert loaded.search("invoice", k=1)

    def embed(text: str) -> list[float]:
        return [1.0 if "refund" in text.lower() else 0.0, 1.0 if "invoice" in text.lower() else 0.0]

    external = SemanticIndex.build(
        [Document(id="a", text="refund policy", path="a.md"), Document(id="b", text="landing page", path="b.md")],
        embed_fn=embed,
    )
    hits = external.search("refund")
    assert hits and hits[0].document.id == "a"


def test_semantic_index_handles_empty_workspace(tmp_path):
    index = SemanticIndex.from_directory(tmp_path)
    assert index.search("anything") == []
    assert index.stats()["documents"] == 0


# ---------------------------------------------------------------- plan approval
def test_plan_draft_for_empty_workspace_uses_starter(tmp_path):
    plan = PlanDrafter(tmp_path).draft("a web app where people sign up and save tasks")
    assert plan.items
    assert any(item.files for item in plan.items)
    rendered = plan.render_markdown()
    assert plan.goal in rendered
    assert "approval" in rendered.lower()


def test_plan_store_approve_reject_and_revise(tmp_path):
    store = PlanStore(tmp_path)
    plan = PlanDrafter(tmp_path).draft("add a health endpoint to the API")
    saved = store.save(plan)
    assert saved.exists()

    assert [item.id for item in store.pending()] == [plan.id]
    approved = store.approve(plan.id, "looks right")
    assert approved.status == "approved"
    assert store.pending() == []

    other_plan = PlanDrafter(tmp_path).draft("rename the storage module")
    store.save(other_plan)
    rejected = store.reject(other_plan.id)
    assert rejected.status == "rejected"

    third = PlanDrafter(tmp_path).draft("split main.py into modules")
    store.save(third)
    revised = store.revise(third.id, ["add a migration step"])
    assert revised.status == "revised"
    assert any("migration" in json.dumps(item.to_dict()) for item in revised.items)


def test_plan_model_draft_uses_json_from_generate_fn(tmp_path):
    def generate(prompt: str) -> str:
        return json.dumps(
            {
                "items": [
                    {"action": "Add retries", "files": ["app/http.py"], "why": "flaky network"},
                    {"action": "Add a test", "files": ["tests/test_http.py"], "why": "prove it"},
                ],
                "risks": ["retry storms"],
                "questions": ["what budget?"],
            }
        )

    plan = PlanDrafter(tmp_path, generate_fn=generate).draft("make HTTP calls resilient", prefer="model")
    assert plan.goal == "make HTTP calls resilient"
    assert plan.items[0].action == "Add retries"
    assert plan.risks == ["retry storms"]
    assert "drafted by the model" in plan.notes


def test_project_plan_serialises_symbols_and_files(tmp_path):
    plan = ProjectPlan.from_dict(
        {
            "goal": "ship it",
            "items": [{"action": "write code", "files": ["a.py"], "why": "because"}],
            "risks": [],
            "questions": [],
            "status": "pending",
        }
    )
    assert plan.files == ["a.py"]
    assert ProjectPlan.from_dict(plan.to_dict()).goal == "ship it"
