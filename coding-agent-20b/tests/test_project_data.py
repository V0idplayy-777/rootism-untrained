"""Training data for whole projects and repair traces (+ the tools that ship them).

Three claims are checked here:

1. **Multi-file project rows** exist, are traceable to a real template render,
   and contain a file tree, tests and a run command - not one pasted file.
2. **Repair rows** are grounded in test evidence: prompt contains the failing
   output, the answer contains a fix, and DPO contrasts a rejected patch.
3. The **agent tools** for scaffolding/deploying/mapping behave like the CLI and
   report failure honestly instead of raising.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.deploy_tool import TOOL_HANDLERS, handle_tool, tool_schemas
from dataset.mixture import DEFAULT_MIXTURE, MixtureSampler
from dataset.project_data import ProjectDatasetGenerator, write_project_dataset
from dataset.repair_traces import RepairTraceGenerator, generate_repair_rows


# ------------------------------------------------------------- project dataset
def test_project_samples_are_multi_file_and_runnable():
    generator = ProjectDatasetGenerator(
        requests=[("fastapi-react", "an app where people sign up and save tasks"), ("cli-tool", "a tool to rename files")]
    )
    samples = generator.samples(include_auth=True)
    assert {sample.template_id for sample in samples} == {"fastapi-react", "cli-tool"}
    assert all(len(sample.files) >= 8 for sample in samples)
    assert any(any(path.startswith("tests/") for path in sample.files) for sample in samples)

    row = samples[0].to_sft_row()
    assert "multi-file" in row["prompt"]
    assert "tests/" in row["response"] and "Run:" in row["response"]
    assert row["metadata"]["has_tests"] is True
    assert row["metadata"]["files"] == len(samples[0].files)
    assert row["metadata"]["test_command"]
    json.dumps(row)


def test_auth_variant_adds_files_and_is_recorded():
    generator = ProjectDatasetGenerator(requests=[("fastapi-react", "a task app")])
    plain = generator.sample("fastapi-react", "a task app")
    with_auth = generator.sample("fastapi-react", "a task app with logins", use_auth=True)
    assert len(with_auth.files) > len(plain.files)
    assert any("auth" in path for path in with_auth.files)


def test_project_dpo_rows_reject_single_file_answers():
    generator = ProjectDatasetGenerator(requests=[("flask-htmx", "an internal ticket tool")])
    sample = generator.samples()[0]
    rows = generator.structure_dpo_rows(sample)
    assert len(rows) >= 2
    assert all(row["chosen"] != row["rejected"] for row in rows)
    assert "omitted" in rows[0]["metadata"]["rejected_reason"] or "does not run" in rows[0]["metadata"]["rejected_reason"]


def test_tests_fix_evidence_and_shard_files(tmp_path):
    generator = ProjectDatasetGenerator(requests=[("cli-tool", "a file tool")])
    sample = generator.samples()[0]
    assert sample.verified is False
    generator.mark_verified(sample, {"ok": True, "command": "python -m pytest -q", "passed": ["3 tests"], "duration": 1.2})
    assert sample.verified is True
    assert sample.evidence["kind"] == "test-run"

    manifest = write_project_dataset(tmp_path / "out", generator)
    assert manifest["sft"] == len(generator.samples())
    assert (tmp_path / "out" / "sft.jsonl").exists()
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["templates"] == ["cli-tool"]


# ------------------------------------------------------------- repair traces
def test_curated_repair_pairs_produce_grounded_rows():
    result = generate_repair_rows()
    assert result["sft"], "expected curated repair rows"
    first = result["sft"][0]
    assert "Test output" in first["prompt"] or "test" in first["prompt"].lower()
    assert "```python" in first["response"]
    assert first["metadata"]["has_test_evidence"] is True

    dpo = result["dpo"][0]
    assert dpo["chosen"] != dpo["rejected"]
    assert "still fails" in dpo["metadata"]["rejected_reason"]


def test_repair_split_is_deterministic_and_disjoint():
    first = generate_repair_rows()
    second = generate_repair_rows()
    assert [row["id"] for row in first["sft"]] == [row["id"] for row in second["sft"]]
    train_ids = {row["id"] for row in first["sft"]}
    heldout_ids = {row["id"] for row in first["sft_heldout"]}
    assert not (train_ids & heldout_ids)


def test_real_verification_trace_becomes_a_row():
    generator = RepairTraceGenerator()
    before = len(generator.examples)
    added = generator.add_trace(
        {
            "green": True,
            "command": "python -m pytest -q",
            "runs": [
                {"output_tail": "FAILED tests/test_core.py::test_add - assert 4 == 5"},
                {"output_tail": "3 passed", "fixed_code": "def add(a, b):\n    return a + b\n", "original_code": "def add(a, b):\n    return a - b\n"},
            ],
            "attempts": [{"round": 1, "changed": True, "diagnosis": "off-by-sign error"}],
        }
    )
    assert added == 1
    assert len(generator.examples) == before + 1
    row = generator.examples[-1].to_sft_row()
    assert row["metadata"]["source"] == "recorded"
    assert row["metadata"]["verified"] is True


def test_incomplete_traces_are_rejected():
    generator = RepairTraceGenerator()
    before = len(generator.examples)
    assert generator.add_trace({"green": False, "runs": [{"output_tail": "boom"}, {"output_tail": "boom"}]}) == 0
    assert generator.add_trace({"green": True, "runs": [{"output_tail": "boom"}]}) == 0
    assert len(generator.examples) == before


def test_repair_source_is_wired_into_the_mixture():
    weights = {source.name: source.weight for source in DEFAULT_MIXTURE}
    assert weights.get("repair-trace-sft", 0) > 0
    assert weights.get("project-scaffold-sft", 0) > 0
    assert abs(sum(weights.values()) - 1.0) < 0.02, weights

    sampler = MixtureSampler({"repair-trace-sft": [{"id": "r1"}], "the-stack-code": [{"id": "c1"}]})
    sampled = list(sampler.sample(20))
    assert len(sampled) == 20 and all(item["id"] in {"r1", "c1"} for item in sampled)
    pipeline = json.loads((Path(__file__).resolve().parent.parent / "configs" / "pipeline.json").read_text())
    assert "repair-trace-sft" in json.dumps(pipeline)


# ------------------------------------------------------------------ agent tools
def test_tool_schemas_cover_the_shipping_tools():
    schemas = tool_schemas()
    for name in ("deploy", "scaffold_project", "project_map", "search_project", "checkpoint"):
        assert name in schemas
        assert schemas[name]["description"]
    assert set(schemas) == set(TOOL_HANDLERS)


def test_scaffold_tool_reports_success_and_refusal(tmp_path):
    payload = handle_tool("scaffold_project", {"request": "a flask app to track tickets", "name": "tickets"}, tmp_path)
    assert payload["success"] is True
    assert payload["template"] == "flask-htmx"
    assert payload["file_count"] > 5

    refused = handle_tool("scaffold_project", {"template": "cli-tool", "name": "other"}, tmp_path)
    assert refused["success"] is False
    assert "not empty" in refused["error"]
    assert refused["hint"]


def test_deploy_tool_dry_run_and_unknown_tool(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>x</title>", encoding="utf-8")
    payload = handle_tool("deploy", {"dry_run": True}, tmp_path, emit=lambda _msg: None)
    assert payload["success"] is True
    assert payload["dry_run"] is True
    assert payload["provider"] in {"archive", "local", "vercel", "github-pages", "fly", "render", "coolify"}
    assert "Deploy plan" in payload["summary"]

    assert handle_tool("not_a_tool", {}, tmp_path) is None


def test_project_tools_survive_bad_input(tmp_path):
    assert handle_tool("scaffold_project", {}, tmp_path)["success"] is False
    assert handle_tool("search_project", {}, tmp_path)["success"] is False
    bad = handle_tool("checkpoint", {"action": "nonsense"}, tmp_path)
    assert bad["success"] is False and "unknown action" in bad["error"]


def test_checkpoint_tool_round_trip(tmp_path):
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    created = handle_tool("checkpoint", {"action": "create", "label": "before"}, tmp_path)
    assert created["success"] is True
    snapshot_id = created["snapshot"]["id"]

    (tmp_path / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("X = 1\n", encoding="utf-8")
    diff = handle_tool("checkpoint", {"action": "diff", "id": snapshot_id}, tmp_path)
    assert "app.py" in diff["modified"] and "new.py" in diff["added"]

    restored = handle_tool("checkpoint", {"action": "rollback", "id": snapshot_id}, tmp_path)
    assert restored["success"] is True
    assert (tmp_path / "app.py").read_text() == "VALUE = 1\n"
    assert not (tmp_path / "new.py").exists()


def test_project_map_tool_returns_budgeted_map(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text("def run() -> int:\n    return 1\n", encoding="utf-8")
    payload = handle_tool("project_map", {"max_chars": 200}, tmp_path)
    assert payload["success"] is True
    assert "pkg/core.py" in payload["map"]
    assert len(payload["map"]) <= 240
    assert payload["stats"]["files"] >= 1
