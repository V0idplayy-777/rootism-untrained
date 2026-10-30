"""The idea -> running app path: catalogue, renderer, taste, corpus and suite."""

from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import pytest

from dataset.app_corpus import AppCorpus, files_block
from dataset.mixture import DEFAULT_MIXTURE
from evaluation.app_suite import AppSuite, variant_guards
from quality.critic import CodeCritic
from quality.taste import taste_of
from scaffold.blueprint import render_blueprint
from scaffold.blueprint_variants import (
    apply_broken_owner_scope,
    apply_incomplete_entities,
    apply_legacy_ui,
)
from scaffold.blueprints import all_blueprints, get_blueprint, split, stats


def _write(files: dict[str, str], root: Path) -> None:
    for path, content in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def test_catalogue_is_large_realistic_and_splits_cleanly():
    catalogue = all_blueprints()
    assert len(catalogue) >= 50
    ids = [item.id for item in catalogue]
    assert len(ids) == len(set(ids))
    counts = stats()
    assert counts["blueprints"] == len(catalogue)
    assert {"crud", "relations", "feature"} <= set(counts["by_tier"])
    assert counts["with_auth"] > 0
    assert counts["with_auth"] < len(catalogue)
    train, held_out = split("app")
    assert train and held_out
    assert not {item.id for item in train} & {item.id for item in held_out}
    assert sorted(item.id for item in train + held_out) == sorted(ids)


def test_renderer_produces_a_complete_project(tmp_path):
    blueprint = get_blueprint("shared-tasks")
    project = render_blueprint(blueprint)
    _write(project.files, tmp_path)
    required = {
        "README.md",
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        "agentdeploy.toml",
        ".env.example",
        "schema.sql",
        "web/index.html",
        "web/app.css",
        "web/app.js",
        "tests/conftest.py",
        "tests/test_api.py",
    }
    assert required <= set(project.files)
    module = blueprint.id.replace("-", "_")
    assert module + "/main.py" in project.files
    assert module + "/store.py" in project.files
    for path, content in project.files.items():
        if path.endswith(".py"):
            ast.parse(content, filename=path)
    assert "@@" not in json.dumps(project.files)


def test_default_ui_passes_taste_and_legacy_ui_fails(tmp_path):
    blueprint = get_blueprint("project-tasks")
    project = render_blueprint(blueprint)
    _write(project.files, tmp_path)
    good = taste_of(tmp_path)
    assert good.ok and good.score >= 90

    legacy_root = tmp_path / "legacy"
    _write(apply_legacy_ui(project.files, blueprint.title), legacy_root)
    legacy = taste_of(legacy_root)
    assert not legacy.ok and legacy.score < 60
    assert any(item.rule == "fixed-width-container" for item in legacy.findings)


def test_variant_guards_reject_the_right_dimensions():
    guards = variant_guards(get_blueprint("project-tasks"))
    by_variant = {(item["variant"], item["gate"]): item["ok"] for item in guards}
    assert by_variant[("default", "tests")] is True
    assert by_variant[("default", "taste")] is True
    assert by_variant[("legacy-ui", "taste")] is True
    assert by_variant[("broken-auth", "tests")] is True
    assert by_variant[("incomplete-entities", "tests")] is True


def test_variants_change_only_what_they_claim(tmp_path):
    blueprint = get_blueprint("project-tasks")
    files = render_blueprint(blueprint).files
    module = blueprint.id.replace("-", "_")
    broken = apply_broken_owner_scope(files, blueprint, auth=True)
    assert "1 = 1" in broken[module + "/store.py"]
    incomplete = apply_incomplete_entities(files, blueprint)
    assert "REFERENCES" not in incomplete[module + "/db.py"]
    assert "_require_parent(" not in incomplete[module + "/store.py"].split("def create_", 1)[1]
    legacy = apply_legacy_ui(files, blueprint.title)
    assert "prefers-color-scheme" not in legacy["web/app.css"]
    assert "!important" not in legacy["web/app.css"]


def test_corpus_rows_are_complete_and_exclude_held_out_prompts(tmp_path):
    train, held_out = split("app")
    held_ids = {item.id for item in held_out}
    corpus = AppCorpus(limit=3, verify=False)
    assert not ({item.id for item in corpus.blueprints} & held_ids)
    samples = corpus.samples()
    sft = corpus.sft_rows(samples)
    preference = corpus.preference_rows(samples)
    repair = corpus.repair_rows(samples)
    taste = corpus.taste_rows(samples)

    assert len(sft) == 3 and len(taste) == 3
    assert preference and repair
    for row in sft:
        assert row["system"] and row["prompt"]
        assert "### " in row["response"] and "python -m pytest -q" in row["response"]
        assert row["metadata"]["kind"] == "app-sft"
        assert row["metadata"]["critic_score"] >= 80
        assert row["metadata"]["taste_score"] >= 70
    for row in preference:
        assert row["chosen"] != row["rejected"]
        assert row["metadata"]["reason"]
        assert row["metadata"]["kind"] == "app-preference"
    for row in repair:
        assert "Diagnosis:" in row["response"]
        assert "```diff" in row["response"]
        assert row["metadata"]["changed_files"]
        assert row["metadata"]["evidence"]["kind"] == "expected"
    assert all("Token layer" in row["response"] for row in taste)


def test_corpus_write_records_manifest(tmp_path):
    corpus = AppCorpus(limit=2, verify=False)
    manifest = corpus.write(tmp_path)
    assert manifest["kind"] == "app-corpus"
    assert set(manifest["files"]) == {
        "app-sft.jsonl",
        "app-preference.jsonl",
        "app-repair.jsonl",
        "app-taste.jsonl",
    }
    for name, count in manifest["files"].items():
        lines = (tmp_path / name).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == count
        assert json.loads(lines[0])["id"]
    assert (tmp_path / "manifest.json").exists()


def test_files_block_prioritises_the_product_surface():
    files = {
        "web/app.css": "css",
        "web/app.js": "js",
        "web/index.html": "html",
        "tests/test_api.py": "tests",
        "README.md": "readme",
    }
    block = files_block(files, max_files=3, max_chars=5000)
    assert "README.md" in block and "web/index.html" in block
    assert "tests/test_api.py" not in block


def test_mixture_upsamples_app_generation_sources():
    weights = {source.name: source.weight for source in DEFAULT_MIXTURE}
    for name in ("app-sft", "app-repair-sft", "app-taste-sft", "app-preference-dpo"):
        assert weights.get(name, 0) >= 0.03
    assert weights["project-scaffold-sft"] > 0.02
    assert weights["repair-trace-sft"] > 0.02
    assert abs(sum(weights.values()) - 1.0) < 0.02
    pipeline = json.loads(
        (Path(__file__).resolve().parent.parent / "configs" / "pipeline.json").read_text()
    )
    text = json.dumps(pipeline)
    for name in ("app-sft", "app-repair-sft", "app-taste-sft", "app-preference-dpo"):
        assert name in text


def test_key_blueprints_score_clean_on_the_code_critic(tmp_path):
    for blueprint_id in ("personal-tasks", "project-tasks", "shared-tasks", "meeting-rooms"):
        project = render_blueprint(get_blueprint(blueprint_id))
        root = tmp_path / blueprint_id
        _write(project.files, root)
        review = CodeCritic(threshold=80).review(root)
        assert review.ok, [(item.rule, item.path) for item in review.blocking]


def test_suite_boots_a_real_app_once():
    suite = AppSuite(blueprints=[get_blueprint("personal-tasks")], which="smoke")
    result = suite.run_one(get_blueprint("personal-tasks"))
    names = {check.name: check for check in result.checks}
    assert names["render"].ok
    assert names["tests"].ok, names["tests"].detail
    assert names["critic"].ok
    assert names["taste"].ok
    assert names["deploy"].ok
    assert names["server"].ok, names["server"].detail
    assert result.ok, result.failures


def test_build_app_reports_every_gate_and_a_preview():
    from scaffold.app_builder import NoBlueprint, build_app, resolve_blueprint

    root = Path(tempfile.mkdtemp(prefix="builder-"))
    report = build_app(root, "a team task list where I can share tasks with teammates")
    assert report.ok, report.failures
    assert report.blueprint_id == "shared-tasks"
    assert report.verify is not None and report.verify.ok
    assert report.critic_score >= 80 and report.critic_ok
    assert report.taste is not None and report.taste.ok
    assert report.deploy_provider
    assert report.module in report.preview_command
    assert (root / "README.md").exists()
    assert report.to_dict()["failures"] == []

    with pytest.raises(NoBlueprint) as exc:
        resolve_blueprint("zzzz qqqq unmatched words")
    assert exc.value.suggestions


def test_repair_restores_a_broken_generated_module_and_never_claims_false_success(tmp_path):
    from scaffold.app_builder import build_app, repair_generated

    blueprint = get_blueprint("shared-tasks")
    files = apply_broken_owner_scope(render_blueprint(blueprint).files, blueprint, auth=True)
    _write(files, tmp_path)
    broken = build_app(tmp_path, blueprint.prompt, blueprint_id=blueprint.id, write_project=False)
    assert not broken.ok
    assert any(item.startswith("tests:") for item in broken.failures)

    repaired = repair_generated(tmp_path, blueprint)
    assert repaired.ok, repaired.after
    assert "shared_tasks/store.py" in repaired.restored
    assert repaired.backup


def test_starter_keywords_still_use_the_template_registry(tmp_path):
    from agent.deploy_tool import handle_tool

    payload = handle_tool(
        "scaffold_project",
        {"request": "a flask app to track tickets", "name": "tickets"},
        tmp_path,
    )
    assert payload["success"] is True
    assert payload["template"] == "flask-htmx"


def test_build_app_tool_reports_gates_without_claiming_success_early(tmp_path):
    from agent.deploy_tool import handle_tool

    payload = handle_tool(
        "build_app",
        {"request": "a team task list where I can share tasks with teammates"},
        tmp_path,
        emit=lambda _message: None,
    )
    assert payload["success"] is True, payload["failures"]
    assert payload["blueprint"] == "shared-tasks"
    assert payload["tests"] is True
    assert payload["critic"]["ok"] and payload["critic"]["score"] >= 80
    assert payload["taste"]["ok"] and payload["taste"]["score"] >= 70
    assert "uvicorn" in payload["preview"]
    assert payload["deploy_provider"]
