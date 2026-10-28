"""Long-horizon agent behaviour: scratch, assumptions, milestones, wrap-up."""

import json

from agent.production_agent import TOOL_SCHEMAS, ProductionAgent
from agent.sandbox import HardenedSandbox, SandboxConfig


def _workspace(tmp_path):
    (tmp_path / "calc.py").write_text(
        "def add(a, b):\n    return a - b\n\ndef double(x):\n    return add(x, x)\n"
    )
    return tmp_path


def _tool(tool, **arguments):
    return "<|tool_call|>\n" + json.dumps({"tool": tool, "arguments": arguments})


def _agent(tmp_path, script, **kwargs):
    sandbox = HardenedSandbox(SandboxConfig(allow_local_fallback=True))
    responses = iter(script)

    def generate(prompt):
        return next(responses, "FINAL ANSWER: out of script")

    return ProductionAgent(
        workspace_path=str(tmp_path),
        sandbox=sandbox,
        generate_fn=generate,
        max_steps=kwargs.pop("max_steps", 14),
        **kwargs,
    )


def _read_extras(agent):
    path = agent._extras_path()
    assert path.exists(), f"extras file missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _schema(name):
    assert name in TOOL_SCHEMAS
    return TOOL_SCHEMAS[name]




def test_control_tool_schemas_exist():
    for name in (
        "record_assumption",
        "verify_assumption",
        "scratch_write",
        "scratch_read",
        "scratch_list",
    ):
        schema = _schema(name)
        assert schema.get("name") == name
        assert schema.get("description")




def test_scratch_tools_persist_and_read(tmp_path):
    _workspace(tmp_path)
    script = [
        _tool("scratch_write", topic="trace", text="hypothesis: off-by-one in add"),
        _tool("scratch_write", topic="trace", text="refuted: add is fine"),
        _tool("scratch_read", topic="trace"),
        _tool("scratch_list"),
        "FINAL ANSWER: traced",
    ]
    agent = _agent(tmp_path, script)
    result = agent.run("trace the bug", verbose=False)
    assert result["success"]
    scratch_dir = tmp_path / ".agent" / "scratch"
    assert (scratch_dir / "trace.md").exists()
    body = (scratch_dir / "trace.md").read_text()
    assert "hypothesis" in body and "refuted" in body


def test_scratch_index_is_injected_into_prompt(tmp_path):
    _workspace(tmp_path)
    seen_prompts = []

    sandbox = HardenedSandbox(SandboxConfig(allow_local_fallback=True))
    script = iter([
        _tool("scratch_write", topic="findings", text="auth middleware sets wrong principal"),
        "FINAL ANSWER: done",
    ])

    def generate(prompt):
        seen_prompts.append(prompt)
        return next(script, "FINAL ANSWER: out of script")

    agent = ProductionAgent(
        workspace_path=str(tmp_path),
        sandbox=sandbox,
        generate_fn=generate,
        max_steps=4,
    )
    agent.run("investigate auth", verbose=False)
    assert any("findings" in p for p in seen_prompts[1:])




def test_assumption_tools_and_ledger_in_prompt(tmp_path):
    _workspace(tmp_path)
    seen_prompts = []
    sandbox = HardenedSandbox(SandboxConfig(allow_local_fallback=True))
    script = iter([
        _tool("record_assumption", text="the API returns JSON"),
        _tool("verify_assumption", id=1, status="refuted", how="repo uses XML"),
        "FINAL ANSWER: adjusted",
    ])

    def generate(prompt):
        seen_prompts.append(prompt)
        return next(script, "FINAL ANSWER: out of script")

    agent = ProductionAgent(
        workspace_path=str(tmp_path),
        sandbox=sandbox,
        generate_fn=generate,
        max_steps=4,
    )
    result = agent.run("consume the API", verbose=False)
    assert result["success"]
    assert any("Refuted" in p or "refuted" in p.lower() for p in seen_prompts[1:])
    extras = _read_extras(agent)
    assert extras.get("assumptions", {}).get("assumptions"), (
        "assumption ledger was not persisted"
    )




def test_update_plan_parses_dependency_markers(tmp_path):
    _workspace(tmp_path)
    script = [
        _tool("update_plan", steps=[
            "read calc.py",
            "[after 1] fix add",
            "[after 2] verify with execute_code",
        ], current=1, evidence="calc.py read"),
        _tool("read_file", path="calc.py"),
        "FINAL ANSWER: planned",
    ]
    agent = _agent(tmp_path, script)
    result = agent.run("plan the fix", verbose=False)
    assert result["success"]
    plan = _read_extras(agent).get("plan") or {}
    steps = plan.get("steps") or []
    assert [s["depends_on"] for s in steps] == [[], [1], [2]]


def test_verified_steps_checkpoint_milestones(tmp_path):
    _workspace(tmp_path)
    script = [
        _tool("update_plan", steps=["read calc", "fix add", "verify"], current=1),
        _tool("read_file", path="calc.py"),
        _tool("update_plan", steps=["read calc", "fix add", "verify"], current=2, evidence="calc.py::add"),
        "FINAL ANSWER: one verified step",
    ]
    agent = _agent(tmp_path, script, enable_milestones=True)
    result = agent.run("fix add", verbose=False)
    assert result["success"]
    milestones = _read_extras(agent).get("milestones") or {}
    assert milestones.get("milestones"), "verified todo did not create a milestone"
    assert milestones["milestones"][0]["label"].startswith("read calc")




def test_wrap_up_injected_when_budget_exhausted(tmp_path):
    _workspace(tmp_path)
    script = [_tool("read_file", path="calc.py")] * 12
    agent = _agent(tmp_path, script, max_steps=6)
    result = agent.run("never finishes", verbose=False)
    assert not result["success"]
    recoveries = result.get("recoveries") or []
    assert any(r.get("action") == "wrap_up" for r in recoveries) or "WRAP" in (
        result.get("answer") or ""
    )




def test_memory_learning_and_prompt_injection(tmp_path):
    _workspace(tmp_path)
    seen_prompts = []
    sandbox = HardenedSandbox(SandboxConfig(allow_local_fallback=True))
    script = iter([
        _tool("note_fact", fact="the suite is pytest -q tests/"),
        _tool("read_file", path="calc.py"),
        "FINAL ANSWER: noted",
    ])

    def generate(prompt):
        seen_prompts.append(prompt)
        return next(script, "FINAL ANSWER: out of script")

    mem_path = tmp_path / "mem.json"
    from agent.memory import MemoryStore

    agent = ProductionAgent(
        workspace_path=str(tmp_path),
        sandbox=sandbox,
        generate_fn=generate,
        max_steps=4,
        memory=MemoryStore(mem_path),
    )
    result = agent.run("learn the repo", verbose=False)
    assert result["success"]
    assert mem_path.exists()

    prompts2 = []

    def generate2(prompt):
        prompts2.append(prompt)
        return "FINAL ANSWER: recalled"

    agent2 = ProductionAgent(
        workspace_path=str(tmp_path),
        sandbox=HardenedSandbox(SandboxConfig(allow_local_fallback=True)),
        generate_fn=generate2,
        max_steps=2,
        memory=MemoryStore(mem_path),
    )
    agent2.run("run the tests", verbose=False)
    assert any("pytest" in p for p in prompts2), "memory did not reach the prompt"




def test_extras_json_saved_on_success(tmp_path):
    _workspace(tmp_path)
    script = [
        _tool("update_plan", steps=["read", "fix"], current=1, evidence="x"),
        "FINAL ANSWER: ok",
    ]
    agent = _agent(tmp_path, script)
    result = agent.run("task", verbose=False)
    assert result["success"]
    state_dir = tmp_path / ".agent"
    assert (state_dir / "memory.json").exists() or (state_dir / "scratch").exists()
    extras = _read_extras(agent)
    assert "plan" in extras and "assumptions" in extras and "milestones" in extras
