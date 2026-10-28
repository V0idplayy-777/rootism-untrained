"""End-to-end agent loop tests with a scripted model (no torch, no network)."""

import json
import os

from agent.production_agent import TOOL_SCHEMAS, ProductionAgent
from agent.repo_context import ContextAssembler, RepoIndex
from agent.sandbox import HardenedSandbox, SandboxConfig


def _workspace(tmp_path):

    (tmp_path / "calc.py").write_text(
        "def add(a, b):\n    return a - b\n\n\ndef double(x):\n    return add(x, x)\n"
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
        max_steps=kwargs.pop("max_steps", 12),
        **kwargs,
    )


def test_plan_facts_and_full_task_flow(tmp_path):

    _workspace(tmp_path)

    script = [
        _tool("update_plan", steps=["read calc", "fix add", "verify"], current=1),
        _tool("note_fact", fact="add must be commutative"),
        _tool("read_file", path="calc.py"),
        _tool("edit_function", path="calc.py", function="add", new_source="def add(a, b):\n    return a + b\n"),
        _tool("update_plan", steps=["read calc", "fix add", "verify"], current=2, evidence="calc.py::add"),
        _tool("execute_code", code="import calc; print(calc.add(2, 3))"),
        "FINAL ANSWER: add() now sums its arguments",
    ]

    agent = _agent(tmp_path, script)

    result = agent.run("fix add", verbose=False)

    assert result["success"]

    assert "now sums" in result["answer"]

    assert result["state"]["facts"] == ["add must be commutative"]

    todos = result["state"]["todos"]["todos"]

    assert todos[0]["status"] == "done"

    assert todos[0]["evidence"] == "calc.py::add"

    assert todos[1]["status"] == "in_progress"

    assert result["state"]["files_touched"] == ["calc.py"]

    assert result["context"]["prompt"]["prompt_tokens"] > 0

    assert "a + b" in (tmp_path / "calc.py").read_text()


def test_failing_tool_triggers_recovery_and_escalates_honestly(tmp_path):

    _workspace(tmp_path)

    failing = _tool("edit_function", path="missing.py", function="nope", new_source="def nope():\n    pass\n")

    script = [failing] * 8

    agent = _agent(tmp_path, script, max_steps=8)

    result = agent.run("edit a file that does not exist", verbose=False)

    assert not result["success"]

    assert "BLOCKED" in result["answer"]

    assert "FINAL ANSWER" not in result["answer"]

    actions = [r["action"] for r in result["recoveries"]]

    assert actions, "recovery policy was never consulted"

    assert "escalate" in actions

    assert result["state"]["step"] <= 8

    assert result["state"]["failures"]


def test_malformed_call_is_repaired_and_counted(tmp_path):

    _workspace(tmp_path)

    script = [
        "sure, let me look\n<|tool_call|>\n{'tool': 'read_file', 'arguments': {'path': 'calc.py',}}",
        "FINAL ANSWER: looked at it",
    ]

    agent = _agent(tmp_path, script)

    result = agent.run("read the file", verbose=False)

    assert result["success"]

    assert result["steps"][0]["tool"] == "read_file"

    assert "add" in result["steps"][0]["result"]["content"]


def test_session_state_is_resumable(tmp_path):

    _workspace(tmp_path)

    state_path = tmp_path / "session.json"

    script = [
        _tool("update_plan", steps=["step one", "step two"], current=1),
        _tool("note_fact", fact="keep the API stable"),
        _tool("read_file", path="calc.py"),
    ]

    agent = _agent(tmp_path, script, max_steps=3, state_path=str(state_path))

    first = agent.run("some task", verbose=False)

    assert not first["success"]

    assert state_path.exists()

    saved = json.loads(state_path.read_text())

    assert saved["task"] == "some task"

    assert saved["facts"] == ["keep the API stable"]

    resumed_calls = []

    def generate(prompt):

        resumed_calls.append(prompt)

        return "FINAL ANSWER: resumed and done"

    resumed = ProductionAgent(
        workspace_path=str(tmp_path),
        sandbox=HardenedSandbox(SandboxConfig(allow_local_fallback=True)),
        generate_fn=generate,
        state_path=str(state_path),
        max_steps=5,
    )

    result = resumed.run("some task", verbose=False)

    assert result["success"]

    assert result["state"]["facts"] == ["keep the API stable"]

    assert "keep the API stable" in resumed_calls[0]

    assert "step one" in resumed_calls[0]


def test_retrieval_context_is_injected_into_the_prompt(tmp_path):

    repo = _workspace(tmp_path)

    (tmp_path / "service.py").write_text(
        "from calc import add, double\n\n\ndef run(payload: dict) -> int:\n    return double(add(payload['a'], payload['b']))\n"
    )

    index = RepoIndex.build(str(repo))

    assembler = ContextAssembler(index)

    prompts = []

    def generate(prompt):

        prompts.append(prompt)

        return "FINAL ANSWER: done"

    agent = ProductionAgent(
        workspace_path=str(repo),
        sandbox=HardenedSandbox(SandboxConfig(allow_local_fallback=True)),
        generate_fn=generate,
        context_assembler=assembler,
        max_steps=3,
    )

    result = agent.run("fix the add function used by run()", verbose=False)

    assert result["success"]

    assert "calc.py" in prompts[0]

    assert "<|repo_map|>" in prompts[0]

    assert result["context"]["retrieval"]["mode"] == "hierarchical"

    assert result["context"]["retrieval"]["tokens"] > 0


def test_control_tools_are_not_dispatched_to_the_sandbox(tmp_path):

    _workspace(tmp_path)

    agent = _agent(tmp_path, [])

    from agent.session import SessionState

    state = SessionState(task="t")

    out = agent._handle_control_tool("update_plan", {"steps": ["a", "b"]}, state)

    assert out["success"] and state.plan == ["a", "b"]

    assert agent._handle_control_tool("read_file", {"path": "x"}, state) is None

    assert agent._handle_control_tool("note_fact", {"fact": "f"}, state)["success"]


def test_tool_schemas_cover_new_control_tools():

    for name in ("update_plan", "note_fact", "revert"):

        assert name in TOOL_SCHEMAS

        assert TOOL_SCHEMAS[name]["parameters"]["type"] == "object"


def test_revert_restores_file_and_stashes(tmp_path):

    repo = _workspace(tmp_path)

    subprocess_env = dict(os.environ)

    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=subprocess_env)

    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=subprocess_env)

    subprocess.run(
        ["git", "-c", "user.email=a@b.c", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=repo,
        check=True,
        env=subprocess_env,
    )

    sandbox = HardenedSandbox(SandboxConfig(allow_local_fallback=True))

    agent = ProductionAgent(
        workspace_path=str(repo),
        sandbox=sandbox,
        generate_fn=lambda prompt: "FINAL ANSWER: noop",
    )

    (repo / "calc.py").write_text("def add(a, b):\n    return 'broken'\n")

    reverted = agent.executor.execute_tool("revert", {"files": ["calc.py"]})

    assert "error" not in reverted

    assert "return a - b" in (repo / "calc.py").read_text()

    (repo / "new.py").write_text("x = 1\n")

    stashed = agent.executor.execute_tool("revert", {"stash": True})

    assert "error" not in stashed

    assert not (repo / "new.py").exists()
