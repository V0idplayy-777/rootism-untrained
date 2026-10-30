"""Mid-task file requests and plan -> skeleton -> fill file synthesis."""

from __future__ import annotations

from pathlib import Path

from agent.production_agent import ProductionAgent
from agent.session import ContextWindow, SessionState


def _fake_generate(prompt: str) -> str:
    if "decomposition plan" in prompt:
        return "1. add(a, b): return the sum\nInvariants:\n- ints only\n"
    if "Implement the module interface" in prompt:
        return '"""Math helpers."""\n\n\ndef add(a, b):\n    """Add two integers."""\n    ...\n'
    if "Write ONLY the body of" in prompt:
        return "    return a + b"
    if "complete corrected Python file" in prompt:
        return '"""Math helpers."""\n\n\ndef add(a, b):\n    """Add two integers."""\n    return a + b\n'
    return "return a + b"


def test_synthesize_file_writes_a_valid_module(tmp_path):
    agent = ProductionAgent(generate_fn=_fake_generate, workspace_path=str(tmp_path))
    state = SessionState(task="write a math helper")
    result = agent._handle_control_tool(
        "synthesize_file",
        {"path": "mathkit.py", "instruction": "add two integers", "max_units": 2},
        state,
    )
    assert result["success"] is True, result
    assert result["units"] >= 1
    written = (tmp_path / "mathkit.py").read_text()
    assert "def add(a, b):" in written
    assert "return a + b" in written
    compile(written, "mathkit.py", "exec")


def test_synthesize_file_refuses_to_write_code_that_does_not_compile(tmp_path):
    def broken(prompt: str) -> str:
        if "decomposition plan" in prompt:
            return "1. broken(: nope\nInvariants:\n"
        if "Implement the module interface" in prompt:
            return "def broken(:\n"
        return "def broken(:\n"

    agent = ProductionAgent(generate_fn=broken, workspace_path=str(tmp_path))
    result = agent._handle_control_tool(
        "synthesize_file",
        {"path": "broken.py", "instruction": "write broken code"},
        SessionState(task="x"),
    )
    assert result["success"] is False
    assert result["errors"]
    assert not (tmp_path / "broken.py").exists()


def test_request_file_reads_by_path_and_pins_the_excerpt(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "auth.py").write_text("def check():\n    return False\n", encoding="utf-8")
    agent = ProductionAgent(generate_fn=_fake_generate, workspace_path=str(tmp_path))
    state = SessionState(task="fix auth")
    result = agent._handle_control_tool(
        "request_file", {"path": "pkg/auth.py"}, state
    )
    assert result["success"] is True
    assert "def check():" in result["content"]

    window = ContextWindow("system", state.task, state, budget_tokens=500, keep_recent_steps=1)
    window.add_observation("request_file", {"path": "pkg/auth.py"}, result)
    for _ in range(12):
        window.add_observation("list_files", {}, {"files": ["a.py", "b.py"]})
    text, stats = window.assemble()
    assert "<|file_chunk|> pkg/auth.py" in text
    assert "def check():" in text
    assert stats["working_set_files"] >= 1
