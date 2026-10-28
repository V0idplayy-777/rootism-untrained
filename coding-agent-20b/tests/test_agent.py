import pytest

from agent.parser import parse_tool_call, repair_prompt
from agent.production_agent import TOOL_SCHEMAS
from agent.safety import check_code_safety, redact_secrets, sanitize_observation
from agent.sandbox import HardenedSandbox, SandboxConfig, SandboxUnavailable
from agent.tools import ToolExecutor


def test_parse_native_fenced_and_repaired_calls():

    call, err = parse_tool_call(
        '<|tool_call|>\n{"tool": "read_file", "arguments": {"path": "a.py"}}',
        TOOL_SCHEMAS,
    )

    assert err is None and call["tool"] == "read_file"

    call, _ = parse_tool_call(
        '```json\n{"tool": "git_log", "arguments": {}}\n```', TOOL_SCHEMAS
    )

    assert call["tool"] == "git_log" and call["arguments"]["n"] == 10

    call, err = parse_tool_call(
        '{"tool": "read_file", "arguments": {"path": "a.py",}}', TOOL_SCHEMAS
    )

    assert call is not None


def test_parse_rejects_unknown_tool_and_bad_types():

    call, err = parse_tool_call('{"tool": "nope", "arguments": {}}', TOOL_SCHEMAS)

    assert call is None and "unknown tool" in err

    assert "corrected call" in repair_prompt(err, "bad")


def test_sandbox_fail_closed_without_docker():

    sandbox = HardenedSandbox(SandboxConfig(allow_local_fallback=False))

    if not sandbox.available:

        with pytest.raises(SandboxUnavailable):

            sandbox.execute("print(1)", ".", timeout=5)


def test_tools_patch_ast_git_and_safety(tmp_path):

    sandbox = HardenedSandbox(SandboxConfig(allow_local_fallback=True))

    ex = ToolExecutor(str(tmp_path), sandbox)

    assert ex.write_file("m.py", "def old():\n    return 1\n")["success"]

    assert ex.edit_function("m.py", "old", "def old():\n    return 42\n")["success"]

    assert "return 42" in ex.read_file("m.py")["content"]

    patch = "--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,2 @@\n def old():\n-    return 42\n+    return 43\n"

    assert ex.apply_patch(patch)["success"]

    assert "return 43" in ex.read_file("m.py")["content"]

    assert ex.execute_code("print(6*7)", language="python")["stdout"].strip() == "42"

    assert check_code_safety("import os; os.system('x')")[0] is True

    assert check_code_safety("rm -rf /")[0] is False

    assert "redacted" in redact_secrets("api_key = hunter2secret")

    assert len(sanitize_observation("x" * 9000)) < 9000


def test_path_escape_blocked(tmp_path):

    sandbox = HardenedSandbox(SandboxConfig(allow_local_fallback=True))

    ex = ToolExecutor(str(tmp_path), sandbox)

    assert "error" in ex.read_file("../outside.txt")


def test_production_agent_repair_loop_end_to_end(tmp_path):

    torch = pytest.importorskip("torch")

    from agent.production_agent import ProductionAgent

    class StubTok:

        next_response = ""

        def encode(self, text, add_special_tokens=True):

            return [1]

        def decode(self, ids, skip_special_tokens=True):

            return StubTok.next_response

    class StubModel:

        def __init__(self):

            self.calls = 0

        def generate(self, input_ids=None, **kw):

            self.calls += 1

            if self.calls == 1:

                StubTok.next_response = (
                    '{"tool": "read_file", "arguments": {"path": 123}}'
                )

            elif self.calls == 2:

                StubTok.next_response = '<|tool_call|>\n{"tool": "write_file", "arguments": {"path": "hi.py", "content": "x = 1\\n"}}'

            else:

                StubTok.next_response = "FINAL ANSWER: done"

            return torch.tensor([[1, 2, 3]])

    sandbox = HardenedSandbox(SandboxConfig(allow_local_fallback=True))

    agent = ProductionAgent(
        StubModel(), StubTok(), workspace_path=str(tmp_path), sandbox=sandbox
    )

    result = agent.run("write hi.py", verbose=False)

    assert result["success"] and result["answer"] == "done"

    assert (tmp_path / "hi.py").exists()
