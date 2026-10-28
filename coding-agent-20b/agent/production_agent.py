"""
Production-Grade Coding Agent

Features:
- Docker-based sandboxing for code execution
- Structured tool calling (JSON schema validation)
- Git operations
- Package management
- Test runners
- Resource limits and timeouts
- Multi-file editing
"""

import json
import logging
from typing import Any

try:

    import docker


except ImportError:

    docker = None


try:

    import git


except ImportError:

    git = None


from .parser import parse_tool_call, repair_prompt
from .recovery import RecoveryAction, RecoveryPolicy, StallDetector
from .sandbox import HardenedSandbox
from .session import ContextWindow, SessionState
from .tools import ToolExecutor

logger = logging.getLogger(__name__)


try:

    import torch


except ImportError:

    torch = None


TOOL_SCHEMAS = {
    "execute_code": {
        "name": "execute_code",
        "description": "Execute Python code in a sandboxed Docker container",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)",
                    "default": 30,
                },
                "install_packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Python packages to install before execution",
                },
            },
            "required": ["code"],
        },
    },
    "read_file": {
        "name": "read_file",
        "description": "Read contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (relative to workspace)",
                }
            },
            "required": ["path"],
        },
    },
    "write_file": {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (relative to workspace)",
                },
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path", "content"],
        },
    },
    "run_tests": {
        "name": "run_tests",
        "description": "Run pytest on the workspace",
        "parameters": {
            "type": "object",
            "properties": {
                "test_path": {
                    "type": "string",
                    "description": "Path to test file or directory",
                    "default": ".",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Verbose output",
                    "default": False,
                },
            },
        },
    },
    "git_status": {
        "name": "git_status",
        "description": "Get git repository status",
        "parameters": {"type": "object", "properties": {}},
    },
    "git_diff": {
        "name": "git_diff",
        "description": "Get git diff of changes",
        "parameters": {
            "type": "object",
            "properties": {
                "staged": {
                    "type": "boolean",
                    "description": "Show staged changes only",
                    "default": False,
                }
            },
        },
    },
    "search_code": {
        "name": "search_code",
        "description": "Search for code patterns in workspace",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex or literal)",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File glob pattern (e.g., '*.py')",
                    "default": "*",
                },
            },
            "required": ["pattern"],
        },
    },
    "list_files": {
        "name": "list_files",
        "description": "List files in the workspace",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "pattern": {"type": "string", "default": "*"},
            },
        },
    },
    "apply_patch": {
        "name": "apply_patch",
        "description": "Apply a unified-diff patch (multi-file, atomic)",
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "Unified diff text"}
            },
            "required": ["patch"],
        },
    },
    "edit_function": {
        "name": "edit_function",
        "description": "Replace one Python function/class by name (AST-aware)",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "function": {"type": "string"},
                "new_source": {"type": "string"},
            },
            "required": ["path", "function", "new_source"],
        },
    },
    "git_log": {
        "name": "git_log",
        "description": "Show recent commit history",
        "parameters": {
            "type": "object",
            "properties": {"n": {"type": "integer", "default": 10}},
        },
    },
    "git_commit": {
        "name": "git_commit",
        "description": "Stage and commit changes",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "add_all": {"type": "boolean", "default": True},
            },
            "required": ["message"],
        },
    },
    "git_branch": {
        "name": "git_branch",
        "description": "List branches, or create (optionally checkout) one",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "checkout": {"type": "boolean", "default": False},
            },
        },
    },
    "git_checkout": {
        "name": "git_checkout",
        "description": "Check out a branch/commit",
        "parameters": {
            "type": "object",
            "properties": {"ref": {"type": "string"}},
            "required": ["ref"],
        },
    },
    "git_show": {
        "name": "git_show",
        "description": "Show a commit summary",
        "parameters": {
            "type": "object",
            "properties": {"ref": {"type": "string", "default": "HEAD"}},
        },
    },
    "revert": {
        "name": "revert",
        "description": (
            "Roll back unverified edits: restore specific files, or stash all "
            "changes including untracked files (nothing is destroyed)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "files": {"type": "array", "items": {"type": "string"}},
                "stash": {"type": "boolean", "default": False},
            },
        },
    },
    "update_plan": {
        "name": "update_plan",
        "description": (
            "Create or revise the ordered plan for this task. Call again when "
            "evidence invalidates the plan (max ~8 short steps)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {"type": "array", "items": {"type": "string"}},
                "current": {
                    "type": "integer",
                    "description": "1-based index of the step you are starting",
                },
                "evidence": {
                    "type": "string",
                    "description": "For the step just finished: test/symbol/file proving it",
                },
            },
            "required": ["steps"],
        },
    },
    "note_fact": {
        "name": "note_fact",
        "description": (
            "Pin a durable fact for the rest of the session (constraint, API "
            "shape, exact failing command). Survives context compaction."
        ),
        "parameters": {
            "type": "object",
            "properties": {"fact": {"type": "string"}},
            "required": ["fact"],
        },
    },
}


class ProductionAgent:
    """Production-grade coding agent with structured tool use.

    Long-horizon behaviour is explicit rather than hoped for:

    - ``SessionState`` tracks the plan, todos with evidence, pinned facts and
      failures; it can be saved/resumed.
    - ``ContextWindow`` compacts observations and older steps into a hard token
      budget, so a 20-40 step session does not silently overflow 8k.
    - ``RecoveryPolicy`` escalates retry -> repair -> change -> switch ->
      reflect -> backtrack -> escalate instead of repeating a failing call.
    - ``ContextAssembler`` (optional) injects retrieved repository evidence and
      a repo map, so SWE-bench style tasks are not limited to one file.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        workspace_path: str = "./workspace",
        max_steps: int = 20,
        sandbox=None,
        max_parse_retries: int = 3,
        generate_fn=None,
        context_assembler=None,
        context_budget_tokens: int = 6000,
        max_prompt_tokens: int = 7600,
        keep_recent_steps: int = 6,
        state_path: str | None = None,
        recovery_policy=None,
    ):

        self.model = model

        self.tokenizer = tokenizer

        self.max_steps = max_steps

        self.max_parse_retries = max_parse_retries

        self.generate_fn = generate_fn

        self.context_assembler = context_assembler

        self.context_budget_tokens = context_budget_tokens

        self.max_prompt_tokens = max_prompt_tokens

        self.keep_recent_steps = keep_recent_steps

        self.state_path = state_path

        self.recovery_policy = recovery_policy or RecoveryPolicy()

        self.sandbox = sandbox if sandbox is not None else HardenedSandbox()

        self.executor = ToolExecutor(workspace_path, self.sandbox)

    def _build_system_prompt(self, context: str = "") -> str:
        """Build system prompt with tool schemas and (optional) repo evidence."""

        tools_json = json.dumps(TOOL_SCHEMAS, indent=2)

        context_block = f"\n{context}\n" if context else ""

        return f"""You are an expert coding assistant with access to tools.

Available tools (JSON Schema):
{tools_json}
{context_block}
Operating rules:
- Start by calling update_plan with 3-6 verifiable steps, then note_fact for any
  constraint that must survive later compaction.
- Emit exactly one tool call per turn, in the native format (no prose inside):
<|tool_call|>
{{ "tool": "tool_name", "arguments": {{ ... }} }}

Example:
<|tool_call|>
{{ "tool": "read_file", "arguments": {{ "path": "src/app.py" }} }}

- Verify with tests or execution before declaring a step done, and pass the
  evidence (test name, file, symbol) when you mark it.
- If a call fails twice, change the approach (different file/selector/scope) or
  revert unverified edits — do not repeat the identical call.
- You may write intermediate reasoning in <|scratchpad|> blocks; keep them short.
- When the task is complete, respond with FINAL ANSWER: followed by the answer.
"""

    def _context_pack(self, task: str):
        """Retrieve repository context for the task (None when unavailable)."""

        if self.context_assembler is None:

            return None

        try:

            return self.context_assembler.build(
                task,
                budget_tokens=self.context_budget_tokens,
                mode="hierarchical",
            )

        except Exception as e:

            logger.warning(f"retrieval failed: {e}")

            return None

    def _generate(self, prompt: str) -> str:
        """One model turn -> text. `generate_fn` is injectable for tests."""

        if self.generate_fn is not None:

            return self.generate_fn(prompt)

        if self.model is None or self.tokenizer is None:

            raise RuntimeError("ProductionAgent needs a model+tokenizer or generate_fn")

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        budget = self.max_prompt_tokens

        if len(input_ids) > budget:

            input_ids = input_ids[:budget]

        output_ids = self.model.generate(
            input_ids=torch.tensor([input_ids]),
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
        )

        return self.tokenizer.decode(output_ids[0][len(input_ids) :])

    def _parse_tool_call(self, text: str):
        """Parse + schema-validate a tool call. Returns (call, error)."""

        return parse_tool_call(text, TOOL_SCHEMAS)

    def _handle_control_tool(self, tool: str, args: dict, state: SessionState) -> dict | None:
        """Plan/notes are state operations, not sandbox operations."""

        if tool == "update_plan":

            steps = [str(s) for s in (args.get("steps") or []) if str(s).strip()]

            if steps:

                state.plan_as(steps[:12])

            current = args.get("current")

            evidence = args.get("evidence", "")

            if isinstance(current, int) and 1 <= current <= len(state.todos.todos):

                if current > 1 and evidence:

                    state.todos.update(current - 1, "done", evidence)

                state.todos.update(current, "in_progress")

            return {
                "success": True,
                "plan": state.todos.render(),
                "note": "plan updated; continue with the current step",
            }

        if tool == "note_fact":

            fact = str(args.get("fact") or "").strip()

            if fact:

                state.note_fact(fact)

            return {"success": True, "facts": state.facts[-5:]}

        return None

    def _load_state(self, task: str) -> SessionState:

        if self.state_path:

            try:

                loaded = SessionState.load(self.state_path)

                if loaded.task == task:

                    logger.info(f"resuming session at step {loaded.step}")

                    return loaded

            except FileNotFoundError:

                pass

            except Exception as e:

                logger.warning(f"could not resume state: {e}")

        return SessionState(task=task, max_steps=self.max_steps)

    def _escalation_answer(self, state: SessionState) -> str:
        """Honest stop: unresolved error + blockers, no success claim."""

        lines = ["BLOCKED: could not complete the task within the step budget."]

        if state.tests_failed:

            lines.append("Failing tests: " + ", ".join(state.tests_failed[:8]))

        if state.recent_failures(1):

            lines.append(
                "Last error: " + (state.recent_failures(1)[0].error or "").splitlines()[0][:300]
            )

        lines.append("Open steps:")

        lines.append(state.todos.render())

        if state.files_touched:

            lines.append("Files changed: " + ", ".join(state.files_touched))

        return "\n".join(lines)

    def run(self, task: str, verbose: bool = True) -> dict[str, Any]:
        """Run the agent loop until a final answer, escalation, or step budget."""

        if verbose:

            print(f"\n{'=' * 60}")

            print(f"Task: {task}")

            print(f"{'=' * 60}\n")

        state = self._load_state(task)

        pack = self._context_pack(task)

        system_prompt = self._build_system_prompt(pack.text if pack else "")

        window = ContextWindow(
            system_prompt,
            task,
            state,
            tokenizer=self.tokenizer,
            budget_tokens=self.max_prompt_tokens,
            keep_recent_steps=self.keep_recent_steps,
        )

        stall = StallDetector(window=self.recovery_policy.stall_window)

        steps: list[dict] = []

        recoveries: list[dict] = []

        context_stats: dict = {"retrieval": pack.to_dict() if pack else {"mode": "none"}}

        while state.step < self.max_steps:

            state.step += 1

            prompt, stats = window.assemble()

            context_stats = {"prompt": stats, "retrieval": pack.to_dict() if pack else {"mode": "none"}}

            if verbose:

                print(
                    f"Step {state.step}/{self.max_steps} "
                    f"(prompt {stats['prompt_tokens']} tok, "
                    f"{stats['steps_compacted']} compacted)"
                )

            response = self._generate(prompt)

            if verbose:

                print(f"Agent: {response[:200]}...")

            if "FINAL ANSWER:" in response:

                answer = response.split("FINAL ANSWER:")[-1].strip()

                state.note_scratch(response.split("FINAL ANSWER:")[0][-400:])

                if self.state_path:

                    state.save(self.state_path)

                if verbose:

                    print(f"\n{'=' * 60}\nFinal Answer: {answer}\n{'=' * 60}\n")

                return {
                    "answer": answer,
                    "steps": steps,
                    "success": True,
                    "state": state.to_dict(),
                    "context": context_stats,
                    "recoveries": recoveries,
                }

            tool_call, parse_error = self._parse_tool_call(response)

            if tool_call is None and parse_error and "no tool call found" not in parse_error:

                window.add_assistant(response)

                state.record_failure(None, None, parse_error)

                decision = self.recovery_policy.decide(
                    state, parse_error=True, error=parse_error
                )

                recoveries.append(decision.to_dict())

                window.add_user(
                    RecoveryPolicy.reflect_prompt(state, decision)
                    if decision.action != RecoveryAction.REPAIR
                    else repair_prompt(parse_error, response)
                )

                steps.append(
                    {
                        "tool": None,
                        "arguments": {},
                        "result": {"error": parse_error},
                        "repaired": False,
                        "recovery": decision.to_dict(),
                        "step": state.step,
                    }
                )

                if decision.action == RecoveryAction.ESCALATE:

                    break

                continue

            if tool_call is None:

                window.add_assistant(response)

                steps.append(
                    {
                        "tool": None,
                        "arguments": {},
                        "result": {"error": "no tool call"},
                        "step": state.step,
                    }
                )

                if stall.observe(state, f"prose:{response[:60]}"):

                    decision = self.recovery_policy.decide(state, stalled=True)

                    recoveries.append(decision.to_dict())

                    window.add_user(RecoveryPolicy.reflect_prompt(state, decision))

                    if decision.action == RecoveryAction.ESCALATE:

                        break

                else:

                    window.add_user(
                        "Continue with exactly one tool call, or FINAL ANSWER: if done."
                    )

                continue

            tool_name = tool_call["tool"]

            arguments = tool_call["arguments"]

            if verbose:

                print(f"  Tool: {tool_name}\n  Args: {arguments}")

            control = self._handle_control_tool(tool_name, arguments, state)

            if control is not None:

                window.add_assistant(response)

                window.add_user(
                    "<|observation|> control\n"
                    + json.dumps(control, indent=2)
                    + "\nContinue with the next verifiable step."
                )

                steps.append(
                    {
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": control,
                        "step": state.step,
                    }
                )

                continue

            result = self.executor.execute_tool(tool_name, arguments)

            if verbose:

                print(f"  Result: {json.dumps(result, indent=2)[:200]}...")

            window.add_assistant(response)

            window.add_observation(tool_name, arguments, result)

            step_record = {
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
                "step": state.step,
            }

            steps.append(step_record)

            if isinstance(result, dict) and result.get("error"):

                decision = self.recovery_policy.decide(
                    state, tool_name, arguments, str(result["error"])
                )

                step_record["recovery"] = decision.to_dict()

                recoveries.append(decision.to_dict())

                if decision.action == RecoveryAction.BACKTRACK:

                    files = arguments.get("path")

                    revert = self.executor.execute_tool(
                        "revert",
                        {"files": [files]} if files else {"stash": True},
                    )

                    step_record["backtrack"] = revert

                    window.add_observation("revert", {}, revert)

                window.add_user(RecoveryPolicy.reflect_prompt(state, decision))

                if decision.action == RecoveryAction.ESCALATE:

                    break

            elif stall.observe(state, f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"):

                decision = self.recovery_policy.decide(state, stalled=True)

                recoveries.append(decision.to_dict())

                window.add_user(RecoveryPolicy.reflect_prompt(state, decision))

                if decision.action == RecoveryAction.ESCALATE:

                    break

        if self.state_path:

            state.save(self.state_path)

        answer = self._escalation_answer(state)

        return {
            "answer": answer,
            "steps": steps,
            "success": False,
            "state": state.to_dict(),
            "context": context_stats,
            "recoveries": recoveries,
        }

    def _format_messages(self, messages: list[dict]) -> str:
        """Legacy flat formatter (kept for callers that build prompts manually)."""

        parts = []

        for msg in messages:

            role = msg["role"]

            content = msg["content"]

            if role == "system":

                parts.append(f"<|system|>\n{content}\n<eos>")

            elif role == "user":

                parts.append(f"<|user|>\n{content}\n<eos>")

            elif role == "assistant":

                parts.append(f"<|assistant|>\n{content}\n<eos>")

            elif role == "tool":

                parts.append(f"<|tool_result|>\n{content}\n<eos>")

        return "\n".join(parts)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    print("Production Agent Example")
    print("=" * 60)
    print("\nTo use the agent:")
    print("""
from agent.production_agent import ProductionAgent
from agent.repo_context import RepoIndex, ContextAssembler

index = RepoIndex.build("./my_project")
agent = ProductionAgent(model, tokenizer, workspace_path="./my_project",
                        context_assembler=ContextAssembler(index, tokenizer))
result = agent.run("Fix the failing expiry test in tests/test_auth.py")
print(result["answer"])
""")
