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
from pathlib import Path
from typing import Any

try:

    import docker


except ImportError:

    docker = None


try:

    import git


except ImportError:

    git = None


from .memory import MemoryStore
from .parser import parse_tool_call, repair_prompt
from .planning import AssumptionLedger, MilestoneCheckpointer, PlanGraph
from .recovery import RecoveryAction, RecoveryPolicy, StallDetector
from .sandbox import HardenedSandbox
from .scratch import ScratchWorkspace
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
    "request_file": {
        "name": "request_file",
        "description": (
            "Ask for a specific file mid-task (path) or the files matching a symbol/query; "
            "the returned excerpt is pinned in the working set so it survives compaction"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace"},
                "query": {"type": "string", "description": "Symbol or phrase when the path is unknown"},
                "max_chars": {"type": "integer", "default": 4000},
            },
        },
    },
    "synthesize_file": {
        "name": "synthesize_file",
        "description": (
            "Generate a whole module with plan -> skeleton -> per-unit fill (bounded calls), "
            "validate it, then write it. Use this instead of one huge write_file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Destination path relative to the workspace"},
                "instruction": {"type": "string", "description": "What the module must do"},
                "constraints": {"type": "string", "description": "Extra constraints (stdlib, typing, ...)"},
                "max_units": {"type": "integer", "default": 8},
            },
            "required": ["path", "instruction"],
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
    
    "project_health": {
        "name": "project_health",
        "description": "Check project health: broken imports, missing tests, circular dependencies",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "default": "."}}},
    },
    "onboarding": {
        "name": "onboarding",
        "description": "Produce architecture summary + entry-point map before editing (required on new repos)",
        "parameters": {"type": "object", "properties": {"subdir": {"type": "string"}, "force": {"type": "boolean", "default": False}}},
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
            "Create or revise the plan for this task. Steps run in dependency "
            "order: prefix a step with \"[after 1,3]\" to make it wait for "
            "steps 1 and 3; plain steps chain sequentially. Call again when "
            "evidence invalidates part of the plan (max ~8 short steps)."
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
    "record_assumption": {
        "name": "record_assumption",
        "description": (
            "Record an assumption you are about to rely on (API shape, file "
            "format, environment, intent). Unverified assumptions are shown "
            "in every prompt until confirmed or refuted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    },
    "verify_assumption": {
        "name": "verify_assumption",
        "description": (
            "Mark an assumption confirmed or refuted, with the evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Assumption id (A<N>)"},
                "status": {"type": "string", "enum": ["confirmed", "refuted"]},
                "how": {"type": "string", "description": "Evidence that decided it"},
            },
            "required": ["id", "status"],
        },
    },
    "scratch_write": {
        "name": "scratch_write",
        "description": (
            "Offload intermediate reasoning/findings to disk (survives "
            "context compaction). The scratch index, not the content, stays "
            "in your prompt."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "text": {"type": "string"},
                "mode": {"type": "string", "enum": ["append", "replace"], "default": "append"},
            },
            "required": ["topic", "text"],
        },
    },
    "scratch_read": {
        "name": "scratch_read",
        "description": "Reload a scratch topic you wrote earlier.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "max_chars": {"type": "integer", "default": 4000},
            },
            "required": ["topic"],
        },
    },
    "scratch_list": {
        "name": "scratch_list",
        "description": "List scratch topics (name, entries, size, summary).",
        "parameters": {"type": "object", "properties": {}},
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


try:

    from .deploy_tool import tool_schemas as _project_tool_schemas

    TOOL_SCHEMAS.update(_project_tool_schemas())

except Exception:  # pragma: no cover - schema import must never break the agent

    pass


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
        scratch: ScratchWorkspace | None = None,
        memory: MemoryStore | None = None,
        milestones: MilestoneCheckpointer | None = None,
        enable_milestones: bool = True,
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

        self.workspace_path = workspace_path

        self.recovery_policy = recovery_policy or RecoveryPolicy()

        self.sandbox = sandbox if sandbox is not None else HardenedSandbox()

        self.executor = ToolExecutor(workspace_path, self.sandbox)


        self.scratch = scratch if scratch is not None else ScratchWorkspace(
            Path(workspace_path) / ".agent" / "scratch"
        )

        self.memory = memory if memory is not None else MemoryStore(
            Path(workspace_path) / ".agent" / "memory.json"
        )

        self.milestones = milestones if milestones is not None else MilestoneCheckpointer(
            workspace_path
        )

        self.enable_milestones = enable_milestones

        self.plan_graph = PlanGraph()

        self.assumptions = AssumptionLedger()


    def _progress_metrics(self, state) -> str:
        """Explicit progress metrics the agent can see (row 5)."""
        try:
            total = len(state.todos.todos)
            done = sum(1 for t in state.todos.todos if t.status == "done")
            pending = total - done
            tests_pass = len(getattr(state, "tests_passed", []) or [])
            tests_fail = len(getattr(state, "tests_failed", []) or [])
            unverified = len([a for a in self.assumptions.assumptions if a.status == "unverified"]) if hasattr(self, "assumptions") else 0
            return (
                f"Progress: {done}/{total} todos done ({pending} pending) | "
                f"tests: {tests_pass} passed, {tests_fail} failed | "
                f"unverified assumptions: {unverified} | "
                f"milestones: {len(self.milestones.milestones) if hasattr(self.milestones, 'milestones') else 0} | "
                f"files touched: {len(getattr(state, 'files_touched', []) or [])}"
            )
        except Exception:
            return "Progress: unknown"

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
- For a new module longer than ~80 lines, call `synthesize_file` (plan -> skeleton -> fill)
  instead of one giant `write_file`; it validates before writing.
- Use `request_file` with a path or a symbol when you need a file that is not in context.
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

    def _request_file(self, args: dict) -> dict:
        """Return a file (or the best match for a query) for mid-task inspection."""
        path = str(args.get("path") or "").strip()
        query = str(args.get("query") or "").strip()
        limit = int(args.get("max_chars") or 4000)
        if not path and query:
            from agent.semantic_index import SemanticIndex

            hits = SemanticIndex.open(self.workspace_path).hybrid_search(query, k=3)
            if not hits:
                return {"success": False, "error": f"no file matches {query!r}"}
            path = hits[0].document.path
        if not path:
            return {"success": False, "error": "path or query is required"}
        result = self.executor.read_file(path)
        if result.get("error"):
            return {"success": False, "error": result["error"], "path": path}
        content = str(result.get("content") or "")
        return {
            "success": True,
            "path": path,
            "content": content[:limit],
            "truncated": len(content) > limit,
            "lines": len(content.splitlines()),
            "pinned": True,
        }

    def _synthesize_file(self, args: dict) -> dict:
        """Plan -> skeleton -> fill a module, then write only if it validates."""
        from model.longform import LongFormSynthesizer

        path = str(args.get("path") or "").strip()
        instruction = str(args.get("instruction") or "").strip()
        if not path or not instruction:
            return {"success": False, "error": "path and instruction are required"}
        synthesiser = LongFormSynthesizer(
            lambda prompt, _max_tokens: self._generate(prompt),
            max_units=int(args.get("max_units") or 8),
        )
        result = synthesiser.synthesise(instruction, str(args.get("constraints") or ""))
        if result.errors:
            return {
                "success": False,
                "error": "generated module does not compile",
                "errors": result.errors[:6],
                "trace": result.trace,
                "chars": len(result.code),
            }
        written = self.executor.write_file(path, result.code)
        if written.get("error"):
            return {"success": False, "error": written["error"], "path": path}
        return {
            "success": True,
            "path": path,
            "chars": len(result.code),
            "units": result.plan.size,
            "repairs": result.repairs,
            "trace": result.trace,
        }

    def _handle_control_tool(self, tool: str, args: dict, state: SessionState) -> dict | None:
        """Plan/notes/assumptions/scratch are state operations, not sandbox."""

        project_result = self._handle_project_tool(tool, args)

        if project_result is not None:

            return project_result

        if tool == "request_file":

            return self._request_file(args)

        if tool == "synthesize_file":

            return self._synthesize_file(args)

        if tool == "update_plan":

            steps = [str(s) for s in (args.get("steps") or []) if str(s).strip()]

            if steps:

                state.plan_as(steps[:12])

                self.plan_graph = PlanGraph.parse(steps[:12])

            current = args.get("current")

            evidence = args.get("evidence", "")

            if isinstance(current, int) and 1 <= current <= len(state.todos.todos):

                if current > 1 and evidence:

                    state.todos.update(current - 1, "done", evidence)



                    self._checkpoint_milestone(state, current - 1, evidence)

                state.todos.update(current, "in_progress")

            ready = [s.id for s in self.plan_graph.ready()]

            return {
                "success": True,
                "plan": state.todos.render(),
                "ready_steps": ready,
                "note": "plan updated; continue with the current step",
            }

        if tool == "note_fact":

            fact = str(args.get("fact") or "").strip()

            if fact:

                state.note_fact(fact)

            return {"success": True, "facts": state.facts[-5:]}

        if tool == "record_assumption":

            text = str(args.get("text") or "").strip()

            if not text:

                return {"success": False, "error": "assumption text is required"}

            assumption = self.assumptions.add(text, step=state.step)

            return {
                "success": True,
                "assumption": assumption.to_dict(),
                "note": f"recorded as A{assumption.id}; verify or refute it with evidence",
            }

        if tool == "verify_assumption":

            try:

                assumption = self.assumptions.record(
                    int(args.get("id")),
                    str(args.get("status")),
                    str(args.get("how") or ""),
                )

            except (ValueError, KeyError, TypeError) as e:

                return {"success": False, "error": str(e)}

            if assumption.status == "refuted":

                return {
                    "success": True,
                    "assumption": assumption.to_dict(),
                    "note": (
                        "refuted assumption: if any completed step relied on "
                        "it, re-plan that part (update_plan) before continuing"
                    ),
                }

            return {"success": True, "assumption": assumption.to_dict()}

        if tool == "scratch_write":

            try:

                receipt = self.scratch.write(
                    str(args.get("topic") or "notes"),
                    str(args.get("text") or ""),
                    mode=str(args.get("mode") or "append"),
                )

            except ValueError as e:

                return {"success": False, "error": str(e)}

            return receipt

        if tool == "scratch_read":

            try:

                text = self.scratch.read(
                    str(args.get("topic") or ""),
                    max_chars=int(args.get("max_chars") or 4000),
                )

            except (KeyError, ValueError) as e:

                return {"success": False, "error": str(e)}

            return {"success": True, "topic": args.get("topic"), "content": text}

        if tool == "scratch_list":

            return {"success": True, "topics": self.scratch.list_topics()}

        return None

    def _handle_project_tool(self, tool: str, args: dict) -> dict | None:
        """Scaffold/deploy/map/search/checkpoint: the shipping tools.

        They reuse the CLI/UI implementations, so an agent deploy and a human
        deploy are the same code path with the same guarantees.
        """

        if tool not in {"deploy", "build_app", "scaffold_project", "project_map", "search_project", "checkpoint"}:

            return None

        from .deploy_tool import handle_tool

        return handle_tool(tool, args, self.workspace_path, emit=self._tool_log)

    def _tool_log(self, message: str) -> None:
        """Progress line for long operations (deploy build stages)."""

        emit = getattr(self, "on_progress", None)

        if callable(emit):

            emit(str(message))

    def _checkpoint_milestone(self, state: SessionState, step_index: int, evidence: str) -> None:
        """Snapshot verified progress (git commit when possible).

        ``step_index`` is the 1-based todo id just marked done (the step
        *before* the one now starting), matching TodoList.update().
        """

        if not self.enable_milestones:

            return

        try:

            todo = state.todos.todos[step_index - 1]

            label = f"{todo.text[:60]} ({evidence[:40]})" if evidence else todo.text[:60]

            self.milestones.checkpoint(
                label,
                step=state.step,
                files_touched=list(state.files_touched),
                tests_passed=list(state.tests_passed),
                executor=self.executor,
            )

        except Exception as e:

            logger.warning(f"milestone checkpoint failed: {e}")

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

        self._load_state_extras()

        pack = self._context_pack(task)

        system_prompt = self._build_system_prompt(pack.text if pack else "")




        try:
            from agent.onboarding import ensure_onboarding
            rep = ensure_onboarding(self.workspace_path, force=False)
            task_block = task + "\n\nOnboarding (auto): " + rep.render(max_chars=2000)
        except Exception:
            task_block = task

        progress_block = self._progress_metrics(state) if 'state' in locals() else ""
        if progress_block:
            task_block += "\n\n" + progress_block

        memory_block = self.memory.render(k=6)

        if memory_block:

            task_block += "\n\n" + memory_block

        scratch_block = self.scratch.render_index(max_chars=500)

        if scratch_block:

            task_block += "\n\n" + scratch_block

        assumption_block = self.assumptions.render()

        if assumption_block:

            task_block += "\n\n" + assumption_block

        window = ContextWindow(
            system_prompt,
            task_block,
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

            wrap = self.recovery_policy.wrap_up_needed(state)

            if wrap is not None and state.step < state.max_steps:

                recoveries.append(wrap.to_dict())

                window.add_user(RecoveryPolicy.reflect_prompt(state, wrap))

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

                self._save_state_extras()

                try:

                    self.memory.learn_from_session(state)

                except Exception as e:

                    logger.warning(f"memory learning failed: {e}")

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



                    milestone = self.milestones.last_verified()

                    if milestone is not None:

                        files, _m = self.milestones.rollback_files(state.files_touched)

                    else:

                        files = [arguments.get("path")] if arguments.get("path") else None

                    revert = self.executor.execute_tool(
                        "revert",
                        {"files": files} if files else {"stash": True},
                    )

                    step_record["backtrack"] = revert

                    step_record["backtrack_milestone"] = (
                        milestone.to_dict() if milestone else None
                    )

                    window.add_observation("revert", {}, revert)



                    current = state.todos.next_pending()

                    if current is not None:

                        affected = self.plan_graph.reset_subtree(current.id)

                        if affected:

                            step_record["replanned_steps"] = sorted(affected)

                window.add_user(RecoveryPolicy.reflect_prompt(state, decision))

                if decision.action == RecoveryAction.ESCALATE:

                    break

            else:

                self.recovery_policy.note_success(tool_name)

                if stall.observe(state, f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"):

                    decision = self.recovery_policy.decide(state, stalled=True)

                    recoveries.append(decision.to_dict())

                    window.add_user(RecoveryPolicy.reflect_prompt(state, decision))

                    if decision.action == RecoveryAction.ESCALATE:

                        break

        if self.state_path:

            state.save(self.state_path)

        self._save_state_extras()

        try:

            self.memory.learn_from_session(state)

        except Exception as e:

            logger.warning(f"memory learning failed: {e}")

        answer = self._escalation_answer(state)

        return {
            "answer": answer,
            "steps": steps,
            "success": False,
            "state": state.to_dict(),
            "context": context_stats,
            "recoveries": recoveries,
            "milestones": self.milestones.to_dict(),
        }

    def _extras_path(self) -> Path:

        if self.state_path:

            return Path(str(self.state_path) + ".extras.json")

        return Path(self.workspace_path) / ".agent" / "session.extras.json"

    def _save_state_extras(self) -> None:

        """Persist plan graph, assumptions and milestones next to the state."""

        try:

            data = {
                "plan": self.plan_graph.to_dict(),
                "assumptions": self.assumptions.to_dict(),
                "milestones": self.milestones.to_dict(),
            }

            path = self._extras_path()

            path.parent.mkdir(parents=True, exist_ok=True)

            tmp = path.with_suffix(".tmp")

            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")

            tmp.replace(path)

        except OSError as e:

            logger.warning(f"could not save session extras: {e}")

    def _load_state_extras(self) -> None:

        path = self._extras_path()

        if not path.is_file():

            return

        try:

            data = json.loads(path.read_text(encoding="utf-8"))

            self.plan_graph = PlanGraph.from_dict(data.get("plan", {}))

            self.assumptions = AssumptionLedger.from_dict(data.get("assumptions", {}))

            self.milestones = MilestoneCheckpointer.from_dict(
                data.get("milestones", {}), str(self.executor.workspace)
            )

        except (json.JSONDecodeError, OSError, ValueError) as e:

            logger.warning(f"could not resume session extras: {e}")

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
