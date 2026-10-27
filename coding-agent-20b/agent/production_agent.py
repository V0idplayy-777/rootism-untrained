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

import os
import json
import subprocess
import tempfile
import shutil
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
try:
    import docker  # noqa: F401  (kept for backwards-compat introspection)
except ImportError:
    docker = None  # type: ignore
try:
    import git  # noqa: F401
except ImportError:
    git = None  # type: ignore

from .sandbox import DockerSandbox, HardenedSandbox, SandboxConfig, SandboxUnavailable  # noqa: E402
from .tools import ToolExecutor  # noqa: E402
from .parser import parse_tool_call, repair_prompt  # noqa: E402
from .safety import sanitize_observation  # noqa: E402

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore


# ============================================================
# TOOL DEFINITIONS (JSON Schema)
# ============================================================

TOOL_SCHEMAS = {
    "execute_code": {
        "name": "execute_code",
        "description": "Execute Python code in a sandboxed Docker container",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)",
                    "default": 30
                },
                "install_packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Python packages to install before execution"
                }
            },
            "required": ["code"]
        }
    },
    "read_file": {
        "name": "read_file",
        "description": "Read contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (relative to workspace)"
                }
            },
            "required": ["path"]
        }
    },
    "write_file": {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (relative to workspace)"
                },
                "content": {
                    "type": "string",
                    "description": "File content"
                }
            },
            "required": ["path", "content"]
        }
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
                    "default": "."
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Verbose output",
                    "default": False
                }
            }
        }
    },
    "git_status": {
        "name": "git_status",
        "description": "Get git repository status",
        "parameters": {
            "type": "object",
            "properties": {}
        }
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
                    "default": False
                }
            }
        }
    },
    "search_code": {
        "name": "search_code",
        "description": "Search for code patterns in workspace",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern (regex or literal)"},
                "file_pattern": {"type": "string", "description": "File glob pattern (e.g., '*.py')", "default": "*"}
            },
            "required": ["pattern"]
        }
    },
    "list_files": {
        "name": "list_files", "description": "List files in the workspace",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "default": "."},
            "pattern": {"type": "string", "default": "*"}}}
    },
    "apply_patch": {
        "name": "apply_patch", "description": "Apply a unified-diff patch (multi-file, atomic)",
        "parameters": {"type": "object", "properties": {
            "patch": {"type": "string", "description": "Unified diff text"}}, "required": ["patch"]}
    },
    "edit_function": {
        "name": "edit_function", "description": "Replace one Python function/class by name (AST-aware)",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "function": {"type": "string"},
            "new_source": {"type": "string"}}, "required": ["path", "function", "new_source"]}
    },
    "git_log": {
        "name": "git_log", "description": "Show recent commit history",
        "parameters": {"type": "object", "properties": {"n": {"type": "integer", "default": 10}}}
    },
    "git_commit": {
        "name": "git_commit", "description": "Stage and commit changes",
        "parameters": {"type": "object", "properties": {
            "message": {"type": "string"}, "add_all": {"type": "boolean", "default": True}}, "required": ["message"]}
    },
    "git_branch": {
        "name": "git_branch", "description": "List branches, or create (optionally checkout) one",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "checkout": {"type": "boolean", "default": False}}}
    },
    "git_checkout": {
        "name": "git_checkout", "description": "Check out a branch/commit",
        "parameters": {"type": "object", "properties": {"ref": {"type": "string"}}, "required": ["ref"]}
    },
    "git_show": {
        "name": "git_show", "description": "Show a commit summary",
        "parameters": {"type": "object", "properties": {"ref": {"type": "string", "default": "HEAD"}}}
    }
}


# ============================================================
# DOCKER SANDBOX
# ============================================================

# DockerSandbox / ToolExecutor live in agent.sandbox / agent.tools
# (re-exported above for backwards compatibility).


# ============================================================
# AGENT
# ============================================================

class ProductionAgent:
    """Production-grade coding agent with structured tool use."""
    
    def __init__(
        self,
        model,
        tokenizer,
        workspace_path: str = "./workspace",
        max_steps: int = 20,
        sandbox=None,
        max_parse_retries: int = 3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_parse_retries = max_parse_retries
        self.sandbox = sandbox if sandbox is not None else HardenedSandbox()
        self.executor = ToolExecutor(workspace_path, self.sandbox)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool schemas."""
        tools_json = json.dumps(TOOL_SCHEMAS, indent=2)
        
        return f"""You are an expert coding assistant with access to tools.

Available tools (JSON Schema):
{tools_json}

Emit tool calls in the native format (no prose inside the call):
<|tool_call|>
{{"tool": "tool_name", "arguments": {{...}}}}

Example:
<|tool_call|>
{{"tool": "execute_code", "arguments": {{"code": "print('Hello')", "timeout": 30}}}}

Always think step-by-step and explain your reasoning before calling tools.
When the task is complete, respond with FINAL ANSWER: followed by your answer.
"""
    
    def _parse_tool_call(self, text: str):
        """Parse + schema-validate a tool call. Returns (call, error)."""
        return parse_tool_call(text, TOOL_SCHEMAS)

    def run(self, task: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Run agent on a task.
        
        Args:
            task: Task description
            verbose: Print step-by-step output
            
        Returns:
            Dict with answer, steps, success
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task}")
            print(f"{'='*60}\n")
        
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": task},
        ]
        
        steps = []
        
        for step_num in range(self.max_steps):
            if verbose:
                print(f"Step {step_num + 1}/{self.max_steps}")
            
            # Generate response
            prompt = self._format_messages(messages)
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            
            output_ids = self.model.generate(
                input_ids=torch.tensor([input_ids]),
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
            )
            
            response = self.tokenizer.decode(output_ids[0][len(input_ids):])
            
            if verbose:
                print(f"Agent: {response[:200]}...")
            
            # Check for final answer
            if "FINAL ANSWER:" in response:
                answer = response.split("FINAL ANSWER:")[-1].strip()
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Final Answer: {answer}")
                    print(f"{'='*60}\n")
                
                return {
                    'answer': answer,
                    'steps': steps,
                    'success': True,
                }
            
            # Parse and execute tool call (with bounded repair loop)
            tool_call, parse_error = self._parse_tool_call(response)
            if tool_call is None and parse_error and "no tool call found" not in parse_error:
                # Model attempted a call but it was malformed/invalid: ask for a fix.
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": repair_prompt(parse_error, response)})
                steps.append({'tool': None, 'arguments': {}, 'result': {'error': parse_error}, 'repaired': False})
                continue
            if tool_call:
                tool_name = tool_call['tool']
                arguments = tool_call['arguments']
                
                if verbose:
                    print(f"  Tool: {tool_name}")
                    print(f"  Args: {arguments}")
                
                # Execute tool
                result = self.executor.execute_tool(tool_name, arguments)
                
                if verbose:
                    result_str = json.dumps(result, indent=2)
                    print(f"  Result: {result_str[:200]}...")
                
                # Add to conversation
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "tool",
                    "content": sanitize_observation(json.dumps(result, indent=2)),
                })
                
                steps.append({
                    'tool': tool_name,
                    'arguments': arguments,
                    'result': result,
                })
            else:
                # No tool call, add to conversation
                messages.append({"role": "assistant", "content": response})
        
        # Max steps reached
        return {
            'answer': 'Max steps reached without final answer',
            'steps': steps,
            'success': False,
        }
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages for model input."""
        parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                parts.append(f"<|system|>\n{content}\n<eos>")
            elif role == 'user':
                parts.append(f"<|user|>\n{content}\n<eos>")
            elif role == 'assistant':
                parts.append(f"<|assistant|>\n{content}\n<eos>")
            elif role == 'tool':
                parts.append(f"<|tool_result|>\n{content}\n<eos>")
        
        return "\n".join(parts)


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Production Agent Example")
    print("="*60)
    print("\nTo use the agent:")
    print("""
from agent.production_agent import ProductionAgent

agent = ProductionAgent(model, tokenizer, workspace_path='./my_project')
result = agent.run("Create a Flask API with tests")
print(result['answer'])
""")
