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
import docker
import git

logger = logging.getLogger(__name__)


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
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex or literal)"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File glob pattern (e.g., '*.py')",
                    "default": "*"
                }
            },
            "required": ["pattern"]
        }
    }
}


# ============================================================
# DOCKER SANDBOX
# ============================================================

class DockerSandbox:
    """Secure code execution in Docker container."""
    
    def __init__(
        self,
        image: str = "python:3.11-slim",
        memory_limit: str = "512m",
        cpu_quota: int = 50000,  # 50% of one CPU
        network_disabled: bool = True,
    ):
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.network_disabled = network_disabled
        
        try:
            self.client = docker.from_env()
            self.available = True
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
            self.available = False
    
    def execute(
        self,
        code: str,
        workspace_path: str,
        timeout: int = 30,
        install_packages: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute code in Docker container.
        
        Args:
            code: Python code to execute
            workspace_path: Path to mount as workspace
            timeout: Execution timeout in seconds
            install_packages: Packages to pip install
            
        Returns:
            Dict with stdout, stderr, exit_code, timed_out
        """
        if not self.available:
            return self._fallback_execute(code, workspace_path, timeout)
        
        try:
            # Create temporary script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                script_path = f.name
            
            # Build pip install command
            install_cmd = ""
            if install_packages:
                packages = " ".join(install_packages)
                install_cmd = f"pip install -q {packages} && "
            
            # Run container
            container = self.client.containers.run(
                self.image,
                command=f"/bin/bash -c '{install_cmd}python /script.py'",
                volumes={
                    workspace_path: {'bind': '/workspace', 'mode': 'rw'},
                    script_path: {'bind': '/script.py', 'mode': 'ro'},
                },
                working_dir='/workspace',
                mem_limit=self.memory_limit,
                cpu_quota=self.cpu_quota,
                network_disabled=self.network_disabled,
                detach=True,
            )
            
            # Wait for completion with timeout
            try:
                result = container.wait(timeout=timeout)
                stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
                stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
                
                return {
                    'stdout': stdout,
                    'stderr': stderr,
                    'exit_code': result['StatusCode'],
                    'timed_out': False,
                }
            except Exception as e:
                # Timeout
                container.kill()
                return {
                    'stdout': '',
                    'stderr': f'Execution timed out after {timeout}s',
                    'exit_code': -1,
                    'timed_out': True,
                }
            finally:
                container.remove(force=True)
                os.unlink(script_path)
        
        except Exception as e:
            logger.error(f"Docker execution error: {e}")
            return {
                'stdout': '',
                'stderr': f'Execution error: {str(e)}',
                'exit_code': -1,
                'timed_out': False,
            }
    
    def _fallback_execute(self, code: str, workspace_path: str, timeout: int) -> Dict:
        """Fallback execution without Docker (less secure)."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                script_path = f.name
            
            result = subprocess.run(
                ['python3', script_path],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            os.unlink(script_path)
            
            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'exit_code': result.returncode,
                'timed_out': False,
            }
        except subprocess.TimeoutExpired:
            return {
                'stdout': '',
                'stderr': f'Execution timed out after {timeout}s',
                'exit_code': -1,
                'timed_out': True,
            }
        except Exception as e:
            return {
                'stdout': '',
                'stderr': f'Execution error: {str(e)}',
                'exit_code': -1,
                'timed_out': False,
            }


# ============================================================
# TOOL IMPLEMENTATIONS
# ============================================================

class ToolExecutor:
    """Execute tools with validation and error handling."""
    
    def __init__(self, workspace_path: str, sandbox: DockerSandbox):
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.sandbox = sandbox
        
        # Initialize git repo if not exists
        try:
            self.repo = git.Repo(self.workspace_path)
        except git.InvalidGitRepositoryError:
            self.repo = git.Repo.init(self.workspace_path)
    
    def execute_code(self, code: str, timeout: int = 30, install_packages: List[str] = None) -> Dict:
        """Execute Python code in sandbox."""
        return self.sandbox.execute(
            code=code,
            workspace_path=str(self.workspace_path),
            timeout=timeout,
            install_packages=install_packages,
        )
    
    def read_file(self, path: str) -> Dict:
        """Read file contents."""
        try:
            file_path = self.workspace_path / path
            if not file_path.exists():
                return {'error': f'File not found: {path}'}
            
            content = file_path.read_text(encoding='utf-8')
            return {'content': content, 'size': len(content)}
        except Exception as e:
            return {'error': str(e)}
    
    def write_file(self, path: str, content: str) -> Dict:
        """Write file contents."""
        try:
            file_path = self.workspace_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
            return {'success': True, 'path': path, 'size': len(content)}
        except Exception as e:
            return {'error': str(e)}
    
    def run_tests(self, test_path: str = ".", verbose: bool = False) -> Dict:
        """Run pytest."""
        cmd = ['python', '-m', 'pytest', test_path]
        if verbose:
            cmd.append('-v')
        
        code = f"""
import subprocess
import sys

result = subprocess.run(
    {cmd},
    capture_output=True,
    text=True,
    cwd='/workspace'
)

print(result.stdout)
if result.stderr:
    print(result.stderr, file=sys.stderr)
sys.exit(result.returncode)
"""
        return self.sandbox.execute(
            code=code,
            workspace_path=str(self.workspace_path),
            timeout=60,
            install_packages=['pytest'],
        )
    
    def git_status(self) -> Dict:
        """Get git status."""
        try:
            status = []
            for item in self.repo.index.diff(None):
                status.append(f"Modified: {item.a_path}")
            for item in self.repo.untracked_files:
                status.append(f"Untracked: {item}")
            
            return {'status': status if status else ['Clean working tree']}
        except Exception as e:
            return {'error': str(e)}
    
    def git_diff(self, staged: bool = False) -> Dict:
        """Get git diff."""
        try:
            if staged:
                diff = self.repo.index.diff(self.repo.head.commit)
            else:
                diff = self.repo.index.diff(None)
            
            diff_text = []
            for item in diff:
                diff_text.append(f"--- {item.a_path}\n+++ {item.b_path}")
                diff_text.append(item.diff.decode('utf-8'))
            
            return {'diff': '\n'.join(diff_text) if diff_text else 'No changes'}
        except Exception as e:
            return {'error': str(e)}
    
    def search_code(self, pattern: str, file_pattern: str = "*") -> Dict:
        """Search for pattern in code."""
        try:
            cmd = ['grep', '-r', '-n', pattern, '--include', file_pattern, '.']
            result = subprocess.run(
                cmd,
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            matches = result.stdout.strip().split('\n') if result.stdout else []
            return {
                'matches': matches[:50],  # Limit to 50 matches
                'total': len(matches),
            }
        except Exception as e:
            return {'error': str(e)}
    
    def execute_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """Execute a tool by name with arguments."""
        tool_map = {
            'execute_code': self.execute_code,
            'read_file': self.read_file,
            'write_file': self.write_file,
            'run_tests': self.run_tests,
            'git_status': self.git_status,
            'git_diff': self.git_diff,
            'search_code': self.search_code,
        }
        
        if tool_name not in tool_map:
            return {'error': f'Unknown tool: {tool_name}'}
        
        try:
            return tool_map[tool_name](**arguments)
        except Exception as e:
            return {'error': f'Tool execution failed: {str(e)}'}


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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        
        # Initialize sandbox and executor
        self.sandbox = DockerSandbox()
        self.executor = ToolExecutor(workspace_path, self.sandbox)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool schemas."""
        tools_json = json.dumps(TOOL_SCHEMAS, indent=2)
        
        return f"""You are an expert coding assistant with access to tools.

Available tools (JSON Schema):
{tools_json}

Use this format for tool calls:
```json
{{
  "tool": "tool_name",
  "arguments": {{...}}
}}
```

Example:
```json
{{
  "tool": "execute_code",
  "arguments": {{
    "code": "print('Hello')",
    "timeout": 30
  }}
}}
```

Always think step-by-step and explain your reasoning before calling tools.
When the task is complete, respond with FINAL ANSWER: followed by your answer.
"""
    
    def _parse_tool_call(self, text: str) -> Optional[Dict]:
        """Parse tool call from model output."""
        import re
        
        # Find JSON in code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if not json_match:
            return None
        
        try:
            tool_call = json.loads(json_match.group(1))
            if 'tool' in tool_call and 'arguments' in tool_call:
                return tool_call
        except json.JSONDecodeError:
            pass
        
        return None
    
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
            
            # Parse and execute tool call
            tool_call = self._parse_tool_call(response)
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
                    "role": "user",
                    "content": f"Tool result:\n```json\n{json.dumps(result, indent=2)}\n```"
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
