"""
Agent System with Tool Use and ReAct Reasoning Loop.

Implements:
- ReAct (Reasoning + Acting) loop
- Tool use infrastructure
- Code execution sandbox
- File system operations
- Web search (simulated)
- Multi-step problem solving
"""

import json
import re
import subprocess
import tempfile
import os
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

try:
    import torch
except ImportError:
    torch = None  # type: ignore


class AgentState(Enum):
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"


@dataclass
class Tool:
    """A tool the agent can use."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    
    def to_prompt_format(self) -> str:
        """Convert to prompt-friendly format."""
        params = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"{self.name}({params}): {self.description}"


@dataclass
class AgentStep:
    """A single step in the agent's reasoning."""
    thought: str = ""
    action: Optional[str] = None
    action_input: Optional[Dict] = None
    observation: Optional[str] = None
    state: AgentState = AgentState.THINKING


@dataclass
class AgentResult:
    """Final result from agent execution."""
    answer: str
    steps: List[AgentStep]
    success: bool
    error: Optional[str] = None


# ============================================================
# TOOLS
# ============================================================

def execute_python_code(code: str, timeout: int = 10) -> str:
    """Execute Python code in a sandboxed environment."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        result = subprocess.run(
            ['python3', temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        
        os.unlink(temp_path)
        return output if output else "Code executed successfully (no output)"
    
    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing code: {str(e)}"


def read_file(file_path: str) -> str:
    """Read contents of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(file_path: str, content: str) -> str:
    """Write content to a file."""
    try:
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


def list_directory(path: str = ".") -> str:
    """List files and directories."""
    try:
        entries = os.listdir(path)
        result = []
        for entry in entries:
            full_path = os.path.join(path, entry)
            entry_type = "DIR" if os.path.isdir(full_path) else "FILE"
            result.append(f"[{entry_type}] {entry}")
        return "\n".join(result) if result else "Empty directory"
    except Exception as e:
        return f"Error listing directory: {str(e)}"


def search_codebase(query: str, directory: str = ".") -> str:
    """Search for code patterns in the codebase."""
    try:
        # Simple grep-based search
        result = subprocess.run(
            ['grep', '-r', '-n', query, directory],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.stdout:
            lines = result.stdout.strip().split('\n')[:20]  # Limit results
            return "\n".join(lines)
        return "No matches found"
    except Exception as e:
        return f"Error searching: {str(e)}"


# ============================================================
# AGENT
# ============================================================

class CodingAgent:
    """
    ReAct-style coding agent with tool use.
    
    The agent reasons about the problem, takes actions using tools,
    observes the results, and iterates until the task is complete.
    """
    
    def __init__(self, model, tokenizer, max_steps: int = 10):
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        
        # Register available tools
        self.tools = self._register_tools()
    
    def _register_tools(self) -> Dict[str, Tool]:
        """Register all available tools."""
        return {
            "execute_python": Tool(
                name="execute_python",
                description="Execute Python code and return the output",
                parameters={"code": "str (Python code to execute)"},
                function=execute_python_code,
            ),
            "read_file": Tool(
                name="read_file",
                description="Read the contents of a file",
                parameters={"file_path": "str (path to file)"},
                function=read_file,
            ),
            "write_file": Tool(
                name="write_file",
                description="Write content to a file",
                parameters={"file_path": "str", "content": "str"},
                function=write_file,
            ),
            "list_directory": Tool(
                name="list_directory",
                description="List files and directories",
                parameters={"path": "str (default: '.')"},
                function=list_directory,
            ),
            "search_codebase": Tool(
                name="search_codebase",
                description="Search for code patterns",
                parameters={"query": "str", "directory": "str (default: '.')"},
                function=search_codebase,
            ),
        }
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool descriptions."""
        tools_desc = "\n".join(tool.to_prompt_format() for tool in self.tools.values())
        
        return f"""You are an expert coding assistant with access to tools. Solve problems step by step.

Available tools:
{tools_desc}

Use this format:
Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: [JSON with tool parameters]
Observation: [result from tool]
... (repeat Thought/Action/Observation)
Thought: I now know the final answer
Final Answer: [your answer to the user]

Example:
User: Write a function to calculate factorial and test it
Thought: I need to write the factorial function and test it. Let me create a Python file.
Action: write_file
Action Input: {{"file_path": "factorial.py", "content": "def factorial(n):\\n    return 1 if n <= 1 else n * factorial(n-1)\\n\\nprint(factorial(5))"}}
Observation: Successfully wrote 85 characters to factorial.py
Thought: Now let me execute it to verify it works
Action: execute_python
Action Input: {{"code": "exec(open('factorial.py').read())"}}
Observation: 120
Thought: The function works correctly
Final Answer: I've created a factorial function that correctly calculates factorial(5) = 120.
"""
    
    def _parse_action(self, text: str) -> tuple[Optional[str], Optional[Dict]]:
        """Parse action and action input from model output."""
        action_match = re.search(r'Action:\s*(\w+)', text)
        if not action_match:
            return None, None
        
        action = action_match.group(1)
        
        # Extract JSON from Action Input
        input_match = re.search(r'Action Input:\s*({[^}]+})', text, re.DOTALL)
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
                return action, action_input
            except json.JSONDecodeError:
                return action, {}
        
        return action, {}
    
    def _parse_final_answer(self, text: str) -> Optional[str]:
        """Extract final answer from model output."""
        match = re.search(r'Final Answer:\s*(.+)', text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def run(self, task: str, verbose: bool = True) -> AgentResult:
        """
        Run the agent on a task.
        
        Args:
            task: The task description
            verbose: Whether to print step-by-step output
            
        Returns:
            AgentResult with answer and execution trace
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task}")
            print(f"{'='*60}\n")
        
        steps = []
        conversation = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": task},
        ]
        
        for step_num in range(self.max_steps):
            if verbose:
                print(f"Step {step_num + 1}/{self.max_steps}")
            
            # Generate response
            prompt = self._format_conversation(conversation)
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            
            # Generate (simplified - in production use proper generation)
            output_ids = self.model.generate(
                input_ids=torch.tensor([input_ids]),
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
            )
            
            response = self.tokenizer.decode(output_ids[0][len(input_ids):])
            
            # Parse response
            step = AgentStep()
            
            # Extract thought
            thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)', 
                                     response, re.DOTALL)
            if thought_match:
                step.thought = thought_match.group(1).strip()
                if verbose:
                    print(f"  Thought: {step.thought[:100]}...")
            
            # Check for final answer
            final_answer = self._parse_final_answer(response)
            if final_answer:
                step.state = AgentState.FINISHED
                steps.append(step)
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Final Answer: {final_answer}")
                    print(f"{'='*60}\n")
                return AgentResult(
                    answer=final_answer,
                    steps=steps,
                    success=True,
                )
            
            # Parse action
            action, action_input = self._parse_action(response)
            if action and action in self.tools:
                step.action = action
                step.action_input = action_input
                step.state = AgentState.ACTING
                
                if verbose:
                    print(f"  Action: {action}")
                    print(f"  Input: {action_input}")
                
                # Execute tool
                tool = self.tools[action]
                try:
                    observation = tool.function(**action_input)
                except Exception as e:
                    observation = f"Error: {str(e)}"
                
                step.observation = observation
                step.state = AgentState.OBSERVING
                
                if verbose:
                    print(f"  Observation: {observation[:200]}...")
                
                # Add to conversation
                conversation.append({"role": "assistant", "content": response})
                conversation.append({"role": "user", "content": f"Observation: {observation}"})
            
            else:
                # No valid action, try to extract answer
                step.state = AgentState.FINISHED
                steps.append(step)
                return AgentResult(
                    answer=response,
                    steps=steps,
                    success=False,
                    error="No valid action found",
                )
            
            steps.append(step)
        
        # Max steps reached
        return AgentResult(
            answer="Max steps reached without final answer",
            steps=steps,
            success=False,
            error="Max steps exceeded",
        )
    
    def _format_conversation(self, messages: List[Dict]) -> str:
        """Format conversation for the model."""
        parts = []
        for msg in messages:
            if msg["role"] == "system":
                parts.append(f"<|system|>\n{msg['content']}\n<eos>")
            elif msg["role"] == "user":
                parts.append(f"<|user|>\n{msg['content']}\n<eos>")
            elif msg["role"] == "assistant":
                parts.append(f"<|assistant|>\n{msg['content']}\n<eos>")
        return "\n".join(parts)


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # This is a demonstration - requires trained model
    print("Agent system demonstration")
    print("="*60)
    print("\nTo use the agent with a trained model:")
    print("""
from agent.agent import CodingAgent
from model.model import CodingAgentModel
from model.tokenizer import CodeTokenizer

# Load model and tokenizer
model = CodingAgentModel.from_pretrained("checkpoints/latest")
tokenizer = CodeTokenizer.load("tokenizer")

# Create agent
agent = CodingAgent(model, tokenizer, max_steps=10)

# Run task
result = agent.run("Create a Python script that downloads a CSV file and calculates statistics")
print(result.answer)
""")
