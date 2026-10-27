"""Agent tools: files, unified-diff patching, AST-aware Python edits, git, tests.

Path safety: every path is resolved inside the workspace (no ../ escapes,
no absolute paths outside it). Patch application is atomic per file
(validate all hunks, then write).
"""
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .safety import check_code_safety
from .sandbox import HardenedSandbox

try:
    import git as gitpython
    HAS_GITPYTHON = True
except ImportError:
    HAS_GITPYTHON = False

LANG_BY_EXT = {".py": "python", ".js": "node", ".mjs": "node", ".sh": "bash",
               ".go": "go", ".rs": "rust"}


class ToolExecutor:
    def __init__(self, workspace_path: str, sandbox: HardenedSandbox):
        self.workspace = Path(workspace_path).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.sandbox = sandbox
        self.repo = None
        if HAS_GITPYTHON:
            try:
                self.repo = gitpython.Repo(self.workspace)
            except Exception:
                try:
                    self.repo = gitpython.Repo.init(self.workspace)
                except Exception:
                    self.repo = None

    # -- paths --------------------------------------------------------------
    def _safe(self, path: str) -> Path:
        p = (self.workspace / path).resolve()
        if p != self.workspace and self.workspace not in p.parents:
            raise ValueError(f"path escapes workspace: {path!r}")
        return p

    # -- files ---------------------------------------------------------------
    def read_file(self, path: str) -> Dict:
        try:
            f = self._safe(path)
            if not f.exists():
                return {"error": f"File not found: {path}"}
            content = f.read_text(encoding="utf-8", errors="replace")
            return {"content": content[:50000], "size": len(content), "truncated": len(content) > 50000}
        except Exception as e:
            return {"error": str(e)}

    def write_file(self, path: str, content: str) -> Dict:
        try:
            f = self._safe(path)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content, encoding="utf-8")
            return {"success": True, "path": path, "size": len(content)}
        except Exception as e:
            return {"error": str(e)}

    def list_files(self, path: str = ".", pattern: str = "*") -> Dict:
        try:
            base = self._safe(path)
            files = [str(p.relative_to(self.workspace)) for p in base.rglob(pattern) if p.is_file()]
            return {"files": files[:500], "count": len(files)}
        except Exception as e:
            return {"error": str(e)}

    def search_code(self, pattern: str, file_pattern: str = "*") -> Dict:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return {"error": f"invalid regex: {e}"}
        hits = []
        for f in self.workspace.rglob(file_pattern):
            if not f.is_file() or ".git" in f.parts:
                continue
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if rx.search(line):
                        hits.append({"file": str(f.relative_to(self.workspace)), "line": i, "text": line[:300]})
                        if len(hits) >= 100:
                            return {"matches": hits, "truncated": True}
            except Exception:
                continue
        return {"matches": hits, "truncated": False}

    # -- patch / AST edits ----------------------------------------------------
    def apply_patch(self, patch: str) -> Dict:
        """Apply a unified diff (multi-file). All files validate before any write."""
        try:
            files = _parse_unified_diff(patch)
        except ValueError as e:
            return {"error": str(e)}
        staged: Dict[Path, str] = {}
        for rel, hunks in files.items():
            try:
                f = self._safe(rel)
                original = f.read_text(encoding="utf-8") if f.exists() else ""
                staged[f] = _apply_hunks(original, hunks, rel)
            except Exception as e:
                return {"error": f"patch failed for {rel}: {e} (no files modified)"}
        for f, content in staged.items():
            f.parent.mkdir(parents=True, exist_ok=True)
            if content is None:  # deletion
                f.unlink(missing_ok=True)
            else:
                f.write_text(content, encoding="utf-8")
        return {"success": True, "files": sorted(files)}

    def edit_function(self, path: str, function: str, new_source: str) -> Dict:
        """AST-aware edit: replace one Python function/class by name."""
        try:
            f = self._safe(path)
            tree = ast.parse(f.read_text(encoding="utf-8"))
            target = None
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == function:
                    target = node
                    break
            if target is None or target.end_lineno is None:
                return {"error": f"symbol {function!r} not found in {path}"}
            try:
                ast.parse(new_source)
            except SyntaxError as e:
                return {"error": f"replacement is not valid Python: {e}"}
            lines = f.read_text(encoding="utf-8").splitlines(keepends=True)
            indent = " " * target.col_offset
            repl = "".join((indent + ln if ln.strip() else ln) for ln in new_source.splitlines(keepends=True))
            if not repl.endswith("\n"):
                repl += "\n"
            new_text = "".join(lines[: target.lineno - 1]) + repl + "".join(lines[target.end_lineno:])
            ast.parse(new_text)  # never write broken Python
            f.write_text(new_text, encoding="utf-8")
            return {"success": True, "path": path, "function": function}
        except Exception as e:
            return {"error": str(e)}

    # -- execution / tests ------------------------------------------------------
    def execute_code(self, code: str, timeout: int = 30, install_packages: List[str] = None,
                     language: str = "python") -> Dict:
        allowed, reason = check_code_safety(code)
        if not allowed:
            return {"error": reason, "exit_code": -1}
        return self.sandbox.execute(code=code, workspace_path=str(self.workspace),
                                    timeout=timeout, install_packages=install_packages, language=language)

    def run_tests(self, test_path: str = ".", verbose: bool = False) -> Dict:
        code = ("import subprocess, sys\n"
                f"r = subprocess.run([sys.executable, '-m', 'pytest', {test_path!r}]"
                f" + (['-v'] if {verbose} else []), capture_output=True, text=True, cwd='/workspace')\n"
                "print(r.stdout[-15000:]); print(r.stderr[-5000:], file=sys.stderr); sys.exit(r.returncode)\n")
        return self.sandbox.execute(code=code, workspace_path=str(self.workspace), timeout=120, language="python")

    # -- git (full surface) -------------------------------------------------------
    def _git(self, *args: str) -> Dict:
        r = subprocess.run(["git", *args], cwd=self.workspace, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return {"error": (r.stderr or r.stdout).strip()[:2000], "exit_code": r.returncode}
        return {"output": r.stdout[:20000], "exit_code": 0}

    def git_status(self) -> Dict:
        return self._git("status", "--short", "--branch")

    def git_diff(self, staged: bool = False) -> Dict:
        return self._git("diff", "--staged") if staged else self._git("diff")

    def git_log(self, n: int = 10) -> Dict:
        return self._git("log", f"-{int(n)}", "--oneline", "--decorate")

    def git_commit(self, message: str, add_all: bool = True) -> Dict:
        if add_all:
            r = self._git("add", "-A")
            if "error" in r:
                return r
        return self._git("commit", "-m", message)

    def git_branch(self, name: str = None, checkout: bool = False) -> Dict:
        if name is None:
            return self._git("branch", "--list")
        return self._git("checkout", "-b", name) if checkout else self._git("branch", name)

    def git_checkout(self, ref: str) -> Dict:
        return self._git("checkout", ref)

    def git_show(self, ref: str = "HEAD") -> Dict:
        return self._git("show", ref, "--stat")

    # -- dispatch -------------------------------------------------------------------
    def execute_tool(self, tool_name: str, arguments: Dict) -> Dict:
        dispatch = {
            "execute_code": self.execute_code, "read_file": self.read_file,
            "write_file": self.write_file, "list_files": self.list_files,
            "search_code": self.search_code, "apply_patch": self.apply_patch,
            "edit_function": self.edit_function, "run_tests": self.run_tests,
            "git_status": self.git_status, "git_diff": self.git_diff, "git_log": self.git_log,
            "git_commit": self.git_commit, "git_branch": self.git_branch,
            "git_checkout": self.git_checkout, "git_show": self.git_show,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return fn(**arguments)
        except TypeError as e:
            return {"error": f"bad arguments for {tool_name}: {e}"}
        except Exception as e:
            return {"error": f"Tool execution failed: {e}"}


def _parse_unified_diff(patch: str) -> Dict[str, list]:
    files: Dict[str, list] = {}
    current, hunks = None, None
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:].strip()
            files[current] = hunks = []
        elif line.startswith("+++ "):
            current = line[4:].strip().removeprefix("b/")
            files[current] = hunks = []
        elif line.startswith("--- "):
            continue
        elif line.startswith("@@") and hunks is not None:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if not m:
                raise ValueError(f"malformed hunk header: {line}")
            hunks.append({"old_start": int(m.group(1)), "lines": []})
        elif hunks:
            hunks[-1]["lines"].append(line)
    if not files:
        raise ValueError("no file hunks found in patch")
    return files


def _apply_hunks(original: str, hunks: list, rel: str) -> Optional[str]:
    lines = original.splitlines(keepends=True)
    offset = 0
    for hunk in hunks:
        idx = (hunk["old_start"] - 1) + offset
        out, consumed = [], 0
        for raw in hunk["lines"]:
            tag, text = (raw[:1], raw[1:] + ("\n" if not raw.endswith("\n") else ""))
            src = raw[1:]
            if tag == " ":
                if idx + consumed >= len(lines) or lines[idx + consumed].rstrip("\n") != src:
                    raise ValueError(f"context mismatch in {rel} near line {idx + consumed + 1}")
                out.append(lines[idx + consumed]); consumed += 1
            elif tag == "-":
                if idx + consumed >= len(lines) or lines[idx + consumed].rstrip("\n") != src:
                    raise ValueError(f"deletion mismatch in {rel} near line {idx + consumed + 1}")
                consumed += 1
            elif tag == "+":
                out.append(src + "\n")
            elif raw.startswith("\\"):
                continue
        lines[idx: idx + consumed] = out
        offset += len(out) - consumed
    return "".join(lines)
