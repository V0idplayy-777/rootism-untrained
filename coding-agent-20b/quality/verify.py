"""Run the test suite, extract the useful failure signal, iterate until green.

This is the "run tests after every meaningful change" loop, made a first-class
object instead of an instruction in a prompt:

* :func:`detect_test_command` finds the project's real test command.
* :func:`run_tests` executes it with a timeout and workspace-confined cwd.
* :func:`parse_output` turns pytest / node:test / jest / go / cargo output into
  structured pass/fail lists plus the **first meaningful assertion per failure**
  (the signal a model can act on, without 200 lines of dots and warnings).
* :class:`VerificationLoop` retries with a ``fix_fn`` until green or the round
  budget is exhausted, recording a transcript that can be used as SFT data.

Nothing here needs a model and nothing writes outside the project directory.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
PYTEST_FAILED = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)(?:\s+-\s+(.*))?$", re.MULTILINE)
PYTEST_COUNTS = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed)", re.I)
PYTEST_ASSERT = re.compile(r"^E\s+(.+)$", re.MULTILINE)
NODE_FAIL = re.compile(r"^not ok \d+ - (.+)$", re.MULTILINE)
JEST_FAIL = re.compile(r"^\s*(?:●|✕)\s+(.+)$", re.MULTILINE)
GO_FAIL = re.compile(r"^--- FAIL: (\S+)", re.MULTILINE)
CARGO_FAIL = re.compile(r"^---- (\S+) stdout ----$", re.MULTILINE)

SAFE_COMMANDS = re.compile(
    r"^(python -m pytest|pytest|python -m unittest|npm (run )?test|pnpm (run )?test|yarn test|"
    r"node --test|npx (jest|vitest)|go test|cargo test|make test|dotnet test|"
    r"python -m (?!pip)\w+)"
)


def _for_allowlist(argv: Sequence[str]) -> str:
    """Command string for the allowlist check, with any interpreter normalised.

    Callers legitimately pass ``sys.executable`` (tests, the agent's own venv);
    that must be judged exactly like ``python`` rather than rejected because the
    path is absolute.
    """

    if not argv:
        return ""
    first = Path(str(argv[0])).name.lower()
    if re.fullmatch(r"python(\d+(\.\d+)?)?(\.exe)?", first):
        first = "python"
    return " ".join([first, *[str(part) for part in argv[1:]]])


@dataclass
class TestRun:
    command: str
    ok: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    duration: float = 0.0
    timed_out: bool = False

    @property
    def output(self) -> str:
        return (self.stdout or "") + (("\n" + self.stderr) if self.stderr else "")

    def failure_signal(self, max_chars: int = 2400) -> str:
        """Compact, model-sized summary of what failed and why."""

        if self.ok:
            return "all tests passed"
        chunks: list[str] = [f"command: {self.command} (exit {self.exit_code})"]
        if self.timed_out:
            chunks.append("the test command timed out")
        for failure in self.failures[:6]:
            chunks.append(f"- {failure.get('test', '?')}: {failure.get('message', '')[:400]}")
        if not self.failures:
            tail = ANSI.sub("", self.output).strip().splitlines()[-12:]
            chunks.append("\n".join(tail))
        return "\n".join(chunks)[:max_chars]

    def render(self, max_lines: int = 40) -> str:
        head = f"{'PASS' if self.ok else 'FAIL'} {self.command} ({self.duration:.1f}s)"
        detail = []
        if self.failed:
            detail.append("failed: " + ", ".join(self.failed[:8]))
        if self.failures:
            detail.extend(f"  {f.get('test')}: {f.get('message', '')[:160]}" for f in self.failures[:5])
        lines = ANSI.sub("", self.output).strip().splitlines()
        if not self.ok and lines:
            detail.append("...")
            detail.extend(lines[-min(max_lines, 20) :])
        return "\n".join([head, *detail])

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "ok": self.ok,
            "exit_code": self.exit_code,
            "passed": list(self.passed),
            "failed": list(self.failed),
            "failures": list(self.failures),
            "duration": round(self.duration, 2),
            "timed_out": self.timed_out,
            "output_tail": ANSI.sub("", self.output)[-4000:],
        }


def detect_test_command(root: str | Path, profile=None) -> list[str] | None:
    """Best real test command for the project, or None when there are no tests."""

    root = Path(root)
    package = root / "package.json"
    if package.exists():
        try:
            scripts = json.loads(package.read_text(encoding="utf-8")).get("scripts") or {}
        except json.JSONDecodeError:
            scripts = {}
        if (root / "pnpm-lock.yaml").exists() and "test" in scripts:
            return ["pnpm", "test"]
        if (root / "yarn.lock").exists() and "test" in scripts:
            return ["yarn", "test"]
        if "test" in scripts:
            return ["npm", "test"]
        if (root / "test").is_dir() or any(root.glob("**/*.test.{js,ts,mjs}")):
            return ["node", "--test"]
    if (root / "go.mod").exists():
        return ["go", "test", "./..."]
    if (root / "Cargo.toml").exists():
        return ["cargo", "test", "--quiet"]
    if (root / "manage.py").exists():
        return ["python", "manage.py", "test"]
    has_python_tests = (root / "tests").is_dir() or any(root.glob("test_*.py")) or any(root.glob("**/test_*.py"))
    if has_python_tests:
        return ["python", "-m", "pytest", "-q"]
    if (root / "pyproject.toml").exists() or (root / "requirements.txt").exists():
        # A Python project with no tests yet: pytest with an explicit "no tests"
        # exit code 5 is still honest signal, not an error.
        return ["python", "-m", "pytest", "-q"]
    return None


def run_tests(
    root: str | Path,
    command: Sequence[str] | str | None = None,
    *,
    timeout: float = 300.0,
    env: dict[str, str] | None = None,
    allow_any: bool = False,
) -> TestRun:
    """Execute the test command and parse the result."""

    root = Path(root)
    if command is None:
        detected = detect_test_command(root)
        command = detected or ["python", "-m", "pytest", "-q"]
    if isinstance(command, str):
        command_str = command
        argv = shlex.split(command)
    else:
        argv = list(command)
        command_str = " ".join(shlex.quote(part) for part in argv)

    if not allow_any and not SAFE_COMMANDS.match(_for_allowlist(argv)):
        return TestRun(
            command=command_str,
            ok=False,
            exit_code=126,
            stderr=(
                "refusing to run an unrecognised test command; use pytest/npm test/"
                "go test/cargo test, or pass allow_any=True"
            ),
        )

    import os
    import sys

    # Run with the interpreter the agent is running under, not whatever
    # "python" happens to be on PATH (venvs, pyenv, CI images).
    if argv and argv[0] in {"python", "python3"}:
        argv = [sys.executable, *argv[1:]]

    merged = dict(os.environ)
    # Never reuse bytecode: an edit within the same second can otherwise be
    # masked by a stale .pyc and the "fix" looks like it did nothing.
    merged.update({"PYTHONUNBUFFERED": "1", "CI": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    merged.update(env or {})
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=merged,
        )
        duration = time.monotonic() - started
        stdout, stderr, code = completed.stdout or "", completed.stderr or "", completed.returncode
        timed_out = False
    except FileNotFoundError:
        return TestRun(
            command=command_str,
            ok=False,
            exit_code=127,
            stderr=f"command not found: {argv[0]}",
            duration=time.monotonic() - started,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - started
        stdout = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        code, timed_out = 124, True

    # pytest exits 5 when no tests were collected.
    ok = code == 0 or (code == 5 and "no tests ran" in (stdout + stderr))
    run = TestRun(
        command=command_str,
        ok=ok,
        exit_code=code,
        stdout=stdout,
        stderr=stderr,
        duration=duration,
        timed_out=timed_out,
    )
    parse_output(run)
    return run


def parse_output(run: TestRun) -> TestRun:
    """Populate passed/failed/failures from the run's output (mutates)."""

    text = ANSI.sub("", run.output)

    for match in PYTEST_FAILED.finditer(text):
        test, message = match.group(1), (match.group(2) or "").strip()
        run.failed.append(test)
        if not message:
            message = _assert_detail(text, test)
        run.failures.append({"test": test, "message": message[:400]})
    if "passed" in text:
        counts = {kind.lower(): int(n) for n, kind in PYTEST_COUNTS.findall(text)}
        passed_count = counts.get("passed", 0)
        if passed_count and not run.passed:
            run.passed = [f"{passed_count} tests"]

    if not run.failed:
        for pattern, label in ((NODE_FAIL, "node"), (JEST_FAIL, "jest"), (GO_FAIL, "go"), (CARGO_FAIL, "cargo")):
            matches = [m.group(1).strip() for m in pattern.finditer(text)]
            if matches:
                run.failed.extend(matches)
                run.failures.extend({"test": name, "message": _generic_detail(text, name)} for name in matches[:8])
                break

    if run.exit_code != 0 and not run.failed:
        # Unknown runner: keep the tail as one synthetic failure so downstream
        # prompts still get signal.
        tail = text.strip().splitlines()[-15:]
        if tail:
            run.failures.append({"test": "test command", "message": "\n".join(tail)[-600:]})
    run.ok = run.ok and run.exit_code in (0, 5) and not run.timed_out
    return run


def _assert_detail(text: str, test: str) -> str:
    """First `E ...` assertion line near a failing test's section."""

    name = test.split("::")[-1]
    section = text
    if name and name in text:
        start = text.find(name)
        section = text[max(0, start - 200) : start + 4000]
    for line in section.splitlines():
        if line.startswith("E ") or line.startswith("AssertionError"):
            return line[2:].strip() if line.startswith("E ") else line.strip()
    return ""


def _generic_detail(text: str, name: str) -> str:
    index = text.find(name)
    if index == -1:
        return ""
    for line in text[index : index + 1200].splitlines()[1:8]:
        if line.strip():
            return line.strip()[:300]
    return ""


class VerificationLoop:
    """Run tests, hand the failure signal to ``fix_fn``, repeat until green."""

    def __init__(
        self,
        root: str | Path,
        *,
        command: Sequence[str] | str | None = None,
        fix_fn: Callable[[str, TestRun], bool] | None = None,
        max_rounds: int = 3,
        timeout: float = 300.0,
        allow_any: bool = False,
    ):
        self.root = Path(root)
        self.command = command
        self.fix_fn = fix_fn
        self.max_rounds = max_rounds
        self.timeout = timeout
        self.allow_any = allow_any
        self.runs: list[TestRun] = []
        self.attempts: list[dict] = []

    def _prompt(self, run: TestRun) -> str:
        return (
            "The test suite failed. Use the failure signal below to make the smallest change "
            "that makes the failing behaviour correct. Do not weaken or delete tests.\n\n"
            + run.failure_signal()
        )

    def run(self) -> dict:
        rounds = 0
        run = run_tests(self.root, self.command, timeout=self.timeout, allow_any=self.allow_any)
        self.runs.append(run)
        while not run.ok and self.fix_fn and rounds < self.max_rounds:
            rounds += 1
            prompt = self._prompt(run)
            started = time.monotonic()
            try:
                changed = bool(self.fix_fn(prompt, run))
            except Exception as exc:  # noqa: BLE001 - fixer failure is data, not a crash
                self.attempts.append(
                    {"round": rounds, "prompt": prompt, "changed": False, "error": f"{type(exc).__name__}: {exc}"}
                )
                break
            self.attempts.append(
                {
                    "round": rounds,
                    "prompt": prompt,
                    "changed": changed,
                    "fix_seconds": round(time.monotonic() - started, 2),
                }
            )
            if not changed:
                break
            run = run_tests(self.root, self.command, timeout=self.timeout, allow_any=self.allow_any)
            self.runs.append(run)
        return {
            "ok": run.ok,
            "rounds": rounds,
            "run": run,
            "runs": self.runs,
            "attempts": self.attempts,
            "command": run.command,
        }

    def to_trace(self) -> dict:
        """A transcript suitable for SFT/DPO rows (failure -> fix -> green)."""

        first = self.runs[0] if self.runs else None
        return {
            "command": first.command if first else "",
            "failure": ("" if first is None or first.ok else first.failure_signal()),
            "final_output": (self.runs[-1].failure_signal() if self.runs and not self.runs[-1].ok else ""),
            "attempts": self.attempts,
            "runs": [run.to_dict() for run in self.runs],
            "green": bool(self.runs and self.runs[-1].ok),
            "rounds": len(self.runs) - 1 if self.runs else 0,
        }
