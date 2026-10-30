"""Held-out model-path suite: plan -> skeleton -> fill -> validate -> test.

The blueprint suite exercises the deterministic app builder. This suite
exercises the *learned* path: any ``generate_fn(prompt, max_tokens) -> str``
(a served checkpoint, a local model wrapper, or a test double) is driven
through :class:`model.longform.LongFormSynthesizer`, and the resulting module
must compile and pass tests written before generation.

There is no trained checkpoint in this repository (training needs a GPU), so
``--backend command`` is how a real run is performed::

    python -m evaluation.model_suite \
        --backend command \
        --command "python tools/serve_model.py --stdin" \
        --json evaluation/reports/model_suite.json

Without a command the suite prints an explicit "no backend" result rather than
pretending a stub is a model. The harness itself is tested with a test double.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from model.longform import LongFormSynthesizer, ReasoningBudget

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelTask:
    """A held-out instruction with tests that were not generated from it."""

    id: str
    instruction: str
    constraints: str
    tests: str

    def to_dict(self) -> dict:
        return {"id": self.id, "instruction": self.instruction}


TASKS: tuple[ModelTask, ...] = (
    ModelTask(
        "slugify",
        "Write a module with `slugify(text: str) -> str` that lowercases, replaces runs of "
        "non-alphanumeric characters with a single hyphen and trims hyphens.",
        "standard library only; no regex backtracking tricks",
        textwrap.dedent(
            """
            from solution import slugify

            def test_basic():
                assert slugify("Hello, World!") == "hello-world"

            def test_runs_and_edges():
                assert slugify("  A  --  B  ") == "a-b"
                assert slugify("") == ""
            """
        ),
    ),
    ModelTask(
        "parse_duration",
        "Write `parse_duration(text: str) -> int` for strings like '1h30m' or '45s', "
        "returning seconds. Raise ValueError on empty or malformed input.",
        "standard library only",
        textwrap.dedent(
            """
            import pytest
            from solution import parse_duration

            def test_units():
                assert parse_duration("1h30m") == 5400
                assert parse_duration("45s") == 45
                assert parse_duration("2m") == 120

            def test_invalid():
                with pytest.raises(ValueError):
                    parse_duration("")
                with pytest.raises(ValueError):
                    parse_duration("abc")
            """
        ),
    ),
    ModelTask(
        "chunk",
        "Write `chunk(items: list, size: int) -> list[list]` that splits a list into "
        "consecutive chunks of at most `size`, rejecting non-positive sizes.",
        "standard library only; do not mutate input",
        textwrap.dedent(
            """
            import pytest
            from solution import chunk

            def test_chunks():
                data = [1, 2, 3, 4, 5]
                assert chunk(data, 2) == [[1, 2], [3, 4], [5]]
                assert data == [1, 2, 3, 4, 5]

            def test_bad_size():
                with pytest.raises(ValueError):
                    chunk([1], 0)
            """
        ),
    ),
    ModelTask(
        "merge_intervals",
        "Write `merge_intervals(intervals)` that merges overlapping closed integer "
        "intervals and returns them sorted.",
        "standard library only; input is a list of (start, end) tuples",
        textwrap.dedent(
            """
            from solution import merge_intervals

            def test_merge():
                assert merge_intervals([(1, 3), (2, 6), (8, 10)]) == [(1, 6), (8, 10)]
                assert merge_intervals([]) == []
            """
        ),
    ),
    ModelTask(
        "rolling_mean",
        "Write `rolling_mean(values: list[float], window: int) -> list[float]`, using "
        "None until the window is full. Reject window <= 0.",
        "standard library only",
        textwrap.dedent(
            """
            import pytest
            from solution import rolling_mean

            def test_values():
                assert rolling_mean([1, 2, 3, 4], 2) == [None, 1.5, 2.5, 3.5]
                assert rolling_mean([1, 2], 3) == [None, None]

            def test_bad_window():
                with pytest.raises(ValueError):
                    rolling_mean([1], 0)
            """
        ),
    ),
    ModelTask(
        "safe_json",
        "Write `loads_or_none(text: str)` returning parsed JSON, or None on invalid JSON. "
        "Do not swallow KeyboardInterrupt/SystemExit.",
        "standard library only",
        textwrap.dedent(
            """
            from solution import loads_or_none

            def test_valid_and_invalid():
                assert loads_or_none('{"a": 1}') == {"a": 1}
                assert loads_or_none("not json") is None
                assert loads_or_none("") is None
            """
        ),
    ),
    ModelTask(
        "dedupe_stable",
        "Write `dedupe(items: list[str]) -> list[str]` preserving first-seen order and "
        "case-insensitive duplicates.",
        "standard library only",
        textwrap.dedent(
            """
            from solution import dedupe

            def test_order():
                assert dedupe(["a", "A", "b", "a", "b"]) == ["a", "b"]
            """
        ),
    ),
    ModelTask(
        "retry_backoff",
        "Write `backoff_seconds(attempt: int, base: float = 0.5, cap: float = 8.0) -> float` "
        "for exponential backoff with jitter omitted; attempt starts at 0.",
        "standard library only",
        textwrap.dedent(
            """
            from solution import backoff_seconds

            def test_backoff():
                assert backoff_seconds(0) == 0.5
                assert backoff_seconds(3) == 4.0
                assert backoff_seconds(20) == 8.0
            """
        ),
    ),
)


@dataclass
class TaskResult:
    id: str
    compiled: bool
    tests_passed: bool
    units: int
    repairs: int
    seconds: float
    errors: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "compiled": self.compiled,
            "tests_passed": self.tests_passed,
            "units": self.units,
            "repairs": self.repairs,
            "seconds": round(self.seconds, 2),
            "errors": self.errors,
            "trace": self.trace[-6:],
        }


def _strip_fence(text: str) -> str:
    body = (text or "").strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        body = "\n".join(lines)
    return body


def evaluate_tasks(
    tasks: Iterable[ModelTask],
    generate_fn: Callable[[str, int], str],
    *,
    timeout: float = 120.0,
    budget: ReasoningBudget | None = None,
) -> dict:
    """Run every task through long-form synthesis, then compile and test."""

    results: list[TaskResult] = []
    for task in tasks:
        started = time.time()
        synth = LongFormSynthesizer(
            generate_fn,
            budget=budget or ReasoningBudget(context_window=8192, prompt_tokens=3000),
            max_units=4,
        )
        try:
            result = synth.synthesise(task.instruction, constraints=task.constraints)
        except Exception as exc:
            results.append(
                TaskResult(task.id, False, False, 0, 0, time.time() - started, errors=[f"backend: {exc}"])
            )
            continue
        code = _strip_fence(result.code)
        compiled = True
        errors = list(result.errors)
        try:
            compile(code, f"{task.id}.py", "exec")
        except SyntaxError as exc:
            compiled = False
            errors.append(f"SyntaxError: {exc}")
        passed = False
        if compiled:
            with tempfile.TemporaryDirectory(prefix=f"model-suite-{task.id}-") as tmp:
                root = Path(tmp)
                (root / "solution.py").write_text(code, encoding="utf-8")
                (root / "test_solution.py").write_text(textwrap.dedent(task.tests), encoding="utf-8")
                try:
                    proc = subprocess.run(
                        ["python", "-m", "pytest", "-q", "test_solution.py"],
                        cwd=root,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    passed = proc.returncode == 0
                    if not passed:
                        tail = (proc.stdout or "")[-1200:] + (proc.stderr or "")[-400:]
                        errors.extend(line for line in tail.splitlines() if line.startswith(("E ", "FAILED")))
                except subprocess.TimeoutExpired:
                    errors.append(f"tests timed out after {timeout:.0f}s")
        results.append(
            TaskResult(
                id=task.id,
                compiled=compiled,
                tests_passed=passed,
                units=len(result.units),
                repairs=result.repairs,
                seconds=time.time() - started,
                errors=errors,
                trace=list(result.trace),
            )
        )

    total = len(results)
    compiled = sum(1 for r in results if r.compiled)
    passed = sum(1 for r in results if r.tests_passed)
    return {
        "version": 1,
        "backend": "callable",
        "tasks": total,
        "results": [r.to_dict() for r in results],
        "metrics": {
            "tasks": total,
            "compiled": compiled,
            "compile_rate": round(compiled / max(1, total), 4),
            "tests_passed": passed,
            "test_pass_rate": round(passed / max(1, total), 4),
            "repairs": sum(r.repairs for r in results),
            "mean_units": round(sum(r.units for r in results) / max(1, total), 2),
            "mean_seconds": round(sum(r.seconds for r in results) / max(1, total), 2),
        },
    }


def command_backend(command: str, timeout: float = 120.0) -> Callable[[str, int], str]:
    """Adapt a stdin/stdout model command to the generator signature."""

    argv = shlex.split(command)

    def generate(prompt: str, max_tokens: int) -> str:
        proc = subprocess.run(
            argv,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"model command exited {proc.returncode}: {proc.stderr[-500:]}")
        return proc.stdout

    return generate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Held-out model-path suite")
    parser.add_argument("--backend", choices=("none", "command"), default="none")
    parser.add_argument("--command", default="", help="stdin/stdout model command")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    tasks = list(TASKS)[: args.limit or None]
    if args.backend == "none":
        report = {
            "version": 1,
            "backend": "none",
            "tasks": len(tasks),
            "metrics": {"tasks": len(tasks), "compile_rate": 0.0, "test_pass_rate": 0.0},
            "note": (
                "No trained checkpoint is present in this checkout; training requires a GPU. "
                "Pass --backend command --command '<stdin/stdout model>' to evaluate a real model."
            ),
        }
        print(report["note"])
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2) + "\n")
        return 2

    if not args.command:
        parser.error("--backend command requires --command")
    report = evaluate_tasks(tasks, command_backend(args.command, timeout=args.timeout), timeout=args.timeout * 2)
    report["backend"] = "command"
    report["command"] = args.command
    metrics = report["metrics"]
    print(
        f"tasks {metrics['tasks']} | compiled {metrics['compiled']} ({metrics['compile_rate']:.0%}) | "
        f"tests {metrics['tests_passed']} ({metrics['test_pass_rate']:.0%}) | repairs {metrics['repairs']}"
    )
    for result in report["results"]:
        if not result["tests_passed"]:
            print(f"  FAIL {result['id']}: {result['errors'][:1]}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report: {args.json}")
    return 0 if metrics["tests_passed"] == metrics["tasks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
