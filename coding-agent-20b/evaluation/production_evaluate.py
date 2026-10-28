"""
Production Evaluation Harness

Features:
- Docker-based sandboxing for safe code execution
- Standard benchmarks: HumanEval, MBPP, LiveCodeBench, SWE-bench
- Parallel evaluation with resource limits
- Detailed metrics and error analysis
- Result caching and resumption
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:

    import docker

    HAS_DOCKER = True


except ImportError:

    HAS_DOCKER = False


logger = logging.getLogger(__name__)


@dataclass
class EvaluationProblem:
    """A single evaluation problem."""

    id: str

    prompt: str

    canonical_solution: str

    test_cases: str

    entry_point: str

    language: str = "python"

    difficulty: str = "medium"

    metadata: dict = None

    def __post_init__(self):

        if self.metadata is None:

            self.metadata = {}


def load_humaneval() -> list[EvaluationProblem]:
    """Load HumanEval benchmark."""

    try:

        from datasets import load_dataset

        dataset = load_dataset("openai_humaneval", split="test")

        problems = []

        for item in dataset:

            problems.append(
                EvaluationProblem(
                    id=item["task_id"],
                    prompt=item["prompt"],
                    canonical_solution=item["canonical_solution"],
                    test_cases=item["test"],
                    entry_point=item["entry_point"],
                    language="python",
                    difficulty="medium",
                    metadata={"source": "humaneval"},
                )
            )

        logger.info(f"Loaded {len(problems)} HumanEval problems")

        return problems

    except Exception as e:

        logger.error(f"Error loading HumanEval: {e}")

        return []


def load_mbpp() -> list[EvaluationProblem]:
    """Load MBPP benchmark."""

    try:

        from datasets import load_dataset

        dataset = load_dataset("mbpp", split="test")

        problems = []

        for i, item in enumerate(dataset):

            prompt = f'"""\n{item["text"]}\n\n{item.get("test_setup_code", "")}\n"""'

            problems.append(
                EvaluationProblem(
                    id=f"mbpp_{i}",
                    prompt=prompt,
                    canonical_solution=item["code"],
                    test_cases="\n".join(item["test_list"]),
                    entry_point=item.get("entry_point", ""),
                    language="python",
                    difficulty="easy",
                    metadata={"source": "mbpp"},
                )
            )

        logger.info(f"Loaded {len(problems)} MBPP problems")

        return problems

    except Exception as e:

        logger.error(f"Error loading MBPP: {e}")

        return []


def load_livecodebench() -> list[EvaluationProblem]:
    """Load LiveCodeBench (competitive programming)."""

    try:

        from datasets import load_dataset

        dataset = load_dataset("livecodebench/code_generation", split="test")

        problems = []

        for i, item in enumerate(dataset):

            problems.append(
                EvaluationProblem(
                    id=item.get("problem_id", f"lcb_{i}"),
                    prompt=item["prompt"],
                    canonical_solution=item.get("canonical_solution", ""),
                    test_cases=item.get("test_cases", ""),
                    entry_point=item.get("entry_point", ""),
                    language="python",
                    difficulty=item.get("difficulty", "medium"),
                    metadata={"source": "livecodebench"},
                )
            )

        logger.info(f"Loaded {len(problems)} LiveCodeBench problems")

        return problems

    except Exception as e:

        logger.error(f"Error loading LiveCodeBench: {e}")

        return []


class SandboxExecutor:
    """Execute code in Docker sandbox with resource limits."""

    def __init__(
        self,
        image: str = "python:3.11-slim",
        memory_limit: str = "1g",
        timeout: int = 30,
        network_disabled: bool = True,
    ):

        self.image = image

        self.memory_limit = memory_limit

        self.timeout = timeout

        self.network_disabled = network_disabled

        if HAS_DOCKER:

            try:

                self.client = docker.from_env()

                self.available = True

            except Exception as e:

                logger.warning(f"Docker not available: {e}")

                self.available = False

        else:

            self.available = False

    def execute(
        self,
        code: str,
        test_cases: str = "",
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Execute code with test cases.

        Returns:
            Dict with passed, failed, error, stdout, stderr, execution_time
        """

        timeout = timeout or self.timeout

        full_code = f"{code}\n\n{test_cases}"

        if not self.available:

            return self._fallback_execute(full_code, timeout)

        start_time = time.time()

        try:

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:

                f.write(full_code)

                script_path = f.name

            container = self.client.containers.run(
                self.image,
                command="python /script.py",
                volumes={script_path: {"bind": "/script.py", "mode": "ro"}},
                mem_limit=self.memory_limit,
                network_disabled=self.network_disabled,
                detach=True,
            )

            try:

                result = container.wait(timeout=timeout)

                stdout = container.logs(stdout=True, stderr=False).decode("utf-8")

                stderr = container.logs(stdout=False, stderr=True).decode("utf-8")

                exit_code = result["StatusCode"]

                execution_time = time.time() - start_time

                passed = exit_code == 0 and not stderr

                return {
                    "passed": passed,
                    "failed": not passed,
                    "error": stderr if not passed else None,
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                    "execution_time": execution_time,
                }

            except Exception as e:

                container.kill()

                return {
                    "passed": False,
                    "failed": True,
                    "error": f"Timeout after {timeout}s",
                    "stdout": "",
                    "stderr": str(e),
                    "exit_code": -1,
                    "execution_time": timeout,
                }

            finally:

                container.remove(force=True)

                os.unlink(script_path)

        except Exception as e:

            return {
                "passed": False,
                "failed": True,
                "error": f"Execution error: {e!s}",
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "execution_time": 0,
            }

    def _fallback_execute(self, code: str, timeout: int) -> dict:
        """Fallback without Docker."""

        start_time = time.time()

        try:

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:

                f.write(code)

                script_path = f.name

            result = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            os.unlink(script_path)

            execution_time = time.time() - start_time

            passed = result.returncode == 0 and not result.stderr

            return {
                "passed": passed,
                "failed": not passed,
                "error": result.stderr if not passed else None,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "execution_time": execution_time,
            }

        except subprocess.TimeoutExpired:

            return {
                "passed": False,
                "failed": True,
                "error": f"Timeout after {timeout}s",
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "execution_time": timeout,
            }

        except Exception as e:

            return {
                "passed": False,
                "failed": True,
                "error": str(e),
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "execution_time": 0,
            }


class CodeGenerator:
    """Generate code completions."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 512):

        self.model = model

        self.tokenizer = tokenizer

        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.95,
        num_samples: int = 1,
    ) -> list[str]:
        """Generate code completions."""

        import torch

        self.model.eval()

        completions = []

        for _ in range(num_samples):

            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

            _dev = (
                next(self.model.parameters()).device
                if hasattr(self.model, "parameters")
                else "cpu"
            )

            input_tensor = torch.tensor([input_ids], device=_dev)

            with torch.no_grad():

                output_ids = self.model.generate(
                    input_tensor,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=(num_samples > 1),
                )

            completion = self.tokenizer.decode(output_ids[0][len(input_ids) :])

            completions.append(completion)

        return completions


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k metric."""

    if n - c < k:

        return 1.0

    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def evaluate_problem(
    problem: EvaluationProblem,
    generator: CodeGenerator,
    executor: SandboxExecutor,
    num_samples: int = 10,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Evaluate a single problem."""

    completions = generator.generate(
        problem.prompt,
        temperature=temperature,
        num_samples=num_samples,
    )

    results = []

    for i, completion in enumerate(completions):

        full_code = problem.prompt + completion

        result = executor.execute(full_code, problem.test_cases)

        results.append(result)

    correct = sum(1 for r in results if r["passed"])

    n = num_samples

    c = correct

    return {
        "id": problem.id,
        "num_samples": n,
        "num_correct": c,
        "pass_at_1": compute_pass_at_k(n, c, 1),
        "pass_at_5": compute_pass_at_k(n, c, 5) if n >= 5 else None,
        "pass_at_10": compute_pass_at_k(n, c, 10) if n >= 10 else None,
        "results": results,
    }


def evaluate_benchmark(
    problems: list[EvaluationProblem],
    generator: CodeGenerator,
    executor: SandboxExecutor,
    num_samples: int = 10,
    temperature: float = 0.2,
    num_workers: int = 4,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate model on a benchmark.

    Args:
        problems: List of evaluation problems
        generator: Code generator
        executor: Sandbox executor
        num_samples: Number of samples per problem
        temperature: Sampling temperature
        num_workers: Number of parallel workers
        cache_dir: Directory to cache results

    Returns:
        Evaluation results with metrics
    """

    results = []

    logger.info(f"Evaluating {len(problems)} problems with {num_samples} samples each")

    cached = {}

    if cache_dir:

        cdir = Path(cache_dir)

        legacy = cdir / "eval_cache.json"

        if legacy.exists():

            with open(legacy) as f:

                cached = json.load(f)

        if cdir.exists():

            for f in cdir.glob("*.json"):

                if f.name == "eval_cache.json":

                    continue

                try:

                    r = json.loads(f.read_text())

                    cached.setdefault(r["id"], r)

                except Exception:

                    continue

        logger.info(f"Loaded {len(cached)} cached results")

    def _one(problem):

        return evaluate_problem(
            problem,
            generator,
            executor,
            num_samples=num_samples,
            temperature=temperature,
        )

    pending = []

    for problem in problems:

        if problem.id in cached:

            results.append(cached[problem.id])

        else:

            pending.append(problem)

    if pending:

        from concurrent.futures import ThreadPoolExecutor as _TPE

        with _TPE(max_workers=max(1, num_workers)) as pool:

            for problem, result in zip(pending, pool.map(_one, pending)):

                results.append(result)

                if cache_dir:

                    cdir = Path(cache_dir)

                    cdir.mkdir(parents=True, exist_ok=True)

                    (cdir / f"{problem.id.replace('/', '_')}.json").write_text(
                        json.dumps(result)
                    )

        by_id = {r["id"]: r for r in results}

        results = [by_id[p.id] for p in problems if p.id in by_id]

    pass_at_1_scores = [r["pass_at_1"] for r in results]

    pass_at_5_scores = [r["pass_at_5"] for r in results if r["pass_at_5"] is not None]

    pass_at_10_scores = [
        r["pass_at_10"] for r in results if r["pass_at_10"] is not None
    ]

    errors = []

    for r in results:

        if r["num_correct"] == 0:

            errors.append(
                {
                    "id": r["id"],
                    "errors": [res["error"] for res in r["results"] if res["error"]],
                }
            )

    metrics = {
        "num_problems": len(problems),
        "num_evaluated": len(results),
        "pass_at_1": np.mean(pass_at_1_scores) if pass_at_1_scores else 0,
        "pass_at_5": np.mean(pass_at_5_scores) if pass_at_5_scores else None,
        "pass_at_10": np.mean(pass_at_10_scores) if pass_at_10_scores else None,
        "num_errors": len(errors),
        "results": results,
        "errors": errors[:10],
    }

    return metrics


@dataclass
class SweBenchProblem:
    """One SWE-bench instance: fix `base_commit` so FAIL_TO_PASS tests pass
    without breaking PASS_TO_PASS tests."""

    instance_id: str

    repo: str

    base_commit: str

    problem_statement: str

    fail_to_pass: list[str]

    pass_to_pass: list[str]

    patch: str = ""

    test_patch: str = ""


def load_swebench(split: str = "test") -> list[SweBenchProblem]:
    """Load SWE-bench Lite from HF (`princeton-nlp/SWE-bench_Lite`)."""

    try:

        import json as _json

        from datasets import load_dataset

        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)

        out = []

        for item in ds:

            def _lst(v):

                if isinstance(v, list):

                    return v

                try:

                    return _json.loads(v)

                except Exception:

                    return [v] if v else []

            out.append(
                SweBenchProblem(
                    instance_id=item["instance_id"],
                    repo=item["repo"],
                    base_commit=item["base_commit"],
                    problem_statement=item["problem_statement"],
                    fail_to_pass=_lst(item.get("FAIL_TO_PASS", [])),
                    pass_to_pass=_lst(item.get("PASS_TO_PASS", [])),
                    patch=item.get("patch", ""),
                    test_patch=item.get("test_patch", ""),
                )
            )

        logger.info(f"Loaded {len(out)} SWE-bench problems")

        return out

    except Exception as e:

        logger.error(f"Error loading SWE-bench: {e}")

        return []


class SweBenchHarness:
    """Run a candidate patch in a disposable clone and check test outcomes.

    Requires git + the repo's own test runner in the eval image; returns
    resolved=True only when every FAIL_TO_PASS passes and no PASS_TO_PASS breaks.
    """

    def __init__(self, work_dir: str, timeout: int = 600):

        self.work_dir = Path(work_dir)

        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.timeout = timeout

    def evaluate(
        self, problem: SweBenchProblem, candidate_patch: str, run_tests_fn=None
    ) -> dict[str, Any]:

        import shutil as _shutil
        import subprocess as _sp

        repo_dir = self.work_dir / problem.instance_id

        try:

            if repo_dir.exists():

                _shutil.rmtree(repo_dir)

            clone = _sp.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    f"https://github.com/{problem.repo}.git",
                    str(repo_dir),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if clone.returncode != 0:

                return {
                    "instance_id": problem.instance_id,
                    "resolved": False,
                    "error": f"clone failed: {clone.stderr[:500]}",
                }

            _sp.run(
                ["git", "checkout", "--quiet", problem.base_commit],
                cwd=repo_dir,
                capture_output=True,
                timeout=120,
            )

            if problem.test_patch:

                _sp.run(
                    ["git", "apply", "-"],
                    cwd=repo_dir,
                    input=problem.test_patch,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

            applied = _sp.run(
                ["git", "apply", "-"],
                cwd=repo_dir,
                input=candidate_patch,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if applied.returncode != 0:

                return {
                    "instance_id": problem.instance_id,
                    "resolved": False,
                    "error": f"patch does not apply: {applied.stderr[:500]}",
                }

            runner = run_tests_fn or (
                lambda d, tests: {
                    "passed": [],
                    "failed": list(tests),
                    "error": "no test runner configured",
                }
            )

            results = runner(str(repo_dir), problem.fail_to_pass + problem.pass_to_pass)

            passed = set(results.get("passed", []))

            resolved = all(t in passed for t in problem.fail_to_pass) and all(
                t in passed for t in problem.pass_to_pass
            )

            return {
                "instance_id": problem.instance_id,
                "resolved": resolved,
                "test_results": results,
            }

        except Exception as e:

            return {
                "instance_id": problem.instance_id,
                "resolved": False,
                "error": str(e),
            }

        finally:

            _shutil.rmtree(repo_dir, ignore_errors=True)


def gold_files_from_patch(patch: str) -> list[str]:
    """Files touched by a gold patch — for *measurement only*, never prompting."""

    import re as _re

    return sorted(
        set(_re.findall(r"^\+\+\+ b/(.+)$", patch or "", flags=_re.M))
    )


def build_swebench_prompt(
    problem: "SweBenchProblem",
    repo_dir: str,
    tokenizer=None,
    budget_tokens: int = 6000,
    mode: str = "hierarchical",
    index=None,
) -> dict[str, Any]:
    """Construct a repo-aware SWE-bench prompt within a token budget.

    `mode` is the retrieval ablation switch: `none` reproduces the
    "local change only" baseline, `flat` adds BM25 chunks, `hierarchical` adds
    the repo map and one-hop import neighbours. The returned dict carries the
    pack stats so runs can report context cost, not just pass rates.
    """

    from agent.repo_context import ContextAssembler, RepoIndex

    index = index or RepoIndex.build(repo_dir)

    assembler = ContextAssembler(index, tokenizer)

    pack = assembler.build(
        problem.problem_statement, budget_tokens=budget_tokens, mode=mode
    )

    prompt = (
        f"Repository: {problem.repo} @ {problem.base_commit}\n"
        f"Issue:\n{problem.problem_statement}\n\n"
        f"{pack.text}\n"
        "Return a unified diff that fixes the issue. Modify as few files as "
        "possible and do not change tests."
    )

    return {
        "prompt": prompt,
        "pack": pack,
        "index": index,
        "stats": {
            "mode": mode,
            "tokens": pack.tokens,
            "map_included": pack.map_included,
            "files_in_context": len({c.path for c in pack.chunks}),
            "chunks": len(pack.chunks),
        },
    }


class RetrievalAblation:
    """Measure retrieval, not vibes.

    For each SWE-bench instance and each context mode, reports:

    - ``gold_file_recall``: fraction of gold-patched files present in context
      (the ceiling for any local fix),
    - ``gold_symbol_recall``: fraction of gold-patched *symbols* retrievable,
    - ``context_tokens``: cost of that context,
    - ``covered``: instances where at least one gold file was retrieved.

    A retriever that does not beat the `none` mode on recall cannot help; one
    that only matches `flat` does not justify its token cost.
    """

    def __init__(self, tokenizer=None, budget_tokens: int = 6000):

        self.tokenizer = tokenizer

        self.budget_tokens = budget_tokens

    def evaluate(
        self,
        problems: list["SweBenchProblem"],
        repo_dirs: dict[str, str],
        modes: tuple[str, ...] = ("none", "flat", "hierarchical"),
        max_instances: int | None = None,
    ) -> dict[str, Any]:

        from agent.repo_context import RepoIndex

        results: dict[str, dict[str, Any]] = {
            mode: {
                "instances": 0,
                "gold_file_recall": 0.0,
                "gold_symbol_recall": 0.0,
                "context_tokens": 0,
                "covered": 0,
                "per_instance": [],
            }
            for mode in modes
        }

        indexes: dict[str, Any] = {}

        for problem in problems[: max_instances or len(problems)]:

            repo_dir = repo_dirs.get(problem.instance_id)

            if not repo_dir:

                continue

            gold_files = gold_files_from_patch(problem.patch)

            if not gold_files:

                continue

            if repo_dir not in indexes:

                indexes[repo_dir] = RepoIndex.build(repo_dir)

            index = indexes[repo_dir]

            for mode in modes:

                prompt_data = build_swebench_prompt(
                    problem,
                    repo_dir,
                    tokenizer=self.tokenizer,
                    budget_tokens=self.budget_tokens,
                    mode=mode,
                    index=index,
                )

                retrieved = {c.path for c in prompt_data["pack"].chunks}

                hit_files = [f for f in gold_files if f in retrieved]

                gold_symbols = set()

                for gold in gold_files:

                    for symbol in index.defined_in(gold):

                        gold_symbols.add(symbol.name)

                retrieved_text = prompt_data["pack"].text

                hit_symbols = {s for s in gold_symbols if s in retrieved_text}

                bucket = results[mode]

                bucket["instances"] += 1

                bucket["gold_file_recall"] += len(hit_files) / len(gold_files)

                bucket["gold_symbol_recall"] += (
                    len(hit_symbols) / len(gold_symbols) if gold_symbols else 0.0
                )

                bucket["context_tokens"] += prompt_data["stats"]["tokens"]

                bucket["covered"] += 1 if hit_files else 0

                bucket["per_instance"].append(
                    {
                        "instance_id": problem.instance_id,
                        "gold_files": gold_files,
                        "hit_files": hit_files,
                        "context_tokens": prompt_data["stats"]["tokens"],
                    }
                )

        summary: dict[str, Any] = {}

        for mode, bucket in results.items():

            n = max(bucket["instances"], 1)

            summary[mode] = {
                "instances": bucket["instances"],
                "gold_file_recall": bucket["gold_file_recall"] / n,
                "gold_symbol_recall": bucket["gold_symbol_recall"] / n,
                "context_tokens": bucket["context_tokens"] / n,
                "covered": bucket["covered"] / n,
                "per_instance": bucket["per_instance"],
            }

        if "hierarchical" in summary and "flat" in summary:

            summary["hierarchical_vs_flat"] = {
                "file_recall_delta": summary["hierarchical"]["gold_file_recall"]
                - summary["flat"]["gold_file_recall"],
                "token_delta": summary["hierarchical"]["context_tokens"]
                - summary["flat"]["context_tokens"],
            }

        return summary


class ContaminationChecker:
    """n-gram overlap of eval prompts/solutions against a training corpus index.

    Builds a set of hashed token 13-grams from training text once, then flags
    any eval item sharing a 13-gram (the GPT-3/Llama decontamination standard).
    """

    def __init__(self, n: int = 13):

        self.n = n

        self.index = set()

    @staticmethod
    def _ngrams(text: str, n: int):

        toks = text.lower().split()

        return (
            {" ".join(toks[i : i + n]) for i in range(max(len(toks) - n + 1, 0))}
            if len(toks) >= n
            else {" ".join(toks)} if toks else set()
        )

    def add_training_text(self, text: str):

        self.index |= self._ngrams(text, self.n)

    def check(self, text: str) -> dict[str, Any]:

        grams = self._ngrams(text, self.n)

        hit = grams & self.index

        return {
            "contaminated": bool(hit),
            "overlap": len(hit) / max(len(grams), 1),
            "shared_ngrams": len(hit),
        }


@dataclass
class Trajectory:

    task: str

    steps: list[dict]

    success: bool

    error: str | None = None

    state: dict | None = None

    recoveries: list[dict] = field(default_factory=list)

    context: dict | None = None


def _longest_failure_streak(steps: list[dict]) -> int:
    """Longest run of consecutive failed/malformed steps."""

    best = current = 0

    for step in steps:

        if step.get("tool") is None:

            current += 1

        else:

            result = step.get("result") or {}

            if isinstance(result, dict) and result.get("error"):

                current += 1

            else:

                current = 0

        best = max(best, current)

    return best


def _loop_rate(steps: list[dict], threshold: int = 2) -> float:
    """Fraction of steps that repeat an identical (tool, arguments) action."""

    seen: dict[str, int] = {}

    repeats = 0

    for step in steps:

        key = json.dumps(
            [step.get("tool"), step.get("arguments") or {}], sort_keys=True
        )

        seen[key] = seen.get(key, 0) + 1

        if seen[key] > threshold:

            repeats += 1

    return repeats / max(len(steps), 1)


def evaluate_trajectories(trajectories: list[Trajectory]) -> dict[str, Any]:
    """Agent-trajectory metrics.

    Measures the things long-horizon tool use actually fails at:

    - success / tool-choice accuracy / malformed-call rate,
    - recovery rate (eventual success after any failure),
    - `recovery_after_3`: success after a streak of 3+ consecutive failures
      (the case where planning, not luck, is required),
    - `loop_rate`: identical call repeated more than twice,
    - `plan_completion` and `state_coherence` from the serialised session state
      (a "done" todo without evidence is not coherent),
    - `escalations`/`backtracks` from the recovery policy trace.
    """

    if not trajectories:

        return {"num_trajectories": 0, "success_rate": 0.0}

    tool_total = tool_correct = malformed = 0

    recovered = recoverable = 0

    hard_recoverable = hard_recovered = 0

    loops = 0.0

    plan_progress: list[float] = []

    coherent = 0

    escalations = backtracks = 0

    for tr in trajectories:

        had_error = False

        for step in tr.steps:

            if step.get("tool") is None:

                malformed += 1

                had_error = True

                continue

            tool_total += 1

            if step.get("expected_tool") is None or step["expected_tool"] == step.get(
                "tool"
            ):

                tool_correct += 1

            result = step.get("result") or {}

            if isinstance(result, dict) and result.get("error"):

                had_error = True

        if had_error:

            recoverable += 1

            if tr.success:

                recovered += 1

        if _longest_failure_streak(tr.steps) >= 3:

            hard_recoverable += 1

            if tr.success:

                hard_recovered += 1

        loops += _loop_rate(tr.steps)

        for record in tr.recoveries:

            action = record.get("action")

            if action == "escalate":

                escalations += 1

            elif action == "backtrack":

                backtracks += 1

        if tr.state:

            completion = tr.state.get("todos", {}).get("completion")

            if isinstance(completion, (int, float)):

                plan_progress.append(float(completion))

            todos = tr.state.get("todos", {}).get("todos", [])

            if all(t.get("evidence") for t in todos if t.get("status") == "done"):

                coherent += 1

    total_calls = tool_total + malformed

    return {
        "num_trajectories": len(trajectories),
        "success_rate": sum(1 for t in trajectories if t.success) / len(trajectories),
        "tool_accuracy": tool_correct / max(tool_total, 1),
        "malformed_rate": malformed / max(total_calls, 1),
        "recovery_rate": recovered / max(recoverable, 1),
        "recovery_after_3": hard_recovered / max(hard_recoverable, 1),
        "hard_trajectories": hard_recoverable,
        "loop_rate": loops / len(trajectories),
        "plan_completion": sum(plan_progress) / max(len(plan_progress), 1),
        "state_coherence": coherent / len(trajectories),
        "escalations": escalations,
        "backtracks": backtracks,
        "avg_steps": sum(len(t.steps) for t in trajectories) / len(trajectories),
    }


def main():
    """Run evaluation on all benchmarks."""

    import argparse

    import torch

    parser = argparse.ArgumentParser(description="Evaluate code generation model")

    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["humaneval", "mbpp", "livecodebench", "swebench", "all"],
    )

    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--tokenizer_path", type=str, default=None)

    parser.add_argument("--num_samples", type=int, default=10)

    parser.add_argument("--temperature", type=float, default=0.2)

    parser.add_argument("--cache_dir", type=str, default="eval_cache")

    parser.add_argument("--output", type=str, default="evaluation_results.json")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger.info(f"Loading model from {args.model_path}")

    from model.config import CodingAgentConfig
    from model.model import CodingAgentModel
    from model.tokenizer import CodeTokenizer

    config = CodingAgentConfig.load(os.path.join(args.model_path, "config.json"))

    model = CodingAgentModel(config)

    checkpoint = torch.load(
        os.path.join(args.model_path, "model.pt"), map_location="cpu"
    )

    model.load_state_dict(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    model.eval()

    tokenizer_path = args.tokenizer_path or args.model_path

    tokenizer = CodeTokenizer.load(tokenizer_path)

    generator = CodeGenerator(model, tokenizer)

    executor = SandboxExecutor()

    all_results = {}

    if args.benchmark in ["humaneval", "all"]:

        logger.info("\n" + "=" * 60)

        logger.info("Evaluating HumanEval")

        logger.info("=" * 60)

        problems = load_humaneval()

        if problems:

            metrics = evaluate_benchmark(
                problems,
                generator,
                executor,
                num_samples=args.num_samples,
                temperature=args.temperature,
                cache_dir=args.cache_dir,
            )

            all_results["humaneval"] = metrics

            logger.info(f"HumanEval pass@1: {metrics['pass_at_1']:.4f}")

    if args.benchmark in ["mbpp", "all"]:

        logger.info("\n" + "=" * 60)

        logger.info("Evaluating MBPP")

        logger.info("=" * 60)

        problems = load_mbpp()

        if problems:

            metrics = evaluate_benchmark(
                problems,
                generator,
                executor,
                num_samples=args.num_samples,
                temperature=args.temperature,
                cache_dir=args.cache_dir,
            )

            all_results["mbpp"] = metrics

            logger.info(f"MBPP pass@1: {metrics['pass_at_1']:.4f}")

    if args.benchmark in ["livecodebench", "all"]:

        logger.info("\n" + "=" * 60)

        logger.info("Evaluating LiveCodeBench")

        logger.info("=" * 60)

        problems = load_livecodebench()

        if problems:

            metrics = evaluate_benchmark(
                problems,
                generator,
                executor,
                num_samples=args.num_samples,
                temperature=args.temperature,
                cache_dir=args.cache_dir,
            )

            all_results["livecodebench"] = metrics

            logger.info(f"LiveCodeBench pass@1: {metrics['pass_at_1']:.4f}")

    with open(args.output, "w") as f:

        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {args.output}")

    print("\n" + "=" * 60)

    print("EVALUATION SUMMARY")

    print("=" * 60)

    for benchmark, metrics in all_results.items():

        print(f"\n{benchmark.upper()}:")

        print(f"  pass@1:  {metrics['pass_at_1']:.4f}")

        if metrics.get("pass_at_5"):

            print(f"  pass@5:  {metrics['pass_at_5']:.4f}")

        if metrics.get("pass_at_10"):

            print(f"  pass@10: {metrics['pass_at_10']:.4f}")

        print(f"  errors:  {metrics['num_errors']}")


if __name__ == "__main__":

    main()
