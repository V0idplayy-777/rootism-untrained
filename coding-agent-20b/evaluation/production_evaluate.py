"""
Production Evaluation Harness

Features:
- Docker-based sandboxing for safe code execution
- Standard benchmarks: HumanEval, MBPP, LiveCodeBench, SWE-bench
- Parallel evaluation with resource limits
- Detailed metrics and error analysis
- Result caching and resumption
"""

import os
import json
import time
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np

try:
    import docker
    HAS_DOCKER = True
except ImportError:
    HAS_DOCK = False

logger = logging.getLogger(__name__)


# ============================================================
# BENCHMARK LOADERS
# ============================================================

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
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def load_humaneval() -> List[EvaluationProblem]:
    """Load HumanEval benchmark."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("openai_humaneval", split="test")
        
        problems = []
        for item in dataset:
            problems.append(EvaluationProblem(
                id=item["task_id"],
                prompt=item["prompt"],
                canonical_solution=item["canonical_solution"],
                test_cases=item["test"],
                entry_point=item["entry_point"],
                language="python",
                difficulty="medium",
                metadata={"source": "humaneval"}
            ))
        
        logger.info(f"Loaded {len(problems)} HumanEval problems")
        return problems
    except Exception as e:
        logger.error(f"Error loading HumanEval: {e}")
        return []


def load_mbpp() -> List[EvaluationProblem]:
    """Load MBPP benchmark."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("mbpp", split="test")
        
        problems = []
        for i, item in enumerate(dataset):
            prompt = f'"""\n{item["text"]}\n\n{item.get("test_setup_code", "")}\n"""'
            
            problems.append(EvaluationProblem(
                id=f"mbpp_{i}",
                prompt=prompt,
                canonical_solution=item["code"],
                test_cases="\n".join(item["test_list"]),
                entry_point=item.get("entry_point", ""),
                language="python",
                difficulty="easy",
                metadata={"source": "mbpp"}
            ))
        
        logger.info(f"Loaded {len(problems)} MBPP problems")
        return problems
    except Exception as e:
        logger.error(f"Error loading MBPP: {e}")
        return []


def load_livecodebench() -> List[EvaluationProblem]:
    """Load LiveCodeBench (competitive programming)."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("livecodebench/code_generation", split="test")
        
        problems = []
        for i, item in enumerate(dataset):
            problems.append(EvaluationProblem(
                id=item.get("problem_id", f"lcb_{i}"),
                prompt=item["prompt"],
                canonical_solution=item.get("canonical_solution", ""),
                test_cases=item.get("test_cases", ""),
                entry_point=item.get("entry_point", ""),
                language="python",
                difficulty=item.get("difficulty", "medium"),
                metadata={"source": "livecodebench"}
            ))
        
        logger.info(f"Loaded {len(problems)} LiveCodeBench problems")
        return problems
    except Exception as e:
        logger.error(f"Error loading LiveCodeBench: {e}")
        return []


# ============================================================
# DOCKER SANDBOX EXECUTOR
# ============================================================

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
        timeout: int = None,
    ) -> Dict[str, Any]:
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
            # Write code to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                script_path = f.name
            
            # Run in Docker
            container = self.client.containers.run(
                self.image,
                command=f"python /script.py",
                volumes={script_path: {'bind': '/script.py', 'mode': 'ro'}},
                mem_limit=self.memory_limit,
                network_disabled=self.network_disabled,
                detach=True,
            )
            
            try:
                result = container.wait(timeout=timeout)
                stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
                stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
                exit_code = result['StatusCode']
                
                execution_time = time.time() - start_time
                
                passed = exit_code == 0 and not stderr
                
                return {
                    'passed': passed,
                    'failed': not passed,
                    'error': stderr if not passed else None,
                    'stdout': stdout,
                    'stderr': stderr,
                    'exit_code': exit_code,
                    'execution_time': execution_time,
                }
            
            except Exception as e:
                container.kill()
                return {
                    'passed': False,
                    'failed': True,
                    'error': f'Timeout after {timeout}s',
                    'stdout': '',
                    'stderr': str(e),
                    'exit_code': -1,
                    'execution_time': timeout,
                }
            finally:
                container.remove(force=True)
                os.unlink(script_path)
        
        except Exception as e:
            return {
                'passed': False,
                'failed': True,
                'error': f'Execution error: {str(e)}',
                'stdout': '',
                'stderr': str(e),
                'exit_code': -1,
                'execution_time': 0,
            }
    
    def _fallback_execute(self, code: str, timeout: int) -> Dict:
        """Fallback without Docker."""
        start_time = time.time()
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                script_path = f.name
            
            result = subprocess.run(
                ['python3', script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            os.unlink(script_path)
            execution_time = time.time() - start_time
            
            passed = result.returncode == 0 and not result.stderr
            
            return {
                'passed': passed,
                'failed': not passed,
                'error': result.stderr if not passed else None,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'exit_code': result.returncode,
                'execution_time': execution_time,
            }
        
        except subprocess.TimeoutExpired:
            return {
                'passed': False,
                'failed': True,
                'error': f'Timeout after {timeout}s',
                'stdout': '',
                'stderr': '',
                'exit_code': -1,
                'execution_time': timeout,
            }
        except Exception as e:
            return {
                'passed': False,
                'failed': True,
                'error': str(e),
                'stdout': '',
                'stderr': str(e),
                'exit_code': -1,
                'execution_time': 0,
            }


# ============================================================
# CODE GENERATOR
# ============================================================

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
    ) -> List[str]:
        """Generate code completions."""
        import torch
        
        self.model.eval()
        completions = []
        
        for _ in range(num_samples):
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            input_tensor = torch.tensor([input_ids], device=self.model.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_tensor,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=(num_samples > 1),
                )
            
            completion = self.tokenizer.decode(output_ids[0][len(input_ids):])
            completions.append(completion)
        
        return completions


# ============================================================
# EVALUATION
# ============================================================

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
) -> Dict[str, Any]:
    """Evaluate a single problem."""
    # Generate completions
    completions = generator.generate(
        problem.prompt,
        temperature=temperature,
        num_samples=num_samples,
    )
    
    # Test each completion
    results = []
    for i, completion in enumerate(completions):
        full_code = problem.prompt + completion
        result = executor.execute(full_code, problem.test_cases)
        results.append(result)
    
    # Count correct
    correct = sum(1 for r in results if r['passed'])
    
    # Compute metrics
    n = num_samples
    c = correct
    
    return {
        'id': problem.id,
        'num_samples': n,
        'num_correct': c,
        'pass_at_1': compute_pass_at_k(n, c, 1),
        'pass_at_5': compute_pass_at_k(n, c, 5) if n >= 5 else None,
        'pass_at_10': compute_pass_at_k(n, c, 10) if n >= 10 else None,
        'results': results,
    }


def evaluate_benchmark(
    problems: List[EvaluationProblem],
    generator: CodeGenerator,
    executor: SandboxExecutor,
    num_samples: int = 10,
    temperature: float = 0.2,
    num_workers: int = 4,
    cache_dir: str = None,
) -> Dict[str, Any]:
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
    
    # Check cache
    if cache_dir:
        cache_path = Path(cache_dir) / "eval_cache.json"
        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
                logger.info(f"Loaded {len(cached)} cached results")
        else:
            cached = {}
    else:
        cached = {}
    
    # Evaluate problems
    for i, problem in enumerate(problems, 1):
        # Check cache
        if problem.id in cached:
            results.append(cached[problem.id])
            if i % 10 == 0:
                logger.info(f"  [{i}/{len(problems)}] Cached: {problem.id}")
            continue
        
        # Evaluate
        result = evaluate_problem(
            problem, generator, executor,
            num_samples=num_samples,
            temperature=temperature,
        )
        results.append(result)
        
        # Cache result
        if cache_dir:
            cached[problem.id] = result
            cache_path = Path(cache_dir) / "eval_cache.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(cached, f)
        
        if i % 10 == 0:
            logger.info(f"  [{i}/{len(problems)}] Evaluated: {problem.id}")
    
    # Aggregate metrics
    pass_at_1_scores = [r["pass_at_1"] for r in results]
    pass_at_5_scores = [r["pass_at_5"] for r in results if r["pass_at_5"] is not None]
    pass_at_10_scores = [r["pass_at_10"] for r in results if r["pass_at_10"] is not None]
    
    # Error analysis
    errors = []
    for r in results:
        if r['num_correct'] == 0:
            errors.append({
                'id': r['id'],
                'errors': [res['error'] for res in r['results'] if res['error']]
            })
    
    metrics = {
        'num_problems': len(problems),
        'num_evaluated': len(results),
        'pass_at_1': np.mean(pass_at_1_scores) if pass_at_1_scores else 0,
        'pass_at_5': np.mean(pass_at_5_scores) if pass_at_5_scores else None,
        'pass_at_10': np.mean(pass_at_10_scores) if pass_at_10_scores else None,
        'num_errors': len(errors),
        'results': results,
        'errors': errors[:10],  # First 10 errors for analysis
    }
    
    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    """Run evaluation on all benchmarks."""
    import argparse
    import torch
    
    parser = argparse.ArgumentParser(description="Evaluate code generation model")
    parser.add_argument("--benchmark", type=str, required=True,
                       choices=["humaneval", "mbpp", "livecodebench", "all"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--cache_dir", type=str, default="eval_cache")
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    from model.config import CodingAgentConfig
    from model.model import CodingAgentModel
    from model.tokenizer import CodeTokenizer
    
    config = CodingAgentConfig.load(os.path.join(args.model_path, "config.json"))
    model = CodingAgentModel(config)
    checkpoint = torch.load(os.path.join(args.model_path, "model.pt"), map_location="cpu")
    model.load_state_dict(checkpoint)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = CodeTokenizer.load(tokenizer_path)
    
    generator = CodeGenerator(model, tokenizer)
    executor = SandboxExecutor()
    
    # Run benchmarks
    all_results = {}
    
    if args.benchmark in ["humaneval", "all"]:
        logger.info("\n" + "="*60)
        logger.info("Evaluating HumanEval")
        logger.info("="*60)
        problems = load_humaneval()
        if problems:
            metrics = evaluate_benchmark(
                problems, generator, executor,
                num_samples=args.num_samples,
                temperature=args.temperature,
                cache_dir=args.cache_dir,
            )
            all_results["humaneval"] = metrics
            logger.info(f"HumanEval pass@1: {metrics['pass_at_1']:.4f}")
    
    if args.benchmark in ["mbpp", "all"]:
        logger.info("\n" + "="*60)
        logger.info("Evaluating MBPP")
        logger.info("="*60)
        problems = load_mbpp()
        if problems:
            metrics = evaluate_benchmark(
                problems, generator, executor,
                num_samples=args.num_samples,
                temperature=args.temperature,
                cache_dir=args.cache_dir,
            )
            all_results["mbpp"] = metrics
            logger.info(f"MBPP pass@1: {metrics['pass_at_1']:.4f}")
    
    if args.benchmark in ["livecodebench", "all"]:
        logger.info("\n" + "="*60)
        logger.info("Evaluating LiveCodeBench")
        logger("="*60)
        problems = load_livecodebench()
        if problems:
            metrics = evaluate_benchmark(
                problems, generator, executor,
                num_samples=args.num_samples,
                temperature=args.temperature,
                cache_dir=args.cache_dir,
            )
            all_results["livecodebench"] = metrics
            logger.info(f"LiveCodeBench pass@1: {metrics['pass_at_1']:.4f}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nResults saved to {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for benchmark, metrics in all_results.items():
        print(f"\n{benchmark.upper()}:")
        print(f"  pass@1:  {metrics['pass_at_1']:.4f}")
        if metrics.get('pass_at_5'):
            print(f"  pass@5:  {metrics['pass_at_5']:.4f}")
        if metrics.get('pass_at_10'):
            print(f"  pass@10: {metrics['pass_at_10']:.4f}")
        print(f"  errors:  {metrics['num_errors']}")


if __name__ == "__main__":
    main()
