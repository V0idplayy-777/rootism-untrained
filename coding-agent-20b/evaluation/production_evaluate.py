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



                                     



            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:



                f.write(full_code)



                script_path = f.name



            



                           



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



            _dev = next(self.model.parameters()).device if hasattr(self.model, "parameters") else "cpu"



            input_tensor = torch.tensor([input_ids], device=_dev)



            



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



    



                   



    correct = sum(1 for r in results if r['passed'])



    



                     



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



        return evaluate_problem(problem, generator, executor,



                                num_samples=num_samples, temperature=temperature)







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



                    (cdir / f"{problem.id.replace('/', '_')}.json").write_text(json.dumps(result))



                                                     



        by_id = {r["id"]: r for r in results}



        results = [by_id[p.id] for p in problems if p.id in by_id]



    



                       



    pass_at_1_scores = [r["pass_at_1"] for r in results]



    pass_at_5_scores = [r["pass_at_5"] for r in results if r["pass_at_5"] is not None]



    pass_at_10_scores = [r["pass_at_10"] for r in results if r["pass_at_10"] is not None]



    



                    



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



        'errors': errors[:10],                                



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



    fail_to_pass: List[str]



    pass_to_pass: List[str]



    patch: str = ""                                               



    test_patch: str = ""











def load_swebench(split: str = "test") -> List[SweBenchProblem]:



    """Load SWE-bench Lite from HF (`princeton-nlp/SWE-bench_Lite`)."""



    try:



        from datasets import load_dataset



        import json as _json



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



            out.append(SweBenchProblem(



                instance_id=item["instance_id"], repo=item["repo"],



                base_commit=item["base_commit"],



                problem_statement=item["problem_statement"],



                fail_to_pass=_lst(item.get("FAIL_TO_PASS", [])),



                pass_to_pass=_lst(item.get("PASS_TO_PASS", [])),



                patch=item.get("patch", ""), test_patch=item.get("test_patch", "")))



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







    def evaluate(self, problem: SweBenchProblem, candidate_patch: str, run_tests_fn=None) -> Dict[str, Any]:



        import shutil as _shutil



        import subprocess as _sp



        repo_dir = self.work_dir / problem.instance_id



        try:



            if repo_dir.exists():



                _shutil.rmtree(repo_dir)



            clone = _sp.run(["git", "clone", "--quiet", f"https://github.com/{problem.repo}.git", str(repo_dir)],



                            capture_output=True, text=True, timeout=self.timeout)



            if clone.returncode != 0:



                return {"instance_id": problem.instance_id, "resolved": False, "error": f"clone failed: {clone.stderr[:500]}"}



            _sp.run(["git", "checkout", "--quiet", problem.base_commit], cwd=repo_dir, capture_output=True, timeout=120)



            if problem.test_patch:



                _sp.run(["git", "apply", "-"], cwd=repo_dir, input=problem.test_patch,



                        capture_output=True, text=True, timeout=60)



            applied = _sp.run(["git", "apply", "-"], cwd=repo_dir, input=candidate_patch,



                              capture_output=True, text=True, timeout=60)



            if applied.returncode != 0:



                return {"instance_id": problem.instance_id, "resolved": False,



                        "error": f"patch does not apply: {applied.stderr[:500]}"}



            runner = run_tests_fn or (lambda d, tests: {"passed": [], "failed": list(tests), "error": "no test runner configured"})



            results = runner(str(repo_dir), problem.fail_to_pass + problem.pass_to_pass)



            passed = set(results.get("passed", []))



            resolved = all(t in passed for t in problem.fail_to_pass) and all(t in passed for t in problem.pass_to_pass)



            return {"instance_id": problem.instance_id, "resolved": resolved, "test_results": results}



        except Exception as e:



            return {"instance_id": problem.instance_id, "resolved": False, "error": str(e)}



        finally:



            _shutil.rmtree(repo_dir, ignore_errors=True)











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



        return {" ".join(toks[i:i + n]) for i in range(max(len(toks) - n + 1, 0))} if len(toks) >= n else {" ".join(toks)} if toks else set()







    def add_training_text(self, text: str):



        self.index |= self._ngrams(text, self.n)







    def check(self, text: str) -> Dict[str, Any]:



        grams = self._ngrams(text, self.n)



        hit = grams & self.index



        return {"contaminated": bool(hit), "overlap": len(hit) / max(len(grams), 1), "shared_ngrams": len(hit)}











@dataclass



class Trajectory:



    task: str



    steps: List[Dict]                                                       



    success: bool



    error: Optional[str] = None











def evaluate_trajectories(trajectories: List[Trajectory]) -> Dict[str, Any]:



    """Agent-trajectory metrics: task success, tool-choice correctness,
    malformed-call rate and error-recovery rate (succeed after a failed step)."""



    if not trajectories:



        return {"num_trajectories": 0, "success_rate": 0.0}



    tool_total = tool_correct = malformed = 0



    recovered = recoverable = 0



    for tr in trajectories:



        had_error = False



        for step in tr.steps:



            if step.get("tool") is None:



                malformed += 1



                had_error = True



                continue



            tool_total += 1



            if step.get("expected_tool") is None or step["expected_tool"] == step.get("tool"):



                tool_correct += 1



            result = step.get("result") or {}



            if isinstance(result, dict) and result.get("error"):



                had_error = True



        if had_error:



            recoverable += 1



            if tr.success:



                recovered += 1



    total_calls = tool_total + malformed



    return {



        "num_trajectories": len(trajectories),



        "success_rate": sum(1 for t in trajectories if t.success) / len(trajectories),



        "tool_accuracy": tool_correct / max(tool_total, 1),



        "malformed_rate": malformed / max(total_calls, 1),



        "recovery_rate": recovered / max(recoverable, 1),



        "avg_steps": sum(len(t.steps) for t in trajectories) / len(trajectories),



    }











                                                              



      



                                                              







def main():



    """Run evaluation on all benchmarks."""



    import argparse



    import torch



    



    parser = argparse.ArgumentParser(description="Evaluate code generation model")



    parser.add_argument("--benchmark", type=str, required=True,



                       choices=["humaneval", "mbpp", "livecodebench", "swebench", "all"])



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



    checkpoint = torch.load(os.path.join(args.model_path, "model.pt"), map_location="cpu")



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



        logger.info("="*60)



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



    



                  



    with open(args.output, 'w') as f:



        json.dump(all_results, f, indent=2)



    



    logger.info(f"\nResults saved to {args.output}")



    



                   



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



