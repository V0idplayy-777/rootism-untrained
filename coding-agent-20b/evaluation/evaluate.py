"""
Evaluation Harness for Code Generation Models.

Implements evaluation on standard benchmarks:
- HumanEval (OpenAI): 164 Python function completion problems
- MBPP (Google): 974 basic Python programming problems
- LiveCodeBench: Recent competitive programming problems
- SWE-bench: Real GitHub issue resolution

Usage:
    # Run HumanEval
    python evaluation/evaluate.py --benchmark humaneval --model_path checkpoints/latest
    
    # Run MBPP
    python evaluation/evaluate.py --benchmark mbpp --model_path checkpoints/latest
    
    # Run all benchmarks
    python evaluation/evaluate.py --benchmark all --model_path checkpoints/latest
"""







import os



import sys



import json



import argparse



import logging



import tempfile



import subprocess



from pathlib import Path



from typing import List, Dict, Any, Optional, Callable



from dataclasses import dataclass



from concurrent.futures import ThreadPoolExecutor, as_completed



import multiprocessing as mp







import torch



import numpy as np







                    



sys.path.insert(0, str(Path(__file__).parent.parent))







logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")



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



            ))



        



        logger.info(f"Loaded {len(problems)} HumanEval problems")



        return problems



    



    except Exception as e:



        logger.error(f"Error loading HumanEval: {e}")



        logger.info("Install datasets: pip install datasets")



        return []











def load_mbpp() -> List[EvaluationProblem]:



    """Load MBPP benchmark."""



    try:



        from datasets import load_dataset



        dataset = load_dataset("mbpp", split="test")



        



        problems = []



        for i, item in enumerate(dataset):



                                                            



            prompt = f'"""{item["text"]}\n\n{item.get("test_setup_code", "")}\n"""'



            



            problems.append(EvaluationProblem(



                id=f"mbpp_{i}",



                prompt=prompt,



                canonical_solution=item["code"],



                test_cases="\n".join(item["test_list"]),



                entry_point=item.get("entry_point", ""),



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



            ))



        



        logger.info(f"Loaded {len(problems)} LiveCodeBench problems")



        return problems



    



    except Exception as e:



        logger.error(f"Error loading LiveCodeBench: {e}")



        return []











                                                              



                 



                                                              







class CodeGenerator:



    """Generate code completions using the model."""



    



    def __init__(self, model, tokenizer, max_new_tokens: int = 512):



        self.model = model



        self.tokenizer = tokenizer



        self.max_new_tokens = max_new_tokens



    



    def generate(self, prompt: str, temperature: float = 0.2, 



                 top_p: float = 0.95, num_samples: int = 1) -> List[str]:



        """Generate code completions."""



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











                                                              



                          



                                                              







def execute_code_with_tests(code: str, test_cases: str, timeout: int = 10) -> Dict[str, Any]:



    """
    Execute code with test cases in a sandboxed environment.
    
    Returns:
        Dict with 'passed', 'failed', 'error', 'stdout', 'stderr'
    """



                            



    full_code = f"{code}\n\n{test_cases}"



    



    try:



        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:



            f.write(full_code)



            temp_path = f.name



        



        result = subprocess.run(



            ['python3', temp_path],



            capture_output=True,



            text=True,



            timeout=timeout,



        )



        



        os.unlink(temp_path)



        



                       



        if result.returncode == 0:



            return {



                "passed": True,



                "failed": False,



                "error": None,



                "stdout": result.stdout,



                "stderr": result.stderr,



            }



        else:



            return {



                "passed": False,



                "failed": True,



                "error": result.stderr,



                "stdout": result.stdout,



                "stderr": result.stderr,



            }



    



    except subprocess.TimeoutExpired:



        return {



            "passed": False,



            "failed": True,



            "error": f"Timeout after {timeout}s",



            "stdout": "",



            "stderr": "",



        }



    except Exception as e:



        return {



            "passed": False,



            "failed": True,



            "error": str(e),



            "stdout": "",



            "stderr": "",



        }











                                                              



                    



                                                              







def compute_pass_at_k(n: int, c: int, k: int) -> float:



    """
    Compute pass@k metric.
    
    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k in pass@k
        
    Returns:
        Probability of at least one correct sample in k draws
    """



    if n - c < k:



        return 1.0



    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))











def evaluate_benchmark(



    problems: List[EvaluationProblem],



    generator: CodeGenerator,



    num_samples: int = 10,



    temperature: float = 0.2,



    num_workers: int = 4,



) -> Dict[str, Any]:



    """
    Evaluate model on a benchmark.
    
    Args:
        problems: List of evaluation problems
        generator: Code generator
        num_samples: Number of samples per problem (for pass@k)
        temperature: Sampling temperature
        num_workers: Number of parallel workers
        
    Returns:
        Evaluation results
    """



    results = []



    



    logger.info(f"Evaluating {len(problems)} problems with {num_samples} samples each")



    



    def process_problem(problem: EvaluationProblem) -> Dict[str, Any]:



        """Process a single problem."""



                              



        completions = generator.generate(



            problem.prompt,



            temperature=temperature,



            num_samples=num_samples,



        )



        



                              



        correct = 0



        for completion in completions:



            full_code = problem.prompt + completion



            result = execute_code_with_tests(full_code, problem.test_cases)



            if result["passed"]:



                correct += 1



        



                                



        n = num_samples



        c = correct



        



        return {



            "id": problem.id,



            "num_samples": n,



            "num_correct": c,



            "pass_at_1": compute_pass_at_k(n, c, 1),



            "pass_at_5": compute_pass_at_k(n, c, 5) if n >= 5 else None,



            "pass_at_10": compute_pass_at_k(n, c, 10) if n >= 10 else None,



        }



    



                                  



    with ThreadPoolExecutor(max_workers=num_workers) as executor:



        futures = {executor.submit(process_problem, p): p for p in problems}



        



        for i, future in enumerate(as_completed(futures), 1):



            try:



                result = future.result()



                results.append(result)



                



                if i % 10 == 0:



                    logger.info(f"  Processed {i}/{len(problems)} problems")



            except Exception as e:



                logger.error(f"Error processing problem: {e}")



    



                       



    pass_at_1_scores = [r["pass_at_1"] for r in results]



    pass_at_5_scores = [r["pass_at_5"] for r in results if r["pass_at_5"] is not None]



    pass_at_10_scores = [r["pass_at_10"] for r in results if r["pass_at_10"] is not None]



    



    metrics = {



        "num_problems": len(problems),



        "num_evaluated": len(results),



        "pass_at_1": np.mean(pass_at_1_scores) if pass_at_1_scores else 0,



        "pass_at_5": np.mean(pass_at_5_scores) if pass_at_5_scores else None,



        "pass_at_10": np.mean(pass_at_10_scores) if pass_at_10_scores else None,



        "results": results,



    }



    



    return metrics











                                                              



      



                                                              







def main():



    parser = argparse.ArgumentParser(description="Evaluate code generation model")



    parser.add_argument("--benchmark", type=str, required=True,



                       choices=["humaneval", "mbpp", "livecodebench", "all"],



                       help="Benchmark to evaluate")



    parser.add_argument("--model_path", type=str, required=True,



                       help="Path to model checkpoint")



    parser.add_argument("--tokenizer_path", type=str, default=None,



                       help="Path to tokenizer (defaults to model_path)")



    parser.add_argument("--num_samples", type=int, default=10,



                       help="Number of samples per problem")



    parser.add_argument("--temperature", type=float, default=0.2,



                       help="Sampling temperature")



    parser.add_argument("--max_new_tokens", type=int, default=512,



                       help="Max tokens to generate")



    parser.add_argument("--num_workers", type=int, default=4,



                       help="Number of parallel workers")



    parser.add_argument("--output", type=str, default="evaluation_results.json",



                       help="Output file for results")



    args = parser.parse_args()



    



                              



    logger.info(f"Loading model from {args.model_path}")



    



    from model.config import CodingAgentConfig



    from model.model import CodingAgentModel



    from model.tokenizer import CodeTokenizer



    



    config_path = os.path.join(args.model_path, "config.json")



    config = CodingAgentConfig.load(config_path)



    



    model = CodingAgentModel(config)



    checkpoint = torch.load(os.path.join(args.model_path, "model.pt"), map_location="cpu")



    model.load_state_dict(checkpoint)



    



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    model = model.to(device)



    model.eval()



    



    tokenizer_path = args.tokenizer_path or args.model_path



    tokenizer = CodeTokenizer.load(tokenizer_path)



    



    generator = CodeGenerator(model, tokenizer, max_new_tokens=args.max_new_tokens)



    



                                  



    all_results = {}



    



    if args.benchmark in ["humaneval", "all"]:



        logger.info("\n" + "="*60)



        logger.info("Evaluating HumanEval")



        logger.info("="*60)



        problems = load_humaneval()



        if problems:



            metrics = evaluate_benchmark(



                problems, generator,



                num_samples=args.num_samples,



                temperature=args.temperature,



                num_workers=args.num_workers,



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



                problems, generator,



                num_samples=args.num_samples,



                temperature=args.temperature,



                num_workers=args.num_workers,



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



                problems, generator,



                num_samples=args.num_samples,



                temperature=args.temperature,



                num_workers=args.num_workers,



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











if __name__ == "__main__":



    main()



