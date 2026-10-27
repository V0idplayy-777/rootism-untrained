# Evaluation Protocol

## Code generation (HumanEval / MBPP / LiveCodeBench)
- Loader: `evaluation/production_evaluate.py` (`load_humaneval`, `load_mbpp`, `load_livecodebench`).
- Generation: temperature 0.2, n=10 samples/problem unless stated; report
  unbiased pass@1/5/10 (`compute_pass_at_k`). Greedy (T=0) pass@1 also reported.
- Execution: sandboxed (`SandboxExecutor`, network off, 30s timeout); a sample
  passes only if the full test block exits 0. Results cached per-problem under
  `eval_cache/` (delete the cache when the model or prompts change).
- Decontamination: run `ContaminationChecker` (13-gram) of every eval prompt
  against the training index; contaminated items are excluded and counted.

## Repo-level (SWE-bench Lite)
`SweBenchHarness.evaluate`: clone at `base_commit`, apply `test_patch`, apply
the candidate patch, run FAIL_TO_PASS + PASS_TO_PASS. Resolved = all
FAIL_TO_PASS pass and zero PASS_TO_PASS regressions. Gold patches are never
included in prompts.

## Agent trajectories
`evaluate_trajectories` reports: task success rate, tool-choice accuracy
(against expected tools), malformed-call rate (parser rejections), and
recovery rate (eventual success after a failed/malformed step). Evaluate on
held-out multi-step tasks with the production sandbox, max 20 steps.

## Reporting checklist
Model checkpoint hash (SHA-256 manifest), tokenizer validation report,
n / temperature / top_p, hardware, cache state, contamination exclusions.
