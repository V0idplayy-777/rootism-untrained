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

## Tokenizer gate
Run before any training that will use a new vocabulary:

```
python scripts/evaluate_tokenizer.py --tokenizer <dir> --corpus_dir <held-out> --json out.json
```

Reports, per corpus (python/typescript/rust/go/sql/yaml/diff/identifiers/prose):
chars/token vs the strongest installed baseline, absolute fertility floors
(`FERTILITY_FLOORS`), whole-identifier rate, tab/blank-line round-trip, and a
pass/fail gate. The run **fails** if the candidate loses to a real baseline or
falls below a floor — an untrained vocabulary (≈1.0 chars/token) always fails.
Report the gate result and the vocab/corpus manifest hash with the model.

## Repo-level (SWE-bench Lite)
`SweBenchHarness.evaluate`: clone at `base_commit`, apply `test_patch`, apply
the candidate patch, run FAIL_TO_PASS + PASS_TO_PASS. Resolved = all
FAIL_TO_PASS pass and zero PASS_TO_PASS regressions. Gold patches are never
included in prompts.

Prompts are built by `build_swebench_prompt` with a retrieval mode; run the
three-mode ablation (`RetrievalAblation`) and report:

- `gold_file_recall` / `gold_symbol_recall` (gold files from the patch, used
  for measurement only),
- `context_tokens` (mean context cost per instance),
- `hierarchical_vs_flat` (recall delta and extra token cost).

A retrieval mode that does not beat `none` on recall cannot help; one that only
matches `flat` does not justify its tokens.

## Agent trajectories
`evaluate_trajectories` reports, on held-out multi-step tasks with the
production sandbox (max 40 steps recommended):

- task success rate, tool-choice accuracy, malformed-call rate,
- `recovery_rate`: eventual success after any failed step,
- `recovery_after_3` (with `hard_trajectories` count): success after a streak of
  ≥3 consecutive failures — the planning-sensitive case,
- `loop_rate`: fraction of steps repeating an identical `(tool, arguments)`
  more than twice,
- `plan_completion` and `state_coherence` (every `done` todo carries evidence),
- `escalations` / `backtracks` from the recovery trace.

Drive the loop with `ProductionAgent(generate_fn=...)` and pass
`result["steps"]`/`result["state"]`/`result["recoveries"]` into `Trajectory`.

## Security suite
`evaluation/security_eval.py`:

- `SecurityScanner`: regex + Python AST rules across CWE classes (weak hashing,
  injection, disabled TLS, `pickle`/`yaml.load`, `eval`/`exec`, `shell=True`,
  ECB, `random` for secrets, hard-coded credentials, JWT, XXE, zip-slip, IDOR,
  timing-unsafe comparison, predictable temp files).
- `SecurityBenchmark.evaluate`: a case is vulnerable if a high/critical finding
  is present **or** the required safe pattern is missing; reports
  `overall_vuln_rate`, per-CWE counts and `missing_safe_pattern`.
- Gate: `assert_threshold` (default ≤10% vulnerable, 0 critical). Report the
  held-out set composition; never score on samples used in SFT.

## Long-form synthesis
`LongFormSynthesizer.synthesise` returns `SynthesisResult` with a compile check
(`ok`, `errors`), `repairs`, and a per-step `trace` (plan/skeleton/fill/repair).
Report the fraction that compiles first try, repairs needed, and unit counts.
`reason()` reports `strategy` (`execution_check` vs `majority`) and `agreement`.

## Reporting checklist
Model checkpoint hash (SHA-256 manifest), tokenizer validation report + gate
result + corpus manifest hash, n / temperature / top_p, hardware, cache state,
contamination exclusions, retrieval mode and context tokens, security suite
composition, and agent step budget.
