# Capability gaps, mitigations, and how each is measured

Six known weaknesses were identified for this stack. This document records, for
each one: the failure mode, the mechanism that now addresses it, how to verify
the mitigation, and what is still *not* solved. Nothing here claims a model
quality jump that cannot be measured from this repository — the deliverables are
training data, training/eval harnesses and agent-side machinery.

| # | Gap | Mechanism | Measured by |
|---|-----|-----------|-------------|
| 1 | Tokenizer: 320-token placeholder, code-aware pre-tok but not competitive | Streaming BPE on a real corpus + exact vocab alignment + competitive gate | `scripts/evaluate_tokenizer.py`, `validation.json`, `benchmark.json` |
| 2 | 8k/32k context vs real repos | AST+block chunking, BM25 retrieval, repo maps, token-budgeted assembly, 64k YaRN profile | `RetrievalAblation` (gold-file/symbol recall, context tokens) |
| 3 | Multi-step tool use: recovery, 15–20 call planning, session state | `SessionState`, `ContextWindow` compaction, `RecoveryPolicy` ladder, `StallDetector`, resumable state | `evaluate_trajectories` (`recovery_after_3`, `loop_rate`, `plan_completion`, `state_coherence`) |
| 4 | Security-critical generation lags general coding | Curated safe/vulnerable pairs (CWE-tagged), review rubric, security SFT/DPO, mixture upsampling | `evaluation/security_eval.py` (`overall_vuln_rate`, per-CWE counts) |
| 5 | Training distribution: unusual phrasing, distant concepts | Phrasing styles, cross-domain synthesis, spec/design/implement ladders, clarification tasks | `dataset/augmentation.py` tests + SFT mixture composition |
| 6 | Long functions / deep reasoning need scratch space | Plan → skeleton → per-unit fill, explicit scratch budget, execution-checked self-consistency | `model/longform.py` traces + `ReasoningBudget` |

---

## 1. Tokenizer

**Problem.** The shipped `CodeTokenizer` is a 79-special + 256-byte vocabulary
(335 entries) with no merges: useful as a code-aware *pre*-tokenizer, useless as
a production vocabulary. A BPE run that silently produces fewer merges than
requested also produces a vocab smaller than the model's embedding table.

**What changed.**

- `scripts/train_production_tokenizer.py` streams `--data_dir` **or** the JSONL
  shards from `scripts/sample_tokenizer_corpus.py` (no corpus cap), applies the
  size/SPDX gates, dedupes by content hash, caps documents per language, and
  writes `manifest.json` with a corpus SHA-256.
- `EfficientBPETrainer` pads every trained vocab to exactly `vocab_size` with
  deterministic `<|reserved_N|>` tokens; `CodeTokenizer.pad_to_size` and
  `CodeTokenizer.load` both fail loudly on size/id-map corruption.
- New reserved specials for the rest of this work: `<|scratchpad|>`, `<|plan|>`,
  `<|repo_map|>`, `<|retrieval|>`, `<|file_chunk|>`, `<|diff|>`, `<|review|>`,
  `<|security|>`, `<|clarify|>`, `<|tool_error|>`, … Existing ids (including
  `<|tool_call|>`=48) are frozen.
- Diff-aware pre-tokenization: `@@ … @@` hunk headers and `---`/`+++` file
  markers are single pre-tokens, so patch output is not shredded into
  punctuation.
- `TokenizerValidator` now reports per-language fertility, whole-identifier
  rate, whitespace/tab round-trip, and unified-diff chars/token.
- `evaluation/tokenizer_bench.py` compares against real baselines (bytes,
  word/punct, and `tiktoken` when installed) with absolute fertility floors, so
  an untrained vocabulary cannot pass.

**Verification.**

```bash
python scripts/sample_tokenizer_corpus.py --dataset bigcode/the-stack-smol \
    --output_dir data/tokenizer_corpus --per_language_docs 2000
python scripts/train_production_tokenizer.py --data_dir data/tokenizer_corpus \
    --output_dir tokenizer --vocab_size 128256 --benchmark_gate
python scripts/evaluate_tokenizer.py --tokenizer tokenizer --corpus_dir <heldout> \
    --json tokenizer/benchmark.json
```

**Still open.** No published 128k vocabulary is checked in (it requires a real
corpus run); the gate is only as strong as the baselines installed (`tiktoken` is
optional). Absolute floors are calibrated on the built-in samples and must be
re-derived from the real held-out corpus.

## 2. Context: repositories larger than the window

**Problem.** SWE-bench Lite is in the protocol, but 8k native (32k YaRN) cannot
hold a repository, so fixes stay local and architectural changes lose.

**What changed.**

- `agent/repo_context.py` builds an in-memory index: Python chunked by AST
  (module header, top-level defs/classes, methods of large classes), other
  languages by blank-line blocks; identifier-aware tokenisation
  (`getUserById` → get/user/by/id; `parse_tool_call` → parse/tool/call).
- BM25 retrieval, exact/suffix symbol lookup, import-graph expansion
  (`related_files`), per-file outlines, and a **budgeted hierarchical repo map**
  (repo → file → signatures, focused on retrieved files).
- `ContextAssembler` packs map + chunks + neighbours into a hard token budget
  with provenance, in three ablation modes: `none`, `flat`, `hierarchical`.
- `build_swebench_prompt` and `RetrievalAblation` compute gold-file and
  gold-symbol recall plus context cost per mode. Gold files come from the patch
  and are used **only for measurement**, never in the prompt.
- `model/config.py`: `get_20b_64k_config()` (YaRN ×8) as headroom for a few
  large files, documented as extrapolation, not native long context.

**Verification.**

```python
from evaluation.production_evaluate import RetrievalAblation, load_swebench
ablation = RetrievalAblation(budget_tokens=6000).evaluate(problems, repo_dirs)
# none: recall 0 | flat: recall R | hierarchical: recall R + map + neighbours
```

**Still open.** Lexical retrieval misses paraphrase-heavy issues; no embeddings
or call-graph depth > 1; repo maps are signature-level, not semantic. Long-context
quality beyond 32k remains unproven without long-context training.

## 3. Long-horizon tool use

**Problem.** 15–20+ calls, many failures, and the model loses the plan, repeats a
broken call, or ignores a constraint stated ten steps ago. The eval protocol
already tracked recovery rate because this was expected to be weak.

**What changed.**

- `agent/session.py`: `SessionState` (plan, todos with **evidence**, pinned
  facts, failures, files, test results, scratchpad) with atomic JSON save/load
  and resume; `ObservationCompactor` (per-tool summaries: failing test ids, first
  assertion error, changed files); `ContextWindow` with two-level compaction
  (pinned layer → one-line step digests → recent steps) inside a hard budget.
- `agent/recovery.py`: ladder `retry → repair → change → switch → reflect →
  backtrack → escalate`, normalised failure signatures (line numbers, paths,
  addresses, hex ids removed), tool fallback chains, `StallDetector` for
  spinning, and an honest escalation answer that reports blockers instead of
  claiming success.
- `agent/tools.py`: `revert` (per-file `git checkout --` or stash incl.
  untracked) for backtracking; `update_plan` / `note_fact` are control tools
  handled by the agent, not the sandbox.
- `ProductionAgent` wires all of it: plan-first prompt, injected repo context,
  recovery prompts, stall checks, resumable state, and a `generate_fn` seam so
  the loop is testable without a GPU.
- `evaluate_trajectories` adds `recovery_after_3`, `hard_trajectories`,
  `loop_rate`, `plan_completion`, `state_coherence`, `escalations`,
  `backtracks`.

**Verification.** `tests/test_agent_loop.py` runs scripted trajectories end to
end: plan/fact/verify flow, 8 consecutive failures → escalation, malformed call
repair, resume from state, retrieval injection, revert semantics.

**Still open.** Recovery is policy-driven, not learned; it bounds damage but
cannot invent a correct hypothesis. Long-session coherence still depends on the
model using `update_plan`/`note_fact`.

## 4. Security-critical generation

**Problem.** General coding ability does not transfer to crypto/auth/sandbox
code; public corpora are full of `md5`, `verify=False`, `shell=True`.

**What changed.**

- `dataset/security_data.py`: 18 curated CWE-tagged samples (password KDFs,
  constant-time compare, SQL injection, IDOR, path traversal, command injection,
  CSPRNG, pickle/YAML, JWT alg pinning, TLS verification, SSRF allow-lists,
  hard-coded secrets, `eval`, XXE, zip-slip, AEAD vs ECB, temp files), each with
  vulnerable **and** safe code plus the reason. Generators emit review / repair /
  implement SFT rows and DPO pairs (`chosen`=safe, `rejected`=vulnerable).
- `dataset/mixture.py`: `security-sft` (weight 0.03, upsample ×3);
  `configs/pipeline.json` upsamples it in midtrain and SFT.
- `evaluation/security_eval.py`: regex + AST scanner (weak hashing, injection,
  disabled TLS, `pickle`, `eval`, shell, ECB, `random` for secrets, hard-coded
  credentials, JWT, XXE, zip-slip, IDOR, temp files), plus **required safe
  patterns** per task ("did not use md5" ≠ "used Argon2"). Scores vulnerability
  rate per CWE and fails a threshold.
- `security_review_rubric()` is both training text and the scoring checklist.

**Verification.** Self-test over the curated set: all 18 vulnerable samples are
flagged, all 18 safe samples pass (`tests/test_security_augment_longform.py`).
`SecurityBenchmark.assert_threshold` gates a release (default ≤10% vulnerable,
0 critical).

**Still open.** Static rules cannot prove absence of vulnerabilities (SSRF needs
taint analysis); the benchmark is a floor, not a certification. Human review
remains mandatory for crypto/auth changes.

## 5. Training distribution

**Problem.** Synthetic instructions + The Stack cover common patterns; unusual
phrasing, ambiguity, and cross-domain composition expose the edges.

**What changed.** `dataset/augmentation.py`:

- eight phrasing styles (terse, casual, verbose, constraint-first, non-native,
  typo, frustrated, partial repro) that preserve technical content;
- `ConceptCombiner`: 10 cross-domain task families (compilers×billing,
  DB×ML, storage×security, …) each with a **compilable** skeleton, explicit
  requirements and acceptance criteria;
- `AbstractionLadder`: the same task as spec / design brief / implementation;
- `AmbiguityGenerator`: underspecified requests plus the targeted clarification
  a good assistant must ask;
- `augment_rows` assembles these into SFT rows; mixture gains `creative-sft`
  (0.04, upsample ×2).

**Verification.** Tests assert determinism, compilable skeletons, schema
compatibility and that originals are preserved.

**Still open.** Template-derived diversity is not human diversity. This expands
coverage; it does not replace real user traffic or red-teaming.

## 6. Long functions and deep reasoning

**Problem.** An 8k window with a 1–2k generation budget cannot emit or reason
through a large artifact; intermediate steps get silently dropped.

**What changed.** `model/longform.py`:

- `ReasoningBudget` splits the window into plan / scratch / answer with explicit
  minimums (and shrinks scratch when the window is tight);
- `LongFormSynthesizer` runs plan → interface skeleton → per-unit fill (each fill
  sees the plan and neighbour **signatures**, not bodies) → AST validation →
  bounded repair; `trace` records every step and `SynthesisResult.ok` is
  compile-based, not vibes-based;
- `missing_definitions` lints undefined calls; `split_long_function` proposes
  extraction points for over-long functions instead of truncating;
- `reason()` implements scratchpad reasoning with execution-checked
  self-consistency, falling back to majority agreement when no checker exists.

**Verification.** Tests cover budget scaling with 8k vs 32k, plan parsing,
per-unit filling on a scripted generator, the repair loop, and self-consistency
selection (execution check beats majority when they disagree).

**Still open.** Decomposition quality is bounded by the planner; if the plan is
wrong, the units compose a plausible but incorrect module. The validator checks
syntax and obvious undefined names — not semantics. Run the tests.

---

## Acceptance criteria for the next milestone

1. Tokenizer trained on ≥20B characters of permissively licensed code, gate
   passing against `tiktoken:cl100k_base` on Python/TS/Rust/Go/diff.
2. `RetrievalAblation` shows hierarchical recall ≥ flat recall at equal or lower
   token cost on ≥100 SWE-bench Lite instances.
3. Agent trajectories (n≥100, max 40 steps): `recovery_after_3` ≥ 0.4,
   `loop_rate` ≤ 0.05, `state_coherence` ≥ 0.95.
4. Security suite: vulnerable rate ≤ 5% overall, 0 critical findings, with a
   held-out set not used in training.
5. Clarification tasks: model asks before coding on ≥80% of ambiguous prompts
   and still solves the unambiguous paraphrase variants.
