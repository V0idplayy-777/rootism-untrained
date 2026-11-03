# CodingAgent-20B — Model Card

## Summary
Decoder-only transformer (60 layers, hidden 6144, GQA 48Q/8KV, SwiGLU, RMSNorm,
RoPE theta=500k) with a 128,256-token byte-level BPE vocabulary, trained for
code generation and multi-step tool-using agent work. Context: 8k native,
32k via YaRN RoPE scaling (`get_20b_32k_config`), 64k YaRN ×8 as headroom
(`get_20b_64k_config`, extrapolation — see `docs/CAPABILITY_GAPS.md`).

Tokenization: 79 hand-written control tokens (ids frozen; `<|tool_call|>`=48,
`<|tool_result|>`=49) plus task tokens for scratchpad/plan/repo-map/retrieval/
diff/review/clarify and tool errors. Unused vocabulary capacity is padded with
deterministic `<|reserved_N|>` specials so the embedding always matches
`vocab_size` exactly. The vocabulary must be *trained* on a real corpus: the
untrained tokenizer is a byte-level fallback (≈1 char/token) and fails the
competitiveness gate by design.

## Training
Stages (see `configs/pipeline.json`, `training/schedules.py`): continuous
pretrain (WSD LR 3e-4, AdamW 0.9/0.95, wd 0.1, bf16, ZeRO-3) → mid-train
anneal on tests/issues/PRs/docs (plus security and distribution-shift sources)
→ SFT (prompt loss masked) → DPO (beta 0.1).

Data: permissive-license code (SPDX gate in `dataset/production_processor.py`,
reused by the tokenizer corpus sampler), MinHash-LSH dedup, attention-aware
packing (`dataset/packing.py`), mixture in `dataset/mixture.py`:
`the-stack-code 0.50 · issues 0.06 · prs 0.08 · tests 0.08 · docs 0.06 ·
instruction 0.08 · security 0.03 (×3) · creative 0.03 (×2) · instruction-induction
0.03 · recovery 0.02 · project-scaffold 0.015 (×1.5) · repair-trace 0.015 (×2)`
(weights sum to 1.00; the last two are the multi-file and fail->fix->green sources).

- **Security** (`dataset/security_data.py`): curated CWE-tagged safe/vulnerable
  pairs with review/repair/implement SFT rows and DPO preference pairs.
- **Distribution shift** (`dataset/augmentation.py`): phrasing styles, ten
  cross-domain task families with acceptance criteria, spec/design/implement
  ladders, and clarification tasks for underspecified requests.

Checkpoints are atomic + SHA-256 verified (`training/checkpoint.py`); recovery
policy in `training/monitor.py`.

### Training throughput and memory

`configs/training_config.json` is the 20B recipe. It is tuned for step time and
resident memory without touching anything that changes the model:

| what | old default | now |
| --- | --- | --- |
| attention | explicit softmax, `[B, 48, S, S]` fp32 weights | fused SDPA, no score matrix |
| LM loss | full `[B, S, 128256]` logits, then CE | chunked projection (1024 rows), loss only |
| micro-batch | 2 | 1, with `gradient_accumulation_steps: "auto"` from the real world size |
| ZeRO-3 state | optimizer + params offloaded to CPU | resident (`{"device": "none"}`) |
| activation checkpointing | whole block, plus DeepSpeed's own block | 2 segments per block, model-side only |
| data feeding | synchronous mmap read + `.to(device)` | threaded prefetch into pinned memory |
| accumulation | loss pre-divided *and* DeepSpeed-divided | DeepSpeed's internal window used correctly |

Loss, gradients, optimizer update, learning-rate schedule, global batch,
sequence length and data order are unchanged; `tests/test_train_perf.py` proves
bit-level agreement with the reference implementations. Enabling `bf16` mixed
precision with `torch.autocast` is *not* required — the recipe stays in bf16
parameter precision as before.

`python scripts/plan_training_memory.py --config configs/training_config.json
--gpus 64 --budget_gb 80` reports the per-rank breakdown (weights, gradients,
fp32 master, Adam moments, activations, logits, ZeRO gather buffers) before a
job is launched, and `--apply` writes back a fitting micro-batch. Full detail:
`docs/TRAINING_PERFORMANCE.md`.

## Evaluation
Protocol: `docs/EVAL_PROTOCOL.md`. Benchmarks: HumanEval, MBPP, LiveCodeBench,
SWE-bench Lite (repo-level, with a retrieval ablation: `none` / `flat` /
`hierarchical`), security suite (`evaluation/security_eval.py`), long-form
synthesis traces (`model/longform.py`), agent-trajectory metrics and 13-gram
contamination screening (`evaluation/production_evaluate.py`). Report pass@k
with n, temperature and cache hash; numbers without them are not comparable.

Tokenizers are evaluated with `scripts/evaluate_tokenizer.py`: per-language
fertility, whole-identifier rate and unified-diff cost against installed
baselines, with absolute floors so an untrained vocabulary cannot pass.

## Limitations & safety
- Agent code execution is fail-closed without Docker; the sandbox is the
  security boundary (read-only workspace, seccomp, cap-drop, quotas).
  `agent/safety.py` redaction/blocking is defence-in-depth, not a guarantee.
- 8k native context still truncates very large repos. Mitigation is retrieval,
  hierarchical repo maps and budgeted context (`agent/repo_context.py`), plus a
  resumable `SessionState` (`agent/session.py`) — not a larger window.
  Long-context quality beyond 32k YaRN is not established.
- Security evaluation is a static-rule floor (regex + AST + required safe
  patterns), not a proof. SSRF-class issues need data-flow analysis and are
  scored by missing safe patterns, not detection. Crypto/auth changes still
  require human review.
- Long-horizon behaviour is policy-bounded (`agent/recovery.py`), not learned:
  the agent stops honestly on repeated failure instead of claiming success.
- No formal verification; benchmark contamination is screened, not proven absent.

## Reproducibility
Pinned deps (`requirements.txt`), fixed tokenizer tie-breaking, seeded
generation/sampling, corpus manifest (`tokenizer/manifest.json`, corpus SHA-256)
and validation/benchmark reports saved beside the vocab. See
`docs/REPRODUCIBILITY.md`.
