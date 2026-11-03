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
packing (`dataset/packing.py`), frozen Corpus v1 mixture in `dataset/mixture.py`
(13 slices, weights sum to 1.00; 73% real / 6% hybrid / 21% synthetic by
document share):
`the-stack-code 0.48 · tests 0.07 · github-prs 0.07 · github-issues 0.06 ·
instruction-sft 0.06 · docs-and-comments 0.05 · app-sft 0.05 ·
repair-trace-sft 0.04 · security-sft 0.03 · app-repair-sft 0.03 ·
app-preference-dpo 0.02 · recovery-sft 0.02 · project-scaffold-sft 0.02`
(upsampling: security ×3, repair-trace ×2.5, app-sft ×2.5, app-repair ×2.5).

**Model-generated text is capped at 5% of corpus tokens**
(`dataset.mixture.synthetic_token_budget`, enforced row-wise by
`dataset.augmentation.cap_rows`): the 21% synthetic document share is a
sampling mix, not a token budget. 18B tokens → 900M synthetic ceiling; the
current plan is 1.05%.

Verified slice contents — 114 CWE-tagged security pairs (28 categories, 46
CWEs, 14 languages), 75 fail→diagnose→fix traces in 7 languages, 65
pytest-verified multi-file apps (18 held out), 372-pattern catalogue, and 8
real-code induction task shapes. `python scripts/corpus_report.py --strict`
recomputes every number above and fails if any invariant regresses.

- **Security** (`dataset/security_data.py`): curated CWE-tagged safe/vulnerable
  pairs with review/repair/implement SFT rows and DPO preference pairs.
- **Distribution shift** (`dataset/augmentation.py`): phrasing styles,
  cross-domain task families with acceptance criteria, spec/design/implement
  ladders, clarification tasks for underspecified requests, and
  constraint-conflict/recovery/dialogue rows — all inside the 5% cap.

Checkpoints are atomic + SHA-256 verified (`training/checkpoint.py`); recovery
policy in `training/monitor.py`.

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
