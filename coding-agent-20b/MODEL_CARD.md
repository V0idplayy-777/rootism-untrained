# CodingAgent-20B — Model Card

## Summary
Decoder-only transformer (60 layers, hidden 6144, GQA 48Q/8KV, SwiGLU, RMSNorm,
RoPE theta=500k) with a 128,256-token byte-level BPE vocabulary, trained for
code generation and multi-step tool-using agent work. Context: 8k native,
32k via YaRN RoPE scaling (`configs`: `get_20b_32k_config`).

## Training
Stages (see `configs/pipeline.json`, `training/schedules.py`): continuous
pretrain (WSD LR 3e-4, AdamW 0.9/0.95, wd 0.1, bf16, ZeRO-3) → mid-train
anneal on tests/issues/PRs/docs → SFT (prompt loss masked) → DPO (beta 0.1).
Data: permissive-license code (SPDX gate in `dataset/production_processor.py`),
MinHash-LSH dedup, attention-aware packing (`dataset/packing.py`), mixture in
`dataset/mixture.py`. Checkpoints are atomic + SHA-256 verified
(`training/checkpoint.py`); recovery policy in `training/monitor.py`.

## Evaluation
Protocol: `docs/EVAL_PROTOCOL.md`. Benchmarks: HumanEval, MBPP, LiveCodeBench,
SWE-bench Lite (repo-level), plus agent-trajectory metrics and 13-gram
contamination screening (`evaluation/production_evaluate.py`). Report pass@k
with n, temperature and cache hash — numbers without them are not comparable.

## Limitations & safety
- Agent code execution is fail-closed without Docker; the sandbox is the
  security boundary (read-only workspace, seccomp, cap-drop, quotas).
  `agent/safety.py` redaction/blocking is defence-in-depth, not a guarantee.
- 8k native context still truncates very large repos; use 32k config + repo
  retrieval. Generated code must be reviewed and tested before use.
- No formal verification; benchmark contamination is screened, not proven absent.

## Reproducibility
Pinned deps (`requirements.txt`), fixed tokenizer tie-breaking, seeded
generation/sampling, data-version + validation report saved with each
tokenizer (`validation.json`). See `docs/REPRODUCIBILITY.md`.
