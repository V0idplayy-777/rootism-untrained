# CodingAgent-3B — Model Card

## Summary
Decoder-only transformer (23 layers, hidden 3072, GQA 24Q/4KV, SwiGLU, RMSNorm,
RoPE theta=500k) with a 128,256-token byte-level BPE vocabulary, trained for
code generation and multi-step tool-using agent work. Context: 8k native,
32k via YaRN RoPE scaling (`get_3b_32k_config`), 64k YaRN ×8 as headroom
(`get_3b_64k_config`, extrapolation — see `docs/CAPABILITY_GAPS.md`).

**This is the EXACT same architecture family as CodingAgent-20B, scaled to ~3B.**
Only width/depth change; all code, tokenizer, data pipeline, agent harness,
and training pipeline are shared.

| Dimension | 20B | 3B (primary) | Scaling | 3B deep variant |
| --- | --- | --- | --- | --- |
| hidden_size | 6144 | 3072 | 0.5× | 3072 |
| intermediate_size | 16384 | 8192 | 0.5× | 8192 |
| num_hidden_layers | 60 | 23 | 0.38× | 26 (~3.33B) |
| num_attention_heads | 48 | 24 | 0.5× | 24 |
| num_kv_heads (GQA) | 8 | 4 | 0.5× | 4 |
| head_dim | 128 | 128 | 1× | 128 |
| GQA ratio (Q/KV) | 6 | 6 | 1× | 6 |
| inter/hidden | 2.666 | 2.666 | 1× | 2.666 |
| vocab_size | 128256 | 128256 | 1× | 128256 |
| sliding_window | 4096 | 4096 | 1× | 4096 |
| rope_theta | 500k | 500k | 1× | 500k |
| max_position | 8192 | 8192 | 1× | 8192 |
| YaRN 32k | yarn×4 | yarn×4 | same | yarn×4 |
| YaRN 64k | yarn×8 | yarn×8 | same | yarn×8 |

Parameter count (via `CodingAgentConfig.total_parameters()`):
- 3B primary (23L): **3,031,053,312 (3.03B)** -- embeddings 0.79B + 23×97.5M
- 3B deep (26L): 3,323,624,448 (3.32B)
- 20B: 24,980,969,472 (24.98B)

Same 79 hand-written control tokens (ids frozen; `|tool_call|`=48,
`|tool_result|`=49) plus task tokens for scratchpad/plan/repo-map/retrieval/
diff/review/clarify and tool errors. Vocab padded to exactly `vocab_size`.

## Training
Stages (see `configs/pipeline.json` + `configs/training_config_3b.json`,
`training/schedules.py`): continuous pretrain (WSD LR 3e-4, AdamW 0.9/0.95,
wd 0.1, bf16, ZeRO-3) → mid-train anneal on tests/issues/PRs/docs (plus
security and distribution-shift sources) → SFT (prompt loss masked) → DPO
(beta 0.1).

For 3B, pretrain tokens 100B vs 300B for 20B, global batch 1024 vs 2048,
`checkpoints-3b/` and `logs/tensorboard-3b`, `wandb_project: coding-agent-3b`.
All other data/mixture settings identical.

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
policy in `training/monitor.py`. 3B uses `ds_config_zero3.json` with
`offload_param: false` (fits on fewer GPUs).

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

3B vs 20B eval: same harness (`evaluation/evaluate.py`,
`evaluation/production_evaluate.py`, `inference/server.py` all accept any
`CodingAgentConfig`). Load 3B via:

```python
from model.config_3b import get_3b_config
from model.model import CodingAgentModel
model = CodingAgentModel(get_3b_config())
# or: from model.model_3b import get_default_3b_model; model = get_default_3b_model()
```

## Limitations & safety
Same as 20B (see MODEL_CARD.md). Highlights:

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
- 3B-specific: expect lower pass@1 than 20B on LiveCodeBench/SWE-bench
  (knowledge capacity scales with params), but identical tool-use, retrieval,
  and session-state machinery. Use 3B for local dev, CI, and fast iteration;
  20B for best quality.

## Reproducibility
Same as 20B: pinned deps (`requirements.txt`), fixed tokenizer tie-breaking,
seeded generation/sampling, corpus manifest (`tokenizer/manifest.json`,
corpus SHA-256) and validation/benchmark reports saved beside the vocab. See
`docs/REPRODUCIBILITY.md`.

3B-specific reproducibility:

```bash
# verify 3B param count (no GPU/torch needed)
python -c "from model.config_3b import get_3b_config; get_3b_config().print_summary()"
# 3.03B

# deep variant
python -c "from model.config_3b import get_3b_26l_config; get_3b_26l_config().print_summary()"
# 3.33B

# train (same pipeline, smaller batch/tokens)
python training/train_production.py --config configs/training_config_3b.json
# serve (OpenAI-compatible, same server handles both)
python -m inference.server --checkpoint checkpoints-3b/latest --port 8000
# model/model_3b.py is a drop-in alias: from model.model_3b import CodingAgentModel
```
