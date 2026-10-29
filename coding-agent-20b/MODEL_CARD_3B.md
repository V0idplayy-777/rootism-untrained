# CodingAgent-3B — Model Card

## Summary
Decoder-only transformer (23 layers, hidden 3072, GQA 24Q/4KV, SwiGLU, RMSNorm,
RoPE theta=500k) with a 128,256-token byte-level BPE vocabulary, designed for
code generation and multi-step tool-using agent work. Context: 8k native,
32k via YaRN RoPE scaling (`get_3b_32k_config`), 64k YaRN ×8 as headroom
(`get_3b_64k_config`, extrapolation — see `docs/CAPABILITY_GAPS.md`).

**This is the EXACT same architecture family as CodingAgent-20B, scaled to ~3B.**
The underlying model code, tokenizer, data components and agent harness are
shared; 3B pretraining has its own budget-aware execution path.

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
The 3B recipe (`configs/training_config_3b.json`) is handled by
`training/train_3b.py` through either existing training entry point. It uses
DeepSpeed ZeRO-3, BF16, AdamW (0.9/0.95, weight decay 0.1), and cosine decay
with warmup from a peak learning rate of 3e-4. It needs CUDA GPUs, DeepSpeed,
and pre-tokenized `train_manifest.json` and `val_manifest.json` shards. For
multi-node runs, the dataset and checkpoint directory must be shared by all ranks.

The **pretraining cap is 18B input tokens** (20B's separate recipe remains at
300B). With global batch 1024 and 8192 tokens per sequence, this is 2,145 full
optimizer steps, or 17,993,564,160 input tokens. The remainder is not processed,
so the cap is never exceeded. Automatic accumulation uses the total number of
ranks: for eight GPUs and micro-batch 4, it is 32 micro-batches per optimizer
step. `--max_steps` can shorten a run but cannot raise the token cap. The 3B
path logs to the console; the production entry point's W&B monitoring is not
implemented for it.

This entry point performs **pretraining only**. The shared `configs/pipeline.json`
has an independent schedule and is not executed by it; mid-train, SFT and DPO
are separate work. The trainer consumes pre-mixed token shards and does not
apply the source upsampling factors itself. Prepare the shards with the desired
mixture before launch; model quality still needs evaluation after training.

The intended data preparation uses permissive-license code (SPDX gate in
`dataset/production_processor.py`), deduplication, and the mixture in
`dataset/mixture.py`. Apply those checks before sharding: the 3B trainer reads
flat token IDs and does not enforce licenses or use document-boundary attention
masks from `dataset/packing.py`. The configured source weights are:
`the-stack-code 0.50 · issues 0.06 · prs 0.08 · tests 0.08 · docs 0.06 ·
instruction 0.08 · security 0.03 (×3) · creative 0.03 (×2) · instruction-induction
0.03 · recovery 0.02 · project-scaffold 0.015 (×1.5) · repair-trace 0.015 (×2)`
(weights sum to 1.00; the last two are the multi-file and fail->fix->green sources).

- **Security** (`dataset/security_data.py`): curated CWE-tagged safe/vulnerable
  pairs with review/repair/implement SFT rows and DPO preference pairs.
- **Distribution shift** (`dataset/augmentation.py`): phrasing styles, ten
  cross-domain task families with acceptance criteria, spec/design/implement
  ladders, and clarification tasks for underspecified requests.

The 3B pretrainer saves sharded DeepSpeed checkpoints in `checkpoints-3b/`
(all ranks participate). Checkpoints record the global token count, data
position, batch topology and training-manifest hash; `--resume` requires those
to match. The recipe keeps the most recent five checkpoints and disables
parameter offload while retaining optimizer offload. These sharded checkpoints
are not the atomic, SHA-256-verified single-file format used by
`training/checkpoint.py`; consolidate them before inference.

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

# From the coding-agent-20b directory, pretrain with 8 BF16-capable GPUs
# (prepare the mixed, tokenized data/processed shards first).
deepspeed --num_gpus=8 training/train.py --config configs/training_config_3b.json

# Resume the same run, on the same number of ranks and training shards
deepspeed --num_gpus=8 training/train.py --config configs/training_config_3b.json --resume

# Optional one-step smoke run in a separate checkpoint directory
deepspeed --num_gpus=8 training/train_production.py \
  --config configs/training_config_3b.json --data_dir data/processed \
  --output_dir checkpoints-3b/smoke --max_steps 1
```

To serve a finished checkpoint, first consolidate its DeepSpeed shards to a
single FP32 state dict and wrap it with the 3B architecture config. This can
require substantial CPU RAM and disk space:

```bash
python - <<'PY'
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from model.config_3b import get_3b_config
from training.checkpoint import save_checkpoint_atomic

weights = get_fp32_state_dict_from_zero_checkpoint("checkpoints-3b")
save_checkpoint_atomic({"config": get_3b_config().to_dict(), "model_state_dict": weights},
                       "checkpoints-3b/export", step=0)
PY
python -m inference.server --checkpoint checkpoints-3b/export/checkpoint_00000000.pt \
  --tokenizer tokenizer --port 8000
```
