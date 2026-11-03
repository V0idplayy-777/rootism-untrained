# CodingAgent-3B — Model Card

## Summary
Decoder-only transformer (20 layers, hidden 3072, GQA 24Q/4KV, SwiGLU 7168,
RMSNorm, RoPE theta=500k) with a 128,256-token byte-level BPE vocabulary,
designed for code generation and multi-step tool-using agent work. Context: 8k
native, 32k via YaRN RoPE scaling (`get_3b_32k_config`), 64k YaRN ×8 as headroom
(`get_3b_64k_config`, extrapolation — see `docs/CAPABILITY_GAPS.md`).
Pretraining runs at 4096 tokens per sequence; the 8k native window and the YaRN
configs are unchanged and remain the inference targets.

**This is the EXACT same architecture family as CodingAgent-20B, scaled to ~3B.**
The underlying model code, tokenizer, data components and agent harness are
shared; 3B pretraining has its own budget-aware execution path.

| Dimension | 20B | 3B (primary) | Scaling | 3B deep variant |
| --- | --- | --- | --- | --- |
| hidden_size | 6144 | 3072 | 0.5× | 3072 |
| intermediate_size | 16384 | 7168 | 0.44× | 7168 |
| num_hidden_layers | 60 | 20 | 0.33× | 26 (~3.08B) |
| num_attention_heads | 48 | 24 | 0.5× | 24 |
| num_kv_heads (GQA) | 8 | 4 | 0.5× | 4 |
| head_dim | 128 | 128 | 1× | 128 |
| GQA ratio (Q/KV) | 6 | 6 | 1× | 6 |
| inter/hidden | 2.666 | 2.333 | 0.88× | 2.333 |
| vocab_size | 128256 | 128256 | 1× | 128256 |
| sliding_window | 4096 | 4096 | 1× | 4096 |
| rope_theta | 500k | 500k | 1× | 500k |
| max_position | 8192 | 8192 | 1× | 8192 |
| YaRN 32k | yarn×4 | yarn×4 | same | yarn×4 |
| YaRN 64k | yarn×8 | yarn×8 | same | yarn×8 |

Parameter count (via `CodingAgentConfig.total_parameters()`):
- 3B primary (20L): **2,549,738,496 (2.55B)** -- embeddings 0.79B + 20×88.1M
- 3B deep (26L): 3,078,257,664 (3.08B)
- 20B: 24,980,969,472 (24.98B)

The 3B class point trades depth (60 → 20) and MLP width (2.666× → 2.333× hidden)
for memory and compute, while hidden size (3072), head_dim (128), the GQA ratio
(6), RoPE theta, sliding window, vocab and the 79 control tokens are unchanged.

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

The recipe is tuned for small GPUs (a single or dual T4): **4096 tokens per
sequence**, `micro_batch_size_per_gpu: 1`, `global_batch_size: 512`, with
gradient accumulation resolved from the total number of ranks. Activation memory
scales with sequence length and micro-batch, so this is the biggest single
memory win; the effective batch is unchanged in kind, only its composition.
Prepare 3B shards with `--seq_length 4096` when the shard generator packs
documents (`scripts/generate_dataset.py`); the trainer re-windows the flat token
stream using its own `sequence_length`, so existing 8192-packed shards still work.

DeepSpeed stays on ZeRO-3 with **both** parameter and optimizer offload to CPU
(`pin_memory: true`, unchanged in `configs/ds_config_zero3.json`). The 3B path
activates activation checkpointing through the model's own
`torch.utils.checkpoint` switch (`deepspeed.activation_checkpointing` in the
recipe), which is why `training/train_3b.py` drops the shared template's
`activation_checkpointing` block instead of double-checkpointing; the template
keeps `partition_activations` and `cpu_checkpointing` for the 20B path. The 3B
recipe additionally lowers `stage3_max_live_parameters` /
`stage3_max_reuse_distance` to 5e8 (from the shared template's 1e9) to cap the
16-bit parameter working set held on the GPU. Those two knobs are read from the
3B recipe only, so the shared 20B template is unchanged.

The **pretraining cap is 18B input tokens** (20B's separate recipe remains at
300B). With global batch 512 and 4096 tokens per sequence, this is 2,097,152
input tokens per optimizer step and 8,583 full optimizer steps, or
17,999,855,616 input tokens. The remainder is not processed, so the cap is never
exceeded. Automatic accumulation uses the total number of ranks: for eight GPUs
and micro-batch 1, it is 64 micro-batches per optimizer step. `--max_steps` can
shorten a run but cannot raise the token cap. The 3B path logs to the console;
the production entry point's W&B monitoring is not implemented for it.

This entry point performs **pretraining only**. The shared `configs/pipeline.json`
has an independent schedule and is not executed by it; mid-train, SFT and DPO
are separate work. The trainer consumes pre-mixed token shards and does not
apply the source upsampling factors itself. Prepare the shards with the desired
mixture before launch; model quality still needs evaluation after training.

The intended data preparation uses permissive-license code (SPDX gate in
`dataset/production_processor.py`), deduplication, and the frozen Corpus v1
mixture in `dataset/mixture.py`. Apply those checks before sharding: the 3B
trainer reads flat token IDs and does not enforce licenses or use
document-boundary attention masks from `dataset/packing.py`.

**Corpus v1 mixture** (13 frozen slices, weights sum to 1.00; 73% real / 6%
hybrid / 21% synthetic by document share):

`the-stack-code 0.48 · tests 0.07 · github-prs 0.07 · github-issues 0.06 ·
instruction-sft 0.06 · docs-and-comments 0.05 · app-sft 0.05 ·
repair-trace-sft 0.04 · security-sft 0.03 · app-repair-sft 0.03 ·
app-preference-dpo 0.02 · recovery-sft 0.02 · project-scaffold-sft 0.02`
(upsampling: security ×3, repair-trace ×2.5, app-sft ×2.5, app-repair ×2.5).

**Synthetic token cap.** The 21% synthetic *document* share is a sampling mix,
not a token budget: model-generated text is capped at **5% of the corpus**.
`dataset.mixture.synthetic_token_budget()` converts the cap into tokens
(18B → 900M ceiling; the current plan is 1.05%), `dataset.augmentation.cap_rows()`
trims augmented rows deterministically (seed 42) to stay under it, and both
`configs/training_config*.json` data blocks record the cap. Report the numbers
with `python scripts/corpus_report.py --strict`.

What each verified slice contains (all reproducible via `scripts/corpus_report.py`):

| Slice | Content | Grounding |
| --- | --- | --- |
| `security-sft` | 114 CWE-tagged safe/vulnerable pairs, 28 categories, 46 CWEs, 14 languages → review/repair/implement SFT + DPO | every vulnerable sample must trip `evaluation/security_eval.py`; safe samples must stay clean |
| `repair-trace-sft` | 75 fail→diagnose→minimal-diff→green traces in 7 languages (python/js/ts/go/rust/java/sql) | python fixes are `ast.parse`-checked; rejected patches always differ from both buggy and fixed |
| `app-sft`, `app-repair-sft`, `app-preference-dpo` | 65 multi-file product blueprints (25–28 files each) → SFT/preference/repair/taste rows | all 65 render a project whose generated pytest suite passes (`--verify`, 65/65); 18 blueprints held out |
| `project-scaffold-sft` | generic starters (fastapi-react, nextjs-fullstack, flask-htmx, cli-tool, discord-bot, static-site) | rendered tree + run command + tests |
| `instruction-sft` | real-corpus induction in **8** task shapes (docstring→code, signature→body, code→explanation, test→contract, example→implementation, imports→module, error-branches→tests, docstring→contract tests) + phrasing/cross-domain augmentation | every row carries `source_path` + SPDX license; rejected licenses never appear |
| `recovery-sft` | tool-failure classification and changed-approach recovery | failure class + anti-pattern recorded per row |

- **Security** (`dataset/security_data.py`): curated CWE-tagged safe/vulnerable
  pairs with review/repair/implement SFT rows and DPO preference pairs.
- **Distribution shift** (`dataset/augmentation.py`): phrasing styles,
  cross-domain task families with acceptance criteria, spec/design/implement
  ladders, clarification tasks for underspecified requests, and
  constraint-conflict / recovery / dialogue rows — all inside the 5% cap.

The 3B pretrainer saves sharded DeepSpeed checkpoints in `checkpoints-3b/`
(all ranks participate). Checkpoints record the global token count, data
position, batch topology and training-manifest hash; `--resume` requires those
to match. The recipe keeps the most recent five checkpoints with both parameter
and optimizer state offloaded to CPU memory. These sharded checkpoints
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
  20B for best quality. The memory-lean 3B point (20 layers, MLP 7168) gives up
  ~16% of parameters versus the earlier 23-layer 8192-MLP variant; the 26-layer
  `get_3b_26l_config` (3.08B) is the option if quality matters more than fit.

## Reproducibility
Same as 20B: pinned deps (`requirements.txt`), fixed tokenizer tie-breaking,
seeded generation/sampling, corpus manifest (`tokenizer/manifest.json`,
corpus SHA-256) and validation/benchmark reports saved beside the vocab. See
`docs/REPRODUCIBILITY.md`.

3B-specific reproducibility:

```bash
# verify 3B param count (no GPU/torch needed)
python -c "from model.config_3b import get_3b_config; get_3b_config().print_summary()"
# 2.55B

# deep variant
python -c "from model.config_3b import get_3b_26l_config; get_3b_26l_config().print_summary()"
# 3.08B

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
