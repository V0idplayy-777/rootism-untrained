# Training performance: what changed, and why nothing got worse

This document covers the throughput and memory work on the pretraining path for
both models (`training/train.py` for 20B, `training/train_3b.py` for 3B). The
rule for every change was the same: **if it can move the loss, it is not
allowed**. Everything below either removes work that produced no gradient, or
moves the same work into a faster kernel.

`tests/test_train_perf.py` is the contract. It compares each fast path against
the straightforward reference implementation it replaced, including gradients,
and the training recipes are checked to keep the learning settings intact.

---

## 1. The LM head no longer exists during a training step

The reference forward built `logits = lm_head(hidden)` — shape
`[B, S, vocab]` — and then threw all of it away except the cross-entropy.
With `vocab = 128256`, a single 20B micro-batch at 8k tokens is **2 GiB** of
logits, plus an fp32 log-softmax temporary of the same size inside
`F.cross_entropy` (another 4 GiB at the same shape). At micro-batch 2 this was
~12 GiB per call, before any activation memory.

Now (`model/fused_ops.py`):

* `chunked_cross_entropy` streams the projection `loss_chunk_size` rows at a
  time (1024 by default). Only one chunk of logits exists at once, and the
  `reduction="sum"` losses are accumulated **before** the single division by
  the token count, so the value is the exact reference batchmean rather than a
  chunk-size-weighted average.
* `sparse_cross_entropy` additionally skips rows whose label is `-100`
  (prompt tokens under `mask_prompt_loss`, padding). Those rows contributed
  nothing to the loss and nothing to the gradient before either.
* `config.return_logits` (default `True`, so nothing else changes) is set to
  `False` by both training recipes. Inference and any caller that asks for
  hidden states still gets the full projection.

Net effect on the 20B at 8k: ~6 GiB less transient VRAM per micro-batch and a
LM head pass that is O(tokens) instead of O(tokens × vocab) resident.

Be precise about which win is which: this is a **memory** win first. It also
saves time, but only in proportion to `vocab × tokens` — on the real 128k
vocabulary that projection is a large share of a step, while on a toy model with
a 4k vocabulary the extra chunk launches can even cost a few percent. The same
holds for activation checkpointing: it buys activation memory with recompute, so
`activation_checkpointing_segments: 0` is the fastest setting *when the step
already fits*. Both knobs are recipe fields, not hard-coded choices.

## 2. Attention is a fused kernel, and nothing quadratic is materialised

`GroupedQueryAttention._compute_attention` used to do
`matmul(q, k.T) / sqrt(d) → softmax(fp32) → matmul(weights, v)`, which builds a
`[B, heads, S, S]` tensor and a second fp32 copy of it for the softmax. At 48
heads and 8k that is **12 GiB** per micro-batch on the 20B.

Now it calls `F.scaled_dot_product_attention`, which dispatches to flash /
memory-efficient attention (linear memory in S) and never materialises the
score matrix. Details that make it *exact*:

* **Causal masking** uses `is_causal=True`, not an additive mask — same values,
  no `[S, S]` subtraction.
* **Padding masks** are passed as boolean "keep" masks (`True == keep`, SDPA's
  own convention) and are *composed with the causal band*, so a padding mask can
  never accidentally widen attention to future tokens.
* **GQA** keeps grouped-query head sharing fused where torch supports it
  (`enable_gqa`, torch ≥ 2.5) and otherwise expands K/V as a view rather than
  the `_repeat_kv` copy.
* The explicit softmax path still exists for the cases fused kernels cannot
  express exactly (structural/code-graph biases, and sequences longer than the
  sliding window), and the sliding-window mask is now built with two broadcasts
  and cached per shape/device instead of a per-row Python loop.
* `attention_backend: "auto"` prefers fused kernels but leaves the math kernel
  available as a fallback; `"flash"` requires them; `"math"` forces the
  reference path for debugging.

## 3. Memory-lean, kernel-count-lean primitives

* **RMSNorm** reduces in fp32 exactly as before but folds the normalisation and
  the learned scale into one expression: one full-size temporary instead of
  three, one kernel instead of four.
* **RoPE** tables are baked in the model dtype (they were recomputed as fp32 and
  cast to bf16 in every layer, every step) and applied directly in the
  `[B, 1, S, D]` broadcast layout, dropping the double transpose and two fp32
  multiplies per layer. This also fixed a real bug: `RotaryEmbedding.forward`
  read the sequence length from `shape[1]`, which is the head count once the
  tensor is `[B, heads, S, D]`.
* **SwiGLU** fuses the gate/up product into the activation with an in-place
  multiply (`silu(gate) * up` → one fewer full-size write).

## 4. Activation checkpointing that only pays for what it saves

`deepspeed.activation_checkpointing` is now a *segment* count
(`activation_checkpointing_segments`, default `2`):

| segments | behaviour |
| --- | --- |
| `0` | no recompute — keep activations, fastest step time |
| `1` | checkpoint the whole block (previous behaviour) |
| `2` | checkpoint the attention and MLP sub-blocks; the two norms stay live |

Recompute recomputes the same function from the same input, so gradients are
unchanged (the recipe trains with dropout 0, so no RNG state matters). Keeping
two cheap `[B, S, H]` norm outputs live removes two sub-layer boundaries per
block from the recompute graph.

The 20B script also stops stacking DeepSpeed's own block-level
`activation_checkpointing` on top of the model's switch — that recomputed every
block twice for no memory benefit — and turns off the `flops_profiler`, which
was running an instrumented step during training.

## 5. Data feeding is off the critical path

`dataset/prefetch.py` runs the mmap read for step *n+1* on a background thread
while step *n* is on the GPU, into pinned host memory, so
`tensor.to(device, non_blocking=True)` is a real asynchronous copy. Each batch
carries the position it was drawn from, so a checkpoint records the batch the
optimizer actually consumed — never the one the prefetcher raced ahead to.

The batch itself is now one `[B, S+1]` int32 token buffer with two shifted
views; before, the loader stacked per-index copies, sliced twice, and upcast
both halves to int64 (`dataset/processor.py:batch_tokens`).

## 6. The optimizer step is actually one optimizer step

Three separate bugs were bounding throughput in the 20B loop:

1. `initialize_deepspeed` derived the accumulation window from
   `--num_gpus` (a per-node hint) instead of the job's real world size, so on
   multi-node runs the global batch was wrong. It is now resolved from
   `WORLD_SIZE` (`resolve_accumulation`), with a fatal divisibility check.
2. `"gradient_accumulation_steps": "auto"` was passed straight to DeepSpeed,
   which is not a legal value there.
3. The loop divided every micro-loss by the accumulation window *and* called
   `model_engine.step()` every micro-batch. DeepSpeed already divides inside
   `backward()` and steps at the right boundary, so every update was
   `1/accumulation` too small and every micro-batch paid a full optimizer step.
   The loop now runs a proper accumulation window and steps once per window.

The LR schedule, warmup, clipping and global batch are unchanged — this fixes
the update to be the one the recipe describes.

## 7. ZeRO residency

Both recipes now keep ZeRO-3 state on the device
(`offload_optimizer` / `offload_param` → `{"device": "none"}`). At 3B the shard
is ~10 GiB on 8 ranks; at 20B on H100/A100-class nodes the shard fits too. The
old configuration pushed optimizer state and parameters over PCIe on every step
and then streamed parameters back for the forward pass — the single largest
throughput loss in the previous recipe.

This is the one setting that can be *wrong on your hardware*. It is per-recipe,
`configs/ds_config_zero3_offload.json` is kept as the drop-in fallback (same
maths, resident set ~2 GiB/rank for 3B), and
`scripts/plan_training_memory.py` tells you which one you need:

```bash
python scripts/plan_training_memory.py --config configs/training_config_3b.json --gpus 8
python scripts/plan_training_memory.py --config configs/training_config_3b.json \
    --gpus 8 --budget_gb 24 --apply     # writes back a fitting micro-batch
```

The planner accounts for bf16 weights, bf16 grads, the fp32 master copy, Adam
moments (fp32 or 8-bit), live + recomputed activations, the (chunked) logits
buffer, math-attention scratch when relevant, and ZeRO gather buffers, and it
never changes precision, LR, schedule, global batch or sequence length.

## 8. Opt-in: 8-bit optimizer and `torch.compile`

Both are off by default because both trade something:

* `deepspeed.zero8bit_optimizer: true` → `AdamW8bit` (same update rule, 8-bit
  moments, halves optimizer-state VRAM).
* `compile.enabled: true` → `torch.compile` on the block stack (few minutes of
  warmup on the first steps).

---

## Measuring it yourself

```bash
# same weights, same tokens: prints tok/s, peak VRAM and the loss delta per stage
python scripts/benchmark_training_step.py --preset 3b --device cuda --seq 4096
python scripts/benchmark_training_step.py --seq 512 --hidden 256 --layers 3   # CPU smoke run

# exactness contract (runs on CPU, no GPU needed)
python -m pytest tests/test_train_perf.py -q
```

Expect the loss column to be identical (`Δ 0.00e+00`) across every variant; the
`reference` row is the old behaviour, the last row is what the recipes now do.
On a 3B-class GPU run the dominant differences are the missing `[B, S, vocab]`
logits buffer (peak VRAM) and fused attention (step time) — with the segment
count as the tuning knob between memory and compute. On a CPU smoke config the
timings are not representative (no fused kernels, no bf16): use that mode to
check the loss column, and a GPU to judge speed.

## What was deliberately *not* done

* No `torch.autocast` around the step: the recipes already train in bf16, and
  autocast would change which ops run in reduced precision.
* No FP8 / quantised weights / approximating attention.
* No change to the data order, the global batch, the schedule or the
  token budget.
* No 2:4 sparsity, no vocabulary trimming, no `head_dim` changes.
* No "fused" kernels written for the sake of it: the MoE dispatch was left as
  the reference masked gather, because batching it would need a transposed copy
  of every expert weight on each step — more traffic under ZeRO, not less.
