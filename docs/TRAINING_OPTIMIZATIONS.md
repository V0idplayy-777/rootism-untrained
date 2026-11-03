# Quality-neutral training acceleration (3B and 20B)

Goal: **blistering speed and very light VRAM, with no model quality loss.**

Everything below changes only *how* the same arithmetic is executed — kernel
selection, memory traffic, allocations, host/device synchronisation, tensor
placement and profiling overhead. Nothing changes the architecture, weights,
parameter count, loss, data order, token budget, batch composition, optimiser,
LR schedule, precision format, seed or number of effective optimizer steps.

**No quantisation, no FP8/INT8, no low-rank adapters, no pruning, no expert
dropping, no approximate attention, no reduced sequence length, no shortened
schedule.**

## What changed

| # | Change | Effect |
|---|---|---|
| 1 | **Chunked fused LM-head cross-entropy** — `scripts`-independent helper `model/fast_ops.py::fused_linear_cross_entropy`. Shifts hidden states/labels internally, computes loss in bounded row × sequence chunks under activation checkpointing, and calls the `lm_head` **module** (not a bare weight) so ZeRO-3 parameter hooks still fire. | Removes the entire `[B, T, V]` logits tensor, its contiguous shift copy, its log-softmax workspace and its gradient. This is 78.3 GiB → 219.2 MiB for the 20B recipe. |
| 2 | **fp32-exact chunk loss** — each chunk uses `log_softmax` in the model dtype, then `nll_loss(..., reduction="sum")` in fp32, summed in fp32, cast back once. | Reproduces PyTorch's CE reduction exactly; no bf16 scalar drift across steps (verified over 20 optimizer steps). |
| 3 | **Shared RoPE tables** — one `cos`/`sin` table per dtype, reused by every layer (`rope_tables` cache); the inverse frequency is kept in a private fp32 buffer so `.bfloat16()` cannot round it. | Removes `layers` redundant table builds and one long-lived `[T, head_dim]` allocation per layer; also fixes a latent bf16 RoPE rounding bug. |
| 4 | **Skip redundant `position_ids`** — training no longer materialises `[B, T]` position ids when they are implicitly `0..T-1`. | Removes an allocation and a slice per layer. Generation/KV-cache still passes explicit positions. |
| 5 | **One-GEMM fused SwiGLU down projection** — dense and routed activations are summed *before* `down_proj`, so one `[T, I]×[I, H]` GEMM instead of two. | Removes one full GEMM (forward + two backward GEMMs) per layer, per micro-batch. |
| 6 | **Sync-free MoE** — replaces the per-expert `if not token_mask.any()` Python branch (a device→host sync) with preallocated dense buffers and a `[T, k, E]` boolean mask. | Removes `4 × layers` GPU→CPU syncs per micro-batch (80 for 3B, 240 for 20B). Identical routing, identical experts, identical output. |
| 7 | **Native GQA SDPA** — uses `F.scaled_dot_product_attention(..., enable_gqa=True)` on torch ≥ 2.5, with a `repeat_kv` fallback on older torch. | Removes the materialised `[B, H, T, D]` key/value expansion. |
| 8 | **DeepSpeed config tuning** — larger all-gather buckets, `communication_data_type=fp16` removed (was downcasting bf16 comms), FLOPs profiler / CPU checkpoint profiling / excessive logging disabled, duplicate LR scheduler removed, VRAM-aware optimizer/param offload policy. | Less host overhead, no fp16 communication downcast, offload only when VRAM is actually tight. |
| 9 | **Data/training-loop hygiene** — preallocated int64 token batches, resolved `"auto"` gradient accumulation (20B was previously broken), one DeepSpeed backward/step per micro-batch, single loss scaling, on-device loss windows and one sync per log interval, on-device eval loss accumulation. | Removes allocation churn, redundant `.item()` syncs and double loss scaling; preserves the effective global batch. |

All flags are independently switchable in `configs/training_config*.json` under
`"optimizations"`. Set any of them to `false` to fall back to the original
implementation. `fused_linear_cross_entropy` is off by default and enabled
only by the two training recipes.

## Proof that quality is preserved

Run either:

```bash
python scripts/verify_no_quality_loss.py
python -m pytest -q tests/test_training_optimizations.py
```

The gate builds two models with identical weights: one executing the fast
path, one executing the reference path. It **also** loads the actual
pre-optimisation upstream `model/model.py` from a pinned git blob
(`git cat-file blob 88823f4…`) and compares against that independent baseline,
so the reference cannot be accidentally "fixed" together with the optimised
code. CI runs the gate with `fetch-depth: 0` so that blob is always available.

| Check | Result |
|---|---|
| fp32 forward loss / all 48 parameter gradients | loss Δ `4.77e-07`, grad rel L2 `2.08e-07` |
| bf16 forward loss / all gradients | loss Δ `0`, grad rel L2 `5.82e-03` (bf16 reassociation only) |
| **vs pristine upstream source**, fp32 (full fast stack + fused CE) | loss Δ `4.77e-07`, grad rel L2 `2.19e-07` |
| **vs pristine upstream source**, bf16 | loss Δ `0`, grad rel L2 `6.34e-03` |
| **strict bf16 mode** (`fuse_mlp_down_projection=false`, fused CE off) | loss Δ `0`, grad rel L2 `0`, **48/48 gradient tensors bitwise identical** |
| fused CE fp32 | loss Δ `4.77e-07`, grad rel L2 `2.17e-07` |
| fused CE bf16 | loss Δ `0`, grad rel L2 `6.12e-03` |
| greedy generation | token-for-token identical |
| 12-step AdamW trajectory, fp32 | mean loss Δ `-2.78e-07`, final weight rel L2 `2.74e-07` |
| **20-step AdamW trajectory, bf16** (the real training dtype) | mean loss Δ `+0.00469`, max `0.03125` = one bf16 ULP, final weight rel L2 `1.71e-03` |
| recipe invariants | token budgets, LR/schedule, betas, eps, weight decay, precision, seed, seq lengths, architectures unchanged |
| parameter count | `179,532` measured == analytic, no parameters added or removed |

`RESULT: PASS — no model quality loss detected`

The only numerical differences are floating-point **reassociation** (the fused
down projection sums two bf16 products before one GEMM) and one-bf16-ULP loss
rounding. There is no change to the function the network computes. If exact
bit-for-bit reproducibility is required, set
`"fuse_mlp_down_projection": false` and `"fused_linear_cross_entropy": false`
— the strict mode above proves that combination is bitwise identical while
still keeping the RoPE, position-id, MoE-sync and DeepSpeed wins.

Full `pytest -q` after the changes has the **same 12 pre-existing failures** as
the untouched baseline (`test_app_generation`, `test_fast_bpe`,
`test_process_provider`, `test_ui` — missing optional Rust/provider/UI
dependencies). No new failures.

### Boundaries of the proof

* The sandbox has **no CUDA GPU**, so parity is measured on deterministic tiny
  CPU models in both fp32 and bf16. The math is device-independent; the
  GPU-specific paths (native GQA SDPA, DeepSpeed ZeRO-3) are gated by
  capability and fall back to exactly the reference implementation.
* `sliding_window` and `code_graph_bias` are ignored by the SDPA branch in the
  **original HEAD code** as well. Both baseline and optimised models use the
  same branch, so there is no quality difference. This is a pre-existing
  modelling limitation, not part of this optimisation.
* Fused CE returns `ModelOutput.logits = None` **only** on a grad-enabled
  forward with `labels` (training *and* eval, both of which consume only
  `loss`; the 20B eval batch is also too large for full logits). Every
  `.logits` consumer in the repo (generation, `scripts/verify_model.py`,
  `training/distillation.py`) calls the model without labels or under
  `torch.no_grad()`, so it still receives full logits. There is a regression
  test for this.
* Real ZeRO-3 multi-GPU execution was not run here. `stage3_max_reuse_distance`
  is set above the `lm_head` shard size so the fused CE's repeated `lm_head`
  calls reuse the already-gathered parameter. If your DeepSpeed version behaves
  differently, set `"fused_linear_cross_entropy": false` for that run.
* The 20B recipe declares `sequence_length=32768` while the model is natively
  `max_position_embeddings=8192` with no YaRN scaling. That is a **pre-existing
  property of the HEAD recipe** and was deliberately left untouched (changing
  it would change the learning problem). A dedicated test trains a tiny model at
  3× its declared context and confirms loss, logits and all gradients are
  identical to the reference, with and without explicit `position_ids`, so the
  optimisation does not introduce any additional long-context error.

## Measured savings

`python scripts/benchmark_training.py` runs baseline and optimised in separate
processes, interleaved three times, on a scaled-down but structurally identical
model (vocab 32768, 2 layers, seq 256, batch 2, gradient checkpointing on,
128-token loss chunks). Representative results (saved in
`coding-agent-20b/evaluation/reports/training_optimization_benchmark.json`):

| Metric | Before | After | Saving (range over repeated runs) |
|---|---|---|---|
| Tensors saved for backward (= live activation memory) | 152,064,500 B (145.0 MiB) | 1,584,640 B (1.5 MiB) | **99.0 %** (stable) |
| Peak process RSS | ~766 MB (731 MiB) | ~690–715 MB (656–682 MiB) | **6–10 %** |
| Median forward+backward step time | 0.21–0.25 s | 0.16–0.18 s | **1.25–1.38× / 20–28 % faster** |
| SwiGLU MLP block fwd+bwd (isolated) | 0.023–0.028 s | 0.021–0.028 s | **1.02–1.18×** |

Timings are the median of three interleaved rounds (both the whole-step and
the isolated MLP measurements); CPU timings on tiny matrices are noisy, hence
the ranges. The saved-for-backward count is exact and deterministic. Peak RSS
only falls ~6–10 % here because PyTorch's runtime and the small weight set
dominate at this scale; the **activation** saving is 99 %, which is what scales
with `V` and `T` on the real models. Exact numbers from the most recent run are
in the JSON report.

## Projected savings on NVIDIA A100

`python scripts/benchmark_training.py --gpu all` (use `--gpu a100-40gb` or
`--gpu a100-80gb` for one). Hardware profiles are explicit and editable at the
top of the script. Both A100 variants have **identical compute** (312 TFLOP/s
dense bf16) — they differ only in HBM bandwidth and VRAM capacity
[NVIDIA A100 datasheet](https://bostondownloads.blob.core.windows.net/downloads/a/d/8/ad8d7991-57f5-4b92-8d58-66f0841e0d2a/NVIDIA-A100-80GB-Datasheet.pdf):

| Profile | HBM | bf16 | VRAM | Usable @85 % |
|---|---|---|---|---|
| A100 40GB (HBM2, SXM/PCIe) | 1.555 TB/s | 312 TFLOP/s | 40 GiB | 34.0 GiB |
| A100 80GB SXM (HBM2e) | 2.039 TB/s | 312 TFLOP/s | 80 GiB | 68.0 GiB |
| A100 80GB PCIe (HBM2e) | 1.935 TB/s | 312 TFLOP/s | 80 GiB | 68.0 GiB |

**Full-run wall-clock on ONE 8-GPU node** (100 % MFU; includes a conservative
upper-bound estimate of ZeRO-Infinity CPU-offload traffic, which the shipped
recipes hard-coded ON and our auto policy now disables where VRAM allows):

| A100 variant | 3B before → after | 20B before → after | 20B fits? |
|---|---|---|---|
| **40GB** | **53.5 h → 43.3 h** (save 10.1 h / 19 %) | **9,853.8 h → 8,591.8 h** (save 1,262 h ≈ 53 d) | **No** |
| **80GB SXM** | **53.5 h → 44.0 h** (save 9.4 h / 18 %) | **9,853.8 h → 8,604.0 h** (save 1,250 h ≈ 52 d) | Yes |
| **80GB PCIe** | 53.5 h → 43.9 h (save 9.6 h) | 9,853.8 h → 8,601.9 h (save 1,252 h) | Yes |

At a realistic **45 % MFU**: 3B ~5 d → ~4 d; 20B ~912 d → ~796 d.

Per-model detail on the 80GB SXM node:

| | 3B (`1 × 4096`) | 20B (`2 × 32768`) |
|---|---|---|
| Logits activation / GPU | 4.9 GiB → 219.2 MiB (**−95.6 %**) | 78.3 GiB → 219.2 MiB (**−99.7 %**) |
| Total activation / GPU | 5.5 GiB → 795.2 MiB (**−85.8 %**) | 126.3 GiB → 48.2 GiB (**−61.8 %**) |
| Est. VRAM / GPU after | **9.8 GiB** (fits, offload auto-disabled) | **49.0 GiB** (fits, offload kept) |
| CPU RAM / GPU for offload | 0 B | 91.5 GiB |
| Original recipe VRAM / GPU | 14.5 GiB | **229.3 GiB → OOM** |
| Est. time / micro-batch | 0.34 s → 0.29 s (15.0 %) | 61.93 s → 54.07 s (12.7 %) |

Time saved per micro-batch = removed logits HBM traffic (15–19 ms / 247–323 ms,
depending on the card), one fewer GEMM per layer (35 ms / 7.61 s), and 80/240
removed GPU→CPU syncs (1.6 ms / 4.8 ms).

**Notes specific to A100:**

* The **slower memory on the 40GB card means it benefits slightly MORE** from
  the memory-traffic optimisation: the 3B saving is 10.1 h there vs 9.4 h on
  the 80GB SXM, and the 20B saving 1,262 h vs 1,250 h.
* **20B does not fit on A100 40GB** even after all optimisations: 49.0 GiB vs
  34.0 GiB usable. Set `micro_batch_size_per_gpu: 1` with doubled gradient
  accumulation (identical effective global batch of 2048 — quality-neutral)
  → ~24.9 GiB/GPU and it fits. Total compute is unchanged, so the time
  estimates above still hold.
* The original 20B recipe would not have run at all on *any* of these cards
  (229 GiB/GPU needed, plus the `"auto"` accumulation `TypeError` and the
  double loss scaling). The fused CE plus state offload is what makes it
  possible.
* 300 B tokens on 8 GPUs is genuinely a ~2.5-year job. Scale world size
  instead — ZeRO-3 shards roughly linearly, so 64 GPUs is ~8× faster (~51 d
  ideal, ~114 d at 45 % MFU). The **percentage** saving is hardware-independent.

> The GPU time figures are **estimates** derived from explicit hardware
> constants, not profiled GPU runs (no GPU in this environment). The CPU
> timings and the 99 % activation/RSS reductions are **measured**. These are
> wall-clock hours on the whole node, not GPU-hours (multiply by 8 for
> GPU-hours). `6·N·T / FLOPs` assumes 100 % MFU, so treat them as lower bounds.

## Reproduce

```bash
cd coding-agent-20b
python scripts/verify_no_quality_loss.py          # quality gate
python -m pytest -q tests/test_training_optimizations.py
python scripts/benchmark_training.py              # measured + all A100 projections
python scripts/benchmark_training.py --gpu a100-40gb   # just the 40GB card
```

20B:

```bash
python -m training.train --model_size 20B --config configs/training_config.json
```

3B (remember the launcher's default is 20B):

```bash
python -m training.train --model_size 3B --config configs/training_config_3b.json
# or: python -m training.train_3b --config configs/training_config_3b.json
```
