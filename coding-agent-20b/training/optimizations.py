"""
Training-side wiring for the quality-neutral fast kernels in ``model.fast_ops``.

This module deliberately contains NO changes to the learning problem.  It only:

* copies execution flags onto ``CodingAgentConfig``;
* removes DeepSpeed settings that cost time or *reduce* gradient precision;
* chooses whether CPU offload is actually necessary (offload is the single
  biggest speed killer in ZeRO-Infinity, and is often unnecessary once the
  logits activation peak is gone);
* provides the arithmetic used by ``scripts/benchmark_training.py`` to report
  measured/estimated VRAM and time savings.

Nothing here changes: token budgets, learning rate or schedule, betas, eps,
weight decay, gradient clipping, batch sizes, sequence lengths, seed, data
order, mixture, precision of weights/gradients, model architecture, parameter
count, or the number of optimiser steps.
"""

from __future__ import annotations

import copy
from typing import Any

from model.fast_ops import (
    format_bytes,
    logits_activation_bytes,
    relative_saving,
    resolve_chunk_tokens,
)

# Defaults are applied to ANY recipe, so both the 3B and 20B training entry
# points get the fast path even if a user supplies a custom config.
DEFAULT_OPTIMIZATIONS: dict[str, Any] = {
    "fast_kernels": True,
    "share_rope_tables": True,
    "skip_redundant_position_ids": True,
    "fuse_mlp_down_projection": True,
    "sync_free_moe": True,
    "native_gqa_sdpa": True,
    "fused_linear_cross_entropy": True,
    "loss_chunk_tokens": 0,
    "loss_chunk_budget_bytes": 192 * 1024 * 1024,
    # DeepSpeed hygiene (no effect on the maths):
    "disable_flops_profiler": True,
    "disable_cpu_activation_checkpointing": True,
    "preserve_bf16_communication": True,
    "auto_offload": True,
}

_BOOL_FLAGS = (
    "fast_kernels",
    "share_rope_tables",
    "skip_redundant_position_ids",
    "fuse_mlp_down_projection",
    "sync_free_moe",
    "native_gqa_sdpa",
    "fused_linear_cross_entropy",
)


def optimization_settings(recipe: dict) -> dict[str, Any]:
    """Merge a recipe's ``optimizations`` block with the defaults."""
    settings = dict(DEFAULT_OPTIMIZATIONS)
    supplied = recipe.get("optimizations") or {}
    unknown = set(supplied) - set(settings)
    if unknown:
        raise ValueError(f"Unknown optimization keys: {sorted(unknown)}")
    settings.update(supplied)
    return settings


def apply_model_optimizations(model_config, recipe: dict) -> dict[str, Any]:
    """Copy execution-only flags onto a ``CodingAgentConfig``.

    The model's *architecture* fields are untouched, so ``state_dict`` keys,
    parameter count and initialisation are identical to the reference model.
    """
    settings = optimization_settings(recipe)
    for key in _BOOL_FLAGS:
        setattr(model_config, key, bool(settings[key]))
    model_config.loss_chunk_tokens = int(settings["loss_chunk_tokens"])
    model_config.loss_chunk_budget_bytes = int(settings["loss_chunk_budget_bytes"])
    return model_config.optimization_summary()


def tune_deepspeed_config(ds_config: dict, recipe: dict) -> dict:
    """Return a DeepSpeed config with time-wasting / precision-harming flags off.

    * ``flops_profiler`` only measures FLOPs, but it adds host overhead every
      step; disabling it cannot change a single gradient.
    * ``communication_data_type: fp16`` downcasts bf16 gradients for the
      reduce-scatter.  Removing it makes the communication use the model's
      dtype (bf16), which is *more* faithful, not less.
    * ``activation_checkpointing.cpu_checkpointing`` moves checkpoints to host
      RAM over PCIe.  Our model uses ``torch.utils.checkpoint`` directly, so
      this setting is inert for us; turning it off removes the footgun.
    """
    result = copy.deepcopy(ds_config)
    settings = optimization_settings(recipe)
    if settings["disable_flops_profiler"] and isinstance(result.get("flops_profiler"), dict):
        result["flops_profiler"]["enabled"] = False
    if settings["preserve_bf16_communication"]:
        result.pop("communication_data_type", None)
    if settings["disable_cpu_activation_checkpointing"]:
        ac = result.get("activation_checkpointing")
        if isinstance(ac, dict) and "cpu_checkpointing" in ac:
            ac["cpu_checkpointing"] = False
    return result


# ---------------------------------------------------------------------------
# VRAM-aware offload choice
# ---------------------------------------------------------------------------
def model_parameter_count(model_config) -> int:
    """Exact parameter count of ``CodingAgentModel(model_config)``.

    ``CodingAgentConfig.total_parameters()`` omits the 4 MoE experts and the
    router in every block, so the live model is larger than the headline
    "3B"/"20B" figure.  Use this for honest VRAM accounting.  It is verified
    against an actually-instantiated tiny model in the test-suite.
    """
    hidden = model_config.hidden_size
    intermediate = model_config.intermediate_size
    heads = model_config.num_attention_heads
    kv_heads = model_config.num_kv_heads
    head_dim = model_config.head_dim
    layers = model_config.num_hidden_layers
    vocab = model_config.vocab_size
    attention = hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim + heads * head_dim * hidden
    mlp = 3 * hidden * intermediate
    moe = 4 * hidden * intermediate + hidden * 4 + 4  # experts + router (bias)
    per_layer = attention + mlp + moe + 2 * hidden
    return 2 * vocab * hidden + layers * per_layer + hidden


def zero3_state_bytes_per_gpu(params: int, world_size: int) -> int:
    """ZeRO-3 per-rank resident training state, in bytes.

    bf16 params (2) + bf16 grads (2) + fp32 master weights (4) +
    fp32 AdamW exp_avg/exp_avg_sq (8) = 16 bytes/param, sharded over ranks.
    """
    return int(params * 16 / max(int(world_size), 1))


def activation_bytes_per_gpu(
    params: int,
    micro_batch: int,
    seq_len: int,
    vocab_size: int,
    layers: int,
    hidden_size: int,
    *,
    fused: bool,
    loss_chunk_tokens: int = 0,
    budget_bytes: int = 192 * 1024 * 1024,
) -> int:
    """Conservative activation estimate with full (per-layer) checkpointing.

    With gradient checkpointing, live activations are one layer's boundary
    tensors plus a bounded recompute window; the logits/CE tensors dominate.
    """
    boundary = micro_batch * seq_len * hidden_size * 2 * layers  # saved inputs
    recompute = micro_batch * seq_len * hidden_size * 2 * 4  # one layer window
    logits = logits_activation_bytes(
        micro_batch,
        seq_len,
        vocab_size,
        fused=fused,
        chunk_tokens=loss_chunk_tokens,
        budget_bytes=budget_bytes,
    )
    return int(boundary + recompute + logits)


def resolve_offload(
    recipe: dict,
    world_size: int,
    params: int,
    micro_batch: int,
    seq_len: int,
    vocab_size: int,
    layers: int,
    hidden_size: int,
    *,
    free_vram_bytes: int | None = None,
    safety_factor: float = 0.85,
) -> tuple[bool, bool, str]:
    """Choose ZeRO-3 CPU offload: off when states+activations comfortably fit.

    Returns ``(offload_optimizer, offload_param, reason)``.  If VRAM cannot be
    queried (CPU, tests, unusual drivers) the recipe's explicit booleans are
    preserved -- a safe, quality-neutral fallback.
    """
    ds = recipe["deepspeed"]
    recipe_optimizer = bool(ds.get("offload_optimizer", True))
    recipe_param = bool(ds.get("offload_param", True))
    policy = ds.get("offload_policy", "auto")
    settings = optimization_settings(recipe)
    if policy == "on" or not settings["auto_offload"]:
        return recipe_optimizer, recipe_param, "explicit/on policy (recipe values kept)"
    if policy == "off":
        return False, False, "explicit/off policy"
    if policy != "auto":
        raise ValueError(f"Unknown deepspeed.offload_policy: {policy!r}")

    if free_vram_bytes is None:
        try:
            import torch

            if not torch.cuda.is_available():
                return recipe_optimizer, recipe_param, "no CUDA device; recipe values kept"
            free_vram_bytes = int(torch.cuda.mem_get_info()[0])
        except Exception:  # pragma: no cover - defensive on unusual drivers
            return recipe_optimizer, recipe_param, "VRAM query unavailable; recipe values kept"

    states = zero3_state_bytes_per_gpu(params, world_size)
    activations = activation_bytes_per_gpu(
        params,
        micro_batch,
        seq_len,
        vocab_size,
        layers,
        hidden_size,
        fused=settings["fused_linear_cross_entropy"],
        loss_chunk_tokens=settings["loss_chunk_tokens"],
        budget_bytes=settings["loss_chunk_budget_bytes"],
    )
    # Working set: gathered ZeRO-3 params for one layer + comm buckets.
    gathered = int(2 * params / max(world_size, 1))
    need = states + activations + gathered
    budget = int(free_vram_bytes * safety_factor)
    if need <= budget:
        return (
            False,
            False,
            f"auto: {format_bytes(need)} fits in {format_bytes(budget)} "
            f"usable VRAM -> CPU offload disabled (faster)",
        )
    return (
        recipe_optimizer,
        recipe_param,
        f"auto: {format_bytes(need)} exceeds {format_bytes(budget)} "
        f"usable VRAM -> CPU offload kept (lean)",
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def savings_report(
    *,
    name: str,
    params: int,
    world_size: int,
    micro_batch: int,
    seq_len: int,
    vocab_size: int,
    layers: int,
    hidden_size: int,
    loss_chunk_tokens: int = 0,
    budget_bytes: int = 192 * 1024 * 1024,
) -> dict[str, Any]:
    """Deterministic activation/state accounting for one micro-batch."""
    before = activation_bytes_per_gpu(
        params, micro_batch, seq_len, vocab_size, layers, hidden_size,
        fused=False, loss_chunk_tokens=loss_chunk_tokens, budget_bytes=budget_bytes,
    )
    after = activation_bytes_per_gpu(
        params, micro_batch, seq_len, vocab_size, layers, hidden_size,
        fused=True, loss_chunk_tokens=loss_chunk_tokens, budget_bytes=budget_bytes,
    )
    logits_before = logits_activation_bytes(micro_batch, seq_len, vocab_size, fused=False)
    logits_after = logits_activation_bytes(
        micro_batch, seq_len, vocab_size, fused=True,
        chunk_tokens=loss_chunk_tokens, budget_bytes=budget_bytes,
    )
    chunk = resolve_chunk_tokens(
        micro_batch * (seq_len - 1), vocab_size, 2, loss_chunk_tokens, budget_bytes
    )
    return {
        "name": name,
        "params": params,
        "world_size": world_size,
        "micro_batch": micro_batch,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "chunk_tokens": chunk,
        "logits_before_bytes": logits_before,
        "logits_after_bytes": logits_after,
        "logits_before": format_bytes(logits_before),
        "logits_after": format_bytes(logits_after),
        "logits_saving": relative_saving(logits_before, logits_after),
        "activations_before_bytes": before,
        "activations_after_bytes": after,
        "activations_before": format_bytes(before),
        "activations_after": format_bytes(after),
        "activations_saving": relative_saving(before, after),
        "state_bytes_per_gpu": zero3_state_bytes_per_gpu(params, world_size),
    }
