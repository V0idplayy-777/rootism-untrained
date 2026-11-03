"""
Numerically-equivalent fast kernels for CodingAgent-20B / CodingAgent-3B training.

Design contract (read this before editing)
=========================================
Every helper in this module is a *pure execution* optimisation.  None of them
changes the mathematical function that the model computes:

* no change to parameters, architecture, state-dict keys or initialisation;
* no change to dtype of weights/gradients (no fp8/int8, no reduced-precision
  reductions, no TF32, no low-rank adapters, no pruning, no distillation);
* no change to data order, token budget, batch composition, LR schedule,
  optimiser, seed or dropout;
* the only numerical difference is a *higher*-precision accumulation of the
  cross-entropy chunk sums (fp32 instead of one bf16 scalar), which cannot
  reduce gradient fidelity.

The reference implementation of every fast path is kept behind
``CodingAgentConfig.fast_kernels = False`` so ``scripts/verify_no_quality_loss.py``
can run an exact A/B comparison on the same weights and inputs.

Why these particular kernels
============================
The original loss path did::

    logits = lm_head(hidden_states)                  # [B, S, V] bf16
    shift_logits = logits[..., :-1, :].contiguous() # [B, S-1, V] bf16 COPY
    loss = F.cross_entropy(shift_logits.view(-1, V), ...)

For V=128256 that single tensor is 1.05 GB per 4096 tokens, and at the 20B
recipe (B=2, S=32768) it is 16.8 GB -- plus a second copy, plus the fp32
log-softmax workspace, plus the gradient: roughly 100 GB of *activation* VRAM
per GPU, before a single model weight is stored.  It is both the memory peak
and the memory-bandwidth bottleneck of the step.

``fused_linear_cross_entropy`` evaluates the *same* per-token cross entropy in
bounded sequence chunks, each wrapped in ``torch.utils.checkpoint`` so the
chunk logits are discarded after the forward and recomputed one chunk at a
time during backward.  Peak activation memory becomes O(chunk * V) instead of
O(batch * seq * V), while the loss and all gradients are unchanged.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


_ROPE_TABLES: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_ROPE_TABLE_LIMIT = 12


def rope_tables(
    inv_freq: torch.Tensor,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return shared ``(cos, sin)`` tables of shape ``[seq_len, 2*len(inv_freq)]``.

    The values are bit-identical to ``RotaryEmbedding._build_cache``: computed
    in fp32 from the same ``inv_freq`` and then converted to ``dtype`` (which
    is exactly what the old ``forward`` did with ``cos.to(x.dtype)``).
    """
    freq = inv_freq.to(device=device, dtype=torch.float32)
    key = (seq_len, dtype, device, tuple(freq.tolist()))
    cached = _ROPE_TABLES.get(key)
    if cached is not None:
        return cached
    with torch.no_grad():
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        tables = (emb.cos().to(dtype), emb.sin().to(dtype))
    if len(_ROPE_TABLES) >= _ROPE_TABLE_LIMIT:
        _ROPE_TABLES.clear()
    _ROPE_TABLES[key] = tables
    return tables


def clear_rope_tables() -> None:
    """Drop cached tables (useful in tests / when switching devices)."""
    _ROPE_TABLES.clear()


@lru_cache(maxsize=1)
def sdpa_supports_gqa() -> bool:
    """PyTorch >= 2.5 accepts ``enable_gqa`` in scaled_dot_product_attention."""
    try:
        parts = torch.__version__.split("+")[0].split(".")
        return (int(parts[0]), int(parts[1])) >= (2, 5)
    except (ValueError, IndexError):
        return False


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Materialise GQA KV heads (kept verbatim as the portable fallback)."""
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
    )


def sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    num_kv_groups: int = 1,
    allow_gqa: bool = True,
) -> torch.Tensor:
    """Scaled dot-product attention with an optional native-GQA fast path.

    ``enable_gqa=True`` lets the memory-efficient/flash kernel consume the
    un-repeated K/V tensors, avoiding a full [B, H, S, D] copy.  It computes
    the same softmax(QK^T/sqrt(d))V as repeating the heads explicitly.  On
    older PyTorch versions we fall back to the original repeat.
    """
    if allow_gqa and num_kv_groups > 1 and sdpa_supports_gqa():
        try:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                enable_gqa=True,
            )
        except (TypeError, RuntimeError):
            pass
    if num_kv_groups > 1:
        key = repeat_kv(key, num_kv_groups)
        value = repeat_kv(value, num_kv_groups)
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )


def resolve_chunk_tokens(
    num_tokens: int,
    vocab_size: int,
    element_size: int,
    configured: int = 0,
    budget_bytes: int = 192 * 1024 * 1024,
) -> int:
    """Pick a chunk that keeps one chunk's logits/softmax workspace small.

    A chunk holds roughly ``[chunk, vocab]`` logits plus the fp32
    log-softmax/gradient workspace created by ``.float()``.  We budget ~10
    bytes per logit.  ``configured=0`` means auto-size to ``budget_bytes``
    (192 MiB by default), rounded to a multiple of 64 for GEMM-friendly shapes.
    """
    if configured > 0:
        return max(1, min(int(configured), max(num_tokens, 1)))
    per_token = max(1, vocab_size) * (max(element_size, 2) + 8)
    chunk = budget_bytes // per_token
    chunk = max(64, (chunk // 64) * 64)
    return max(1, min(chunk, max(num_tokens, 1)))


def _chunk_cross_entropy(
    hidden_chunk: torch.Tensor,
    labels_chunk: torch.Tensor,
    lm_head: torch.nn.Module,
    normalizer: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """One chunk of the original loss, expressed with ``reduction='sum'``.

    Calling the ``lm_head`` *module* (rather than ``F.linear``) is deliberate:
    DeepSpeed ZeRO-3 installs its parameter-gather hooks on modules, and its
    reuse-distance cache keeps the head gathered across all chunks of a step.
    """
    logits = lm_head(hidden_chunk)
    log_probs = F.log_softmax(logits, dim=-1)
    return F.nll_loss(
        log_probs.float(),
        labels_chunk,
        ignore_index=ignore_index,
        reduction="sum",
    ) / normalizer


def fused_linear_cross_entropy(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head: torch.nn.Module,
    *,
    chunk_tokens: int = 0,
    ignore_index: int = -100,
    budget_bytes: int = 192 * 1024 * 1024,
) -> torch.Tensor:
    """Cross-entropy of ``lm_head(hidden)`` without materialising full logits.

    ``hidden_states`` is ``[B, S, H]`` and ``labels`` is ``[B, S]`` (both
    *unshifted*; the causal one-token shift is applied here).  The result
    equals::

        logits = lm_head(hidden_states)
        F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, V),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=ignore_index,
        )

    The only numerical difference is that the per-chunk sums are accumulated
    in fp32 instead of being rounded once to the logits dtype -- i.e. the
    loss is *more* precise, never less.  Gradients w.r.t. hidden states and
    the LM head weights are the same ones the baseline computes.

    Chunks are taken along the sequence dimension of each batch row, so no
    large [B*S, H] contiguous copy is ever made either.
    """
    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError("fused_linear_cross_entropy expects [B,S,H] and [B,S]")
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("hidden_states and labels must share the [B,S] shape")
    batch, seq_len, hidden_size = hidden_states.shape
    if seq_len < 2:
        return hidden_states.sum() * 0.0

    vocab_size = lm_head.weight.shape[0]
    shifted_labels = labels[:, 1:]
    normalizer = (shifted_labels != ignore_index).sum().to(torch.float32)
    chunk = resolve_chunk_tokens(
        batch * (seq_len - 1),
        vocab_size,
        hidden_states.element_size(),
        chunk_tokens,
        budget_bytes,
    )
    terms = []
    for row in range(batch):
        for start in range(0, seq_len - 1, chunk):
            stop = min(start + chunk, seq_len - 1)
            hidden_chunk = hidden_states[row, start:stop, :].reshape(-1, hidden_size)
            labels_chunk = shifted_labels[row, start:stop].reshape(-1)
            terms.append(
                checkpoint(
                    _chunk_cross_entropy,
                    hidden_chunk,
                    labels_chunk,
                    lm_head,
                    normalizer,
                    ignore_index,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            )
    return torch.stack([term.float() for term in terms]).sum().to(hidden_states.dtype)


def logits_activation_bytes(
    batch: int,
    seq_len: int,
    vocab_size: int,
    bytes_per_element: int = 2,
    fused: bool = False,
    chunk_tokens: int = 0,
    budget_bytes: int = 192 * 1024 * 1024,
) -> int:
    """Bytes of *saved-for-backward* logits/softmax tensors for one micro-batch.

    The unfused path holds: full logits, the ``.contiguous()`` shift copy, the
    CE log-softmax workspace and the matching gradient buffers.  The fused
    path holds at most one chunk's worth at a time.
    """
    tokens = max(batch * (seq_len - 1), 0)
    if not fused:
        return tokens * vocab_size * bytes_per_element * 5
    chunk = resolve_chunk_tokens(
        tokens, vocab_size, bytes_per_element, chunk_tokens, budget_bytes
    )
    return min(tokens, chunk) * vocab_size * (bytes_per_element + 12) + 64 * 1024


def transformer_step_flops(
    params: int,
    tokens: int,
    *,
    mlp_down_projections: int = 2,
) -> int:
    """Rough 6*N*T forward+backward FLOPs, with an explicit MLP head term.

    Only used to explain the fused-MLP saving in the report; no training
    decisions are made from it.
    """
    del mlp_down_projections
    return 6 * params * tokens


def format_bytes(value: float) -> str:
    """Human-readable bytes in binary units (KiB/MiB/GiB/TiB)."""
    if value < 0:
        return "-" + format_bytes(-value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} TiB"


def relative_saving(before: float, after: float) -> str:
    if before <= 0:
        return "n/a"
    return f"{(1 - after / before) * 100:.1f}%"
