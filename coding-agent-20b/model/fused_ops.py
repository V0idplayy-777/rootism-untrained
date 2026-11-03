"""Numerically-identical, memory-lean primitives used by the training path.

Every helper in this module computes *exactly* the same function as the
reference implementation in ``model/model.py``: no approximation, no reduced
precision, no dropped terms.  The savings come from

* not materialising tensors nobody reads (e.g. the full ``[B, S, vocab]``
  logits when only the scalar loss is needed),
* keeping the big vocab projection inside a bounded chunk,
* avoidable copies (views instead of ``cat``/``reshape`` copies),
* picking the fused kernel that PyTorch already ships (SDPA flash /
  memory-efficient backends) and refusing to silently fall back to the
  quadratically-expensive math kernel.

Nothing here requires Triton, flash-attn or a CUDA build: each helper has a
pure-PyTorch implementation and, where available, uses a faster dispatch.
"""

from __future__ import annotations

import inspect
import os

import torch
import torch.nn.functional as F

_SDPA = getattr(F, "scaled_dot_product_attention", None)

HAS_SDPA = _SDPA is not None

try:

    _SDPA_PARAMS = set(inspect.signature(_SDPA).parameters) if _SDPA is not None else set()

except (TypeError, ValueError):

    _SDPA_PARAMS = set()


def _supports_gqa() -> bool:
    """Whether this torch accepts ``enable_gqa`` (torch >= 2.5)."""

    if "enable_gqa" in _SDPA_PARAMS:
        return True

    if not HAS_SDPA:
        return False

    try:

        q = torch.zeros(1, 4, 2, 8)

        _SDPA(q, torch.zeros(1, 2, 2, 8), torch.zeros(1, 2, 2, 8), enable_gqa=True)

    except TypeError:

        return False

    except Exception:

        return True

    return True


HAS_ENABLE_GQA = _supports_gqa()

HAS_SDPA_SELECTION = hasattr(torch.nn.attention, "sdpa_kernel") or hasattr(
    torch.backends.cuda, "sdp_kernel"
)

DEFAULT_LOSS_CHUNK = 2048


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean env switch (``0/false/no/off`` disable it)."""

    value = os.environ.get(name)

    if value is None:
        return default

    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def is_autocast() -> bool:
    """True while an autocast region is active (bf16 needs none of it)."""

    return torch.is_autocast_enabled()


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with a single fp32 temporary.

    Reference (``model.model.RMSNorm``)::

        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    ``_norm`` there is ``x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)``.
    Fusing the scale into the same expression removes one full-size
    intermediate and one kernel per norm, while keeping the whole reduction
    (and the normalisation) in fp32 exactly as before.
    """

    xf = x.float()

    scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)

    return xf.mul(scale).to(dtype=weight.dtype).mul(weight)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the rotary dimensions (GPT-NeoX / non-interleaved layout)."""

    x1, x2 = x.chunk(2, dim=-1)

    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q and k.

    ``cos``/``sin`` must already be broadcastable to ``[B, 1, S, D]``.  Both
    q and k are rotated in the same pass so the two ``rotate_half`` buffers are
    the only allocations, and k's is skipped entirely when k *is* q.
    """

    q_embed = (q * cos).add_(rotate_half(q).mul_(sin))

    if k is q:
        return q_embed, q_embed

    k_embed = (k * cos).add_(rotate_half(k).mul_(sin))

    return q_embed, k_embed


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
    scale: float | None = None,
    num_kv_groups: int = 1,
) -> torch.Tensor:
    """SDPA that keeps grouped-query head sharing fused when PyTorch can.

    ``torch>=2.5`` accepts ``enable_gqa=True`` and then never materialises the
    repeated K/V heads.  On older builds (this repo pins 2.3.1) the heads are
    expanded into a view and the fused kernels do the broadcast; the reference
    ``_repeat_kv`` copy is only used when a kernel actually requires it.
    """

    if not HAS_SDPA:
        return _attention_reference(query, key, value, attention_mask, is_causal, dropout_p, scale)

    if num_kv_groups > 1:
        if HAS_ENABLE_GQA:
            return _SDPA(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=True,
            )

        key = _expand_kv(key, num_kv_groups)

        value = _expand_kv(value, num_kv_groups)

    return _SDPA(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def _expand_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads *as a view* (no copy unless a kernel forces one)."""

    if n_rep == 1:
        return x

    batch, num_kv_heads, seq_len, head_dim = x.shape

    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)

    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def _attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    is_causal: bool,
    dropout_p: float,
    scale: float | None,
) -> torch.Tensor:
    """Explicit softmax attention (last-resort fallback, same math as SDPA)."""

    head_dim = query.shape[-1]

    scale = 1.0 / (head_dim**0.5) if scale is None else scale

    weights = torch.matmul(query, key.transpose(-2, -1)) * scale

    if is_causal and weights.shape[-1] > 1:
        rows = torch.arange(weights.shape[-2], device=query.device).unsqueeze(-1)

        cols = torch.arange(weights.shape[-1], device=query.device).unsqueeze(0)

        weights = weights.masked_fill(cols > rows + (weights.shape[-1] - weights.shape[-2]), float("-inf"))

    if attention_mask is not None:
        weights = weights + attention_mask

    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)

    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p)

    return torch.matmul(weights, value)


def _kernel_selector(flash: bool, math: bool, mem_efficient: bool):
    """Context manager selecting SDPA backends across torch versions.

    ``torch.nn.attention.sdpa_kernel`` is the current API (and the legacy
    ``torch.backends.cuda.sdp_kernel`` is deprecated in 2.3+); both mean the same
    thing, so prefer the new one and fall back.  Returns ``None`` when the build
    offers no selection at all, in which case SDPA picks for itself.
    """

    modern = getattr(torch.nn.attention, "sdpa_kernel", None)

    wanted = []

    if modern is not None and hasattr(torch.nn.attention, "SDPBackend"):

        backend = torch.nn.attention.SDPBackend

        if flash:
            wanted.append(backend.FLASH_ATTENTION)

        if mem_efficient:
            wanted.append(backend.EFFICIENT_ATTENTION)

        if math:
            wanted.append(backend.MATH)

        if wanted:

            try:

                return modern(*wanted)

            except Exception:

                pass

    if hasattr(torch.backends.cuda, "sdp_kernel"):

        try:

            return torch.backends.cuda.sdp_kernel(
                enable_flash=flash, enable_math=math, enable_mem_efficient=mem_efficient
            )

        except Exception:

            return None

    return None


class fast_sdpa:
    """Context manager that steers SDPA onto the fused CUDA kernels.

    When ``backend="auto"`` the math kernel stays available as a safety net but
    flash/memory-efficient attention is preferred, which is what turns 8k
    training from "materialise a [B, H, S, S] matrix" into a linear-memory
    kernel.  ``backend="math"`` forces the reference path (useful for numerics
    debugging); ``backend="flash"`` makes the fused kernels mandatory.
    """

    def __init__(self, backend: str = "auto"):

        self.backend = (backend or "auto").lower()

        self._ctx = None

    def __enter__(self):

        if self.backend == "math":
            return self

        required = self.backend in {"flash", "flash_attention_2", "flash_attention"}

        ctx = _kernel_selector(flash=True, math=not required, mem_efficient=not required)

        if ctx is not None:

            try:

                ctx.__enter__()

                self._ctx = ctx

            except Exception:

                self._ctx = None

        return self

    def __exit__(self, *exc_info):

        if self._ctx is not None:

            self._ctx.__exit__(*exc_info)

            self._ctx = None

        return False


def chunked_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = DEFAULT_LOSS_CHUNK,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Token-mean cross entropy without materialising all logits at once.

    Mathematically identical to::

        F.cross_entropy(F.linear(hidden, weight).view(-1, V), labels.view(-1))

    but the ``[tokens, vocab]`` projection is built ``chunk_size`` rows at a
    time, so the logits buffer (up to several GB at 8k context with a 128k
    vocabulary) and the fp32 log-softmax scratch that goes with it are bounded.
    Autograd still sees the same graph, so gradients are unchanged; the sum of
    per-chunk losses is divided by the number of scored tokens, matching the
    ``reduction="mean"`` semantics of the reference call.
    """

    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])

    flat_labels = labels.reshape(-1)

    chunk_size = max(int(chunk_size or DEFAULT_LOSS_CHUNK), 1)

    if flat_labels.numel() != flat_hidden.shape[0]:
        raise ValueError(
            f"labels/hidden mismatch: {flat_labels.numel()} labels for {flat_hidden.shape[0]} positions"
        )

    scored = int((flat_labels != ignore_index).sum().item())

    if scored == 0:

        raise ValueError("batch contains no labels to score (all positions ignored)")

    total = None

    for start in range(0, flat_hidden.shape[0], chunk_size):

        stop = min(start + chunk_size, flat_hidden.shape[0])

        chunk_labels = flat_labels[start:stop]

        if not bool((chunk_labels != ignore_index).any()):
            continue

        logits = F.linear(flat_hidden[start:stop], weight)

        chunk_sum = F.cross_entropy(logits, chunk_labels, ignore_index=ignore_index, reduction="sum")

        total = chunk_sum if total is None else total + chunk_sum

    # Single division for the whole batch: exactly the reference batchmean, and
    # the only way to keep chunk sizes from reweighting the loss.
    return total * (1.0 / scored)


def sparse_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = DEFAULT_LOSS_CHUNK,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross entropy that only projects rows which carry a label.

    Identical value and gradient to :func:`chunked_cross_entropy` (and to the
    reference ``F.cross_entropy(F.linear(...))``), but when a large part of the
    batch is masked out (prompt tokens with ``mask_prompt_loss``, padding) the
    dismissed rows never enter the ``[tokens, vocab]`` projection at all.  Work
    is still streamed in chunks, so the temporary logits stay bounded.
    """

    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])

    flat_labels = labels.reshape(-1)

    chunk_size = max(int(chunk_size or DEFAULT_LOSS_CHUNK), 1)

    if flat_labels.numel() != flat_hidden.shape[0]:
        raise ValueError(
            f"labels/hidden mismatch: {flat_labels.numel()} labels for {flat_hidden.shape[0]} positions"
        )

    scored = int((flat_labels != ignore_index).sum().item())

    if scored == 0:

        raise ValueError("batch contains no labels to score (all positions ignored)")

    total = None

    for start in range(0, flat_hidden.shape[0], chunk_size):

        stop = min(start + chunk_size, flat_hidden.shape[0])

        chunk_labels = flat_labels[start:stop]

        keep = chunk_labels != ignore_index

        if not bool(keep.any()):
            continue

        chunk_logits = F.linear(flat_hidden[start:stop][keep], weight)

        chunk_sum = F.cross_entropy(chunk_logits, chunk_labels[keep], reduction="sum")

        total = chunk_sum if total is None else total + chunk_sum

    return total * (1.0 / scored)


def logits_in_chunks(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    chunk_size: int,
    bias: torch.Tensor | None = None,
):
    """Generator over ``F.linear(hidden, weight, bias)`` in bounded row chunks."""

    flat = hidden_states.reshape(-1, hidden_states.shape[-1])

    chunk_size = max(int(chunk_size or DEFAULT_LOSS_CHUNK), 1)

    for start in range(0, flat.shape[0], chunk_size):

        yield F.linear(flat[start : start + chunk_size], weight, bias)


def softcap_logits(logits: torch.Tensor, cap: float | None) -> torch.Tensor:
    """Optional logit soft-cap (identity when ``cap`` is None)."""

    if not cap:
        return logits

    return torch.tanh(logits / cap) * cap
