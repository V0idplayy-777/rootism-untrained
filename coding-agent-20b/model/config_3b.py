"""
Configuration for the 3B-class Coding Agent Model.

EXACT same architecture family as CodingAgent-20B, scaled to the 3B class
(2.55B parameters) with a memory-lean profile for small GPUs.

Architecture decisions (mirrors model/config.py):

- 20B: 60 layers, hidden=6144, intermediate=16384, 48Q/8KV (GQA), head_dim=128
- 3B:  20 layers, hidden=3072, intermediate=7168, 24Q/4KV (GQA), head_dim=128
  (26-layer 3.08B deep variant also available as get_3b_26l_config)

Both share:

- Vocabulary size: 128256 (power-of-2 aligned, 79 frozen control tokens)
- Context length: 8192 native, 32768 via YaRN x4, 65536 via YaRN x8
- RoPE theta=500_000, sliding window 4096, RMSNorm, SwiGLU (SiLU), GQA
- Same tokenizer, data-building components and agent harness; 3B uses a
  budget-aware pretraining path

Why these numbers?
------------------
20B params (measured via CodingAgentConfig.total_parameters()) is ~24.98B with
vocab 128256 (embed*2 dominates). Halving width gives ~4x reduction per
layer (hidden*hidden), so depth must be trimmed less aggressively to land in
the 3B class.

Chosen 3B point (primary):

- hidden_size 3072 = 6144 / 2 (width halved -> attention + MLP 4x smaller).
  Width is deliberately NOT reduced further: it costs more quality per byte
  of activation memory than depth or MLP width.
- intermediate_size 7168 (SwiGLU ratio 2.33x hidden, down from 2.666x). A
  slightly thinner MLP is a standard efficiency move; it removes ~9.4M
  params per layer and most of the per-layer activation peak.
- num_attention_heads 24 = 48 / 2, num_kv_heads 4 = 8 / 2 (GQA ratio 6 preserved)
- head_dim 128 unchanged (3072/24 = 128 == 6144/48)
- num_hidden_layers 20 (down from the earlier 23) trims ~13% of compute and,
  together with the thinner MLP, lands at 2.55B params.
  A 26-layer 3.08B deep variant is available via get_3b_26l_config.

What is deliberately unchanged: vocab 128256, head_dim 128, GQA ratio 6,
RoPE theta 500k, sliding window 4096, bf16, and the 79 frozen control tokens.

Trade-offs vs 20B:

- Same head_dim, RoPE, sliding window and vocab allow a shared tokenizer and
  model code; the different layer count/width make model weights incompatible.
- YaRN 32k/64k scaling identical logic, just applied to the 3B base; pretraining
  runs at 4096 tokens per sequence, the long-context configs stay available for
  inference.
- The 3B pretrainer uses cosine decay with warmup (peak 3e-4, AdamW
  0.9/0.95, bf16 ZeRO-3 with full CPU offload), a global batch of 512 with a
  micro-batch of 1 per GPU, and a shorter pretrain target (18B vs 300B tokens)
  -- see configs/training_config_3b.json.

Parameter breakdown (vocab 128256, 20-layer primary, dense SwiGLU — MoE disabled):

- Embeddings (token + LM head counted as *2): 0.788B
- Per-layer: attn 22.02M (Q 9.44M + KV 3.15M + O 9.44M) + dense MLP 66.06M
  (gate+up+down) + norms -> ~88.1M/layer
- 20 layers -> 1.76B transformer + 0.79B embeddings = 2.550B (primary)
- 26 layers -> 2.29B transformer + 0.79B embeddings = 3.078B (deep variant)

MoE (Mixture of Experts) is OPT-IN. Setting ``use_moe=True`` with 4 experts
adds ~88.1M parameters per layer (4 extra up-projections + a router) and
would inflate the 20-layer model to ~4.31B. The primary 3B / 20B configs
ship with ``use_moe=False`` so their live PyTorch parameter counts match
the headline 2.55B / 24.98B figures.

Verification:

    from model.config_3b import get_3b_config
    assert 2.45e9 < get_3b_config().total_parameters() < 2.65e9
    get_3b_config().print_summary()  # 2.55B
    from model.config_3b import get_3b_26l_config
    assert 3.0e9 < get_3b_26l_config().total_parameters() < 3.15e9
"""

import json
from dataclasses import dataclass

from .config import CodingAgentConfig


def get_3b_config() -> CodingAgentConfig:
    """Get the primary 3B-class configuration (20 layers, ~2.55B).

    Primary model: preserves the 20B family's GQA ratio (6 = 24/4 vs 48/8),
    head_dim 128, RoPE theta 500k, sliding window 4096 and vocab 128256,
    while trimming depth (60 -> 20) and MLP width (inter/hidden 2.666 -> 2.33).
    Hidden size stays 3072: width is the most quality-dense dimension, so the
    memory savings come from depth, MLP width and the training recipe instead.
    `max_position_embeddings` remains 8192 (native) even though pretraining
    runs at 4096 tokens per sequence, so the model keeps its 8k native window.
    """

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=3072,
        intermediate_size=7168,
        num_hidden_layers=20,
        num_attention_heads=24,
        num_kv_heads=4,
        max_position_embeddings=8192,
        sliding_window=4096,
        rope_theta=500000.0,
    )


def get_3b_tight_config() -> CodingAgentConfig:
    """Alias for get_3b_config (20 layers, 2.55B) - kept for backwards compat."""

    return get_3b_config()


def get_3b_26l_config() -> CodingAgentConfig:
    """Deeper 3B variant (26 layers, ~3.08B) - same width, deeper.

    Same width and MLP size as get_3b_config but 26 layers for more reasoning
    depth at the cost of 3.08B params. Use when you can afford the extra
    compute and prefer depth over the memory-lean 20-layer primary.
    """

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=3072,
        intermediate_size=7168,
        num_hidden_layers=26,
        num_attention_heads=24,
        num_kv_heads=4,
        max_position_embeddings=8192,
        sliding_window=4096,
        rope_theta=500000.0,
    )


def get_3b_32k_config() -> CodingAgentConfig:
    """3B with YaRN-scaled RoPE for 32k agent trajectories (train 8k, extend 4x).

    Mirrors get_20b_32k_config: same base as get_3b_config (20 layers) but
    with max_position_embeddings=32768, rope_scaling_type=yarn, factor 4.0.
    The layer count and MLP width must match the trained checkpoint exactly,
    only the context window changes.
    """

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=3072,
        intermediate_size=7168,
        num_hidden_layers=20,
        num_attention_heads=24,
        num_kv_heads=4,
        max_position_embeddings=32768,
        sliding_window=4096,
        rope_theta=500000.0,
        rope_scaling_type="yarn",
        rope_scaling_factor=4.0,
        original_max_position_embeddings=8192,
    )


def get_3b_64k_config() -> CodingAgentConfig:
    """3B with YaRN x8 for 64k windows (long files, multi-file diffs).

    Mirrors get_20b_64k_config: extrapolation headroom. Same caveat as 20B
    64k -- not native long-context training, expect degradation on mid-window
    recall; retrieval + repo maps remain primary. Uses the 20-layer base.
    """

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=3072,
        intermediate_size=7168,
        num_hidden_layers=20,
        num_attention_heads=24,
        num_kv_heads=4,
        max_position_embeddings=65536,
        sliding_window=8192,
        rope_theta=500000.0,
        rope_scaling_type="yarn",
        rope_scaling_factor=8.0,
        original_max_position_embeddings=8192,
        kv_cache_max_tokens=65536,
    )


def get_3b_32k_tight_config() -> CodingAgentConfig:
    """Alias for get_3b_32k_config (20 layers, 2.55B base)."""

    return get_3b_32k_config()


def get_3b_64k_tight_config() -> CodingAgentConfig:
    """Alias for get_3b_64k_config (20 layers, 2.55B base)."""

    return get_3b_64k_config()


def get_3b_26l_32k_config() -> CodingAgentConfig:
    """Deeper 26-layer 3.08B + 32k YaRN."""

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=3072,
        intermediate_size=7168,
        num_hidden_layers=26,
        num_attention_heads=24,
        num_kv_heads=4,
        max_position_embeddings=32768,
        sliding_window=4096,
        rope_theta=500000.0,
        rope_scaling_type="yarn",
        rope_scaling_factor=4.0,
        original_max_position_embeddings=8192,
    )


def get_3b_26l_64k_config() -> CodingAgentConfig:
    """Deeper 26-layer 3.08B + 64k YaRN."""

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=3072,
        intermediate_size=7168,
        num_hidden_layers=26,
        num_attention_heads=24,
        num_kv_heads=4,
        max_position_embeddings=65536,
        sliding_window=8192,
        rope_theta=500000.0,
        rope_scaling_type="yarn",
        rope_scaling_factor=8.0,
        original_max_position_embeddings=8192,
        kv_cache_max_tokens=65536,
    )


def verify_3b_params(tight: bool = False) -> None:
    """Assert the 3B config lands in expected range (for CI/tests)."""

    cfg = get_3b_tight_config() if tight else get_3b_config()
    total = cfg.total_parameters()
    if tight:
        assert 2.45e9 < total < 2.65e9, f"tight 3B params {total/1e9:.3f}B out of range"
    else:
        assert 2.45e9 < total < 3.15e9, f"3B params {total/1e9:.3f}B out of range"
    print(f"OK: {total:,} params ({total/1e9:.2f}B)")


if __name__ == "__main__":
    print("=== 3B Primary (20 layers, ~2.55B) ===")
    get_3b_config().print_summary()
    print("\n=== 3B Deep (26 layers, ~3.08B) ===")
    get_3b_26l_config().print_summary()
    print("\n=== 3B 32k (20 layers, YaRN x4) ===")
    get_3b_32k_config().print_summary()
    print("\n=== 3B 64k (20 layers, YaRN x8) ===")
    get_3b_64k_config().print_summary()
