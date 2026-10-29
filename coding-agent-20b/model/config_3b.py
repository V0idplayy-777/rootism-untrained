"""
Configuration for the 3B parameter Coding Agent Model.

EXACT same architecture family as CodingAgent-20B, scaled to ~3B parameters.

Architecture decisions (mirrors model/config.py):

- 20B: 60 layers, hidden=6144, intermediate=16384, 48Q/8KV (GQA), head_dim=128
- 3B:  23 layers, hidden=3072, intermediate=8192, 24Q/4KV (GQA), head_dim=128
  (26-layer 3.33B deep variant also available as get_3b_26l_config)

Both share:

- Vocabulary size: 128256 (power-of-2 aligned, 79 frozen control tokens)
- Context length: 8192 native, 32768 via YaRN x4, 65536 via YaRN x8
- RoPE theta=500_000, sliding window 4096, RMSNorm, SwiGLU (SiLU), GQA
- Same tokenizer, data pipeline, training stages, agent harness

Why these numbers?
------------------
20B params (measured via CodingAgentConfig.total_parameters()) is ~24.98B with
vocab 128256 (embed*2 dominates). Halving width gives ~4x reduction per
layer (hidden*hidden), so depth must be trimmed less aggressively to land near
3B.

Chosen 3B point (primary):

- hidden_size 3072 = 6144 / 2 (width halved -> attention + MLP 4x smaller)
- intermediate_size 8192 = 16384 / 2 (keeps SwiGLU ratio 2.666x hidden)
- num_attention_heads 24 = 48 / 2, num_kv_heads 4 = 8 / 2 (GQA ratio 6 preserved)
- head_dim 128 unchanged (3072/24 = 128 == 6144/48)
- num_hidden_layers 23 tuned to hit 3.03B exactly (26-layer 3.33B deep variant
  also available via get_3b_26l_config for depth lovers).

Trade-offs vs 20B:

- Same head_dim, same RoPE, same sliding window, same vocab -> checkpoints
  and tokenizer are drop-in compatible (only layer count/width differ).
- YaRN 32k/64k scaling identical logic, just applied to 3B base.
- Training recipe identical (WSD 3e-4, AdamW 0.9/0.95, bf16 ZeRO-3) with
  smaller global batch (1024 vs 2048) and shorter pretrain (100B vs 300B
  tokens) -- see configs/training_config_3b.json.

Parameter breakdown (vocab 128256, tight 23-layer):

- Embeddings (token + LM head counted as *2): 0.79B
- Per-layer: attn 22.0M (Q 9.44M + KV 3.15M + O 9.44M) + MLP 75.50M + norms
  -> ~97.5M/layer
- 23 layers -> 2.24B transformer + 0.79B embeddings = 3.031B (exact 3B)
- 26 layers -> 2.54B transformer + 0.79B embeddings = 3.33B (deep variant)

Verification:

    from model.config_3b import get_3b_config
    assert 2.9e9 < get_3b_config().total_parameters() < 3.2e9
    get_3b_config().print_summary()  # 3.03B
    from model.config_3b import get_3b_26l_config
    assert 3.2e9 < get_3b_26l_config().total_parameters() < 3.5e9
"""

import json
from dataclasses import dataclass

from .config import CodingAgentConfig


def get_3b_config() -> CodingAgentConfig:
    """Get the 3B parameter configuration (23 layers, ~3.03B).

    Primary 3B model: preserves the 20B family's width ratio (inter/hidden
    = 2.666 = 8192/3072 vs 16384/6144), GQA ratio (6 = 24/4 vs 48/8),
    head_dim 128, RoPE theta 500k, sliding window 4096, vocab 128256.
    Depth 23 is chosen to land at 3.03B exactly (halved width + trimmed
    depth). This is the exact 3B point; a deeper 26-layer 3.33B variant is
    available as get_3b_26l_config() if you prefer depth over tight budget.
    """

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=23,
        num_attention_heads=24,
        num_kv_heads=4,
        max_position_embeddings=8192,
        sliding_window=4096,
        rope_theta=500000.0,
    )


def get_3b_tight_config() -> CodingAgentConfig:
    """Alias for get_3b_config (23 layers, 3.03B) - kept for backwards compat."""

    return get_3b_config()


def get_3b_26l_config() -> CodingAgentConfig:
    """Deeper 3B variant (26 layers, ~3.33B) - halved width, deeper.

    Same width as get_3b_config but 26 layers for more reasoning depth at
    the cost of 3.33B params (still 3B class, ~10% over). Use when you
    want depth comparable to 20B's 60 layers scaled proportionally.
    """

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=26,
        num_attention_heads=24,
        num_kv_heads=4,
        max_position_embeddings=8192,
        sliding_window=4096,
        rope_theta=500000.0,
    )


def get_3b_32k_config() -> CodingAgentConfig:
    """3B with YaRN-scaled RoPE for 32k agent trajectories (train 8k, extend 4x).

    Mirrors get_20b_32k_config: same base as get_3b_config (23 layers) but
    with max_position_embeddings=32768, rope_scaling_type=yarn, factor 4.0.
    """

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=23,
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
    recall; retrieval + repo maps remain primary. Uses 23-layer base.
    """

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=23,
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
    """Alias for get_3b_32k_config (23 layers, 3.03B base)."""

    return get_3b_32k_config()


def get_3b_64k_tight_config() -> CodingAgentConfig:
    """Alias for get_3b_64k_config (23 layers, 3.03B base)."""

    return get_3b_64k_config()


def get_3b_26l_32k_config() -> CodingAgentConfig:
    """Deeper 26-layer 3.33B + 32k YaRN."""

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=3072,
        intermediate_size=8192,
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
    """Deeper 26-layer 3.33B + 64k YaRN."""

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=3072,
        intermediate_size=8192,
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
        assert 2.95e9 < total < 3.15e9, f"tight 3B params {total/1e9:.3f}B out of range"
    else:
        assert 2.9e9 < total < 3.5e9, f"3B params {total/1e9:.3f}B out of range"
    print(f"OK: {total:,} params ({total/1e9:.2f}B)")


if __name__ == "__main__":
    print("=== 3B Primary (23 layers, ~3.03B) ===")
    get_3b_config().print_summary()
    print("\n=== 3B Deep (26 layers, ~3.33B) ===")
    get_3b_26l_config().print_summary()
    print("\n=== 3B 32k (23 layers, YaRN x4) ===")
    get_3b_32k_config().print_summary()
    print("\n=== 3B 64k (23 layers, YaRN x8) ===")
    get_3b_64k_config().print_summary()
