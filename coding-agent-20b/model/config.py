"""
Configuration for the 20B parameter Coding Agent Model.

Architecture decisions:
- 20B parameters achieved via: 60 layers, hidden_size=6144, num_heads=48, num_kv_heads=8 (GQA)
- Vocabulary size: 128256 (power-of-2 aligned for efficient compute)
- Context length: 8192 tokens (suitable for code files and long conversations)
- Uses RoPE for positional embeddings
- Uses RMSNorm (more stable than LayerNorm for large models)
- Uses SwiGLU activation (better than GELU/ReLU for LLMs)
- Uses Grouped Query Attention (GQA) for efficient inference
- Sliding window attention for long contexts
"""

import json
from dataclasses import dataclass


@dataclass
class CodingAgentConfig:
    """Full configuration for the 20B Coding Agent Model."""

    vocab_size: int = 128256

    hidden_size: int = 6144

    intermediate_size: int = 16384

    num_hidden_layers: int = 60

    num_attention_heads: int = 48

    num_kv_heads: int = 8

    head_dim: int = 128

    max_position_embeddings: int = 8192

    sliding_window: int | None = 4096

    rms_norm_eps: float = 1e-5

    norm_type: str = "rmsnorm"

    hidden_act: str = "silu"

    rope_theta: float = 500000.0

    attention_dropout: float = 0.0

    attention_bias: bool = False

    rope_scaling_type: str | None = None

    rope_scaling_factor: float = 1.0

    original_max_position_embeddings: int = 8192

    tool_call_token_id: int = 48

    tool_result_token_id: int = 49

    kv_cache_max_tokens: int = 32768

    initializer_range: float = 0.02

    tie_word_embeddings: bool = False

    use_gradient_checkpointing: bool = True

    gradient_checkpointing_ratio: float = 1.0

    bos_token_id: int = 1

    eos_token_id: int = 2

    pad_token_id: int = 0

    deepspeed_stage: int = 3

    mixed_precision: str = "bf16"

    max_seq_length: int = 8192

    def __post_init__(self):

        assert (
            self.hidden_size % self.num_attention_heads == 0
        ), f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"

        assert (
            self.num_attention_heads % self.num_kv_heads == 0
        ), f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"

        self.head_dim = self.hidden_size // self.num_attention_heads

    def total_parameters(self) -> int:
        """Estimate total parameter count."""

        embed_params = self.vocab_size * self.hidden_size * 2

        q_proj = self.hidden_size * self.num_attention_heads * self.head_dim

        kv_proj = 2 * self.hidden_size * self.num_kv_heads * self.head_dim

        o_proj = self.num_attention_heads * self.head_dim * self.hidden_size

        attn_params = q_proj + kv_proj + o_proj

        mlp_params = (
            self.hidden_size * self.intermediate_size * 2
            + self.intermediate_size * self.hidden_size
        )

        norm_params = 2 * self.hidden_size

        layer_params = attn_params + mlp_params + norm_params

        total_transformer = self.num_hidden_layers * layer_params

        final_norm = self.hidden_size

        total = embed_params + total_transformer + final_norm

        return total

    def to_dict(self) -> dict:
        """Convert config to dictionary."""

        return {k: v for k, v in self.__dict__.items()}

    def save(self, path: str):
        """Save config to JSON file."""

        with open(path, "w") as f:

            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "CodingAgentConfig":
        """Create config from dictionary."""

        return cls(**d)

    @classmethod
    def load(cls, path: str) -> "CodingAgentConfig":
        """Load config from JSON file."""

        with open(path, "r") as f:

            d = json.load(f)

        return cls.from_dict(d)

    def print_summary(self):
        """Print model architecture summary."""

        total_params = self.total_parameters()

        print("=" * 60)

        print("  CODING AGENT MODEL - ARCHITECTURE SUMMARY")

        print("=" * 60)

        print(f"  Total Parameters:      {total_params:,} ({total_params / 1e9:.2f}B)")

        print(f"  Hidden Size:           {self.hidden_size}")

        print(f"  Num Layers:            {self.num_hidden_layers}")

        print(f"  Num Attention Heads:   {self.num_attention_heads}")

        print(f"  Num KV Heads (GQA):    {self.num_kv_heads}")

        print(f"  Head Dimension:        {self.head_dim}")

        print(f"  Intermediate Size:     {self.intermediate_size}")

        print(f"  Vocabulary Size:       {self.vocab_size:,}")

        print(f"  Max Sequence Length:   {self.max_position_embeddings}")

        print(f"  Sliding Window:        {self.sliding_window}")

        print(f"  RoPE Theta:            {self.rope_theta}")

        print("  Activation:            SwiGLU (SiLU)")

        print("  Normalization:         RMSNorm")

        print(f"  Gradient Checkpoint:   {self.use_gradient_checkpointing}")

        print(f"  Mixed Precision:       {self.mixed_precision}")

        print(f"  DeepSpeed Stage:       ZeRO-{self.deepspeed_stage}")

        print("=" * 60)


def get_20b_config() -> CodingAgentConfig:
    """Get the 20B parameter configuration."""

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=6144,
        intermediate_size=16384,
        num_hidden_layers=60,
        num_attention_heads=48,
        num_kv_heads=8,
        max_position_embeddings=8192,
        sliding_window=4096,
        rope_theta=500000.0,
    )


def get_7b_config() -> CodingAgentConfig:
    """Get a 7B parameter configuration (for testing/debugging)."""

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        max_position_embeddings=8192,
        sliding_window=4096,
        rope_theta=500000.0,
    )


def get_20b_32k_config() -> CodingAgentConfig:
    """20B with YaRN-scaled RoPE for 32k agent trajectories (train 8k, extend 4x)."""

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=6144,
        intermediate_size=16384,
        num_hidden_layers=60,
        num_attention_heads=48,
        num_kv_heads=8,
        max_position_embeddings=32768,
        sliding_window=4096,
        rope_theta=500000.0,
        rope_scaling_type="yarn",
        rope_scaling_factor=4.0,
        original_max_position_embeddings=8192,
    )


def get_20b_16k_config() -> CodingAgentConfig:
    """20B with YaRN-scaled RoPE for 16k agent trajectories (train 8k, extend 2x)."""

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=6144,
        intermediate_size=16384,
        num_hidden_layers=60,
        num_attention_heads=48,
        num_kv_heads=8,
        max_position_embeddings=16384,
        sliding_window=4096,
        rope_theta=500000.0,
        rope_scaling_type="yarn",
        rope_scaling_factor=2.0,
        original_max_position_embeddings=8192,
    )


def get_20b_64k_config() -> CodingAgentConfig:
    """20B with YaRN x8 for 64k windows (long files, multi-file diffs).

    Honest caveat: 64k via YaRN is an extrapolation, not native long-context
    training. Expect degradation on tasks needing precise recall across the
    middle of the window; retrieval + repo maps (agent/repo_context.py) remain
    the primary mechanism, with 64k as headroom for a few large files.
    """

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=6144,
        intermediate_size=16384,
        num_hidden_layers=60,
        num_attention_heads=48,
        num_kv_heads=8,
        max_position_embeddings=65536,
        sliding_window=8192,
        rope_theta=500000.0,
        rope_scaling_type="yarn",
        rope_scaling_factor=8.0,
        original_max_position_embeddings=8192,
        kv_cache_max_tokens=65536,
    )


def get_20b_16k_config() -> CodingAgentConfig:
    """20B with YaRN-scaled RoPE for 16k agent trajectories (train 8k, extend 2x)."""

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=6144,
        intermediate_size=16384,
        num_hidden_layers=60,
        num_attention_heads=48,
        num_kv_heads=8,
        max_position_embeddings=16384,
        sliding_window=4096,
        rope_theta=500000.0,
        rope_scaling_type="yarn",
        rope_scaling_factor=2.0,
        original_max_position_embeddings=8192,
    )


def get_20b_64k_config() -> CodingAgentConfig:
    """20B with YaRN x8 for 64k windows (long files, multi-file diffs).

    Honest caveat: 64k via YaRN is an extrapolation, not native long-context
    training. Expect degradation on tasks needing precise recall across the
    middle of the window; retrieval + repo maps (agent/repo_context.py) remain
    the primary mechanism, with 64k as headroom for a few large files.
    """

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=6144,
        intermediate_size=16384,
        num_hidden_layers=60,
        num_attention_heads=48,
        num_kv_heads=8,
        max_position_embeddings=65536,
        sliding_window=8192,
        rope_theta=500000.0,
        rope_scaling_type="yarn",
        rope_scaling_factor=8.0,
        original_max_position_embeddings=8192,
        kv_cache_max_tokens=65536,
    )


def get_20b_64k_config() -> CodingAgentConfig:
    """20B with YaRN x8 for 64k windows (long files, multi-file diffs).

    Honest caveat: 64k via YaRN is an extrapolation, not native long-context
    training. Expect degradation on tasks needing precise recall across the
    middle of the window; retrieval + repo maps (agent/repo_context.py) remain
    the primary mechanism, with 64k as headroom for a few large files.
    """

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=6144,
        intermediate_size=16384,
        num_hidden_layers=60,
        num_attention_heads=48,
        num_kv_heads=8,
        max_position_embeddings=65536,
        sliding_window=8192,
        rope_theta=500000.0,
        rope_scaling_type="yarn",
        rope_scaling_factor=8.0,
        original_max_position_embeddings=8192,
        kv_cache_max_tokens=65536,
    )


def get_tiny_config(vocab_size: int = 320) -> CodingAgentConfig:
    """CPU smoke-test config (fast forward/generate in CI)."""

    return CodingAgentConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        max_position_embeddings=256,
        sliding_window=128,
        rope_theta=10000.0,
    )


def get_1b_config() -> CodingAgentConfig:
    """Get a 1B parameter configuration (for smoke testing)."""

    return CodingAgentConfig(
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=5504,
        num_hidden_layers=22,
        num_attention_heads=16,
        num_kv_heads=4,
        max_position_embeddings=4096,
        sliding_window=2048,
        rope_theta=500000.0,
    )
