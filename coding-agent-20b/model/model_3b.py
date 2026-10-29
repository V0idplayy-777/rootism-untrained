"""
3B Parameter Coding Agent Model - Full Transformer Architecture

EXACT same code as model/model.py, but intended to be used with the 3B
config (model/config_3b.py). The transformer itself is architecture-agnostic:
Grouped Query Attention, SwiGLU, RMSNorm, RoPE, sliding window, etc. are
identical to the 20B model -- only the config numbers change.

- 20B: 60 layers, hidden 6144, 24Q? actually 48Q/8KV, intermediate 16384
- 3B:  23 layers, hidden 3072, 24Q/4KV, intermediate 8192

All other components (RotaryEmbedding, GroupedQueryAttention, SwiGLUMLP,
TransformerBlock, PagedKVCache, CodingAgentModel) are re-exported from
model.model so this file stays byte-for-byte compatible with the 20B
implementation. Importing from here vs model.model yields identical
behavior; the only difference is the default config helper.

Usage:

    from model.model_3b import CodingAgentModel
    from model.config_3b import get_3b_config

    model = CodingAgentModel(get_3b_config())
    out = model(input_ids)
    print(model.num_parameters() / 1e9)  # ~3.03B

Or via convenience:

    from model.model_3b import get_default_3b_model
    model = get_default_3b_model()

See model/config_3b.py for the full scaling table and
configs/training_config_3b.json for the 3B training recipe.
"""

from .config_3b import (
    get_3b_26l_32k_config,
    get_3b_26l_64k_config,
    get_3b_26l_config,
    get_3b_32k_config,
    get_3b_32k_tight_config,
    get_3b_64k_config,
    get_3b_64k_tight_config,
    get_3b_config,
    get_3b_tight_config,
)

__all__ = [
    "CodingAgentModel",
    "CodingAgent3BModel",
    "GroupedQueryAttention",
    "ModelOutput",
    "PagedKVCache",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLUMLP",
    "TransformerBlock",
    "apply_rotary_pos_emb",
    "get_3b_26l_32k_config",
    "get_3b_26l_64k_config",
    "get_3b_26l_config",
    "get_3b_32k_config",
    "get_3b_32k_tight_config",
    "get_3b_64k_config",
    "get_3b_64k_tight_config",
    "get_3b_config",
    "get_3b_tight_config",
    "get_default_3b_model",
    "rotate_half",
]


def __getattr__(name: str):
    if name in {
        "CodingAgentModel",
        "CodingAgent3BModel",
        "GroupedQueryAttention",
        "ModelOutput",
        "PagedKVCache",
        "RMSNorm",
        "RotaryEmbedding",
        "SwiGLUMLP",
        "TransformerBlock",
        "apply_rotary_pos_emb",
        "rotate_half",
    }:
        from .model import (
            CodingAgentModel as _CodingAgentModel,
            GroupedQueryAttention as _GQA,
            ModelOutput as _MO,
            PagedKVCache as _Cache,
            RMSNorm as _RMS,
            RotaryEmbedding as _RoPE,
            SwiGLUMLP as _MLP,
            TransformerBlock as _Block,
            apply_rotary_pos_emb as _apply,
            rotate_half as _rot,
        )

        mapping = {
            "CodingAgentModel": _CodingAgentModel,
            "CodingAgent3BModel": _CodingAgentModel,
            "GroupedQueryAttention": _GQA,
            "ModelOutput": _MO,
            "PagedKVCache": _Cache,
            "RMSNorm": _RMS,
            "RotaryEmbedding": _RoPE,
            "SwiGLUMLP": _MLP,
            "TransformerBlock": _Block,
            "apply_rotary_pos_emb": _apply,
            "rotate_half": _rot,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_default_3b_model(*, device: str | None = None):
    """Instantiate CodingAgentModel with the 3B config (23 layers, 3.03B).

    This is the direct 3B equivalent of `CodingAgentModel(get_20b_config())`
    for the 20B model. All model code paths are shared; only the config
    differs.
    """

    from .model import CodingAgentModel

    cfg = get_3b_config()
    model = CodingAgentModel(cfg)
    if device is not None:
        import torch

        model = model.to(device)
    return model
