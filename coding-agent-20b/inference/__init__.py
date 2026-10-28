"""Inference: serving + quantization."""

from .quantize import (
    benchmark_model,
    dynamic_int8,
    load_weight_only_int8,
    weight_only_int8_state_dict,
)

__all__ = [
    "benchmark_model",
    "dynamic_int8",
    "load_weight_only_int8",
    "weight_only_int8_state_dict",
]
