"""Inference: serving + quantization."""







from .quantize import dynamic_int8, weight_only_int8_state_dict, load_weight_only_int8, benchmark_model















__all__ = ["dynamic_int8", "weight_only_int8_state_dict", "load_weight_only_int8", "benchmark_model"]







