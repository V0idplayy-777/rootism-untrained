"""
Utility functions for training and evaluation.
"""















import os







import json







import time







import math







import random







import numpy as np







import torch







from typing import Dict, Any























def set_seed(seed: int):







    """Set random seeds for reproducibility."""







    random.seed(seed)







    np.random.seed(seed)







    torch.manual_seed(seed)







    if torch.cuda.is_available():







        torch.cuda.manual_seed_all(seed)























def get_device() -> torch.device:







    if torch.cuda.is_available():







        return torch.device('cuda')







    return torch.device('cpu')























def count_parameters(model: torch.nn.Module) -> Dict[str, int]:







    counts = {'total': 0, 'trainable': 0, 'embedding': 0, 'attention': 0, 'mlp': 0, 'norm': 0}







    for name, param in model.named_parameters():







        numel = param.numel()







        counts['total'] += numel







        if param.requires_grad:







            counts['trainable'] += numel







        name_lower = name.lower()







        if 'embed' in name_lower or 'lm_head' in name_lower:







            counts['embedding'] += numel







        elif 'attn' in name_lower:







            counts['attention'] += numel







        elif 'mlp' in name_lower or 'gate' in name_lower:







            counts['mlp'] += numel







        elif 'norm' in name_lower:







            counts['norm'] += numel







    return counts























def format_number(n: float) -> str:







    if n >= 1e12: return f"{n/1e12:.2f}T"







    elif n >= 1e9: return f"{n/1e9:.2f}B"







    elif n >= 1e6: return f"{n/1e6:.2f}M"







    elif n >= 1e3: return f"{n/1e3:.2f}K"







    return f"{n:.2f}"























def estimate_training_time(num_params, num_tokens, tokens_per_second, num_gpus):







    total_seconds = num_tokens / tokens_per_second







    total_hours = total_seconds / 3600







    return {







        "total_flops": 6 * num_params * num_tokens,







        "total_hours": total_hours,







        "total_days": total_hours / 24,







        "gpu_hours": total_hours * num_gpus,







    }























def compute_perplexity(loss: float) -> float:







    return math.exp(min(loss, 20))























class Timer:







    def __init__(self, name=""):







        self.name = name







        self.elapsed = 0







    def __enter__(self):







        self.start = time.time()







        return self







    def __exit__(self, *args):







        self.elapsed = time.time() - self.start























class MetricsTracker:







    def __init__(self, window_size=100):







        self.window_size = window_size







        self.metrics = {}







    def update(self, **kwargs):







        for k, v in kwargs.items():







            if k not in self.metrics:







                self.metrics[k] = []







            self.metrics[k].append(v)







            if len(self.metrics[k]) > self.window_size:







                self.metrics[k] = self.metrics[k][-self.window_size:]







    def get(self, key):







        vals = self.metrics.get(key, [])







        if not vals: return {"mean": 0, "count": 0}







        return {"mean": sum(vals)/len(vals), "min": min(vals), "max": max(vals), "count": len(vals)}























def save_training_log(log_data, filepath):







    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)







    with open(filepath, 'a') as f:







        f.write(json.dumps(log_data) + '\n')























def get_gpu_memory_info():







    if not torch.cuda.is_available():







        return {"error": "CUDA not available"}







    info = {}







    for i in range(torch.cuda.device_count()):







        allocated = torch.cuda.memory_allocated(i) / 1024**3







        total = torch.cuda.get_device_properties(i).total_memory / 1024**3







        info[f"gpu_{i}"] = {







            "allocated_gb": round(allocated, 2),







            "total_gb": round(total, 2),







            "utilization_pct": round(allocated / total * 100, 1),







        }







    return info







