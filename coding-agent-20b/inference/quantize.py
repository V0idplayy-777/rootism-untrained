"""Deployment quantization paths.

- dynamic_int8: torch dynamic quantization of Linear layers (CPU, zero calibration).
- weight_only_int8: per-channel symmetric int8 weight packing with scales kept
  in the state dict (dequantize on load) — the portable checkpoint format.
Full GPTQ/AWQ need calibration data + CUDA kernels; the interfaces here match
that flow (calibrate -> quantize -> save) so kernels can be swapped in.
"""



from __future__ import annotations







import copy



from pathlib import Path



from typing import Dict







import torch



import torch.nn as nn











def dynamic_int8(model: nn.Module) -> nn.Module:



    m = copy.deepcopy(model)



    m.eval()



    return torch.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)











def weight_only_int8_state_dict(model: nn.Module) -> Dict:



    """Pack every 2D Linear weight as int8 + per-row scale (round-trip exact API)."""



    packed = {}



    for name, tensor in model.state_dict().items():



        if tensor.dtype in (torch.float32, torch.float16, torch.bfloat16) and tensor.dim() == 2 and name.endswith("weight"):



            w = tensor.float()



            scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-8)



            q = torch.round(w / scale).clamp(-128, 127).to(torch.int8)



            packed[name] = q



            packed[name + ".scale"] = scale



        else:



            packed[name] = tensor



    return packed











def load_weight_only_int8(model: nn.Module, packed: Dict) -> nn.Module:



    state = {}



    for name, tensor in packed.items():



        if name.endswith(".scale"):



            continue



        if tensor.dtype == torch.int8 and (name + ".scale") in packed:



            state[name] = (tensor.float() * packed[name + ".scale"]).to(torch.float32)



        else:



            state[name] = tensor



    model.load_state_dict(state, strict=False)



    return model











def save_quantized(model: nn.Module, path: str, method: str = "weight_only_int8") -> Path:



    p = Path(path)



    p.parent.mkdir(parents=True, exist_ok=True)



    if method == "weight_only_int8":



        torch.save({"method": method, "state_dict": weight_only_int8_state_dict(model)}, p)



    elif method == "dynamic_int8":



        torch.save({"method": method, "state_dict": dynamic_int8(model).state_dict()}, p)



    else:



        raise ValueError(f"unknown quantization method {method!r}")



    return p











def benchmark_model(model: nn.Module, example_ids: torch.Tensor, repeats: int = 3) -> Dict:



    import time



    model.eval()



    params = sum(x.numel() for x in model.parameters())



    with torch.no_grad():



        start = time.time()



        for _ in range(repeats):



            model(input_ids=example_ids)



        elapsed = (time.time() - start) / max(repeats, 1)



    return {"params": params, "forward_seconds": elapsed,



            "tokens_per_second": example_ids.numel() / max(elapsed, 1e-9)}



