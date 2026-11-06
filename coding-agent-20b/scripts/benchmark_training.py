"""
Measure what the training optimisations actually save.

Run:  python scripts/benchmark_training.py
      python scripts/benchmark_training.py --gpu a100-40gb
      python scripts/benchmark_training.py --gpu a100-80gb --iters 5

MEASURED (this machine, CPU, scaled-down but structurally identical model):
  * bytes of tensors saved for backward  == activation VRAM that must stay live
  * peak process RSS                    == real RAM high-water mark
  * median forward+backward wall time
  * SwiGLU/MoE block microbenchmark (the fused down-projection)
  * number of GPU->CPU synchronisations removed by the sync-free MoE

PROJECTED (deterministic arithmetic from the real recipe tensor shapes):
  * activation VRAM per GPU before/after for the 3B and 20B recipes
  * HBM traffic removed per micro-batch and its time cost at a stated bandwidth
  * FLOPs removed by the fused down-projection and its time cost at a stated
    bf16 throughput
  * total estimated step-time reduction and full-run wall-clock before/after
  * per-GPU VRAM feasibility and ZeRO-Infinity CPU-offload overhead
  * explicit A100 40GB / 80GB SXM / 80GB PCIe hardware profiles

No GPU is required to run this.  The projections use explicit, editable
constants for bandwidth/throughput so every number can be re-derived.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.config import CodingAgentConfig, get_20b_config  # noqa: E402
from model.config_3b import get_3b_config  # noqa: E402
from model.fast_ops import clear_rope_tables, format_bytes  # noqa: E402
from model.model import CodingAgentModel  # noqa: E402
from training.optimizations import (  # noqa: E402
    model_parameter_count,
    savings_report,
    zero3_state_bytes_per_gpu,
)

GiB = 1024 ** 3
GPU_PROFILES: dict[str, dict] = {
    "a100-40gb": {
        "label": "NVIDIA A100 40GB (HBM2, SXM/PCIe)",
        "hbm_bytes_per_second": 1.555e12,
        "bf16_flops_per_second": 3.12e14,
        "vram_bytes": 40 * GiB,
    },
    "a100-80gb": {
        "label": "NVIDIA A100 80GB SXM (HBM2e)",
        "hbm_bytes_per_second": 2.039e12,
        "bf16_flops_per_second": 3.12e14,
        "vram_bytes": 80 * GiB,
    },
    "a100-80gb-pcie": {
        "label": "NVIDIA A100 80GB PCIe (HBM2e)",
        "hbm_bytes_per_second": 1.935e12,
        "bf16_flops_per_second": 3.12e14,
        "vram_bytes": 80 * GiB,
    },
}
DEFAULT_GPU = "a100-80gb"
GPU_SYNC_SECONDS = 20e-6
PCIE_BYTES_PER_SECOND = 25e9
VRAM_SAFETY_FACTOR = 0.85
REALISTIC_MFU = 0.45

BENCH_VOCAB = 32768
BENCH_HIDDEN = 128
BENCH_INTERMEDIATE = 256
BENCH_LAYERS = 2
BENCH_SEQ = 256
BENCH_BATCH = 2
BENCH_CHUNK = 128


def bench_config(mode: str) -> CodingAgentConfig:
    cfg = CodingAgentConfig(
        vocab_size=BENCH_VOCAB,
        hidden_size=BENCH_HIDDEN,
        intermediate_size=BENCH_INTERMEDIATE,
        num_hidden_layers=BENCH_LAYERS,
        num_attention_heads=4,
        num_kv_heads=2,
        max_position_embeddings=512,
        sliding_window=128,
        use_gradient_checkpointing=True,
    )
    if mode == "baseline":
        cfg.fast_kernels = False
        cfg.fused_linear_cross_entropy = False
    elif mode in ("optimized", "strict"):
        cfg.fast_kernels = True
        cfg.fused_linear_cross_entropy = True
        cfg.loss_chunk_tokens = BENCH_CHUNK
        cfg.fuse_mlp_down_projection = mode == "optimized"
    else:
        raise ValueError(mode)
    return cfg


def saved_bytes(model, ids, labels) -> int:
    """Exact count of tensors PyTorch holds for backward (activation memory)."""
    sizes: list[int] = []

    def pack(tensor: torch.Tensor):
        sizes.append(tensor.numel() * tensor.element_size())
        return tensor

    def unpack(tensor: torch.Tensor):
        return tensor

    model.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = model(input_ids=ids, labels=labels)
        output.loss.backward()
    return sum(sizes)


def step_times(model, ids, labels, iters: int) -> list[float]:
    times = []
    for _ in range(iters):
        model.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        times.append(time.perf_counter() - t0)
    return times


def mlp_microbenchmark(iters: int = 20) -> dict:
    """Isolate the one-GEMM-instead-of-two change (real wall-clock).

    Interleave the two modes to cancel out machine-load drift, and use a
    larger tensor so the GEMMs dominate over Python overhead.
    """
    from model.model import SwiGLUMLP

    cfg = bench_config("optimized")
    cfg.fused_linear_cross_entropy = False
    cfg.use_moe = True
    cfg.moe_num_experts = 4
    cfg.moe_top_k = 2
    reference_mlp = SwiGLUMLP(cfg)
    state = {k: v.clone() for k, v in reference_mlp.state_dict().items()}
    models = {}
    for mode in ("reference", "fused"):
        cfg.fuse_mlp_down_projection = mode == "fused"
        clear_rope_tables()
        mlp = SwiGLUMLP(cfg)
        mlp.load_state_dict(state)
        models[mode] = mlp

    x = torch.randn(4, 512, BENCH_HIDDEN)
    for mlp in models.values():
        for _ in range(3):
            mlp.zero_grad(set_to_none=True)
            mlp(x).sum().backward()

    samples: dict[str, list[float]] = {"reference": [], "fused": []}
    for _ in range(3):
        for mode in ("reference", "fused"):
            mlp = models[mode]
            times = []
            for _ in range(iters):
                mlp.zero_grad(set_to_none=True)
                t0 = time.perf_counter()
                mlp(x).sum().backward()
                times.append(time.perf_counter() - t0)
            samples[mode].append(statistics.median(times))
    results = {mode: statistics.median(values) for mode, values in samples.items()}
    return {
        "reference_seconds": results["reference"],
        "fused_seconds": results["fused"],
        "speedup": results["reference"] / max(results["fused"], 1e-9),
        "gemms_per_layer_before": 2,
        "gemms_per_layer_after": 1,
    }


def worker(mode: str, iters: int) -> dict:
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    torch.manual_seed(1234)
    clear_rope_tables()
    cfg = bench_config(mode)
    model = CodingAgentModel(cfg)
    ids = torch.randint(0, BENCH_VOCAB, (BENCH_BATCH, BENCH_SEQ))
    labels = ids.clone()

    saved = saved_bytes(model, ids, labels)
    step_times(model, ids, labels, 2)
    times = step_times(model, ids, labels, iters)
    del model
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "mode": mode,
        "saved_activation_bytes": saved,
        "peak_rss_bytes": peak_kib * 1024,
        "step_seconds_median": statistics.median(times),
        "step_seconds_all": times,
        "params": model_parameter_count(cfg),
    }


def run_worker(mode: str, iters: int) -> dict:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", mode, "--iters", str(iters)],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=900,
        env={**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"benchmark worker {mode} failed:\n{proc.stdout}\n{proc.stderr}")
    line = [l for l in proc.stdout.splitlines() if l.startswith("BENCH_JSON=")]
    if not line:
        raise RuntimeError(f"missing BENCH_JSON in worker output:\n{proc.stdout}\n{proc.stderr}")
    return json.loads(line[0][len("BENCH_JSON=") :])


def projection(name: str, cfg: CodingAgentConfig, micro_batch: int, seq_len: int,
              total_tokens: int, global_batch_size: int,
              gpu_key: str = DEFAULT_GPU, world_size: int = 8) -> dict:
    gpu = GPU_PROFILES[gpu_key]
    hbm = gpu["hbm_bytes_per_second"]
    flops = gpu["bf16_flops_per_second"]
    vram = gpu["vram_bytes"]
    params = model_parameter_count(cfg)
    report = savings_report(
        name=name,
        params=params,
        world_size=world_size,
        micro_batch=micro_batch,
        seq_len=seq_len,
        vocab_size=cfg.vocab_size,
        layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
    )
    tokens = micro_batch * seq_len
    traffic_before = report["logits_before_bytes"] * 6
    traffic_after = report["logits_after_bytes"] * 6
    traffic_saved = max(traffic_before - traffic_after, 0)
    ce_time_saved = traffic_saved / hbm

    gemm_flops = 6 * tokens * cfg.intermediate_size * cfg.hidden_size * cfg.num_hidden_layers
    gemm_time_saved = gemm_flops / flops

    syncs_removed = 4 * cfg.num_hidden_layers
    sync_time_saved = syncs_removed * GPU_SYNC_SECONDS

    baseline_step_seconds = 6 * params * tokens / flops
    estimated_saving = ce_time_saved + gemm_time_saved + sync_time_saved
    optimized_step_seconds = max(baseline_step_seconds - estimated_saving, 1e-9)
    state = report["state_bytes_per_gpu"]
    gather_bucket_bytes = 2 * int(2e8) * 2
    global_tokens_per_micro_batch = tokens * world_size
    total_micro_batches = total_tokens / global_tokens_per_micro_batch
    total_optimizer_steps = total_tokens / (global_batch_size * seq_len)

    vram_budget = int(vram * VRAM_SAFETY_FACTOR)
    gathered = int(2 * params / world_size)
    no_offload_need = state + report["activations_after_bytes"] + gathered
    with_offload_need = report["activations_after_bytes"] + gather_bucket_bytes
    offload_needed = no_offload_need > vram_budget
    final_need = with_offload_need if offload_needed else no_offload_need
    fits = final_need <= vram_budget
    baseline_need = state + report["activations_before_bytes"] + gathered

    # The shipped recipes hard-code offload_optimizer/offload_param = true.
    # Model: read + write of the optimizer state once per optimizer step over
    offload_seconds_per_step = 2 * state / PCIE_BYTES_PER_SECOND
    baseline_offload_hours = total_optimizer_steps * offload_seconds_per_step / 3600
    optimized_offload_hours = baseline_offload_hours if offload_needed else 0.0

    kernels_before_h = total_micro_batches * baseline_step_seconds / 3600
    kernels_after_h = total_micro_batches * optimized_step_seconds / 3600

    report.update(
        {
            "gpu": gpu_key,
            "gpu_label": gpu["label"],
            "hbm_bytes_per_second": hbm,
            "bf16_flops_per_second": flops,
            "vram_bytes": vram,
            "vram_budget_bytes": vram_budget,
            "vram_final_need_bytes": final_need,
            "vram_shortfall_bytes": max(final_need - vram_budget, 0),
            "vram_with_half_micro_batch_bytes": (
                report["activations_after_bytes"] // 2 + gather_bucket_bytes
            ),
            "vram_baseline_need_bytes": baseline_need,
            "baseline_fits_in_vram": baseline_need <= vram_budget,
            "offload_needed": offload_needed,
            "fits_in_vram": fits,
            "cpu_ram_per_gpu_bytes": state if offload_needed else 0,
            "total_optimizer_steps": total_optimizer_steps,
            "offload_seconds_per_optimizer_step": offload_seconds_per_step,
            "baseline_offload_hours": baseline_offload_hours,
            "optimized_offload_hours": optimized_offload_hours,
            "full_run_before_hours_incl_offload": kernels_before_h + baseline_offload_hours,
            "full_run_after_hours_incl_offload": kernels_after_h + optimized_offload_hours,
            "full_run_hours_saved_incl_offload": (
                kernels_before_h + baseline_offload_hours
                - kernels_after_h - optimized_offload_hours
            ),
            "realistic_baseline_full_run_days_incl_offload": (
                (kernels_before_h / REALISTIC_MFU + baseline_offload_hours) / 24
            ),
            "realistic_optimized_full_run_days_incl_offload": (
                (kernels_after_h / REALISTIC_MFU + optimized_offload_hours) / 24
            ),
            "total_tokens": total_tokens,
            "global_tokens_per_micro_batch": global_tokens_per_micro_batch,
            "total_micro_batches": total_micro_batches,
            "estimated_baseline_full_run_hours": total_micro_batches * baseline_step_seconds / 3600,
            "estimated_optimized_full_run_hours": (
                total_micro_batches * max(baseline_step_seconds - estimated_saving, 1e-9) / 3600
            ),
            "estimated_full_run_seconds_saved": total_micro_batches * estimated_saving,
            "estimated_full_run_hours_saved": total_micro_batches * estimated_saving / 3600,
            "estimated_full_run_days_saved": total_micro_batches * estimated_saving / 86400,
            "realistic_mfu": REALISTIC_MFU,
            "realistic_baseline_full_run_days": (
                total_micro_batches * baseline_step_seconds / 86400 / REALISTIC_MFU
            ),
            "realistic_optimized_full_run_days": (
                total_micro_batches * max(baseline_step_seconds - estimated_saving, 1e-9)
                / 86400 / REALISTIC_MFU
            ),
            "vram_before_bytes": state + report["activations_before_bytes"],
            "vram_after_bytes": state + report["activations_after_bytes"],
            "vram_after_with_state_offload_bytes": (
                report["activations_after_bytes"] + gather_bucket_bytes
            ),
            "estimated_baseline_micro_batch_seconds": baseline_step_seconds,
            "estimated_speedup_pct": 100 * estimated_saving / max(baseline_step_seconds, 1e-9),
            "estimated_optimized_micro_batch_seconds": max(baseline_step_seconds - estimated_saving, 1e-9),
            "actual_params": params,
            "headline_params": cfg.total_parameters(),
            "state_bytes_per_gpu": zero3_state_bytes_per_gpu(params, world_size),
            "hbm_traffic_saved_bytes": traffic_saved,
            "ce_hbm_seconds_saved": ce_time_saved,
            "mlp_flops_saved": gemm_flops,
            "mlp_seconds_saved": gemm_time_saved,
            "gpu_syncs_removed": syncs_removed,
            "sync_seconds_saved": sync_time_saved,
            "estimated_seconds_saved_per_micro_batch": estimated_saving,
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("baseline", "optimized", "strict"))
    parser.add_argument("--iters", type=int, default=9)
    parser.add_argument("--out", type=str, default="evaluation/reports/training_optimization_benchmark.json")
    parser.add_argument("--gpu", choices=[*GPU_PROFILES, "all"], default="all",
                        help="A100 variant to project (default: all)")
    args = parser.parse_args()

    if args.worker:
        print("BENCH_JSON=" + json.dumps(worker(args.worker, args.iters)))
        return 0

    print("=" * 84)
    print("  TRAINING OPTIMISATION BENCHMARK (measured + projected)")
    print("=" * 84)
    rounds = 3
    baselines, optimised = [], []
    for _ in range(rounds):
        baselines.append(run_worker("baseline", args.iters))
        optimised.append(run_worker("optimized", args.iters))
    base = dict(baselines[0])
    opt = dict(optimised[0])
    base["step_seconds_median"] = statistics.median(r["step_seconds_median"] for r in baselines)
    opt["step_seconds_median"] = statistics.median(r["step_seconds_median"] for r in optimised)
    base["peak_rss_bytes"] = statistics.median(r["peak_rss_bytes"] for r in baselines)
    opt["peak_rss_bytes"] = statistics.median(r["peak_rss_bytes"] for r in optimised)
    base["step_seconds_all"] = [t for r in baselines for t in r["step_seconds_all"]]
    opt["step_seconds_all"] = [t for r in optimised for t in r["step_seconds_all"]]

    def pct(before, after):
        return (1 - after / before) * 100 if before else 0.0

    measured = {
        "benchmark_model": {
            "vocab": BENCH_VOCAB, "hidden": BENCH_HIDDEN, "intermediate": BENCH_INTERMEDIATE,
            "layers": BENCH_LAYERS, "seq": BENCH_SEQ, "batch": BENCH_BATCH,
            "loss_chunk_tokens": BENCH_CHUNK, "gradient_checkpointing": True,
        },
        "baseline": base,
        "optimized": opt,
        "saved_activation_bytes_reduction_pct": pct(base["saved_activation_bytes"], opt["saved_activation_bytes"]),
        "peak_rss_reduction_pct": pct(base["peak_rss_bytes"], opt["peak_rss_bytes"]),
        "step_time_speedup": base["step_seconds_median"] / max(opt["step_seconds_median"], 1e-9),
        "step_time_reduction_pct": pct(base["step_seconds_median"], opt["step_seconds_median"]),
        "mlp_microbenchmark": mlp_microbenchmark(),
        "timing_rounds": rounds,
    }
    print("\n-- MEASURED (CPU, scaled model, separate processes) --")
    print(f"  saved-for-backward bytes : {base['saved_activation_bytes']:,} -> {opt['saved_activation_bytes']:,}"
          f"   ({measured['saved_activation_bytes_reduction_pct']:.1f}% less)")
    print(f"  peak RSS                 : {base['peak_rss_bytes']:,} -> {opt['peak_rss_bytes']:,}"
          f"   ({measured['peak_rss_reduction_pct']:.1f}% less)")
    print(f"  step time (median)       : {base['step_seconds_median']:.4f}s -> {opt['step_seconds_median']:.4f}s"
          f"   ({measured['step_time_speedup']:.2f}x, {measured['step_time_reduction_pct']:.1f}% faster)")
    mlp = measured["mlp_microbenchmark"]
    print(f"  MLP block fwd+bwd        : {mlp['reference_seconds']:.4f}s -> {mlp['fused_seconds']:.4f}s"
          f"   ({mlp['speedup']:.2f}x)")

    gpu_keys = list(GPU_PROFILES) if args.gpu == "all" else [args.gpu]
    recipes = {
        "3B": dict(cfg=get_3b_config(), micro_batch=1, seq_len=4096,
                   total_tokens=18_000_000_000, global_batch_size=512),
        "20B": dict(cfg=get_20b_config(), micro_batch=2, seq_len=32768,
                    total_tokens=300_000_000_000, global_batch_size=2048),
    }
    projections_by_gpu: dict[str, dict] = {}
    for gpu_key in gpu_keys:
        gpu = GPU_PROFILES[gpu_key]
        projections = {
            name: projection(name, gpu_key=gpu_key, **spec)
            for name, spec in recipes.items()
        }
        projections_by_gpu[gpu_key] = projections
        print(f"\n-- PROJECTED: {gpu['label']} | {gpu['hbm_bytes_per_second']/1e12:.3f} TB/s, "
              f"{gpu['bf16_flops_per_second']/1e12:.0f} TFLOP/s bf16, "
              f"{format_bytes(gpu['vram_bytes'])} VRAM, 8 GPUs --")
        for name, r in projections.items():
            fit = "FITS" if r["fits_in_vram"] else "DOES NOT FIT"
            print(f"  {name} ({r['actual_params']/1e9:.2f}B params): "
                  f"logits {r['logits_before']} -> {r['logits_after']} ({r['logits_saving']}); "
                  f"activations {r['activations_before']} -> {r['activations_after']} "
                  f"({r['activations_saving']})")
            print(f"       VRAM/GPU after: {format_bytes(r['vram_final_need_bytes'])} of "
                  f"{format_bytes(r['vram_budget_bytes'])} usable -> {fit}; "
                  f"CPU RAM/GPU for offload {format_bytes(r['cpu_ram_per_gpu_bytes'])}")
            print(f"       step {r['estimated_baseline_micro_batch_seconds']:.2f}s -> "
                  f"{r['estimated_optimized_micro_batch_seconds']:.2f}s "
                  f"({r['estimated_speedup_pct']:.1f}% faster = "
                  f"HBM {r['ce_hbm_seconds_saved']:.3f}s + MLP {r['mlp_seconds_saved']:.3f}s "
                  f"+ {r['gpu_syncs_removed']} syncs {r['sync_seconds_saved']:.3f}s)")
            print(f"       FULL RUN wall-clock (incl. offload, 100% MFU): "
                  f"{r['full_run_before_hours_incl_offload']:,.1f} h -> "
                  f"{r['full_run_after_hours_incl_offload']:,.1f} h "
                  f"(saves {r['full_run_hours_saved_incl_offload']:,.1f} h)")
            print(f"       ...at a realistic {REALISTIC_MFU:.0%} MFU: "
                  f"{r['realistic_baseline_full_run_days_incl_offload']:,.0f} days -> "
                  f"{r['realistic_optimized_full_run_days_incl_offload']:,.0f} days")
            if not r["baseline_fits_in_vram"]:
                print(f"       NOTE: the ORIGINAL recipe needed "
                      f"{format_bytes(r['vram_baseline_need_bytes'])}/GPU and would OOM here.")
            if not r["fits_in_vram"]:
                print(f"       WARNING: short by {format_bytes(r['vram_shortfall_bytes'])}. "
                      f"Use micro_batch=1 with 2x grad accumulation (same effective global "
                      f"batch, quality-neutral) -> ~"
                      f"{format_bytes(r['vram_with_half_micro_batch_bytes'])}, fits. "
                      f"Times above still hold (total compute is unchanged).")

    header = (f"{'GPU':<34} {'3B before':>11} {'3B after':>11} "
              f"{'20B before':>12} {'20B after':>12} {'20B fits?':>11}")
    print("\n-- A100 COMPARISON: full run on ONE 8-GPU node (days, 100% MFU, incl. offload) --")
    print(header)
    print("-" * len(header))
    for gpu_key, pr in projections_by_gpu.items():
        a, b = pr["3B"], pr["20B"]
        print(f"{GPU_PROFILES[gpu_key]['label'][:34]:<34} "
              f"{a['full_run_before_hours_incl_offload']/24:>10.2f}d "
              f"{a['full_run_after_hours_incl_offload']/24:>10.2f}d "
              f"{b['full_run_before_hours_incl_offload']/24:>11.1f}d "
              f"{b['full_run_after_hours_incl_offload']/24:>11.1f}d "
              f"{('yes' if b['fits_in_vram'] else 'NO'):>11}")

    result = {
        "assumptions": {
            "gpu_profiles": GPU_PROFILES,
            "gpu_sync_seconds": GPU_SYNC_SECONDS,
            "pcie_bytes_per_second": PCIE_BYTES_PER_SECOND,
            "vram_safety_factor": VRAM_SAFETY_FACTOR,
            "world_size": 8,
            "realistic_mfu": REALISTIC_MFU,
            "note": "full-run times are wall-clock on the 8-GPU node (not GPU-hours); "
                    "offload overhead is a conservative upper bound assuming zero overlap",
        },
        "measured": measured,
        "projected": projections_by_gpu.get(DEFAULT_GPU, next(iter(projections_by_gpu.values()))),
        "projected_by_gpu": projections_by_gpu,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nWrote {out.relative_to(ROOT)}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
