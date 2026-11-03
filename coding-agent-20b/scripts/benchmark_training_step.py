"""Measure a training step with each fast path on and off.

The point is to show what each optimisation actually buys, and that turning
them all on changes the loss by nothing.  Runs anywhere: on CPU it is a
correctness+relative-cost harness, on a GPU it reports real VRAM and tok/s.

    python scripts/benchmark_training_step.py --layers 4 --hidden 512 --seq 1024
    python scripts/benchmark_training_step.py --preset 3b --device cuda
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from model.config import CodingAgentConfig, get_20b_config  # noqa: E402
from model.config_3b import get_3b_config  # noqa: E402
from model.model import CodingAgentModel  # noqa: E402

PRESETS = {
    "tiny": dict(num_hidden_layers=4, hidden_size=512, intermediate_size=1408, num_attention_heads=8, num_kv_heads=2),
    "3b": None,
    "20b": None,
}


def build_config(args) -> CodingAgentConfig:

    if args.preset == "3b":
        return get_3b_config()

    if args.preset == "20b":
        return get_20b_config()

    return CodingAgentConfig(
        vocab_size=args.vocab,
        hidden_size=args.hidden,
        intermediate_size=int(args.hidden * 2.75),
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_kv_heads=max(args.heads // 4, 1),
        max_position_embeddings=max(args.seq, 4096),
        sliding_window=args.sliding_window,
        rope_theta=500_000.0,
    )


def run(config: CodingAgentConfig, args, **overrides) -> dict:

    config = copy.deepcopy(config)

    for key, value in overrides.items():
        setattr(config, key, value)

    device = torch.device(args.device)

    # Same weights, same tokens, same order for every variant: the loss column
    # below then isolates the fast paths instead of comparing two random inits.
    torch.manual_seed(0)

    dtype = torch.bfloat16 if args.device == "cuda" and args.bf16 else torch.float32

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = CodingAgentModel(config).to(device)

    model.train()

    ids = torch.randint(0, config.vocab_size, (args.micro_batch, args.seq), device=device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)

    if args.device == "cuda" and dtype is torch.bfloat16:
        autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast = torch.autocast("cpu", enabled=False)

    # warmup (allocator + kernel selection)
    for _ in range(args.warmup):
        with autocast:
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    for _ in range(args.steps):
        with autocast:
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)

    elapsed = time.perf_counter() - start

    tokens = args.micro_batch * args.seq * args.steps

    peak = (
        torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else float("nan")
    )

    return {
        "loss": float(loss.detach().float()),
        "tokens_per_second": tokens / elapsed,
        "peak_mib": peak,
    }


def main() -> int:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--preset", choices=["tiny", "3b", "20b"], default="tiny")

    parser.add_argument("--layers", type=int, default=4)

    parser.add_argument("--hidden", type=int, default=512)

    parser.add_argument("--heads", type=int, default=8)

    parser.add_argument("--vocab", type=int, default=32000)

    parser.add_argument("--seq", type=int, default=1024)

    parser.add_argument("--micro_batch", type=int, default=1)

    parser.add_argument("--sliding_window", type=int, default=None, help="override (default: keep the config's)")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--bf16", action="store_true", default=True)

    parser.add_argument("--steps", type=int, default=5)

    parser.add_argument("--warmup", type=int, default=2)

    args = parser.parse_args()

    config = build_config(args)

    if args.sliding_window is not None:
        config.sliding_window = args.sliding_window

    print(f"device={args.device} seq={args.seq} micro_batch={args.micro_batch} "
          f"layers={config.num_hidden_layers} hidden={config.hidden_size} vocab={config.vocab_size}")

    variants = [
        ("reference (full logits, math attention, no ckpt)", dict(
            return_logits=True, attention_backend="math", gradient_checkpointing_segments=0,
            use_gradient_checkpointing=False, loss_chunk_size=0,
        )),
        ("chunked loss only", dict(
            return_logits=False, attention_backend="math", gradient_checkpointing_segments=0,
            use_gradient_checkpointing=False, loss_chunk_size=1024,
        )),
        ("+ fused SDPA", dict(
            return_logits=False, attention_backend="auto", gradient_checkpointing_segments=0,
            use_gradient_checkpointing=False, loss_chunk_size=1024,
        )),
        ("+ segmented activation checkpointing", dict(
            return_logits=False, attention_backend="auto", gradient_checkpointing_segments=2,
            use_gradient_checkpointing=True, loss_chunk_size=1024,
        )),
    ]

    reference_loss = None

    for label, overrides in variants:

        result = run(config, args, **overrides)

        if reference_loss is None:
            reference_loss = result["loss"]

        delta = abs(result["loss"] - reference_loss)

        print(
            f"  {label:<52} {result['tokens_per_second']:10,.0f} tok/s   "
            f"peak {result['peak_mib']:8.0f} MiB   loss {result['loss']:.6f} (Δ {delta:.2e})"
        )

    return 0


if __name__ == "__main__":

    raise SystemExit(main())
