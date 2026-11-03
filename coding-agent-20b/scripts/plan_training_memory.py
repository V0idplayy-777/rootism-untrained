"""Report (and optionally apply) the memory/throughput plan for a recipe.

Examples
--------
    # what does the 3B recipe need per rank on 8 GPUs?
    python scripts/plan_training_memory.py --config configs/training_config_3b.json --gpus 8

    # I have 24 GB cards: show the batch/accumulation that fits, without editing files
    python scripts/plan_training_memory.py --config configs/training_config_3b.json \\
        --gpus 8 --budget_gb 24

    # write the resolved fields back into the recipe (comments/keys are preserved)
    python scripts/plan_training_memory.py --config configs/training_config_3b.json \\
        --gpus 8 --budget_gb 24 --apply

Nothing here changes the learning recipe: precision, learning rate, schedule,
global batch, sequence length and data order are inputs, never outputs.  The
plan only decides how much work is in flight at once (micro-batch,
accumulation, checkpointed segments, loss chunking).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import CodingAgentConfig, get_1b_config, get_20b_config  # noqa: E402
from model.config_3b import get_3b_config  # noqa: E402
from training.perf import estimate, largest_micro_batch, plan_recipe  # noqa: E402


def needs_window_scratch(model: CodingAgentConfig, seq_length: int) -> bool:
    """True when a sliding window shorter than the sequence forces the explicit path.

    See ``training/perf.py``: the model applies its window only in the explicit
    softmax branch, so a windowed run at this sequence length materialises the
    quadratic score matrix -- and therefore needs that memory budgeted.
    """

    window = getattr(model, "sliding_window", None)

    return bool(window) and window < seq_length


def model_for(recipe: dict) -> CodingAgentConfig:

    size = recipe.get("model_size", "20B")

    if size == "3B":
        return get_3b_config()

    if size == "1B":
        return get_1b_config()

    return get_20b_config()


def main() -> int:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--config", required=True, help="training recipe (configs/training_config*.json)")

    parser.add_argument("--gpus", type=int, default=8, help="total ranks in the job")

    parser.add_argument("--budget_gb", type=float, default=None, help="per-rank VRAM budget, GiB")

    parser.add_argument("--reserve_gb", type=float, default=1.5, help="headroom kept free, GiB")

    parser.add_argument("--apply", action="store_true", help="write the resolved fields into the recipe")

    args = parser.parse_args()

    path = Path(args.config)

    recipe = json.loads(path.read_text())

    model = model_for(recipe)

    batching = recipe["batching"]

    deepspeed = recipe["deepspeed"]

    training = recipe["training"]

    print(f"{path.name}: {recipe.get('model_name', 'model')} "
          f"({model.total_parameters() / 1e9:.2f}B params), {args.gpus} ranks")

    breakdown = estimate(
        model,
        world_size=args.gpus,
        micro_batch=batching["micro_batch_size_per_gpu"],
        seq_length=batching["sequence_length"],
        zero_stage=deepspeed.get("stage", 3),
        zero8bit_optimizer=bool(deepspeed.get("zero8bit_optimizer", False)),
        offload_optimizer=bool(deepspeed.get("offload_optimizer", False)),
        offload_param=bool(deepspeed.get("offload_param", False)),
        stage3_max_live_parameters=deepspeed.get("stage3_max_live_parameters"),
        activation_segments=deepspeed.get("activation_checkpointing_segments", 2),
        chunked_loss=bool(training.get("chunked_loss", True)),
        loss_chunk_size=int(training.get("loss_chunk_size", 1024)),
        fused_attention=training.get("attention_backend", "auto") != "math",
        window_materialised=needs_window_scratch(model, batching["sequence_length"]),
    )

    print(breakdown.report())

    if breakdown.extra.get("host_gib"):
        print(f"  host RAM (offloaded state)  {breakdown.extra['host_gib']:8.3f} GiB")

    if args.budget_gb is not None and breakdown.total / (1024**3) > args.budget_gb - args.reserve_gb:

        plan = plan_recipe(recipe, model, args.budget_gb, world_size=args.gpus, reserve_gb=args.reserve_gb)

        print("\ncurrent configuration does not fit the budget; plan:")

        print(json.dumps({k: v for k, v in plan.items() if k != "estimate_gib"}, indent=2))

        if plan.get("fits"):

            sequence_length = batching["sequence_length"]

            biggest = largest_micro_batch(
                model, args.budget_gb, world_size=args.gpus, seq_length=sequence_length
            )

            print(f"largest micro-batch that fits: {biggest}")

        else:

            print("still does not fit: add ranks, shorten sequence_length, or enable "
                  "deepspeed.offload_optimizer / offload_param in the recipe")

        if args.apply and plan.get("fits"):

            batching["micro_batch_size_per_gpu"] = plan["micro_batch_size_per_gpu"]

            batching["gradient_accumulation_steps"] = "auto"

            deepspeed["activation_checkpointing_segments"] = plan["activation_checkpointing_segments"]

            path.write_text(json.dumps(recipe, indent=2) + "\n")

            print(f"updated {path}")

    return 0


if __name__ == "__main__":

    raise SystemExit(main())
