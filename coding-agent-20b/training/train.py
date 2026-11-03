"""
Training Script for the 20B Parameter Coding Agent Model.

This is the main training entry point that:
1. Initializes the 20B model with DeepSpeed ZeRO-3
2. Sets up mixed precision training (BF16/FP16)
3. Configures activation/gradient checkpointing
4. Loads tokenized dataset shards
5. Runs the training loop with logging and checkpointing
6. Handles distributed training across multiple GPUs/nodes

Usage:
    # Single node, multiple GPUs:
    deepspeed --num_gpus=8 training/train.py --config configs/training_config.json

    # Multi-node:
    deepspeed --hostfile=hostfile.txt --num_gpus=8 training/train.py --config configs/training_config.json

    # Quick test with smaller model:
    python training/train.py --config configs/training_config.json --model_size 1b --max_steps 100
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


from dataset.prefetch import PrefetchStream, to_device
from dataset.processor import MemoryMappedDataset
from model.config import CodingAgentConfig, get_1b_config, get_7b_config, get_20b_config
from model.model import CodingAgentModel

try:

    import deepspeed
    from deepspeed.runtime.zero.stage3 import (
        estimate_zero3_model_states_mem_needs_all_live,
    )

    DEEPSPEED_AVAILABLE = True


except ImportError:

    DEEPSPEED_AVAILABLE = False

    print("WARNING: DeepSpeed not available. Install with: pip install deepspeed")


def setup_logging(rank: int = 0, log_dir: str = "logs"):
    """Setup logging with file and console handlers."""

    os.makedirs(log_dir, exist_ok=True)

    log_level = logging.INFO if rank == 0 else logging.WARNING

    logging.basicConfig(
        level=log_level,
        format=f"[%(asctime)s] [Rank {rank}] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"train_rank{rank}.log")),
            logging.StreamHandler() if rank == 0 else logging.NullHandler(),
        ],
    )

    return logging.getLogger(__name__)


class CosineDecayWithWarmup:
    """
    Learning rate schedule with linear warmup and cosine decay.

    This is the standard schedule for training large language models:
    1. Linear warmup from 0 to max_lr
    2. Cosine decay from max_lr to min_lr
    """

    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ):

        self.max_lr = max_lr

        self.min_lr = min_lr

        self.warmup_steps = warmup_steps

        self.total_steps = total_steps

    def get_lr(self, step: int) -> float:

        if step < self.warmup_steps:

            return self.max_lr * (step + 1) / self.warmup_steps

        if step >= self.total_steps:

            return self.min_lr

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)

        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )


@dataclass
class TrainingState:
    """Tracks training progress and metrics."""

    step: int = 0

    tokens_processed: int = 0

    epoch: int = 0

    best_val_loss: float = float("inf")

    train_loss_history: list = None

    val_loss_history: list = None

    throughput_history: list = None

    start_time: float = 0.0

    def __post_init__(self):

        if self.train_loss_history is None:

            self.train_loss_history = []

        if self.val_loss_history is None:

            self.val_loss_history = []

        if self.throughput_history is None:

            self.throughput_history = []


class StreamingDataLoader:
    """
    Efficient data loader for memory-mapped token datasets.

    Features:
    - Streams from disk without loading full dataset into RAM
    - Prefetches next batch while GPU processes current
    - Handles distributed data parallelism
    """

    def __init__(
        self,
        dataset: MemoryMappedDataset,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 42,
    ):

        self.dataset = dataset

        self.batch_size = batch_size

        self.rank = rank

        self.world_size = world_size

        self.shuffle = shuffle

        self.seed = seed

        self.epoch = 0

        self.total_sequences = len(dataset)

        self.sequences_per_rank = self.total_sequences // world_size

        self.start_idx = rank * self.sequences_per_rank

        self.end_idx = self.start_idx + self.sequences_per_rank

        self.batch_in_epoch = 0

    def __iter__(self):

        self.batch_in_epoch = 0

        indices = list(range(self.start_idx, self.end_idx))

        if self.shuffle:

            rng = np.random.RandomState(self.seed + self.epoch)

            rng.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):

            batch_indices = indices[i : i + self.batch_size]

            if len(batch_indices) < self.batch_size:

                continue

            tokens = torch.from_numpy(self.dataset.batch_tokens(batch_indices))

            self.batch_in_epoch += 1

            # One token tensor, two shifted views: the old path copied the
            # whole batch twice (once per shift) and upcast both to int64.
            yield {
                "input_ids": tokens[:, :-1],
                "labels": tokens[:, 1:],
            }

        self.epoch += 1

    @property
    def position(self) -> tuple[int, int]:
        """Epoch and batch position of the *next* yielded batch."""

        return self.epoch, self.batch_in_epoch

    def __len__(self) -> int:

        return self.sequences_per_rank // self.batch_size


def create_model(config: CodingAgentConfig) -> CodingAgentModel:
    """Create and initialize the model."""

    model = CodingAgentModel(config)

    if config.use_gradient_checkpointing:

        model.enable_gradient_checkpointing(config.gradient_checkpointing_ratio)

    return model


def resolve_accumulation(batching_config: dict, micro_batch: int, world_size: int) -> int:
    """Gradient-accumulation window that makes the global batch land exactly.

    ``"auto"`` in the recipe means "derive it from the job's real world size".
    A wrong window is the difference between the intended global batch and
    whatever this rank happened to hold, so the divisibility check is fatal
    rather than silently rounded.
    """

    global_batch = batching_config.get("global_batch_size")

    configured = batching_config.get("gradient_accumulation_steps", "auto")

    if not global_batch or configured != "auto":

        return int(configured if configured != "auto" else 1) or 1

    if global_batch % (micro_batch * world_size):

        raise ValueError(
            f"global_batch_size={global_batch} is not divisible by "
            f"micro_batch_size_per_gpu * world_size={micro_batch * world_size}"
        )

    return global_batch // (micro_batch * world_size)


def initialize_deepspeed(
    model: CodingAgentModel,
    config: dict[str, Any],
    ds_config_path: str,
    args: argparse.Namespace,
):
    """
    Initialize DeepSpeed engine with ZeRO-3, mixed precision,
    and activation checkpointing.
    """

    with open(ds_config_path) as f:

        ds_config = json.load(f)

    training_config = config.get("training", {})

    batching_config = config.get("batching", {})

    ds_config["train_micro_batch_size_per_gpu"] = batching_config.get(
        "micro_batch_size_per_gpu", 2
    )

    ds_config["gradient_accumulation_steps"] = batching_config.get(
        "gradient_accumulation_steps", 1
    )

    # "auto" in the recipe means "derive it": the accumulation window must make
    # the global batch land exactly, using all ranks in the job (not the
    # --num_gpus hint, which is only the per-node default).
    world_size = int(os.environ.get("WORLD_SIZE", args.num_gpus))

    micro_batch = ds_config["train_micro_batch_size_per_gpu"]

    ds_config["gradient_accumulation_steps"] = resolve_accumulation(
        batching_config, micro_batch, world_size
    )

    ds_config["train_batch_size"] = (
        ds_config["train_micro_batch_size_per_gpu"]
        * ds_config["gradient_accumulation_steps"]
        * world_size
    )

    if "optimizer" in ds_config:

        ds_config["optimizer"]["params"]["lr"] = training_config.get(
            "learning_rate", 3e-4
        )

    if "scheduler" in ds_config:

        total_steps = config.get("training", {}).get("total_steps", 100000)

        warmup_steps = config.get("training", {}).get("warmup_steps", 2000)

        ds_config["scheduler"]["params"]["warmup_num_steps"] = warmup_steps

        ds_config["scheduler"]["params"]["total_num_steps"] = total_steps

        ds_config["scheduler"]["params"]["warmup_max_lr"] = training_config.get(
            "learning_rate", 3e-4
        )

    zero = ds_config.setdefault("zero_optimization", {})

    for key in ("offload_optimizer", "offload_param"):

        enabled = config.get("deepspeed", {}).get(key, False)

        if isinstance(zero.get(key), dict):
            zero[key]["device"] = "cpu" if enabled else "none"
        else:
            zero[key] = {"device": "cpu" if enabled else "none"}

    for key, auto in (
        ("reduce_bucket_size", model.config.hidden_size * model.config.hidden_size),
        ("stage3_prefetch_bucket_size", int(0.9 * model.config.hidden_size * model.config.hidden_size)),
    ):
        if zero.get(key) == "auto":
            zero[key] = auto

    for key in ("stage3_max_live_parameters", "stage3_max_reuse_distance"):
        value = config.get("deepspeed", {}).get(key)
        if value is not None:
            zero[key] = value

    if config.get("deepspeed", {}).get("zero8bit_optimizer") and "optimizer" in ds_config:
        ds_config["optimizer"]["type"] = "AdamW8bit"

    # The model checkpoints its own segments (see
    # CodingAgentModel._checkpointed_layer); leaving DeepSpeed's
    # activation-checkpointing block in place would recompute every block twice.
    ds_config.pop("activation_checkpointing", None)
    ds_config.pop("flops_profiler", None)

    precision = training_config.get("precision", "bf16")

    if precision == "bf16":

        ds_config["bf16"] = {"enabled": True}

        ds_config["fp16"] = {"enabled": False}

    else:

        ds_config["bf16"] = {"enabled": False}

        ds_config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
        }

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        config=ds_config,
    )

    model_engine.optimizer.accumulation = int(ds_config["gradient_accumulation_steps"])

    return model_engine, optimizer, scheduler, ds_config


@torch.no_grad()
def evaluate(
    model,
    eval_loader: StreamingDataLoader,
    max_batches: int = 50,
    device: torch.device = None,
) -> dict[str, float]:
    """Run evaluation and compute metrics."""

    model.eval()

    total_loss = 0.0

    total_tokens = 0

    num_batches = 0

    for batch in eval_loader:

        if num_batches >= max_batches:

            break

        input_ids = to_device(batch["input_ids"], device)

        labels = to_device(batch["labels"], device)

        if hasattr(model, "module"):

            output = model.module(input_ids=input_ids, labels=labels)

        else:

            output = model(input_ids=input_ids, labels=labels)

        loss = output.loss if hasattr(output, "loss") else output[0]

        total_loss += loss.item() * labels.numel()

        total_tokens += labels.numel()

        num_batches += 1

    model.train()

    avg_loss = total_loss / max(total_tokens, 1)

    perplexity = math.exp(min(avg_loss, 20))

    return {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
        "eval_tokens": total_tokens,
        "eval_batches": num_batches,
    }


class CheckpointManager:
    """Manages model checkpoints with rotation."""

    def __init__(
        self,
        save_dir: str,
        keep_last_n: int = 5,
        rank: int = 0,
    ):

        self.save_dir = Path(save_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = keep_last_n

        self.rank = rank

    def save(
        self,
        model,
        optimizer,
        state: TrainingState,
        config: CodingAgentConfig,
        step: int,
    ):
        """Save a checkpoint."""

        if self.rank != 0:

            return

        ckpt_dir = self.save_dir / f"step_{step:08d}"

        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "save_checkpoint"):

            model.save_checkpoint(str(ckpt_dir), tag=f"step_{step}")

        else:

            torch.save(model.state_dict(), ckpt_dir / "model.pt")

        state_dict = {
            "step": state.step,
            "tokens_processed": state.tokens_processed,
            "epoch": state.epoch,
            "best_val_loss": state.best_val_loss,
            "train_loss_history": state.train_loss_history[-1000:],
            "val_loss_history": state.val_loss_history[-100:],
        }

        torch.save(state_dict, ckpt_dir / "training_state.pt")

        config.save(str(ckpt_dir / "config.json"))

        self._rotate_checkpoints()

        logging.getLogger(__name__).info(f"Checkpoint saved: {ckpt_dir}")

    def _rotate_checkpoints(self):
        """Keep only the last N checkpoints."""

        checkpoints = sorted(self.save_dir.glob("step_*"))

        while len(checkpoints) > self.keep_last_n:

            old = checkpoints.pop(0)

            import shutil

            shutil.rmtree(old)

            logging.getLogger(__name__).info(f"Removed old checkpoint: {old}")

    def load_latest(self, model, optimizer=None) -> TrainingState | None:
        """Load the latest checkpoint."""

        checkpoints = sorted(self.save_dir.glob("step_*"))

        if not checkpoints:

            return None

        latest = checkpoints[-1]

        if hasattr(model, "load_checkpoint"):

            model.load_checkpoint(str(latest))

        else:

            model_path = latest / "model.pt"

            if model_path.exists():

                model.load_state_dict(torch.load(model_path, map_location="cpu"))

        state_path = latest / "training_state.pt"

        if state_path.exists():

            state_dict = torch.load(state_path, map_location="cpu")

            state = TrainingState(**state_dict)

            return state

        return None


def train(args: argparse.Namespace):
    """Main training function."""

    with open(args.config) as f:
        recipe = json.load(f)
    recipe_size = recipe.get("model_size", "20B")
    model_size = args.model_size or recipe_size
    if recipe_size == "3B" or model_size == "3B":
        if recipe_size != "3B" or model_size != "3B":
            raise ValueError("The 3B model and training configuration must be used together")
        from training.train_3b import train_3b

        return train_3b(args, recipe)

    if DEEPSPEED_AVAILABLE:

        deepspeed.init_distributed()

    rank = int(os.environ.get("RANK", 0))

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    logger = setup_logging(rank=rank)

    if rank == 0:

        logger.info("=" * 60)

        logger.info("  CODING AGENT MODEL - TRAINING START")

        logger.info("=" * 60)

    with open(args.config) as f:

        config = json.load(f)

    model_size = args.model_size or config.get("model_size", "20B")

    if model_size == "20B":

        model_config = get_20b_config()

    elif model_size == "7B":

        model_config = get_7b_config()

    elif model_size == "1B":

        model_config = get_1b_config()

    else:

        model_config = get_20b_config()

    deepspeed_config = config["deepspeed"]

    training_config = config["training"]

    model_config.use_gradient_checkpointing = deepspeed_config["activation_checkpointing"]

    model_config.mixed_precision = training_config["precision"]

    model_config.deepspeed_stage = deepspeed_config["stage"]

    # Lossless throughput switches, driven entirely by the recipe.
    model_config.return_logits = False

    model_config.attention_backend = str(training_config.get("attention_backend", "auto"))

    model_config.logit_softcap = training_config.get("logit_softcap")

    model_config.loss_chunk_size = int(
        training_config.get("loss_chunk_size", model_config.loss_chunk_size)
    )

    segments = deepspeed_config.get("activation_checkpointing_segments")

    model_config.gradient_checkpointing_segments = (
        int(segments) if segments is not None else (2 if model_config.use_gradient_checkpointing else 0)
    )

    if rank == 0:

        model_config.print_summary()

        logger.info(f"World size: {world_size} GPUs")

    logger.info("Creating model...")

    model = create_model(model_config)

    if rank == 0:

        total_params = model.num_parameters()

        logger.info(f"Model parameters: {total_params:,} ({total_params / 1e9:.2f}B)")

    if DEEPSPEED_AVAILABLE and rank == 0:

        try:

            estimate_zero3_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=args.num_gpus,
                num_nodes=world_size // args.num_gpus,
            )

        except Exception as e:

            logger.warning(f"Memory estimation failed: {e}")

    ds_config_path = os.path.join(
        os.path.dirname(args.config),
        config["deepspeed"]["config_file"],
    )

    if not os.path.isabs(ds_config_path):

        ds_config_path = os.path.join(
            os.path.dirname(os.path.abspath(args.config)), "..", ds_config_path
        )

    logger.info(f"Initializing DeepSpeed with config: {ds_config_path}")

    model_engine, optimizer, _scheduler, ds_config = initialize_deepspeed(
        model, config, ds_config_path, args
    )

    ds_config_accumulation = int(ds_config["gradient_accumulation_steps"])

    tokens_per_optimizer_step = (
        ds_config["train_batch_size"] * config["batching"]["sequence_length"]
    )

    device = (
        model_engine.device
        if hasattr(model_engine, "device")
        else torch.device(f"cuda:{local_rank}")
    )

    logger.info("Loading datasets...")

    data_dir = config["data"]["data_dir"]

    train_dataset = MemoryMappedDataset(
        data_dir, split="train", seq_length=config["batching"]["sequence_length"]
    )

    eval_dataset = MemoryMappedDataset(
        data_dir, split="val", seq_length=config["batching"]["sequence_length"]
    )

    train_loader = StreamingDataLoader(
        train_dataset,
        batch_size=config["batching"]["micro_batch_size_per_gpu"],
        rank=rank,
        world_size=world_size,
        seed=config["training"]["seed"],
    )

    eval_loader = StreamingDataLoader(
        eval_dataset,
        batch_size=config["evaluation"]["eval_batch_size"],
        rank=rank,
        world_size=world_size,
        shuffle=False,
    )

    # Reads + host->device staging on a background thread, bounded queue.
    prefetch_train = PrefetchStream(
        train_loader,
        depth=int(config["data"].get("prefetch_depth", 2) or 0),
        pin_memory=bool(config["data"].get("pin_memory", True)),
        name=f"20b-prefetch-rank{rank}",
    )

    training_config = config["training"]

    total_tokens = training_config["total_tokens"]

    seq_length = config["batching"]["sequence_length"]

    global_batch_size = config["batching"]["global_batch_size"]

    tokens_per_step = global_batch_size * seq_length

    total_steps = total_tokens // tokens_per_step

    warmup_steps = training_config["warmup_tokens"] // tokens_per_step

    lr_scheduler = CosineDecayWithWarmup(
        max_lr=training_config["learning_rate"],
        min_lr=training_config["min_learning_rate"],
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    if rank == 0:

        logger.info(f"Total training steps: {total_steps:,}")

        logger.info(f"Warmup steps: {warmup_steps:,}")

        logger.info(f"Tokens per step: {tokens_per_step:,}")

    ckpt_manager = CheckpointManager(
        save_dir=config["checkpointing"]["save_dir"],
        keep_last_n=config["checkpointing"]["keep_last_n"],
        rank=rank,
    )

    state = TrainingState(start_time=time.time())

    if args.resume:

        loaded_state = ckpt_manager.load_latest(model_engine, optimizer)

        if loaded_state:

            state = loaded_state

            logger.info(f"Resumed from step {state.step}")

    logger.info("Starting training loop...")

    model_engine.train()

    tokens_per_batch = (
        config["batching"]["micro_batch_size_per_gpu"] * seq_length * world_size
    )

    grad_accum_steps = ds_config_accumulation

    if grad_accum_steps < 1:
        raise ValueError("gradient_accumulation_steps must be resolved to a positive integer")

    train_iter = iter(prefetch_train)

    step_start_time = time.time()

    max_steps = args.max_steps or total_steps

    optimizer_steps = 0

    last_loss = float("nan")

    while state.step < max_steps:

        lr = lr_scheduler.get_lr(optimizer_steps)

        for param_group in optimizer.param_groups:

            param_group["lr"] = lr

        micro_losses = []

        for _ in range(grad_accum_steps):

            try:

                batch = next(train_iter)

            except StopIteration:

                # Only reachable if the prefetcher was built without restarts.
                state.epoch += 1

                prefetch_train.close()

                prefetch_train = PrefetchStream(
                    train_loader,
                    depth=int(config["data"].get("prefetch_depth", 2) or 0),
                    pin_memory=bool(config["data"].get("pin_memory", True)),
                    name=f"20b-prefetch-rank{rank}",
                )

                train_iter = iter(prefetch_train)

                batch = next(train_iter)

            input_ids = to_device(batch["input_ids"], device)

            labels = to_device(batch["labels"], device)

            output = model_engine(input_ids=input_ids, labels=labels)

            loss = output.loss if hasattr(output, "loss") else output[0]

            # DeepSpeed applies gradient accumulation internally, so backward
            # takes the unscaled micro-loss.  Scaling it here *and* letting the
            # engine divide again shrank every update by 1/accumulation, while
            # the optimizer step fired every micro-batch instead of every
            # accumulation window.
            model_engine.backward(loss)

            micro_losses.append(loss.detach())

            state.step += 1

            state.tokens_processed += tokens_per_batch

            if state.step % grad_accum_steps == 0:

                model_engine.step()

                optimizer_steps += 1

                last_loss = float(torch.stack(micro_losses).mean())

                state.train_loss_history.append(last_loss)

                micro_losses = []

                if optimizer_steps % config["logging"]["log_every_steps"] == 0 and rank == 0:

                    elapsed = time.time() - step_start_time

                    tokens_per_sec = (
                        tokens_per_optimizer_step
                        * config["logging"]["log_every_steps"]
                        / max(elapsed, 1e-9)
                    )

                    step_start_time = time.time()

                    throughput = tokens_per_sec / 1e6

                    state.throughput_history.append(throughput)

                    recent_losses = state.train_loss_history[-100:]

                    avg_loss = sum(recent_losses) / len(recent_losses)

                    elapsed_total = time.time() - state.start_time

                    optimizer_steps_remaining = max(
                        max_steps // grad_accum_steps - optimizer_steps, 0
                    )

                    eta_hours = (
                        elapsed_total / max(optimizer_steps, 1)
                    ) * optimizer_steps_remaining / 3600

                    logger.info(
                        f"Step {state.step:,}/{max_steps:,} "
                        f"({optimizer_steps:,} optimizer steps) | "
                        f"Loss: {last_loss:.4f} (avg: {avg_loss:.4f}) | "
                        f"LR: {lr:.2e} | "
                        f"Throughput: {throughput:.2f}M tok/s | "
                        f"Tokens: {state.tokens_processed / 1e9:.2f}B | "
                        f"Epoch: {state.epoch} | "
                        f"ETA: {eta_hours:.1f}h"
                    )

        if (
            config["logging"]["eval_every_steps"] > 0
            and optimizer_steps > 0
            and optimizer_steps % config["logging"]["eval_every_steps"] == 0
        ):

            eval_metrics = evaluate(
                model_engine,
                eval_loader,
                max_batches=config["evaluation"]["eval_tokens"]
                // (config["evaluation"]["eval_batch_size"] * seq_length),
                device=device,
            )

            state.val_loss_history.append(eval_metrics["eval_loss"])

            if rank == 0:

                logger.info(
                    f"EVAL Step {state.step} | "
                    f"Loss: {eval_metrics['eval_loss']:.4f} | "
                    f"Perplexity: {eval_metrics['eval_perplexity']:.2f}"
                )

                if eval_metrics["eval_loss"] < state.best_val_loss:

                    state.best_val_loss = eval_metrics["eval_loss"]

                    logger.info(f"New best validation loss: {state.best_val_loss:.4f}")

            model_engine.train()

        save_every = config["checkpointing"]["save_every_steps"]

        if save_every > 0 and optimizer_steps > 0 and optimizer_steps % save_every == 0:

            ckpt_manager.save(model_engine, optimizer, state, model_config, state.step)

    prefetch_train.close()

    if rank == 0:

        logger.info("Training complete! Saving final checkpoint...")

        ckpt_manager.save(model_engine, optimizer, state, model_config, state.step)

        total_time = time.time() - state.start_time

        logger.info(f"Total training time: {total_time / 3600:.1f} hours")

        logger.info(f"Total tokens processed: {state.tokens_processed / 1e9:.2f}B")

        if state.train_loss_history:

            logger.info(f"Final train loss: {state.train_loss_history[-1]:.4f}")

        if state.val_loss_history:

            logger.info(f"Best val loss: {state.best_val_loss:.4f}")


def parse_args():

    parser = argparse.ArgumentParser(description="Train the Coding Agent Model")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.json",
        help="Path to training configuration file",
    )

    parser.add_argument(
        "--model_size",
        type=str,
        choices=["1B", "3B", "7B", "20B"],
        default=None,
        help="Override model size",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (3B cannot exceed the configured token budget)",
    )

    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )

    parser.add_argument(
        "--num_gpus", type=int, default=8, help="Number of GPUs per node"
    )

    parser.add_argument(
        "--local_rank", "--local-rank", type=int, default=-1, help="Local rank for distributed training"
    )

    if DEEPSPEED_AVAILABLE:

        parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    train(args)
