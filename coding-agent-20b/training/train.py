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

import os
import sys
import json
import time
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import CodingAgentConfig, get_20b_config, get_7b_config, get_1b_config
from model.model import CodingAgentModel
from dataset.processor import MemoryMappedDataset

# DeepSpeed import (will be available during training)
try:
    import deepspeed
    from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("WARNING: DeepSpeed not available. Install with: pip install deepspeed")


# ============================================================
# LOGGING SETUP
# ============================================================

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


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

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
            # Linear warmup
            return self.max_lr * (step + 1) / self.warmup_steps

        if step >= self.total_steps:
            return self.min_lr

        # Cosine decay
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================
# TRAINING STATE
# ============================================================

@dataclass
class TrainingState:
    """Tracks training progress and metrics."""
    step: int = 0
    tokens_processed: int = 0
    epoch: int = 0
    best_val_loss: float = float('inf')
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


# ============================================================
# DATA LOADER
# ============================================================

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

        # Total sequences across all ranks
        self.total_sequences = len(dataset)
        self.sequences_per_rank = self.total_sequences // world_size
        self.start_idx = rank * self.sequences_per_rank
        self.end_idx = self.start_idx + self.sequences_per_rank

    def __iter__(self):
        indices = list(range(self.start_idx, self.end_idx))

        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            rng.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if len(batch_indices) < self.batch_size:
                continue

            inputs, targets = self.dataset.get_batch(self.batch_size, batch_indices)
            yield {
                "input_ids": torch.from_numpy(inputs).long(),
                "labels": torch.from_numpy(targets).long(),
            }

        self.epoch += 1

    def __len__(self) -> int:
        return self.sequences_per_rank // self.batch_size


# ============================================================
# MODEL INITIALIZATION
# ============================================================

def create_model(config: CodingAgentConfig) -> CodingAgentModel:
    """Create and initialize the model."""
    model = CodingAgentModel(config)

    if config.use_gradient_checkpointing:
        model.enable_gradient_checkpointing(config.gradient_checkpointing_ratio)

    return model


def initialize_deepspeed(
    model: CodingAgentModel,
    config: Dict[str, Any],
    ds_config_path: str,
    args: argparse.Namespace,
):
    """
    Initialize DeepSpeed engine with ZeRO-3, mixed precision,
    and activation checkpointing.
    """
    # Load DeepSpeed config
    with open(ds_config_path) as f:
        ds_config = json.load(f)

    # Override with training config values
    training_config = config.get("training", {})
    batching_config = config.get("batching", {})

    ds_config["train_micro_batch_size_per_gpu"] = batching_config.get("micro_batch_size_per_gpu", 2)
    ds_config["gradient_accumulation_steps"] = batching_config.get("gradient_accumulation_steps", 1)
    ds_config["train_batch_size"] = (
        ds_config["train_micro_batch_size_per_gpu"]
        * ds_config["gradient_accumulation_steps"]
        * args.num_gpus
    )

    # Set optimizer LR
    if "optimizer" in ds_config:
        ds_config["optimizer"]["params"]["lr"] = training_config.get("learning_rate", 3e-4)

    # Set scheduler
    if "scheduler" in ds_config:
        total_steps = config.get("training", {}).get("total_steps", 100000)
        warmup_steps = config.get("training", {}).get("warmup_steps", 2000)
        ds_config["scheduler"]["params"]["warmup_num_steps"] = warmup_steps
        ds_config["scheduler"]["params"]["total_num_steps"] = total_steps
        ds_config["scheduler"]["params"]["warmup_max_lr"] = training_config.get("learning_rate", 3e-4)

    # BF16/FP16 settings
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

    # Create dummy parameter groups for DeepSpeed
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # Initialize DeepSpeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        config=ds_config,
    )

    return model_engine, optimizer, scheduler


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    eval_loader: StreamingDataLoader,
    max_batches: int = 50,
    device: torch.device = None,
) -> Dict[str, float]:
    """Run evaluation and compute metrics."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for batch in eval_loader:
        if num_batches >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        if hasattr(model, 'module'):
            output = model.module(input_ids=input_ids, labels=labels)
        else:
            output = model(input_ids=input_ids, labels=labels)

        loss = output.loss if hasattr(output, 'loss') else output[0]

        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()
        num_batches += 1

    model.train()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))  # Cap to prevent overflow

    return {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
        "eval_tokens": total_tokens,
        "eval_batches": num_batches,
    }


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

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

        # Save model
        if hasattr(model, 'save_checkpoint'):
            # DeepSpeed model
            model.save_checkpoint(str(ckpt_dir), tag=f"step_{step}")
        else:
            torch.save(model.state_dict(), ckpt_dir / "model.pt")

        # Save training state
        state_dict = {
            "step": state.step,
            "tokens_processed": state.tokens_processed,
            "epoch": state.epoch,
            "best_val_loss": state.best_val_loss,
            "train_loss_history": state.train_loss_history[-1000:],
            "val_loss_history": state.val_loss_history[-100:],
        }
        torch.save(state_dict, ckpt_dir / "training_state.pt")

        # Save config
        config.save(str(ckpt_dir / "config.json"))

        # Rotate old checkpoints
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

    def load_latest(self, model, optimizer=None) -> Optional[TrainingState]:
        """Load the latest checkpoint."""
        checkpoints = sorted(self.save_dir.glob("step_*"))
        if not checkpoints:
            return None

        latest = checkpoints[-1]

        # Load model
        if hasattr(model, 'load_checkpoint'):
            model.load_checkpoint(str(latest))
        else:
            model_path = latest / "model.pt"
            if model_path.exists():
                model.load_state_dict(torch.load(model_path, map_location='cpu'))

        # Load training state
        state_path = latest / "training_state.pt"
        if state_path.exists():
            state_dict = torch.load(state_path, map_location='cpu')
            state = TrainingState(**state_dict)
            return state

        return None


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train(args: argparse.Namespace):
    """Main training function."""
    # Initialize distributed
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

    # Load training config
    with open(args.config) as f:
        config = json.load(f)

    # Create model config
    model_size = args.model_size or config.get("model_size", "20B")
    if model_size == "20B":
        model_config = get_20b_config()
    elif model_size == "7B":
        model_config = get_7b_config()
    elif model_size == "1B":
        model_config = get_1b_config()
    else:
        model_config = get_20b_config()

    model_config.use_gradient_checkpointing = config["deepspeed"]["activation_checkpointing"]
    model_config.mixed_precision = config["training"]["precision"]
    model_config.deepspeed_stage = config["deepspeed"]["stage"]

    if rank == 0:
        model_config.print_summary()
        logger.info(f"World size: {world_size} GPUs")

    # Create model
    logger.info("Creating model...")
    model = create_model(model_config)

    if rank == 0:
        total_params = model.num_parameters()
        logger.info(f"Model parameters: {total_params:,} ({total_params / 1e9:.2f}B)")

    # Memory estimation
    if DEEPSPEED_AVAILABLE and rank == 0:
        try:
            estimate_zero3_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=args.num_gpus,
                num_nodes=world_size // args.num_gpus,
            )
        except Exception as e:
            logger.warning(f"Memory estimation failed: {e}")

    # Initialize DeepSpeed
    ds_config_path = os.path.join(
        os.path.dirname(args.config),
        config["deepspeed"]["config_file"],
    )
    if not os.path.isabs(ds_config_path):
        ds_config_path = os.path.join(os.path.dirname(os.path.abspath(args.config)), "..", ds_config_path)

    logger.info(f"Initializing DeepSpeed with config: {ds_config_path}")
    model_engine, optimizer, scheduler = initialize_deepspeed(
        model, config, ds_config_path, args
    )

    device = model_engine.device if hasattr(model_engine, 'device') else torch.device(f'cuda:{local_rank}')

    # Load datasets
    logger.info("Loading datasets...")
    data_dir = config["data"]["data_dir"]

    train_dataset = MemoryMappedDataset(data_dir, split="train", seq_length=config["batching"]["sequence_length"])
    eval_dataset = MemoryMappedDataset(data_dir, split="val", seq_length=config["batching"]["sequence_length"])

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

    # Learning rate scheduler
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

    # Checkpoint manager
    ckpt_manager = CheckpointManager(
        save_dir=config["checkpointing"]["save_dir"],
        keep_last_n=config["checkpointing"]["keep_last_n"],
        rank=rank,
    )

    # Training state
    state = TrainingState(start_time=time.time())

    # Try to resume from checkpoint
    if args.resume:
        loaded_state = ckpt_manager.load_latest(model_engine, optimizer)
        if loaded_state:
            state = loaded_state
            logger.info(f"Resumed from step {state.step}")

    # ============ TRAINING LOOP ============
    logger.info("Starting training loop...")
    model_engine.train()

    tokens_per_batch = config["batching"]["micro_batch_size_per_gpu"] * seq_length * world_size
    grad_accum_steps = config["batching"]["gradient_accumulation_steps"]

    train_iter = iter(train_loader)
    step_start_time = time.time()

    max_steps = args.max_steps or total_steps

    while state.step < max_steps:
        # Get next batch (restart epoch if needed)
        try:
            batch = next(train_iter)
        except StopIteration:
            state.epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Move to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Update learning rate
        lr = lr_scheduler.get_lr(state.step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        output = model_engine(input_ids=input_ids, labels=labels)
        loss = output.loss if hasattr(output, 'loss') else output[0]

        # Scale loss for gradient accumulation
        loss = loss / grad_accum_steps

        # Backward pass
        model_engine.backward(loss)

        # Optimizer step (DeepSpeed handles gradient accumulation internally)
        if (state.step + 1) % grad_accum_steps == 0:
            model_engine.step()

        # Update state
        state.step += 1
        state.tokens_processed += tokens_per_batch
        loss_value = loss.item() * grad_accum_steps
        state.train_loss_history.append(loss_value)

        # Logging
        if state.step % config["logging"]["log_every_steps"] == 0 and rank == 0:
            elapsed = time.time() - step_start_time
            tokens_per_sec = tokens_per_batch * config["logging"]["log_every_steps"] / elapsed
            step_start_time = time.time()

            throughput = tokens_per_sec / 1e6  # M tokens/sec
            state.throughput_history.append(throughput)

            # Recent average loss
            recent_losses = state.train_loss_history[-100:]
            avg_loss = sum(recent_losses) / len(recent_losses)

            elapsed_total = time.time() - state.start_time
            hours_elapsed = elapsed_total / 3600
            steps_remaining = max_steps - state.step
            eta_hours = (elapsed_total / max(state.step, 1)) * steps_remaining / 3600

            logger.info(
                f"Step {state.step:,}/{max_steps:,} | "
                f"Loss: {loss_value:.4f} (avg: {avg_loss:.4f}) | "
                f"LR: {lr:.2e} | "
                f"Throughput: {throughput:.2f}M tok/s | "
                f"Tokens: {state.tokens_processed / 1e9:.2f}B | "
                f"Epoch: {state.epoch} | "
                f"ETA: {eta_hours:.1f}h"
            )

        # Evaluation
        if state.step % config["logging"]["eval_every_steps"] == 0:
            eval_metrics = evaluate(
                model_engine,
                eval_loader,
                max_batches=config["evaluation"]["eval_tokens"] // (config["evaluation"]["eval_batch_size"] * seq_length),
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

        # Checkpointing
        save_every = config["checkpointing"]["save_every_steps"]
        if save_every > 0 and state.step % save_every == 0:
            ckpt_manager.save(model_engine, optimizer, state, model_config, state.step)

    # Final checkpoint
    if rank == 0:
        logger.info("Training complete! Saving final checkpoint...")
        ckpt_manager.save(model_engine, optimizer, state, model_config, state.step)

        total_time = time.time() - state.start_time
        logger.info(f"Total training time: {total_time / 3600:.1f} hours")
        logger.info(f"Total tokens processed: {state.tokens_processed / 1e9:.2f}B")
        logger.info(f"Final train loss: {state.train_loss_history[-1]:.4f}")
        if state.val_loss_history:
            logger.info(f"Best val loss: {state.best_val_loss:.4f}")


# ============================================================
# ENTRY POINT
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Coding Agent Model")

    parser.add_argument("--config", type=str, default="configs/training_config.json",
                        help="Path to training configuration file")
    parser.add_argument("--model_size", type=str, choices=["1B", "7B", "20B"], default=None,
                        help="Override model size")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum training steps (overrides config)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs per node")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")

    # DeepSpeed will add its own arguments
    if DEEPSPEED_AVAILABLE:
        parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
