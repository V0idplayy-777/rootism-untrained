"""3B-only DeepSpeed pretraining with a hard, global input-token budget.

The shared staged-pipeline configuration is deliberately not used here. This
entry point consumes already-tokenized, pre-mixed train/val shards and counts
complete global batches, including gradient accumulation across all ranks.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import shutil
import time
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from dataset.processor import MemoryMappedDataset
from model.config_3b import get_3b_config
from model.model import CodingAgentModel
from training.schedules import cosine_lr


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingPlan:
    world_size: int
    global_batch_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    sequence_length: int
    tokens_per_step: int
    total_steps: int
    warmup_steps: int
    run_steps: int

    @property
    def planned_tokens(self) -> int:
        return self.total_steps * self.tokens_per_step


def build_training_plan(config: dict, world_size: int, max_steps: int | None = None) -> TrainingPlan:
    """Resolve accumulation using the *total* number of distributed ranks."""
    if config.get("model_size") != "3B":
        raise ValueError("The 3B trainer requires a config with model_size='3B'")
    batch = config["batching"]
    train = config["training"]
    global_batch = batch["global_batch_size"]
    micro_batch = batch["micro_batch_size_per_gpu"]
    seq_length = batch["sequence_length"]
    total_tokens = train["total_tokens"]
    warmup_tokens = train["warmup_tokens"]
    for name, value in (
        ("world_size", world_size),
        ("global_batch_size", global_batch),
        ("micro_batch_size_per_gpu", micro_batch),
        ("sequence_length", seq_length),
        ("total_tokens", total_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if seq_length < 2:
        raise ValueError("sequence_length must be at least 2 for causal language modeling")
    if isinstance(warmup_tokens, bool) or not isinstance(warmup_tokens, int) or not 0 <= warmup_tokens < total_tokens:
        raise ValueError("warmup_tokens must be a non-negative integer smaller than total_tokens")

    global_micro_batch = micro_batch * world_size
    if global_batch % global_micro_batch:
        raise ValueError(
            f"global_batch_size={global_batch} must be divisible by "
            f"micro_batch_size_per_gpu * world_size={global_micro_batch}"
        )
    accumulation = global_batch // global_micro_batch
    configured_accumulation = batch.get("gradient_accumulation_steps", "auto")
    if configured_accumulation != "auto" and (
        isinstance(configured_accumulation, bool)
        or not isinstance(configured_accumulation, int)
        or configured_accumulation != accumulation
    ):
        raise ValueError(f"gradient_accumulation_steps must be 'auto' or {accumulation}")

    tokens_per_step = global_batch * seq_length
    total_steps = total_tokens // tokens_per_step
    if total_steps < 1:
        raise ValueError("total_tokens must cover at least one complete global batch")
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0):
        raise ValueError("max_steps must be a positive integer")
    run_steps = min(total_steps, max_steps) if max_steps is not None else total_steps
    warmup_steps = min(warmup_tokens // tokens_per_step, total_steps - 1)
    return TrainingPlan(
        world_size,
        global_batch,
        micro_batch,
        accumulation,
        seq_length,
        tokens_per_step,
        total_steps,
        warmup_steps,
        run_steps,
    )


def resolve_deepspeed_path(config_path: str, configured_path: str) -> Path:
    """Resolve recipe paths both from the project directory and config location."""
    path = Path(configured_path).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        recipe_dir = Path(config_path).resolve().parent
        candidates = [Path.cwd() / path, recipe_dir / path, recipe_dir.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"DeepSpeed config {configured_path!r} not found (recipe: {config_path})")


def build_deepspeed_config(template: dict, config: dict, plan: TrainingPlan, hidden_size: int) -> dict:
    """Resolve 3B-specific DeepSpeed settings without modifying the shared file."""
    result = copy.deepcopy(template)
    train = config["training"]
    ds = config["deepspeed"]
    if ds["stage"] != 3:
        raise ValueError("The 3B pretrainer currently requires DeepSpeed ZeRO-3")
    result["train_batch_size"] = plan.global_batch_size
    result["train_micro_batch_size_per_gpu"] = plan.micro_batch_size
    result["gradient_accumulation_steps"] = plan.gradient_accumulation_steps
    result["gradient_clipping"] = train["max_grad_norm"]

    precision = train["precision"].lower()
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"Unsupported precision: {precision}")
    result["bf16"] = {"enabled": precision == "bf16"}
    result["fp16"] = {"enabled": precision == "fp16"}
    result.pop("communication_data_type", None)

    zero = result.setdefault("zero_optimization", {})
    zero["stage"] = 3
    for key, enabled in (("offload_optimizer", ds["offload_optimizer"]), ("offload_param", ds["offload_param"])):
        if enabled:
            zero[key] = {**zero.get(key, {}), "device": "cpu"}
        else:
            zero.pop(key, None)
    auto_values = {
        "reduce_bucket_size": hidden_size * hidden_size,
        "stage3_prefetch_bucket_size": int(0.9 * hidden_size * hidden_size),
        "stage3_param_persistence_threshold": 10 * hidden_size,
    }
    for key, value in auto_values.items():
        if zero.get(key) == "auto":
            zero[key] = value

    optimizer = result.setdefault("optimizer", {"type": "AdamW", "params": {}})
    optimizer["type"] = "AdamW"
    optimizer.setdefault("params", {}).update(
        {
            "lr": train["learning_rate"],
            "betas": [train["beta1"], train["beta2"]],
            "eps": train["eps"],
            "weight_decay": train["weight_decay"],
        }
    )
    # One cosine schedule is applied at optimizer-step boundaries in the loop.
    result.pop("scheduler", None)
    result.pop("activation_checkpointing", None)
    result.pop("flops_profiler", None)
    return result


def build_3b_model_config(config: dict):
    if config.get("model_size") != "3B":
        raise ValueError("The 3B trainer requires a 3B recipe")
    model_config = get_3b_config()
    if config["batching"]["sequence_length"] > model_config.max_position_embeddings:
        raise ValueError("3B pretraining sequence length exceeds the model's native context")
    model_config.use_gradient_checkpointing = config["deepspeed"]["activation_checkpointing"]
    model_config.deepspeed_stage = config["deepspeed"]["stage"]
    model_config.mixed_precision = config["training"]["precision"]
    return model_config


class Pretraining3BModel(CodingAgentModel):
    """Return a tensor so ZeRO-3 can attach its backward hooks to the output.

    The base model returns a dataclass, which DeepSpeed's ZeRO-3 hooks do not
    traverse. Subclassing preserves the original model's state-dict keys.
    """

    def forward(self, input_ids, labels):
        return super().forward(input_ids=input_ids, labels=labels).loss


class TokenBatchStream:
    """Deterministic, disjoint rank batches from the memory-mapped token shards."""

    def __init__(
        self,
        dataset: MemoryMappedDataset,
        batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
        *,
        shuffle: bool = True,
        epoch: int = 0,
        batch_in_epoch: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = epoch
        self.batch_in_epoch = batch_in_epoch
        self.sequence_length = dataset.seq_length
        if not 0 <= rank < world_size:
            raise ValueError("rank must be within world_size")
        if dataset.shard_offsets[-1] != dataset.total_tokens:
            raise ValueError("dataset manifest token count does not match the shard files")
        self.num_sequences = dataset.total_tokens // self.sequence_length
        self.batches_per_epoch = self.num_sequences // (batch_size * world_size)
        if self.batches_per_epoch < 1:
            raise ValueError("dataset does not contain one full micro-batch per rank")
        if epoch < 0 or not 0 <= batch_in_epoch <= self.batches_per_epoch:
            raise ValueError("invalid saved dataset position")
        self._order = None

    def _sequence(self, index: int) -> np.ndarray:
        start = index * self.sequence_length
        end = start + self.sequence_length
        shard = bisect_right(self.dataset.shard_offsets, start) - 1
        pieces = []
        while start < end:
            shard_end = self.dataset.shard_offsets[shard + 1]
            stop = min(end, shard_end)
            pieces.append(
                self.dataset.shards[shard][
                    start - self.dataset.shard_offsets[shard] : stop - self.dataset.shard_offsets[shard]
                ]
            )
            start = stop
            shard += 1
        return np.concatenate(pieces).astype(np.int64, copy=False)

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        if self.batch_in_epoch == self.batches_per_epoch:
            self.epoch += 1
            self.batch_in_epoch = 0
            self._order = None
        if self._order is None:
            self._order = np.arange(self.num_sequences, dtype=np.int64)
            if self.shuffle:
                np.random.default_rng(self.seed + self.epoch).shuffle(self._order)
        global_micro_batch = self.batch_size * self.world_size
        start = self.batch_in_epoch * global_micro_batch + self.rank * self.batch_size
        indices = self._order[start : start + self.batch_size]
        self.batch_in_epoch += 1
        batch = np.stack([self._sequence(int(index)) for index in indices])
        return torch.from_numpy(batch)


@dataclass
class PretrainState:
    step: int = 0
    tokens_processed: int = 0
    epoch: int = 0
    batch_in_epoch: int = 0


def checkpoint_metadata(plan: TrainingPlan, stream: TokenBatchStream, manifest_path: Path) -> dict:
    return {
        "model_size": "3B",
        "world_size": plan.world_size,
        "global_batch_size": plan.global_batch_size,
        "micro_batch_size": plan.micro_batch_size,
        "gradient_accumulation_steps": plan.gradient_accumulation_steps,
        "sequence_length": plan.sequence_length,
        "dataset_tokens": stream.dataset.total_tokens,
        "dataset_seed": stream.seed,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def restore_3b_checkpoint(engine, checkpoint_dir: Path, plan: TrainingPlan, metadata: dict) -> PretrainState:
    load_path, client_state = engine.load_checkpoint(str(checkpoint_dir))
    if not load_path or not client_state:
        raise RuntimeError(f"No 3B checkpoint could be loaded from {checkpoint_dir}")
    if client_state.get("pretrain_3b_metadata") != metadata:
        raise ValueError("3B checkpoint does not match the current batch, world size, or dataset")
    saved_state = client_state.get("pretrain_3b_state")
    if not isinstance(saved_state, dict):
        raise ValueError("Checkpoint is missing 3B pretraining state")
    state = PretrainState(**saved_state)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in asdict(state).values()):
        raise ValueError("Checkpoint has invalid 3B counters")
    if state.tokens_processed != state.step * plan.tokens_per_step:
        raise ValueError("Checkpoint has inconsistent 3B step/token accounting")
    if state.step > plan.total_steps:
        raise ValueError("Checkpoint already exceeds the configured 3B token budget")
    if getattr(engine, "global_steps", state.step) != state.step:
        raise ValueError("Checkpoint's DeepSpeed step does not match its 3B token count")
    return state


def save_3b_checkpoint(engine, checkpoint_dir: Path, state: PretrainState, metadata: dict, keep_last: int, rank: int):
    """All ZeRO ranks must participate; only rank 0 removes old checkpoints."""
    tag = f"step_{state.step:08d}"
    saved = engine.save_checkpoint(
        str(checkpoint_dir),
        tag=tag,
        client_state={
            "pretrain_3b_state": asdict(state),
            "pretrain_3b_metadata": metadata,
        },
    )
    if not saved:
        raise RuntimeError(f"Failed to save 3B checkpoint {tag}")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        (checkpoint_dir / tag / ".3b-pretrain").write_text("3B pretraining checkpoint\n")
        if keep_last > 0:
            checkpoints = sorted(
                p
                for p in checkpoint_dir.glob("step_*")
                if p.is_dir() and not p.is_symlink() and p.name[5:].isdigit() and (p / ".3b-pretrain").is_file()
            )
            for old in checkpoints[:-keep_last]:
                shutil.rmtree(old)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def evaluate_3b(engine, dataset, config: dict, plan: TrainingPlan, rank: int) -> float | None:
    eval_batch = config["evaluation"]["eval_batch_size"]
    max_batches = config["evaluation"]["eval_tokens"] // (eval_batch * plan.sequence_length * plan.world_size)
    try:
        stream = TokenBatchStream(dataset, eval_batch, rank, plan.world_size, config["training"]["seed"], shuffle=False)
    except ValueError as exc:
        if "one full micro-batch" not in str(exc):
            raise
        return None
    max_batches = min(max_batches, stream.batches_per_epoch)
    if max_batches < 1:
        return None
    engine.eval()
    losses = []
    with torch.no_grad():
        for _ in range(max_batches):
            inputs = next(stream).to(engine.device)
            losses.append(engine(input_ids=inputs, labels=inputs).detach().float())
    loss = torch.stack(losses).mean()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(loss)
        loss /= plan.world_size
    engine.train()
    return loss.item()


def run_3b_training(
    engine,
    optimizer,
    stream: TokenBatchStream,
    config: dict,
    plan: TrainingPlan,
    checkpoint_dir: Path,
    metadata: dict,
    rank: int,
    *,
    state: PretrainState | None = None,
    eval_dataset: MemoryMappedDataset | None = None,
) -> PretrainState:
    """Run complete optimizer steps, never exceeding the configured token cap."""
    state = state or PretrainState()
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in asdict(state).values())
        or state.tokens_processed != state.step * plan.tokens_per_step
        or state.step > plan.total_steps
        or state.batch_in_epoch > stream.batches_per_epoch
        or state.epoch * stream.batches_per_epoch + state.batch_in_epoch
        != state.step * plan.gradient_accumulation_steps
    ):
        raise ValueError("Invalid initial 3B training state")
    if (stream.batch_size, stream.world_size, stream.sequence_length) != (
        plan.micro_batch_size,
        plan.world_size,
        plan.sequence_length,
    ):
        raise ValueError("3B data stream does not match the training plan")
    stream.epoch, stream.batch_in_epoch = state.epoch, state.batch_in_epoch
    stream._order = None
    train = config["training"]
    ckpt = config["checkpointing"]
    log_every = config["logging"]["log_every_steps"]
    eval_every = config["logging"]["eval_every_steps"]
    save_steps = ckpt.get("save_every_steps", 0)
    save_tokens = ckpt.get("save_every_tokens", 0)
    last_saved_step = state.step if state.step else -1
    engine.train()
    start_time = time.monotonic()
    tokens_at_start = state.tokens_processed

    while state.step < plan.run_steps:
        lr = cosine_lr(
            state.step, train["learning_rate"], train["min_learning_rate"], plan.warmup_steps, plan.total_steps
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        losses = []
        for _ in range(plan.gradient_accumulation_steps):
            inputs = next(stream).to(engine.device)
            if tuple(inputs.shape) != (plan.micro_batch_size, plan.sequence_length):
                raise ValueError("3B data stream returned an incomplete micro-batch")
            loss = engine(input_ids=inputs, labels=inputs)
            losses.append(loss.detach().float())
            # DeepSpeed scales the loss and steps the optimizer at its own
            engine.backward(loss)
            engine.step()
        if getattr(engine, "global_steps", state.step + 1) != state.step + 1:
            raise RuntimeError("DeepSpeed did not advance at the expected accumulation boundary")
        previous_tokens = state.tokens_processed
        state.step += 1
        state.tokens_processed += plan.tokens_per_step
        state.epoch, state.batch_in_epoch = stream.epoch, stream.batch_in_epoch

        if rank == 0 and log_every > 0 and state.step % log_every == 0:
            elapsed = max(time.monotonic() - start_time, 1e-9)
            logger.info(
                "3B step %d/%d | loss %.4f | lr %.3g | tokens %s/%s | %.0f tokens/s",
                state.step,
                plan.run_steps,
                torch.stack(losses).mean().item(),
                lr,
                f"{state.tokens_processed:,}",
                f"{train['total_tokens']:,}",
                (state.tokens_processed - tokens_at_start) / elapsed,
            )
        if eval_dataset is not None and eval_every > 0 and state.step % eval_every == 0:
            val_loss = evaluate_3b(engine, eval_dataset, config, plan, rank)
            if rank == 0:
                logger.info(
                    "3B validation loss: %s",
                    f"{val_loss:.4f}" if val_loss is not None else "skipped (not enough tokens)",
                )
        due_steps = save_steps > 0 and state.step % save_steps == 0
        due_tokens = save_tokens > 0 and previous_tokens // save_tokens < state.tokens_processed // save_tokens
        if due_steps or due_tokens:
            save_3b_checkpoint(engine, checkpoint_dir, state, metadata, ckpt["keep_last_n"], rank)
            last_saved_step = state.step

    if state.step and state.step != last_saved_step:
        save_3b_checkpoint(engine, checkpoint_dir, state, metadata, ckpt["keep_last_n"], rank)
    if rank == 0:
        logger.info(
            "3B pretraining stopped at %s input tokens (%d global steps)", f"{state.tokens_processed:,}", state.step
        )
    return state


def train_3b(args, config: dict) -> PretrainState:
    """Entrypoint shared by the two existing training CLIs, only for 3B."""
    try:
        import deepspeed
    except ImportError as exc:
        raise RuntimeError("3B pretraining requires DeepSpeed; install the training dependencies") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("3B pretraining requires a CUDA GPU; none was detected")
    deepspeed.init_distributed()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    plan = build_training_plan(config, world_size, getattr(args, "max_steps", None))
    model_config = build_3b_model_config(config)
    ds_path = resolve_deepspeed_path(args.config, config["deepspeed"]["config_file"])
    ds_config = build_deepspeed_config(json.loads(ds_path.read_text()), config, plan, model_config.hidden_size)

    data_dir = Path(getattr(args, "data_dir", None) or config["data"]["data_dir"])
    output_dir = Path(getattr(args, "output_dir", None) or config["checkpointing"]["save_dir"])
    resume = getattr(args, "resume", False)
    if not config["checkpointing"].get("save_optimizer_state", True):
        raise ValueError("3B DeepSpeed checkpoints must save optimizer state for resume")
    if resume and not (output_dir / "latest").is_file():
        raise FileNotFoundError(f"No latest 3B checkpoint exists in {output_dir}")
    if not resume and (output_dir / "latest").exists():
        raise FileExistsError(f"3B checkpoint already exists in {output_dir}; use --resume or another output directory")

    dataset = MemoryMappedDataset(str(data_dir), split="train", seq_length=plan.sequence_length)
    stream = TokenBatchStream(dataset, plan.micro_batch_size, rank, world_size, config["training"]["seed"])
    metadata = checkpoint_metadata(plan, stream, data_dir / "train_manifest.json")
    eval_dataset = None
    if config["logging"]["eval_every_steps"] > 0:
        eval_dataset = MemoryMappedDataset(str(data_dir), split="val", seq_length=plan.sequence_length)

    torch.manual_seed(config["training"]["seed"])
    torch.cuda.manual_seed_all(config["training"]["seed"])
    if rank == 0:
        logger.info(
            "3B pretraining: %d ranks, accumulation %d, %d global steps, %s planned input tokens (cap %s)",
            world_size,
            plan.gradient_accumulation_steps,
            plan.total_steps,
            f"{plan.planned_tokens:,}",
            f"{config['training']['total_tokens']:,}",
        )
    with deepspeed.zero.Init(config_dict_or_path=ds_config):
        model = Pretraining3BModel(model_config)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        dist_init_required=False,
    )
    state = restore_3b_checkpoint(engine, output_dir, plan, metadata) if resume else PretrainState()
    return run_3b_training(
        engine, optimizer, stream, config, plan, output_dir, metadata, rank, state=state, eval_dataset=eval_dataset
    )
