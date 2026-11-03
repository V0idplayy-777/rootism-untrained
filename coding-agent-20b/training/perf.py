"""Analytic memory model + auto-tuning for the pretraining recipes.

"Why is this OOM" and "what micro-batch fits" are the two questions that decide
whether a long run finishes, and both are answerable before touching a GPU.  The
numbers here come from the standard accounting for ZeRO + activation
checkpointing:

============================  ==========================================
state                         per-rank bytes
============================  ==========================================
bf16 weights                  ``2P`` (ZeRO-3: ``2P / world``)
bf16 gradients                ``2P`` (ZeRO-3: ``2P / world``)
fp32 master copy              ``4P`` (ZeRO-3: ``4P / world``)
Adam moments (fp32/8-bit)     ``8P`` / ``2P`` (ZeRO-3: divided)
live activations              ``B*S*H*2 * (live + recompute)``
logits                        ``B*S*V*2`` -- or ``B*chunk*V*8`` chunked
attention scratch             ``0`` (fused SDPA) / ``B*heads*S*S*4``
============================  ==========================================

The activation and attention terms are the ones that decide whether a step
fits: an 8k sequence with the math attention kernel needs a
``[B, heads, 8192, 8192]`` fp32 matrix (2 GB per head-pair at 48 heads), while
fused SDPA keeps only ``O(S)`` working memory per row block.  The planner is
deliberately explicit about that, and about the logits buffer that disappears
once the loss is computed in chunks.

:func:`plan_recipe` turns a memory budget into the recipe fields that make it
fit *without touching the numerics*: it only changes how much work is in flight
(micro-batch, accumulation, checkpointed segments, chunk size), never which
tokens, which precision, or which weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

GB = 1024**3


@dataclass
class MemoryBreakdown:
    """Per-rank memory estimate, in bytes."""

    weights_bf16: float = 0.0
    gradients_bf16: float = 0.0
    master_fp32: float = 0.0
    optimizer_state: float = 0.0
    activations: float = 0.0
    logits: float = 0.0
    attention_scratch: float = 0.0
    communication: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        return (
            self.weights_bf16
            + self.gradients_bf16
            + self.master_fp32
            + self.optimizer_state
            + self.activations
            + self.logits
            + self.attention_scratch
            + self.communication
        )

    @property
    def optimizer_total(self) -> float:
        return self.master_fp32 + self.optimizer_state

    def as_gib(self) -> dict:
        data = {key: value / GB for key, value in asdict(self).items() if key != "extra"}
        data["total"] = self.total / GB
        return {key: round(value, 3) for key, value in data.items()}

    def report(self, title: str = "per-rank training memory") -> str:
        rows = [
            ("bf16 weights", self.weights_bf16),
            ("bf16 gradients", self.gradients_bf16),
            ("fp32 master", self.master_fp32),
            ("optimizer moments", self.optimizer_state),
            ("activations", self.activations),
            ("logits", self.logits),
            ("attention scratch", self.attention_scratch),
            ("communication buffers", self.communication),
        ]
        width = max(len(name) for name, _ in rows)
        lines = [f"{title} (GiB):"]
        for name, value in rows:
            lines.append(f"  {name:<{width}}  {value / GB:8.3f}")
        lines.append(f"  {'TOTAL':<{width}}  {self.total / GB:8.3f}")
        return "\n".join(lines)


def _layer_activation_bytes(
    batch: int,
    seq: int,
    hidden: int,
    intermediate: int,
    live_segments: int,
) -> tuple[float, float]:
    """(stored, recompute-peak) activation bytes for one transformer block.

    With checkpointing, what stays resident per segment is its *input* only
    (``B*S*H*2`` bytes); the peak during backward is one recomputed segment,
    i.e. the two MLP projections that are multiplied together
    (``2 * B*S*I`` each) plus the attention working set, none of which is
    quadratic in S once SDPA is fused.
    """

    stored = 2 * batch * seq * hidden * max(live_segments, 0)

    recompute = 2 * batch * seq * intermediate * 2 + 2 * batch * seq * hidden * 4

    return stored, recompute


def estimate(
    model,
    *,
    world_size: int = 1,
    micro_batch: int = 1,
    seq_length: int = 8192,
    zero_stage: int = 3,
    zero8bit_optimizer: bool = False,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    stage3_max_live_parameters: float | None = None,
    activation_segments: int = 2,
    chunked_loss: bool = True,
    loss_chunk_size: int = 1024,
    fused_attention: bool = True,
    gradient_checkpointing: bool = True,
) -> MemoryBreakdown:
    """Estimate per-rank memory for one training configuration.

    ``model`` is a ``CodingAgentConfig`` (only the architecture fields are read).
    Offloaded state is reported in ``extra["host_gib"]`` instead of the device
    total, because that is where it actually lives.
    """

    params = model.total_parameters()

    shard = world_size if zero_stage == 3 else 1

    optimizer_shard = world_size if zero_stage >= 1 else 1

    breakdown = MemoryBreakdown()

    host = 0.0

    if offload_param:

        host += 2 * params

    else:

        breakdown.weights_bf16 = 2 * params / shard

    breakdown.gradients_bf16 = 2 * params / shard

    optimizer_bytes = (4 + (2 if zero8bit_optimizer else 8)) * params / optimizer_shard

    if offload_optimizer:

        host += optimizer_bytes

    else:

        breakdown.master_fp32 = 4 * params / optimizer_shard

        breakdown.optimizer_state = (2 if zero8bit_optimizer else 8) * params / optimizer_shard

    stored, recompute = 0.0, 0.0

    if gradient_checkpointing and activation_segments > 0:

        per_layer_stored, recompute = _layer_activation_bytes(
            micro_batch, seq_length, model.hidden_size, model.intermediate_size, activation_segments + 1
        )

        stored = per_layer_stored * model.num_hidden_layers

    else:

        stored = (
            2 * micro_batch * seq_length * (model.hidden_size * 8 + model.intermediate_size * 3)
        ) * model.num_hidden_layers

    breakdown.activations = stored + recompute

    if chunked_loss:

        # one chunk of bf16 logits plus the fp32 log-softmax scratch built from it
        breakdown.logits = 6 * micro_batch * min(seq_length, max(int(loss_chunk_size), 1)) * model.vocab_size

    else:

        breakdown.logits = 2 * micro_batch * seq_length * model.vocab_size

    if not fused_attention and seq_length > 1:

        breakdown.attention_scratch = (
            4 * micro_batch * model.num_attention_heads * seq_length * seq_length
        )

    # ZeRO-3 streams parameters in for the layer being computed.  The cap is
    # explicit on 20B+ (stage3_max_live_parameters) and defaults to two layers.
    live_cap = stage3_max_live_parameters or (2 * params / max(model.num_hidden_layers, 1))

    breakdown.communication = 2 * min(params, live_cap) / shard + 2 * (
        model.hidden_size * model.hidden_size * 8
    )

    breakdown.extra = {"host_gib": round(host / GB, 3), "shard": shard}

    return breakdown


def fits(breakdown: MemoryBreakdown, budget_gb: float, reserve_gb: float = 1.5) -> bool:
    """True when the estimate lands inside ``budget_gb`` with a safety reserve."""

    return breakdown.total / GB <= max(budget_gb - reserve_gb, 0.0)


def largest_micro_batch(
    model,
    budget_gb: float,
    *,
    world_size: int = 1,
    seq_length: int = 8192,
    max_micro_batch: int = 16,
    reserve_gb: float = 1.5,
    **kwargs,
) -> int:
    """Largest micro-batch that still fits ``budget_gb`` (at least 1)."""

    best = 1

    for candidate in range(1, max_micro_batch + 1):

        breakdown = estimate(model, world_size=world_size, micro_batch=candidate, seq_length=seq_length, **kwargs)

        if fits(breakdown, budget_gb, reserve_gb):

            best = candidate

        else:

            break

    return best


def plan_recipe(
    config: dict,
    model,
    budget_gb: float,
    *,
    world_size: int = 1,
    reserve_gb: float = 1.5,
) -> dict:
    """Return recipe fields that fit ``budget_gb`` without changing the maths.

    Constraints that keep the optimizer update identical:

    * ``global_batch_size`` is preserved, so ``gradient_accumulation_steps``
      absorbs every micro-batch change.  The gradient of a sum over the same
      global batch is the same sum, just evaluated in more pieces.
    * precision, learning rate, betas, weight decay, clipping, sequence length
      and data order are untouched.
    * activation checkpointing is only *tightened* (more segments) if that is
      what makes the batch fit, which trades compute but never values.
    """

    batching = config["batching"]

    deepspeed = config.get("deepspeed", {})

    seq_length = batching["sequence_length"]

    global_batch = batching["global_batch_size"]

    zero_stage = deepspeed.get("stage", 3)

    offload_optimizer = bool(deepspeed.get("offload_optimizer", False))

    offload_param = bool(deepspeed.get("offload_param", False))

    segments = int(deepspeed.get("activation_checkpointing_segments", 2) or 0)

    micro = batching["micro_batch_size_per_gpu"]

    attempts = []

    while micro >= 1:

        breakdown = estimate(
            model,
            world_size=world_size,
            micro_batch=micro,
            seq_length=seq_length,
            zero_stage=zero_stage,
            zero8bit_optimizer=bool(deepspeed.get("zero8bit_optimizer", False)),
            offload_optimizer=offload_optimizer,
            offload_param=offload_param,
            stage3_max_live_parameters=deepspeed.get("stage3_max_live_parameters"),
            activation_segments=segments if deepspeed.get("activation_checkpointing", True) else 0,
            chunked_loss=bool(config["training"].get("chunked_loss", True)),
            loss_chunk_size=int(config["training"].get("loss_chunk_size", 2048)),
            fused_attention=config["training"].get("attention_backend", "auto") != "math",
        )

        attempts.append((micro, segments, breakdown))

        if fits(breakdown, budget_gb, reserve_gb) and global_batch % (micro * world_size) == 0:

            return {
                "fits": True,
                "micro_batch_size_per_gpu": micro,
                "global_batch_size": global_batch,
                "gradient_accumulation_steps": global_batch // (micro * world_size),
                "activation_checkpointing_segments": segments,
                "activation_checkpointing": segments > 0,
                "estimate_gib": breakdown.as_gib(),
            }

        # Try tightening activation checkpointing before shrinking the batch:
        # recompute is cheaper than a smaller micro-batch for the same tokens.
        if deepspeed.get("activation_checkpointing", True) and segments < 2:

            segments = 2

            continue

        micro -= 1

    smallest = attempts[-1][2] if attempts else None

    return {
        "fits": False,
        "micro_batch_size_per_gpu": 1,
        "global_batch_size": global_batch,
        "gradient_accumulation_steps": max(global_batch // max(world_size, 1), 1),
        "activation_checkpointing_segments": max(segments, 1),
        "activation_checkpointing": True,
        "estimate_gib": smallest.as_gib() if smallest else {},
    }
