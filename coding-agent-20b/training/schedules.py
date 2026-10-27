"""LR schedules, sequence-length curriculum and staged-training configs.

Tuned for the 20B recipe (AdamW b1=0.9 b2=0.95 wd=0.1, bf16, grad-clip 1.0):
pretrain 3e-4 -> mid-train 1.5e-4 -> SFT 2e-5 -> DPO 5e-7 (beta 0.1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


def wsd_lr(step: int, max_lr: float, min_lr: float, warmup: int, stable_end: int, total: int) -> float:
    """Warmup-Stable-Decay (used by MiniCPM/DeepSeek recipes; easier to extend
    than cosine when total tokens change mid-run)."""
    if step < warmup:
        return max_lr * (step + 1) / max(warmup, 1)
    if step < stable_end:
        return max_lr
    if step >= total:
        return min_lr
    progress = (step - stable_end) / max(total - stable_end, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def cosine_lr(step: int, max_lr: float, min_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(warmup, 1)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def seq_length_at(step: int, schedule: List[tuple] = None) -> int:
    """Sequence-length curriculum: short+cheap first, full 8k later."""
    schedule = schedule or [(0, 2048), (2000, 4096), (6000, 8192)]
    current = schedule[0][1]
    for start, length in schedule:
        if step >= start:
            current = length
    return current


@dataclass
class StageConfig:
    name: str
    kind: str  # pretrain | midtrain | sft | dpo
    learning_rate: float
    min_lr: float
    warmup_steps: int
    total_steps: int
    seq_length: int = 8192
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    extra: Dict = field(default_factory=dict)


def default_pipeline() -> List[StageConfig]:
    """Continuous pretrain -> mid-train -> SFT -> preference (DPO) pipeline."""
    return [
        StageConfig("pretrain", "pretrain", 3e-4, 3e-5, 2000, 100000, seq_length=8192),
        StageConfig("midtrain", "midtrain", 1.5e-4, 1.5e-5, 500, 20000, seq_length=8192,
                    extra={"anneal_sources": ["tests", "issues", "prs", "docs"]}),
        StageConfig("sft", "sft", 2e-5, 2e-6, 200, 6000, seq_length=8192,
                    extra={"mask_prompt_loss": True}),
        StageConfig("dpo", "dpo", 5e-7, 5e-8, 50, 1500, seq_length=4096, weight_decay=0.0,
                    extra={"beta": 0.1, "reference_model": "sft"}),
    ]


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta: float = 0.1):
    """Direct Preference Optimization loss (Rafailov et al. 2023). Torch tensors in/out."""
    import torch
    import torch.nn.functional as F
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    losses = -F.logsigmoid(chosen_rewards - rejected_rewards)
    return losses.mean(), {"chosen_reward": chosen_rewards.mean().detach(),
                            "rejected_reward": rejected_rewards.mean().detach(),
                            "margin": (chosen_rewards - rejected_rewards).mean().detach()}
