"""Modern sampling + structured-output controls (pure functions, unit-testable).

All filters operate on raw logits [vocab] or [batch, vocab] and return filtered
logits with -inf for removed tokens. Kept dependency-light (torch only).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class GenerationConfig:

    max_new_tokens: int = 512

    temperature: float = 0.7

    top_p: float = 0.95

    top_k: int = 0

    min_p: float = 0.0

    typical_p: float = 1.0

    repetition_penalty: float = 1.0

    frequency_penalty: float = 0.0

    presence_penalty: float = 0.0

    do_sample: bool = True

    seed: int | None = None

    stop_strings: list[str] = field(default_factory=list)

    stop_token_ids: list[int] = field(default_factory=list)

    json_mode: bool = False

    grammar_validator: Callable[[str], bool] | None = None


def apply_penalties(
    logits: torch.Tensor, generated_ids: Sequence[int], cfg: GenerationConfig
) -> torch.Tensor:

    logits = logits.clone()

    if not generated_ids:

        return logits

    counts: dict = {}

    for t in generated_ids:

        counts[int(t)] = counts.get(int(t), 0) + 1

    for tid, c in counts.items():

        if cfg.repetition_penalty != 1.0:

            logits[..., tid] = (
                logits[..., tid] / cfg.repetition_penalty
                if logits[..., tid] > 0
                else logits[..., tid] * cfg.repetition_penalty
            )

        if cfg.frequency_penalty:

            logits[..., tid] -= cfg.frequency_penalty * c

        if cfg.presence_penalty:

            logits[..., tid] -= cfg.presence_penalty

    return logits


def filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:

    if k <= 0:

        return logits

    k = min(k, logits.size(-1))

    thresh = torch.topk(logits, k, dim=-1).values[..., -1, None]

    return torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)


def filter_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:

    if p >= 1.0:

        return logits

    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)

    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    remove = cum - F.softmax(sorted_logits, dim=-1) >= p

    remove[..., 0] = False

    out = logits.clone()

    out.scatter_(
        -1,
        sorted_idx,
        torch.where(
            remove, torch.full_like(sorted_logits, float("-inf")), sorted_logits
        ),
    )

    return out


def filter_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:

    if min_p <= 0:

        return logits

    probs = F.softmax(logits, dim=-1)

    thresh = probs.max(dim=-1, keepdim=True).values * min_p

    return torch.where(probs < thresh, torch.full_like(logits, float("-inf")), logits)


def filter_typical(logits: torch.Tensor, typical_p: float) -> torch.Tensor:

    if typical_p >= 1.0:

        return logits

    probs = F.softmax(logits, dim=-1)

    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1, keepdim=True)

    shifted = torch.abs(-torch.log(probs.clamp_min(1e-12)) - entropy)

    _sorted_shifted, sorted_idx = torch.sort(shifted, dim=-1)

    sorted_probs = probs.gather(-1, sorted_idx)

    cum = torch.cumsum(sorted_probs, dim=-1)

    remove = (cum - sorted_probs) >= typical_p

    remove[..., 0] = False

    out = logits.clone()

    out.scatter_(
        -1,
        sorted_idx,
        torch.where(remove, torch.full_like(logits, float("-inf")), logits),
    )

    return out


def apply_sampling_filters(logits: torch.Tensor, cfg: GenerationConfig) -> torch.Tensor:

    if cfg.temperature != 1.0 and cfg.temperature > 0:

        logits = logits / cfg.temperature

    logits = filter_top_k(logits, cfg.top_k)

    logits = filter_min_p(logits, cfg.min_p)

    logits = filter_typical(logits, cfg.typical_p)

    logits = filter_top_p(logits, cfg.top_p)

    return logits


def sample_next(
    logits: torch.Tensor,
    generated_ids: Sequence[int],
    cfg: GenerationConfig,
    generator: torch.Generator | None = None,
) -> int:

    logits = apply_penalties(logits.float(), generated_ids, cfg)

    if not cfg.do_sample or cfg.temperature == 0:

        return int(torch.argmax(logits, dim=-1).item())

    filtered = apply_sampling_filters(logits, cfg)

    probs = F.softmax(filtered, dim=-1)

    return int(torch.multinomial(probs, 1, generator=generator).item())


def find_stop(text: str, stop_strings: Sequence[str]) -> int | None:
    """Return cut index of the earliest stop string in text, or None."""

    best = None

    for s in stop_strings:

        if s and s in text:

            i = text.index(s)

            best = i if best is None else min(best, i)

    return best


def validate_structured(text: str, cfg: GenerationConfig) -> bool:

    if cfg.json_mode:

        try:

            json.loads(text.strip())

        except Exception:

            return False

    if cfg.grammar_validator is not None:

        return bool(cfg.grammar_validator(text))

    return True
