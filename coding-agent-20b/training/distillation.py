"""Knowledge distillation: smaller student from the 20B teacher (KL + CE)."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, labels=None,
                      temperature: float = 2.0, alpha_kd: float = 0.7) -> Dict:
    """alpha_kd * T^2 * KL(teacher||student) + (1-alpha_kd) * CE(labels)."""
    log_p = F.log_softmax(student_logits / temperature, dim=-1)
    q = F.softmax(teacher_logits / temperature, dim=-1)
    kd = F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)
    total = kd * alpha_kd
    out = {"kd_loss": kd.detach(), "loss": total}
    if labels is not None:
        ce = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)),
                             labels.view(-1), ignore_index=-100)
        out["ce_loss"] = ce.detach()
        out["loss"] = total + ce * (1 - alpha_kd)
    return out


@torch.no_grad()
def _teacher_logits(teacher, input_ids):
    return teacher(input_ids=input_ids).logits


def distillation_step(student, teacher, batch: Dict, optimizer, temperature: float = 2.0,
                      alpha_kd: float = 0.7) -> Dict:
    student.train()
    teacher.eval()
    input_ids = batch["input_ids"]
    labels = batch.get("labels")
    with torch.no_grad():
        t_logits = _teacher_logits(teacher, input_ids)
    s_logits = student(input_ids=input_ids).logits
    # Shift for next-token loss alignment.
    shift_labels = None
    if labels is not None:
        shift_labels = torch.cat([labels[:, 1:], torch.full_like(labels[:, :1], -100)], dim=1)
    losses = distillation_loss(s_logits, t_logits, labels=shift_labels,
                               temperature=temperature, alpha_kd=alpha_kd)
    losses["loss"].backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {k: (float(v) if hasattr(v, "item") else v) for k, v in losses.items()}
