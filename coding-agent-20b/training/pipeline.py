"""Staged training pipeline: pretrain -> mid-train -> SFT -> DPO.

Each stage declares its data mixture, LR schedule and checkpoint lineage;
`run_stage` is intentionally thin (DeepSpeed launch stays in
train_production.py) — this module owns stage *sequencing*, collation and
the packing-aware batch contract so stages cannot drift apart.
"""







from __future__ import annotations















from dataclasses import dataclass







from typing import Dict, List















import numpy as np















from .schedules import StageConfig, default_pipeline























@dataclass







class Pipeline:







    stages: List[StageConfig]















    @classmethod







    def default(cls) -> "Pipeline":







        return cls(default_pipeline())















    def stage(self, name: str) -> StageConfig:







        for s in self.stages:







            if s.name == name:







                return s







        raise KeyError(f"unknown stage {name!r}; have {[s.name for s in self.stages]}")















    def lineage(self) -> List[str]:







        """Checkpoint each stage initialises from (previous stage's output)."""







        return [f"{s.name} <- {self.stages[i - 1].name if i else 'random-init'}"







                for i, s in enumerate(self.stages)]























class PackingCollator:







    """Batch PackedSequence dicts -> torch tensors with pad loss masked.

    labels already carry -100 on pads (see dataset.packing); position_ids and
    doc_ids pass through so the model/loss can block cross-document attention.
    """















    def __call__(self, batch: List[Dict[str, np.ndarray]]) -> Dict:







        import torch







        keys = ["input_ids", "labels", "position_ids", "doc_ids"]







        return {k: torch.from_numpy(np.stack([b[k] for b in batch])).long() for k in keys if k in batch[0]}























def sft_mask_prompt(input_ids, prompt_len: int, eos_id: int) -> np.ndarray:







    """Labels for SFT: -100 over the prompt, real ids over the completion."""







    labels = np.array(input_ids, dtype=np.int64)







    labels[:prompt_len] = -100







    return labels







