"""Data mixture ratios, high-value upsampling and curriculum scheduling."""







from __future__ import annotations















import random







from dataclasses import dataclass, field







from typing import Dict, Iterator, List, Sequence























@dataclass







class SourceSpec:







    name: str







    weight: float                                                         







    upsample: float = 1.0                                                     







    kind: str = "code"                                                    























                                                                              







DEFAULT_MIXTURE: List[SourceSpec] = [







    SourceSpec("the-stack-code", 0.62, 1.0, "code"),







    SourceSpec("github-issues", 0.06, 2.0, "issues"),







    SourceSpec("github-prs", 0.08, 2.0, "prs"),







    SourceSpec("tests", 0.08, 2.5, "tests"),







    SourceSpec("docs-and-comments", 0.06, 1.5, "docs"),







    SourceSpec("instruction-sft", 0.10, 1.0, "instruction"),







]























class MixtureSampler:







    """Weighted sampling over named document pools with upsampling."""















    def __init__(self, pools: Dict[str, Sequence], mixture: List[SourceSpec] = None,







                 seed: int = 42):







        self.pools = pools







        self.mixture = mixture or DEFAULT_MIXTURE







        self.rng = random.Random(seed)







        total = sum(s.weight * s.upsample for s in self.mixture if s.name in pools)







        if total <= 0:







            raise ValueError("mixture has no available sources")







        self._weights = [(s, s.weight * s.upsample / total) for s in self.mixture if s.name in pools]







        self._cursors = {name: 0 for name in pools}















    def _next_from(self, name: str):







        pool = self.pools[name]







        i = self._cursors[name] % len(pool)







        self._cursors[name] += 1







        if self._cursors[name] % len(pool) == 0:







                                                                                 







            pass







        return pool[i]















    def sample(self, n: int) -> Iterator:







        names = [s.name for s, _ in self._weights]







        weights = [w for _, w in self._weights]







        for _ in range(n):







            name = self.rng.choices(names, weights=weights, k=1)[0]







            yield self._next_from(name)























def curriculum_order(docs: List[dict], step: int, total_steps: int) -> List[dict]:







    """Curriculum: early steps favour short/high-quality docs, late steps
    expose the full distribution (sorted blend controlled by progress)."""







    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)







    scored = sorted(docs, key=lambda d: (-(d.get("quality_score", 0.5)), len(d.get("content", ""))))







                                                                                   







    n_hard = int(len(scored) * progress * 0.5)







    hard = scored[len(scored) - n_hard:] if n_hard else []







    easy = scored[: len(scored) - n_hard] if n_hard else scored







    out: List[dict] = []







    hi = iter(hard)







    for i, d in enumerate(easy):







        out.append(d)







        if progress > 0 and i % 4 == 3:







            try:







                out.append(next(hi))







            except StopIteration:







                break







    out.extend(list(hi))







    return out







