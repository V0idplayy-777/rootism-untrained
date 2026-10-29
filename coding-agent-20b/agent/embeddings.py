"""Local embeddings for semantic retrieval (row 7).

Attempts to use a real embedding model (sentence-transformers or local).
Falls back to HashingEmbedder (deterministic, no network) if nothing is available.
Keeps hybrid BM25 + call-graph approach, just swaps the vector source.

Usage:
  from agent.embeddings import get_embedder
  embed_fn = get_embedder(dimension=384)  # returns callable or None
  index = SemanticIndex.from_directory(root, embed_fn=embed_fn)

Embedder contract: (text: str) -> list[float] (l2-normalized)
If the model cannot be loaded, embed_fn is None and HashingEmbedder is used.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable


def _try_sentence_transformers(model_name: str = "all-MiniLM-L6-v2") -> Callable[[str], list[float]] | None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        model = SentenceTransformer(model_name, device="cpu")
        vec = model.encode(["hello world"], normalize_embeddings=True)
        if vec is None or len(vec) == 0:
            return None

        def embed(text: str) -> list[float]:
            v = model.encode([text], normalize_embeddings=True)[0]
            return [float(x) for x in v]

        return embed
    except Exception:
        return None


def _try_huggingface_transformers(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Callable[[str], list[float]] | None:
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        import torch.nn.functional as F
    except ImportError:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
        model.eval()

        def embed(text: str) -> list[float]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1)
                summed = (outputs.last_hidden_state * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                mean = summed / counts
                norm = F.normalize(mean, p=2, dim=1)
                return norm[0].tolist()

        return embed
    except Exception:
        return None


def get_embed_fn(preferred: str | None = None, dim: int = 384) -> Callable[[str], list[float]] | None:
    """Return a local embed function if available, else None (caller falls back to hashing)."""
    import os

    if os.environ.get("AGENT_EMBEDDINGS") == "hashing":
        return None
    if preferred == "hashing":
        return None
    fn = _try_sentence_transformers()
    if fn is not None:
        return fn
    fn = _try_huggingface_transformers()
    if fn is not None:
        return fn
    return None


def get_embedder_info() -> dict:
    """For diagnostics: what embeddings are available."""
    has_st = False
    has_transformers = False
    try:
        import sentence_transformers  # noqa: F401
        has_st = True
    except ImportError:
        pass
    try:
        import transformers  # noqa: F401
        has_transformers = True
    except ImportError:
        pass
    fn = get_embed_fn()
    return {
        "sentence_transformers": has_st,
        "transformers": has_transformers,
        "active": "external" if fn else "hashing",
        "note": "local embeddings auto-detected; set AGENT_EMBEDDINGS=hashing to force hashing",
    }
