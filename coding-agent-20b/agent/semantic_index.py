"""Local semantic search without a model download or a vector database.

`RepoIndex` already fuses BM25 with character-ngram cosine. This module adds the
pieces that make "find code by meaning" work offline and in CI:

* :class:`HashingEmbedder` - feature hashing over word unigrams/bigrams and
  character 3-grams, with IDF weighting learned from the corpus. It is
  deterministic, dependency-free and fast enough to index a repository in
  seconds.
* :class:`SemanticIndex` - sparse cosine search plus an optional ``embed_fn``
  seam, so a real embedding model (or a hosted API) can be dropped in without
  changing call sites.
* Persistence (``.agent/semantic_index.json.gz``) and incremental updates.
* ``hybrid_search`` - Reciprocal Rank Fusion of semantic and keyword rankings,
  which is what recovers exact identifier matches that pure embeddings miss.

Honest limitation, stated in the report too: hashed lexical/char features
capture paraphrase only through shared vocabulary and substrings. This is not a
neural embedding; it is a strong offline baseline with a clean upgrade path.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
IDENT = re.compile(r"[A-Z][a-z0-9]+|[a-z0-9]+")


def split_identifier(word: str) -> list[str]:
    """``getUserById`` -> get / user / by / id ; ``parse_tool_call`` -> parse/tool/call."""

    parts: list[str] = []
    for piece in re.split(r"[_\W]+", word):
        parts.extend(IDENT.findall(piece) or ([piece] if piece else []))
    return [part.lower() for part in parts if part]


def tokenise(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN.findall(text or ""):
        if "_" in raw or any(char.isupper() for char in raw[1:]):
            pieces = split_identifier(raw)
            tokens.extend(pieces)
            tokens.append(raw.lower())
        else:
            tokens.append(raw.lower())
    return tokens


def _stable_hash(feature: str, dim: int) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


class HashingEmbedder:
    """Sparse hashed features with optional IDF weights."""

    def __init__(self, dim: int = 1024, char_ngrams: bool = True):
        self.dim = dim
        self.char_ngrams = char_ngrams
        self.idf: dict[int, float] = {}

    def _features(self, text: str) -> list[tuple[int, float]]:
        tokens = tokenise(text)
        feats: list[tuple[int, float]] = []
        for token in tokens:
            feats.append((_stable_hash(f"w:{token}", self.dim), 1.0))
        for left, right in zip(tokens, tokens[1:]):
            feats.append((_stable_hash(f"b:{left}_{right}", self.dim), 0.6))
        if self.char_ngrams:
            compact = re.sub(r"\s+", " ", (text or "").lower())
            for ngram in _ngrams(compact, 3):
                feats.append((_stable_hash(f"c:{ngram}", self.dim), 0.25))
        return feats

    def fit(self, documents: Iterable[str]) -> "HashingEmbedder":
        df: dict[int, int] = {}
        total = 0
        for text in documents:
            total += 1
            for index in {index for index, _ in self._features(text)}:
                df[index] = df.get(index, 0) + 1
        self.idf = {
            index: math.log((1 + total) / (1 + count)) + 1.0 for index, count in df.items()
        }
        return self

    def embed(self, text: str) -> dict[int, float]:
        vector: dict[int, float] = {}
        for index, weight in self._features(text):
            vector[index] = vector.get(index, 0.0) + weight * self.idf.get(index, 1.0)
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {index: value / norm for index, value in vector.items() if value}


def _ngrams(text: str, n: int) -> Iterable[str]:
    if len(text) < n:
        return []
    return (text[i : i + n] for i in range(len(text) - n + 1))


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(index, 0.0) for index, value in a.items())


@dataclass
class Document:
    id: str
    text: str
    path: str = ""
    start_line: int = 0
    end_line: int = 0
    kind: str = "chunk"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        return cls(
            id=data["id"],
            text=data["text"],
            path=data.get("path", ""),
            start_line=int(data.get("start_line", 0)),
            end_line=int(data.get("end_line", 0)),
            kind=data.get("kind", "chunk"),
        )


@dataclass
class SearchHit:
    document: Document
    score: float
    source: str = "semantic"

    def to_dict(self) -> dict:
        return {
            "id": self.document.id,
            "path": self.document.path,
            "start_line": self.document.start_line,
            "end_line": self.document.end_line,
            "score": round(self.score, 4),
            "source": self.source,
            "preview": self.document.text[:200],
        }

    def render(self, width: int = 120) -> str:
        head = f"{self.document.path}:{self.document.start_line}-{self.document.end_line} ({self.score:.3f})"
        body = " ".join(self.document.text.split())[:width]
        return f"{head}\n  {body}"


class SemanticIndex:
    def __init__(
        self,
        documents: list[Document] | None = None,
        embedder: HashingEmbedder | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
    ):
        self.documents: list[Document] = documents or []
        self.embedder = embedder or HashingEmbedder()
        self.embed_fn = embed_fn
        self.vectors: list[dict[int, float]] = []
        self._external: list[list[float]] = []
        self.updated_at = 0.0

    @classmethod
    def build(cls, documents: list[Document], embed_fn=None, dim: int = 1024) -> "SemanticIndex":
        index = cls(documents=documents, embedder=HashingEmbedder(dim=dim), embed_fn=embed_fn)
        index.reindex()
        return index

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        patterns: tuple[str, ...] = (".py", ".js", ".ts", ".tsx", ".go", ".rs", ".md"),
        max_files: int = 800,
        chunk_lines: int = 40,
        embed_fn: Callable[[str], list[float]] | None = None,
        use_embeddings: bool | None = None,
    ) -> "SemanticIndex":
        root = Path(root)
        if embed_fn is None and (use_embeddings is None or use_embeddings):
            try:
                from .embeddings import get_embed_fn

                embed_fn = get_embed_fn()
            except Exception:
                embed_fn = None
        docs: list[Document] = []
        for path in sorted(root.rglob("*")):
            if len(docs) >= max_files:
                break
            if not path.is_file() or path.suffix not in patterns:
                continue
            rel = path.relative_to(root)
            if any(part in {".git", "node_modules", ".venv", "__pycache__", ".agent", "dist", "build"} for part in rel.parts[:-1]):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = text.splitlines()
            if not lines:
                continue
            for start in range(0, len(lines), chunk_lines):
                block = "\n".join(lines[start : start + chunk_lines])
                if block.strip():
                    docs.append(
                        Document(
                            id=f"{rel}:{start + 1}",
                            text=block,
                            path=str(rel),
                            start_line=start + 1,
                            end_line=min(start + chunk_lines, len(lines)),
                        )
                    )
        return cls.build(docs, embed_fn=embed_fn)

    def reindex(self) -> "SemanticIndex":
        import time

        if self.embed_fn is None:
            self.embedder.fit(document.text for document in self.documents)
            self.vectors = [self.embedder.embed(document.text) for document in self.documents]
        else:
            self._external = [list(self.embed_fn(document.text)) for document in self.documents]
        self.updated_at = time.time()
        return self

    def search(self, query: str, k: int = 8) -> list[SearchHit]:
        if not self.documents:
            return []
        if self.embed_fn is None:
            query_vector = self.embedder.embed(query)
            scored = [
                SearchHit(self.documents[i], cosine(query_vector, vector))
                for i, vector in enumerate(self.vectors)
            ]
        else:
            query_vector = list(self.embed_fn(query))
            scored = [
                SearchHit(self.documents[i], _cosine_dense(query_vector, vector))
                for i, vector in enumerate(self._external)
            ]
        scored = [hit for hit in scored if hit.score > 0]
        scored.sort(key=lambda hit: -hit.score)
        return scored[:k]

    def keyword_search(self, query: str, k: int = 8) -> list[SearchHit]:
        """Cheap BM25-ish lexical fallback used by hybrid search."""

        terms = tokenise(query)
        if not terms:
            return []
        scored: list[SearchHit] = []
        for document in self.documents:
            tokens = tokenise(document.text)
            if not tokens:
                continue
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            score = sum(counts.get(term, 0) / (1 + len(tokens) / 200) for term in terms)
            exact_bonus = sum(2.0 for term in terms if term in document.path.lower())
            if score or exact_bonus:
                scored.append(SearchHit(document, score + exact_bonus, source="keyword"))
        scored.sort(key=lambda hit: -hit.score)
        return scored[:k]

    def hybrid_search(self, query: str, k: int = 8, rrf_k: int = 60) -> list[SearchHit]:
        semantic = self.search(query, k=k * 2)
        keyword = self.keyword_search(query, k=k * 2)
        fused: dict[str, SearchHit] = {}
        scores: dict[str, float] = {}
        for ranking in (semantic, keyword):
            for rank, hit in enumerate(ranking, 1):
                fused.setdefault(hit.document.id, hit)
                scores[hit.document.id] = scores.get(hit.document.id, 0.0) + 1.0 / (rrf_k + rank)
        ranked = sorted(fused.values(), key=lambda hit: -scores[hit.document.id])
        for hit in ranked:
            hit.score = scores[hit.document.id]
            hit.source = "hybrid"
        return ranked[:k]

    def add(self, document: Document) -> None:
        self.documents.append(document)
        if self.embed_fn is None:
            self.vectors.append(self.embedder.embed(document.text))
        else:
            self._external.append(list(self.embed_fn(document.text)))

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "dim": self.embedder.dim,
            "updated_at": self.updated_at,
            "idf": {str(key): value for key, value in self.embedder.idf.items()},
            "external": self.embed_fn is not None,
            "documents": [document.to_dict() for document in self.documents],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict()).encode("utf-8")
        target.write_bytes(gzip.compress(payload))
        return target

    @classmethod
    def load(cls, path: str | Path, embed_fn=None) -> "SemanticIndex":
        data = json.loads(gzip.decompress(Path(path).read_bytes()).decode("utf-8"))
        embedder = HashingEmbedder(dim=int(data.get("dim", 1024)))
        embedder.idf = {int(key): float(value) for key, value in (data.get("idf") or {}).items()}
        documents = [Document.from_dict(item) for item in data.get("documents") or []]
        index = cls(documents=documents, embedder=embedder, embed_fn=embed_fn)
        index.reindex()
        return index

    @classmethod
    def open(cls, root: str | Path, path: str | Path | None = None) -> "SemanticIndex":
        root = Path(root)
        target = Path(path) if path else root / ".agent" / "semantic_index.json.gz"
        if target.exists():
            try:
                return cls.load(target)
            except (OSError, ValueError, KeyError):
                pass
        return cls.from_directory(root)

    def incremental_update(self, changed_files: list[str], root: str | Path) -> list[str]:
        """Refresh only changed files; keep index accurate after every write/patch (row 4)."""
        root = Path(root)
        changed_set = set(changed_files)
        before = len(self.documents)
        self.documents = [d for d in self.documents if d.path not in changed_set]
        added = 0
        for rel in changed_files:
            path = root / rel
            if not path.is_file():
                continue
            if path.suffix not in (".py", ".js", ".ts", ".tsx", ".go", ".rs", ".md"):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = text.splitlines()
            for start in range(0, len(lines), 40):
                block = "\n".join(lines[start : start + 40])
                if block.strip():
                    doc = Document(id=f"{rel}:{start+1}", text=block, path=str(rel), start_line=start+1, end_line=min(start+40, len(lines)))
                    self.documents.append(doc)
                    added += 1
        if added or len(self.documents) != before:
            self.reindex()
            return changed_files
        return []

    def stats(self) -> dict:
        return {
            "documents": len(self.documents),
            "dim": self.embedder.dim,
            "backend": "external-embed-fn" if self.embed_fn else "hashing",
            "updated_at": self.updated_at,
        }


def _cosine_dense(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def build_index(root: str | Path, save: bool = True, embed_fn: Callable[[str], list[float]] | None = None) -> SemanticIndex:
    index = SemanticIndex.from_directory(root, embed_fn=embed_fn)
    if save:
        index.save(Path(root) / ".agent" / "semantic_index.json.gz")
    return index
