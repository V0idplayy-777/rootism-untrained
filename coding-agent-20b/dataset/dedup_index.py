"""Constant-memory, disk-backed document dedup index (exact SHA-256 + MinHash LSH).

``dataset.production_processor.Deduplicator`` keeps every signature in RAM. That
is fine for a single fetch run of a few hundred thousand files, but a real
permissive-code backbone is 10^7 - 10^8 documents: at 16-128 signatures per
document the in-memory index costs tens of gigabytes and the pipeline dies
before it produces a shard.

``DedupIndex`` is the scale-out path. It keeps the *exact* SHA-256 of every
normalised document plus a MinHash signature and its band buckets in SQLite, so
RAM stays flat (a few MB of page cache) while the index grows on disk. It is
deterministic: the same documents in the same order produce the same verdicts,
and a reopened index produces the verdicts a single run would have.

Why both stages:

- **exact** — normalised SHA-256 catches byte-identical (or whitespace-only
  different) copies. This is the cheap, non-probabilistic stage.
- **near** — MinHash over 5-gram shingles, `bands` bands of `rows` permutations.
  Banding only decides *which* pairs are candidates; candidates are then
  verified against their stored signature, so the index never reports a
  near-duplicate that is not actually above ``threshold``. The banding curve
  can miss a pair below the LSH threshold (a false negative), never invent one.

Usage::

    from dataset.dedup_index import DedupIndex

    with DedupIndex("data/corpus_v1/dedup.sqlite") as index:
        duplicate, reason = index.check_and_add(content, path)
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

SCHEMA_VERSION = 1

DEFAULT_THRESHOLD = 0.85
DEFAULT_NUM_PERM = 16
DEFAULT_BANDS = 4
SHINGLE_SIZE = 5
MAX_SHINGLES = 512
_SHINGLE_BASE = 0x100000001B3
MASK64 = 0xFFFFFFFFFFFFFFFF
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)

_WS = re.compile(r"\s+")


def normalise_document(content: str) -> str:
    """Whitespace-collapsed document text (the dedup key space)."""
    return _WS.sub(" ", content.strip())


def shingle_hashes(content: str, *, max_shingles: int = MAX_SHINGLES) -> np.ndarray:
    """Deterministic 64-bit hashes of a document's 5-gram shingles.

    The hash of the shingle starting at ``i`` is a polynomial
    ``sum(byte * B**k)`` over its five bytes, computed with numpy shifts so the
    cost is linear in the document length rather than one ``hashlib`` call per
    position.

    Sampling is **content-addressed**: when a document has more than
    ``max_shingles`` shingles, shingles are kept by a hash-bit test, never by
    position. Positional sampling (``text[i:i+5] for i in range(0, n, step)``)
    samples a different slice of a shifted document, which drives the estimated
    Jaccard of two near-identical documents to zero - the exact bug a dedup
    index must not have.
    """

    text = normalise_document(content)
    if not text:
        return np.empty(0, dtype=np.uint64)

    data = text.encode("utf-8", "ignore")
    if len(data) < SHINGLE_SIZE:
        return np.array([_fnv1a64(data)], dtype=np.uint64)

    raw = np.frombuffer(data, dtype=np.uint8).astype(np.uint64)
    windows = raw.size - SHINGLE_SIZE + 1
    base = np.uint64(_SHINGLE_BASE)
    with np.errstate(over="ignore"):
        digest = np.zeros(windows, dtype=np.uint64)
        power = np.uint64(1)
        for offset in range(SHINGLE_SIZE):
            digest += raw[offset : offset + windows] * power
            power = power * base
    unique = np.unique(digest)
    if unique.size <= max_shingles:
        return unique

    bits = 1
    while (unique.size >> bits) > max_shingles:
        bits += 1
    keep = (unique & np.uint64((1 << bits) - 1)) == 0
    sampled = unique[keep]
    if sampled.size == 0:
        sampled = np.sort(unique)[:max_shingles]
    return sampled


def _fnv1a64(data: bytes) -> int:
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & MASK64
    return value


def _mix64_vector(values: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser, vectorised over a uint64 array (wrapping arithmetic)."""
    values = values + _GOLDEN
    values = (values ^ (values >> np.uint64(30))) * _M1
    values = (values ^ (values >> np.uint64(27))) * _M2
    return values ^ (values >> np.uint64(31))


def minhash_signature(
    content: str,
    num_perm: int = DEFAULT_NUM_PERM,
    *,
    max_shingles: int = MAX_SHINGLES,
) -> tuple[int, ...]:
    """MinHash signature of the whitespace-normalised 5-gram shingles of a document.

    ``perm`` i hashes every shingle with a different seed and keeps the minimum,
    which is the standard MinHash estimator of Jaccard similarity.
    """

    hashes = shingle_hashes(content, max_shingles=max_shingles)
    if hashes.size == 0:
        return tuple()

    base_seed = int.from_bytes(hashlib.blake2b(b"minhash-perm", digest_size=8).digest(), "little")
    golden = int(_GOLDEN)
    seeds = np.array(
        [(base_seed + perm * golden) & MASK64 for perm in range(num_perm)],
        dtype=np.uint64,
    )
    signature = np.empty(num_perm, dtype=np.uint64)
    for perm in range(num_perm):
        signature[perm] = _mix64_vector(hashes ^ seeds[perm]).min()
    return tuple(int(value) for value in signature)


def pack_signature(signature: Sequence[int]) -> bytes:
    return b"".join(int(value).to_bytes(8, "little") for value in signature)


def unpack_signature(blob: bytes) -> tuple[int, ...]:
    return tuple(
        int.from_bytes(blob[i : i + 8], "little") for i in range(0, len(blob), 8)
    )


def jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    """Estimated Jaccard similarity from two MinHash signatures (or two shingle sets)."""
    left_len, right_len = len(left), len(right)
    if left_len == 0 or right_len == 0 or left_len != right_len:
        return 0.0
    equal = sum(1 for a, b in zip(left, right) if a == b)
    return equal / left_len


def true_jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    """Exact Jaccard similarity of two shingle collections (used to audit MinHash)."""
    lhs, rhs = set(int(v) for v in left), set(int(v) for v in right)
    if not lhs or not rhs:
        return 0.0
    return len(lhs & rhs) / len(lhs | rhs)


def band_keys(
    signature: Sequence[int], bands: int = DEFAULT_BANDS
) -> list[tuple[int, int]]:
    """``(band_index, band_key)`` pairs used as LSH bucket keys."""
    if not signature:
        return []
    rows = max(1, len(signature) // bands)
    keys: list[tuple[int, int]] = []
    for band in range(bands):
        chunk = signature[band * rows : (band + 1) * rows]
        if not chunk:
            continue
        digest = hashlib.blake2b(
            pack_signature(chunk), digest_size=8
        ).digest()
        keys.append((band, int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF))
    return keys


class DedupIndex:
    """Disk-backed exact + MinHash near-duplicate index (constant RAM).

    ``path`` may be ``":memory:"`` for tests and short-lived runs. Documents are
    namespaced (``train`` vs ``heldout``) so a held-out document can be
    registered without ever making a training document look duplicated.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        threshold: float = DEFAULT_THRESHOLD,
        num_perm: int = DEFAULT_NUM_PERM,
        bands: int = DEFAULT_BANDS,
        namespace: str = "train",
        batch_size: int = 2000,
    ):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if num_perm < bands or bands < 1:
            raise ValueError("num_perm must be >= bands >= 1")

        self.path = str(path)
        self.threshold = float(threshold)
        self.num_perm = int(num_perm)
        self.bands = int(bands)
        self.namespace = namespace
        self.batch_size = max(1, int(batch_size))

        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        self._db = sqlite3.connect(self.path, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=MEMORY")
        self._db.execute("PRAGMA synchronous=OFF")
        self._db.execute("PRAGMA temp_store=MEMORY")
        self._db.execute("PRAGMA cache_size=-32768")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                namespace TEXT NOT NULL,
                digest    TEXT NOT NULL,
                doc_id    TEXT NOT NULL,
                signature BLOB NOT NULL,
                PRIMARY KEY (namespace, digest)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS bands (
                namespace TEXT NOT NULL,
                band      INTEGER NOT NULL,
                key       INTEGER NOT NULL,
                digest    TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS bands_lookup
                ON bands (namespace, band, key);
            CREATE TABLE IF NOT EXISTS metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self._db.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._pending_docs: list[tuple[str, str, str, bytes]] = []
        self._pending_bands: list[tuple[str, int, int, str]] = []
        self._pending_digests: set[str] = set()
        self._pending_signatures: dict[tuple[int, int], list[bytes]] = {}

        self.total_checked = 0
        self.exact_duplicates = 0
        self.near_duplicates = 0
        self._band_probes = 0

    def __enter__(self) -> "DedupIndex":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.flush()
        try:
            self._db.commit()
        finally:
            self._db.close()

    def __len__(self) -> int:
        return self.count()

    def count(self) -> int:
        self.flush()
        row = self._db.execute(
            "SELECT COUNT(*) FROM documents WHERE namespace = ?", (self.namespace,)
        ).fetchone()
        return int(row[0]) if row else 0

    def flush(self) -> None:
        if self._pending_docs:
            self._db.executemany(
                "INSERT OR IGNORE INTO documents (namespace, digest, doc_id, signature) "
                "VALUES (?, ?, ?, ?)",
                self._pending_docs,
            )
            self._pending_docs.clear()
        if self._pending_bands:
            self._db.executemany(
                "INSERT INTO bands (namespace, band, key, digest) VALUES (?, ?, ?, ?)",
                self._pending_bands,
            )
            self._pending_bands.clear()
        self._pending_digests.clear()
        self._pending_signatures.clear()

    def is_duplicate(self, doc_id: str, content: str, *, remember: bool = True) -> bool:
        """``Deduplicator.is_duplicate``-compatible wrapper.

        ``DataProcessor`` and ``CorpusWriter`` speak this two-argument API, so
        the disk-backed index can be swapped in wherever the in-RAM one is used
        without touching either caller.
        """

        found, _reason = self.check_and_add(content, doc_id, remember=remember)
        return found

    def check(
        self, content: str, doc_id: str = ""
    ) -> tuple[bool, str]:
        """Is ``content`` already indexed? Never mutates the index."""

        return self.check_and_add(content, doc_id, remember=False)

    def check_and_add(
        self,
        content: str,
        doc_id: str = "",
        *,
        remember: bool = True,
    ) -> tuple[bool, str]:
        """Return ``(is_duplicate, reason)``; index the document when new.

        ``reason`` is one of ``"exact"``, ``"near"``, ``""`` (new) or
        ``"empty"`` (a blank document is a duplicate of nothing and is rejected
        upstream, but the index refuses to hash it).
        """

        if not content or not content.strip():
            return True, "empty"

        self.total_checked += 1
        normalised = normalise_document(content)
        digest = hashlib.sha256(normalised.encode("utf-8", "ignore")).hexdigest()

        if digest in self._pending_digests:
            self.exact_duplicates += 1
            return True, "exact"

        known = self._db.execute(
            "SELECT 1 FROM documents WHERE namespace = ? AND digest = ? LIMIT 1",
            (self.namespace, digest),
        ).fetchone()
        if known:
            self.exact_duplicates += 1
            return True, "exact"

        signature = minhash_signature(content, self.num_perm)
        if signature and self.bands > 0:
            for band, key in band_keys(signature, self.bands):
                for pending_signature in self._pending_signatures.get((band, key), ()):
                    if jaccard(signature, unpack_signature(pending_signature)) >= self.threshold:
                        self.near_duplicates += 1
                        return True, "near"
                candidates = self._db.execute(
                    "SELECT digest FROM bands WHERE namespace = ? AND band = ? AND key = ? "
                    "LIMIT 8",
                    (self.namespace, band, key),
                ).fetchall()
                self._band_probes += 1
                if not candidates:
                    continue
                digests = [row[0] for row in candidates]
                placeholders = ",".join("?" for _ in digests)
                rows = self._db.execute(
                    f"SELECT signature FROM documents WHERE namespace = ? "
                    f"AND digest IN ({placeholders})",
                    (self.namespace, *digests),
                ).fetchall()
                for (blob,) in rows:
                    if jaccard(signature, unpack_signature(blob)) >= self.threshold:
                        self.near_duplicates += 1
                        return True, "near"

        if not remember:
            return False, ""

        packed = pack_signature(signature)
        self._pending_docs.append((self.namespace, digest, doc_id, packed))
        self._pending_digests.add(digest)
        for band, key in band_keys(signature, self.bands):
            self._pending_bands.append((self.namespace, band, key, digest))
            self._pending_signatures.setdefault((band, key), []).append(packed)
        if len(self._pending_docs) >= self.batch_size:
            self.flush()
        return False, ""

    def stats(self) -> dict[str, Any]:
        removed = self.exact_duplicates + self.near_duplicates
        return {
            "index": self.path,
            "namespace": self.namespace,
            "threshold": self.threshold,
            "num_perm": self.num_perm,
            "bands": self.bands,
            "checked": self.total_checked,
            "indexed": self.count(),
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "duplicates": removed,
            "dedup_rate": round(removed / max(self.total_checked, 1), 6),
            "band_probes": self._band_probes,
        }

    def merge(self, other: "DedupIndex", namespace: str | None = None) -> int:
        """Copy every document of ``other`` into this index; returns rows added."""

        other.flush()
        self.flush()
        target = namespace or self.namespace
        rows = other._db.execute(
            "SELECT digest, doc_id, signature FROM documents WHERE namespace = ?",
            (other.namespace,),
        ).fetchall()
        before = self.count()
        self._db.executemany(
            "INSERT OR IGNORE INTO documents (namespace, digest, doc_id, signature) "
            "VALUES (?, ?, ?, ?)",
            [(target, digest, doc_id, blob) for digest, doc_id, blob in rows],
        )
        self._db.executemany(
            "INSERT INTO bands (namespace, band, key, digest) VALUES (?, ?, ?, ?)",
            [
                (target, band, key, digest)
                for digest, blob in ((row[0], row[2]) for row in rows)
                for band, key in band_keys(unpack_signature(blob), self.bands)
            ],
        )
        return self.count() - before
