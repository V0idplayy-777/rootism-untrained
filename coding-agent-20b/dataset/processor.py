"""
Dataset Processor - Tokenization, Sharding, and Streaming.

Handles:
- Tokenizing raw text into binary format
- Sharding dataset for distributed training
- Creating memory-mapped datasets for efficient loading
- Data deduplication and quality filtering
- Creating train/val/test splits
"""

import hashlib
import json
import logging
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""

    max_seq_length: int = 8192

    train_split: float = 0.95

    val_split: float = 0.04

    test_split: float = 0.01

    shard_size_mb: int = 256

    min_sequence_length: int = 32

    dedup_threshold: float = 0.85

    num_workers: int = 4


class TextDeduplicator:
    """
    Near-duplicate detection using MinHash and LSH.
    Removes semantically similar documents to improve data quality.
    """

    def __init__(self, num_hashes: int = 128, num_bands: int = 16):

        self.num_hashes = num_hashes

        self.num_bands = num_bands

        self.rows_per_band = num_hashes // num_bands

        self.signatures: dict[str, np.ndarray] = {}

        self.buckets: dict[tuple[int, ...], list[str]] = {}

    def _shingles(self, text: str, k: int = 5) -> set:
        """Generate k-character shingles from text."""

        text = text.lower().strip()

        if len(text) < k:

            return {text}

        return {text[i : i + k] for i in range(len(text) - k + 1)}

    def _minhash(self, shingles: set) -> np.ndarray:
        """Compute MinHash signature for a set of shingles."""

        signature = np.full(self.num_hashes, np.inf)

        for shingle in shingles:

            h = hashlib.sha256(shingle.encode()).digest()

            hash_val = int.from_bytes(h[:4], "big")

            for i in range(self.num_hashes):

                permuted = (hash_val ^ ((i * 2654435761) & 0xFFFFFFFF)) & 0xFFFFFFFF

                signature[i] = min(signature[i], permuted)

        return signature

    def add(self, doc_id: str, text: str) -> bool:
        """
        Add a document. Returns True if it's NOT a near-duplicate.
        """

        shingles = self._shingles(text)

        if not shingles:

            return False

        signature = self._minhash(shingles)

        for band_idx in range(self.num_bands):

            start = band_idx * self.rows_per_band

            end = start + self.rows_per_band

            band = tuple(signature[start:end].astype(np.int64))

            bucket_key = (band_idx, band)

            if bucket_key in self.buckets:

                for existing_id in self.buckets[bucket_key]:

                    existing_sig = self.signatures[existing_id]

                    similarity = np.mean(signature == existing_sig)

                    if similarity > 0.85:

                        return False

        self.signatures[doc_id] = signature

        for band_idx in range(self.num_bands):

            start = band_idx * self.rows_per_band

            end = start + self.rows_per_band

            band = tuple(signature[start:end].astype(np.int64))

            bucket_key = (band_idx, band)

            self.buckets.setdefault(bucket_key, []).append(doc_id)

        return True


class QualityFilter:
    """Filter low-quality code samples based on heuristics."""

    @staticmethod
    def is_quality_code(text: str) -> tuple[bool, str]:
        """
        Check if text is quality code. Returns (passes, reason).
        """

        lines = text.strip().split("\n")

        if len(lines) < 3:

            return False, "Too short (< 3 lines)"

        unique_lines = set(lines)

        repetition_ratio = 1 - len(unique_lines) / len(lines)

        if repetition_ratio > 0.5:

            return False, f"High repetition ({repetition_ratio:.1%})"

        code_indicators = [
            "def ",
            "class ",
            "import ",
            "from ",
            "return ",
            "function ",
            "const ",
            "let ",
            "var ",
            "if ",
            "for ",
            "while ",
            "try ",
            "catch ",
            "async ",
            "=>",
            "::",
            "->",
        ]

        has_code = any(indicator in text for indicator in code_indicators)

        if not has_code:

            return False, "No code indicators found"

        brackets = {"(": ")", "[": "]", "{": "}"}

        stack = []

        for char in text:

            if char in brackets:

                stack.append(brackets[char])

            elif char in brackets.values():

                if not stack or stack[-1] != char:

                    return False, "Unbalanced brackets"

                stack.pop()

        if len(stack) > len(text) * 0.05:

            return False, "Too many unclosed brackets"

        return True, "OK"


class DatasetSharder:
    """
    Splits tokenized dataset into shards for efficient distributed loading.
    Each shard is a binary file containing packed token sequences.
    """

    def __init__(self, output_dir: str, shard_size_tokens: int = 1_000_000):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.shard_size_tokens = shard_size_tokens

    def create_shards(
        self,
        token_ids: list[int],
        prefix: str = "train",
    ) -> list[str]:
        """
        Split token IDs into binary shard files.

        Each shard is a flat array of uint32 token IDs.
        """

        shard_paths = []

        total_tokens = len(token_ids)

        num_shards = (
            total_tokens + self.shard_size_tokens - 1
        ) // self.shard_size_tokens

        logger.info(f"Creating {num_shards} shards from {total_tokens:,} tokens")

        for i in range(num_shards):

            start = i * self.shard_size_tokens

            end = min(start + self.shard_size_tokens, total_tokens)

            shard_tokens = token_ids[start:end]

            shard_path = self.output_dir / f"{prefix}_{i:05d}.bin"

            arr = np.array(shard_tokens, dtype=np.uint32)

            arr.tofile(str(shard_path))

            shard_paths.append(str(shard_path))

            logger.info(f"  Shard {i}: {len(shard_tokens):,} tokens -> {shard_path}")

        manifest = {
            "num_shards": num_shards,
            "total_tokens": total_tokens,
            "shard_size_tokens": self.shard_size_tokens,
            "shards": [
                {
                    "path": os.path.basename(p),
                    "tokens": (
                        self.shard_size_tokens
                        if i < num_shards - 1
                        else total_tokens - (num_shards - 1) * self.shard_size_tokens
                    ),
                }
                for i, p in enumerate(shard_paths)
            ],
        }

        manifest_path = self.output_dir / f"{prefix}_manifest.json"

        with open(manifest_path, "w") as f:

            json.dump(manifest, f, indent=2)

        return shard_paths


class MemoryMappedDataset:
    """
    Memory-mapped dataset for efficient training data loading.

    Uses numpy memory mapping to avoid loading entire dataset into RAM.
    Supports random access and sequential iteration.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        seq_length: int = 8192,
    ):

        self.data_dir = Path(data_dir)

        self.split = split

        self.seq_length = seq_length

        manifest_path = self.data_dir / f"{split}_manifest.json"

        with open(manifest_path) as f:

            self.manifest = json.load(f)

        self.shards = []

        self.shard_offsets = [0]

        for shard_info in self.manifest["shards"]:

            shard_path = self.data_dir / shard_info["path"]

            mmap = np.memmap(str(shard_path), dtype=np.uint32, mode="r")

            self.shards.append(mmap)

            self.shard_offsets.append(self.shard_offsets[-1] + len(mmap))

        self.total_tokens = self.manifest["total_tokens"]

        self.num_sequences = self.total_tokens // self.seq_length

        logger.info(
            f"Loaded {split} dataset: {self.total_tokens:,} tokens, "
            f"{self.num_sequences:,} sequences of length {seq_length}"
        )

    def __len__(self) -> int:

        return self.num_sequences

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get a sequence by index."""

        start = idx * self.seq_length

        end = start + self.seq_length + 1

        tokens = self._get_tokens(start, end)

        return tokens

    def _get_tokens(self, start: int, end: int) -> np.ndarray:
        """Get tokens spanning potentially multiple shards."""

        shard_idx = 0

        for i in range(len(self.shard_offsets) - 1):

            if self.shard_offsets[i] <= start < self.shard_offsets[i + 1]:

                shard_idx = i

                break

        local_start = start - self.shard_offsets[shard_idx]

        local_end = min(
            end - self.shard_offsets[shard_idx], len(self.shards[shard_idx])
        )

        tokens = self.shards[shard_idx][local_start:local_end].copy()

        remaining = (end - start) - len(tokens)

        if remaining > 0 and shard_idx + 1 < len(self.shards):

            next_tokens = self.shards[shard_idx + 1][:remaining].copy()

            tokens = np.concatenate([tokens, next_tokens])

        return tokens.astype(np.int64)

    def get_batch(
        self, batch_size: int, indices: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get a batch of sequences."""

        if indices is None:

            indices = random.sample(range(len(self)), batch_size)

        batch = np.stack([self[i] for i in indices])

        inputs = batch[:, :-1]

        targets = batch[:, 1:]

        return inputs, targets

    def iterate_epoch(
        self, batch_size: int, shuffle: bool = True
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterate over one epoch of data."""

        indices = list(range(len(self)))

        if shuffle:

            random.shuffle(indices)

        for i in range(0, len(indices), batch_size):

            batch_indices = indices[i : i + batch_size]

            if len(batch_indices) < batch_size:

                continue

            yield self.get_batch(batch_size, batch_indices)


class DatasetProcessor:
    """
    Complete dataset processing pipeline.

    Steps:
    1. Quality filtering
    2. Deduplication
    3. Tokenization
    4. Packing into sequences
    5. Sharding
    6. Train/val/test split
    """

    def __init__(self, config: DatasetConfig, tokenizer):

        self.config = config

        self.tokenizer = tokenizer

        self.deduplicator = TextDeduplicator()

        self.quality_filter = QualityFilter()

    def process(
        self,
        raw_texts: list[dict[str, Any]],
        output_dir: str,
    ) -> dict[str, Any]:
        """
        Process raw text data into sharded token dataset.

        Args:
            raw_texts: List of dicts with 'text', 'category', etc.
            output_dir: Directory to write processed shards

        Returns:
            Processing statistics
        """

        stats = {
            "total_documents": len(raw_texts),
            "filtered_quality": 0,
            "filtered_duplicate": 0,
            "total_tokens": 0,
            "sequences_created": 0,
        }

        all_tokens = []

        logger.info(f"Processing {len(raw_texts)} documents...")

        for i, doc in enumerate(raw_texts):

            if (i + 1) % 10000 == 0:

                logger.info(f"  Processed {i + 1}/{len(raw_texts)} documents")

            text = doc.get("text", "")

            passes, _reason = self.quality_filter.is_quality_code(text)

            if not passes:

                stats["filtered_quality"] += 1

                continue

            doc_id = hashlib.md5(text[:200].encode()).hexdigest()

            if not self.deduplicator.add(doc_id, text):

                stats["filtered_duplicate"] += 1

                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=True)

            tokens.append(self.tokenizer.eos_token_id)

            if len(tokens) < self.config.min_sequence_length:

                continue

            all_tokens.extend(tokens)

        stats["total_tokens"] = len(all_tokens)

        logger.info(f"Total tokens after filtering: {len(all_tokens):,}")

        total = len(all_tokens)

        train_end = int(total * self.config.train_split)

        val_end = train_end + int(total * self.config.val_split)

        splits = {
            "train": all_tokens[:train_end],
            "val": all_tokens[train_end:val_end],
            "test": all_tokens[val_end:],
        }

        sharder = DatasetSharder(
            output_dir,
            shard_size_tokens=1_000_000,
        )

        for split_name, tokens in splits.items():

            if tokens:

                paths = sharder.create_shards(tokens, prefix=split_name)

                stats[f"{split_name}_shards"] = len(paths)

                stats[f"{split_name}_tokens"] = len(tokens)

        stats["sequences_created"] = total // self.config.max_seq_length

        stats_path = os.path.join(output_dir, "processing_stats.json")

        with open(stats_path, "w") as f:

            json.dump(stats, f, indent=2)

        logger.info(f"Processing complete: {json.dumps(stats, indent=2)}")

        return stats


class StreamingDataset(MemoryMappedDataset):
    """Backwards-compatible name used by training/train_production.py.

    Extra rank/world_size args are accepted (sharding is handled by the
    sampler); the underlying storage is the memory-mapped shards.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        seq_length: int = 8192,
        rank: int = 0,
        world_size: int = 1,
        **kwargs,
    ):

        super().__init__(data_dir=data_dir, split=split, seq_length=seq_length)

        self.rank = rank

        self.world_size = world_size

    def __iter__(self):

        n = len(self)

        idxs = [i for i in range(n) if i % max(self.world_size, 1) == self.rank]

        for i in idxs:

            tokens = self[i]

            yield {"input_ids": tokens[:-1], "labels": tokens[1:]}
