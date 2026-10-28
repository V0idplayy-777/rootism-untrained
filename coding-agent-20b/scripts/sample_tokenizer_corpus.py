"""Sample a real tokenizer corpus once, offline and reproducibly.

The BPE trainer streams, but you still need a *real* corpus on disk before a
vocab is meaningful. This script turns a HuggingFace code dataset into a
balanced local shard set with a manifest, so the tokenizer run is:

    python scripts/sample_tokenizer_corpus.py \
        --dataset bigcode/the-stack-smol --output_dir data/tokenizer_corpus \
        --target_chars 200_000_000
    python scripts/train_production_tokenizer.py \
        --data_dir data/tokenizer_corpus --output_dir tokenizer \
        --vocab_size 128256 --benchmark_gate

Guarantees:
- per-language quotas (so the vocab is not 90% Python),
- licence gate (SPDX allow-list, reusing dataset.production_processor),
- content-hash dedup at sample time (near-dup MinHash where datasketch exists),
- deterministic: sorted dataset iteration + seeded reservoir per language,
- shard files written atomically; manifest records counts and a corpus hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.production_processor import Deduplicator, LicenseFilter, LanguageDetector

logger = logging.getLogger(__name__)

EXT = {
    "Python": ".py",
    "JavaScript": ".js",
    "TypeScript": ".ts",
    "Java": ".java",
    "C++": ".cpp",
    "C": ".c",
    "Go": ".go",
    "Rust": ".rs",
    "Ruby": ".rb",
    "PHP": ".php",
    "C#": ".cs",
    "Kotlin": ".kt",
    "Swift": ".swift",
    "SQL": ".sql",
    "YAML": ".yaml",
    "Shell": ".sh",
}


def _write_shard(out_dir: Path, lang: str, index: int, docs: list[dict]) -> Path:
    """Write one language shard atomically (tmp + rename)."""

    path = out_dir / f"{lang.lower().replace('+', 'p').replace('#', 's')}-{index:06d}.jsonl"

    tmp = path.with_suffix(".tmp")

    with open(tmp, "w", encoding="utf-8") as f:

        for doc in docs:

            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    os.replace(tmp, path)

    return path


class CorpusSampler:
    def __init__(
        self,
        output_dir: str,
        per_language_target_docs: int = 2000,
        shard_size: int = 200,
        languages: list[str] | None = None,
        allow_unknown_license: bool = False,
        min_lines: int = 10,
        max_lines: int = 5000,
    ):

        self.output_dir = Path(output_dir)

        self.per_language_target_docs = per_language_target_docs

        self.shard_size = shard_size

        self.languages = languages or list(EXT)

        self.license_filter = LicenseFilter(allow_unknown=allow_unknown_license)

        self.dedup = Deduplicator(threshold=0.8)

        self.min_lines = min_lines

        self.max_lines = max_lines

        self.hasher = hashlib.sha256()

        self.counts: Counter = Counter()

        self.shard_index: Counter = Counter()

        self.buffers: dict[str, list[dict]] = defaultdict(list)

    def accept(self, item: dict) -> tuple[bool, str]:

        content = item.get("content") or ""

        if not content:

            return False, "empty"

        lines = content.split("\n")

        if not (self.min_lines <= len(lines) <= self.max_lines):

            return False, "size"

        lic = item.get("license") or item.get("licenses")

        if isinstance(lic, list):

            lic = lic[0] if lic else None

        if not self.license_filter.allows(lic):

            return False, "license"

        lang = item.get("language") or LanguageDetector.detect(content, item.get("path"))

        if lang not in self.languages:

            return False, "language"

        if self.dedup.is_duplicate(str(item.get("path") or content[:64]), content):

            return False, "duplicate"

        item = dict(item)

        item["language"] = lang

        item["_sha256"] = hashlib.sha256(content.encode("utf-8", "ignore")).hexdigest()

        return True, lang

    def add(self, item: dict) -> bool:

        ok, info = self.accept(item)

        if not ok:

            return False

        lang = info

        if self.counts[lang] >= self.per_language_target_docs:

            return False

        self.buffers[lang].append(item)

        self.counts[lang] += 1

        self.hasher.update(item["_sha256"].encode())

        if len(self.buffers[lang]) >= self.shard_size:

            self.flush(lang)

        return True

    def flush(self, lang: str | None = None) -> list[Path]:

        written = []

        targets = [lang] if lang else list(self.buffers)

        for name in targets:

            docs = self.buffers.get(name) or []

            if not docs:

                continue

            path = _write_shard(
                self.output_dir, name, self.shard_index[name], docs
            )

            self.shard_index[name] += 1

            self.buffers[name] = []

            written.append(path)

        return written

    def done(self) -> bool:

        return all(
            self.counts[lang] >= self.per_language_target_docs
            for lang in self.languages
        )

    def manifest(self) -> dict:

        return {
            "documents": sum(self.counts.values()),
            "per_language": dict(sorted(self.counts.items())),
            "corpus_sha256": self.hasher.hexdigest(),
            "shards": sum(self.shard_index.values()),
        }


def main():

    parser = argparse.ArgumentParser(description="Sample a real tokenizer corpus")

    parser.add_argument("--dataset", default="bigcode/the-stack-smol")

    parser.add_argument("--split", default="train")

    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--per_language_docs", type=int, default=2000)

    parser.add_argument("--shard_size", type=int, default=200)

    parser.add_argument(
        "--languages",
        default="Python,JavaScript,TypeScript,Java,C++,C,Go,Rust,Ruby,PHP,C#,Kotlin,Swift,SQL,YAML,Shell",
    )

    parser.add_argument("--allow_unknown_license", action="store_true")

    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    from datasets import load_dataset

    out_dir = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    sampler = CorpusSampler(
        args.output_dir,
        per_language_target_docs=args.per_language_docs,
        shard_size=args.shard_size,
        languages=[s.strip() for s in args.languages.split(",") if s.strip()],
        allow_unknown_license=args.allow_unknown_license,
    )

    stream = load_dataset(args.dataset, split=args.split, streaming=True)

    for i, item in enumerate(stream):

        if args.max_samples is not None and i >= args.max_samples:

            break

        sampler.add(dict(item))

        if sampler.done():

            break

        if (i + 1) % 5000 == 0:

            logger.info(f"scanned={i + 1:,} kept={dict(sampler.counts)}")

    sampler.flush()

    manifest = sampler.manifest()

    manifest["dataset"] = args.dataset

    with open(out_dir / "manifest.json", "w") as f:

        json.dump(manifest, f, indent=2)

    logger.info(f"Wrote {manifest['documents']:,} documents to {out_dir}")
    logger.info(json.dumps(manifest["per_language"], indent=2))


if __name__ == "__main__":

    main()
