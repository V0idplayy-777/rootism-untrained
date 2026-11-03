"""Large-Scale Real-Code Dataset Ingestion & Subsampler (Corpus v1).

Streams and filters real permissively licensed code at multi-billion-token scale
from:
- HuggingFace streaming datasets (``the-stack-v2``, ``the-stack-dedup``,
  ``starcoderdata``, ``the-stack-github-issues``, ``opencoder``),
- Local JSONL shards fetched by ``scripts/fetch_real_corpus.py``
  (``--local_jsonl_dir``),
- Local checked-out Git repositories (``--local_repos``).

All documents pass through ``dataset.production_processor.DataProcessor``:
1. Fail-closed SPDX permissive license gate (unknown / copyleft rejected)
2. Hard quality & structural complexity filters (line bounds, byte bounds,
   no minified/vendored/generated/lockfiles, syntax & complexity floor)
3. Document-level exact SHA-256 + MinHash LSH near-duplicate removal
4. Held-out isolation registry (0 overlap with ``--heldout_dir``)
5. Streaming token-budget cap / deterministic subsampling (e.g. 25B+ raw tokens
   subsampled cleanly to the 18B-token 3B budget) without buffering the corpus
   in RAM, routed straight into ``ProductionShardBuilder`` (4096-token uint32
   shards + ``corpus_manifest.json``) or sharded JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.production_processor import (
    DataProcessor,
    Document,
    HeldoutRegistry,
    ProductionShardBuilder,
)
from dataset.dedup_index import DedupIndex
from scripts.fetch_real_corpus import (
    CorpusWriter,
    HeldoutWriter,
    classify_license_text,
    is_heldout_document,
    merged_github_repos,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSourceSpec:
    name: str
    huggingface_id: str
    description: str
    languages: tuple[str, ...]
    estimated_tokens: str
    default_license: str | None
    slice_hint: str | None = None


DATASETS: dict[str, DatasetSourceSpec] = {
    "the-stack-v2": DatasetSourceSpec(
        name="The Stack v2 (Permissive)",
        huggingface_id="bigcode/the-stack-v2-dedup",
        description="Multi-hundred-billion-token permissively licensed source code",
        languages=("python", "typescript", "javascript", "go", "rust", "java", "cpp", "c", "csharp", "kotlin", "swift", "ruby", "php", "scala", "sql", "shell"),
        estimated_tokens="600B+",
        default_license=None,
    ),
    "the-stack": DatasetSourceSpec(
        name="The Stack Dedup",
        huggingface_id="bigcode/the-stack-dedup",
        description="Permissively licensed source code from GitHub (SPDX per file)",
        languages=("python", "typescript", "javascript", "go", "rust", "java", "cpp", "c", "ruby", "php", "csharp", "sql", "shell"),
        estimated_tokens="300B+",
        default_license=None,
    ),
    "starcoderdata": DatasetSourceSpec(
        name="StarCoderData",
        huggingface_id="bigcode/starcoderdata",
        description="Filtered code, GitHub issues, and commits/PRs",
        languages=("python", "typescript", "javascript", "go", "rust", "java", "cpp", "c", "csharp", "kotlin", "swift", "sql", "shell"),
        estimated_tokens="250B+",
        default_license=None,
    ),
    "github-issues": DatasetSourceSpec(
        name="The Stack GitHub Issues",
        huggingface_id="bigcode/the-stack-github-issues",
        description="Engineering issue and PR discussion threads from permissive repos",
        languages=("markdown",),
        estimated_tokens="20B+",
        default_license=None,
        slice_hint="github-issues",
    ),
    "opencoder": DatasetSourceSpec(
        name="OpenCoder Filtered SFT",
        huggingface_id="OpenCoder-LLM/opc-sft-stage1",
        description="Apache-2.0 filtered coding instruction & reasoning samples",
        languages=("python", "typescript", "javascript", "java", "cpp", "go", "rust"),
        estimated_tokens="5B",
        default_license="Apache-2.0",
        slice_hint="instruction-sft",
    ),
}


def format_instruction(sample: dict[str, Any]) -> str:
    parts: list[str] = []
    if sample.get("instruction"):
        parts.append(f"<|user|>\n{sample['instruction']}")
    if sample.get("input"):
        parts.append(f"\n```\n{sample['input']}\n```")
    if sample.get("output"):
        parts.append(f"\n<|assistant|>\n{sample['output']}")
    return "\n".join(parts)


def stream_huggingface_dataset(
    spec: DatasetSourceSpec,
    *,
    languages: list[str] | None = None,
    max_samples: int | None = None,
    subsample_rate: float = 1.0,
    seed: int = 42,
) -> Iterator[dict[str, Any]]:
    """Stream raw items from HuggingFace without loading dataset into disk/RAM."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("HuggingFace streaming requires `datasets` (`pip install datasets`)") from exc

    allow = {lang.lower() for lang in (languages or spec.languages)}
    dataset = load_dataset(spec.huggingface_id, split="train", streaming=True)
    emitted = 0
    for idx, sample in enumerate(dataset):
        if subsample_rate < 1.0:
            h = int(hashlib.sha256(f"{seed}:{spec.name}:{idx}".encode()).hexdigest()[:8], 16)
            if (h / 0xFFFFFFFF) > subsample_rate:
                continue

        lang = str(sample.get("lang") or sample.get("language") or "").lower().strip()
        if lang and allow and lang not in allow:
            continue

        content = (
            sample.get("content")
            or sample.get("text")
            or sample.get("code")
            or (format_instruction(sample) if "instruction" in sample else None)
        )
        if not content:
            continue

        raw_lic = (
            sample.get("license")
            or sample.get("max_stars_repo_licenses")
            or sample.get("licenses")
            or spec.default_license
        )
        path = str(
            sample.get("max_stars_repo_path")
            or sample.get("path")
            or f"{spec.name}:{idx}"
        )
        repo = str(sample.get("max_stars_repo_name") or sample.get("repo_name") or spec.name)

        yield {
            "content": str(content),
            "language": lang or "unknown",
            "license": raw_lic,
            "source": f"hf:{spec.name}:{repo}",
            "path": f"{repo}:{path}",
            "slice": spec.slice_hint,
        }
        emitted += 1
        if max_samples is not None and emitted >= max_samples:
            break


def stream_local_jsonl(jsonl_dir: str | Path) -> Iterator[dict[str, Any]]:
    """Stream JSONL shards written by ``scripts/fetch_real_corpus.py``."""
    root = Path(jsonl_dir)
    if not root.exists():
        raise FileNotFoundError(f"JSONL directory not found: {root}")
    for shard in sorted(root.glob("*.jsonl")):
        if shard.name in {"sources.jsonl"} or shard.name.startswith("heldout"):
            continue
        with open(shard, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("content"):
                    yield row


def _detect_repo_license(repo_dir: Path) -> str | None:
    for candidate in sorted(repo_dir.iterdir()):
        if not candidate.is_file():
            continue
        if candidate.name.lower().startswith(("license", "copying")):
            try:
                text = candidate.read_text(encoding="utf-8", errors="ignore")[:10000]
            except OSError:
                continue
            got = classify_license_text(text)
            if got:
                return got
    return None


def stream_local_repos(repos_dir: str | Path) -> Iterator[dict[str, Any]]:
    """Stream source files from local Git repositories with fail-closed LICENSE detection."""
    root = Path(repos_dir)
    if not root.exists():
        raise FileNotFoundError(f"Local repos directory not found: {root}")

    candidates = [d for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith(".")]
    if not candidates or (root / ".git").exists() or any(f.is_file() and f.name.lower().startswith("license") for f in root.iterdir()):
        candidates = [root]

    skip_parts = {"node_modules", ".git", "__pycache__", ".venv", "venv", "build", "dist", "vendor", "third_party", "target"}
    for repo in candidates:
        license_id = _detect_repo_license(repo)
        if not license_id:
            logger.info("Skipping local repo %s: no permissive LICENSE detected", repo.name)
            continue
        for filepath in sorted(repo.rglob("*")):
            if not filepath.is_file():
                continue
            rel = filepath.relative_to(repo)
            if any(part in skip_parts for part in rel.parts):
                continue
            ext = filepath.suffix.lower()
            if ext not in {
                ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".cpp", ".cc", ".c", ".h",
                ".go", ".rs", ".rb", ".php", ".cs", ".kt", ".swift", ".scala", ".sql", ".sh", ".md",
            }:
                continue
            try:
                content = filepath.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            yield {
                "content": content,
                "language": "",
                "license": license_id,
                "source": f"local:{repo.name}",
                "path": f"{repo.name}:{rel}",
            }


class DatasetBuilder:
    """Streaming real-code dataset builder backed by ``ProductionShardBuilder``."""

    def __init__(
        self,
        output_dir: str | Path,
        tokenizer,
        *,
        seq_length: int = 4096,
        heldout_dir: str | Path | None = None,
        min_quality: float = 0.25,
        dedup_threshold: float = 0.85,
        seed: int = 42,
        dedup_index_path: str | Path | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.seed = seed
        self.heldout_dir = Path(heldout_dir) if heldout_dir else (self.output_dir.parent / "heldout")

        self.heldout = HeldoutRegistry()
        self.heldout.load_directory(self.heldout_dir)
        self.processor = DataProcessor(
            min_quality=min_quality,
            dedup_threshold=dedup_threshold,
            heldout=self.heldout,
        )
        # At real-corpus scale the in-RAM signature index is the limiting factor,
        # so the builder can hand both filters a SQLite-backed index that keeps
        # RAM flat and survives across runs.
        self.dedup_index = (
            DedupIndex(dedup_index_path, threshold=dedup_threshold, namespace="train")
            if dedup_index_path is not None
            else None
        )
        if self.dedup_index is not None:
            self.processor.dedup = self.dedup_index

    def build(
        self,
        *,
        dataset_names: list[str] | None = None,
        local_jsonl_dir: str | Path | None = None,
        local_repos: str | Path | None = None,
        emit_jsonl_dir: str | Path | None = None,
        target_tokens: int = 18_000_000_000,
        max_samples_per_dataset: int | None = None,
        languages: list[str] | None = None,
        subsample_rate: float = 1.0,
        heldout_every: int = 20,
    ) -> dict[str, Any]:
        shard_builder = ProductionShardBuilder(
            self.output_dir,
            self.tokenizer,
            seq_length=self.seq_length,
            heldout=self.heldout,
            seed=self.seed,
        )
        jsonl_writer = (
            CorpusWriter(Path(emit_jsonl_dir), dedup_index=self.dedup_index)
            if emit_jsonl_dir
            else None
        )
        heldout_writer = HeldoutWriter(self.heldout_dir)

        per_source_docs: Counter[str] = Counter()
        total_tokens = 0

        def _consume_stream(label: str, items: Iterator[dict[str, Any]]) -> None:
            nonlocal total_tokens
            for raw in items:
                if total_tokens >= target_tokens:
                    break
                path = str(raw.get("path") or raw.get("source") or label)
                if is_heldout_document(path, heldout_every=heldout_every):
                    canon_lic = self.processor.license_filter.canonicalize(raw.get("license"))
                    if canon_lic and not self.heldout.is_heldout(raw.get("content", ""), path):
                        doc_tmp = Document(
                            content=raw["content"],
                            language=raw.get("language") or "unknown",
                            source=str(raw.get("source") or label),
                            license=canon_lic,
                            path=path,
                        )
                        row = doc_tmp.to_jsonl_row()
                        heldout_writer.add(row)
                        self.heldout.register_text(row["content"], row["path"])
                    continue

                doc = self.processor.process_document(
                    raw["content"],
                    str(raw.get("source") or label),
                    raw.get("license"),
                    path=path,
                    language_hint=raw.get("language"),
                    slice_hint=raw.get("slice"),
                )
                if doc is None:
                    continue
                if languages and doc.language not in {l.lower() for l in languages}:
                    continue

                if jsonl_writer is not None:
                    jsonl_writer.add(doc.language, doc.to_jsonl_row())
                added = shard_builder.add_document(doc)
                total_tokens += added
                per_source_docs[label] += 1

        if local_jsonl_dir:
            _consume_stream("local_jsonl", stream_local_jsonl(local_jsonl_dir))

        if local_repos and total_tokens < target_tokens:
            _consume_stream("local_repos", stream_local_repos(local_repos))

        for name in dataset_names or []:
            if total_tokens >= target_tokens:
                break
            if name not in DATASETS:
                logger.warning("Unknown dataset %r (known: %s)", name, ", ".join(sorted(DATASETS)))
                continue
            spec = DATASETS[name]
            logger.info("Streaming %s (%s)...", spec.name, spec.huggingface_id)
            _consume_stream(
                name,
                stream_huggingface_dataset(
                    spec,
                    languages=languages,
                    max_samples=max_samples_per_dataset,
                    subsample_rate=subsample_rate,
                    seed=self.seed,
                ),
            )

        if jsonl_writer is not None:
            jsonl_writer.flush()
        if heldout_writer.rows:
            heldout_writer.flush()

        dedup_stats = self.dedup_index.stats() if self.dedup_index is not None else None
        processor_stats = self.processor.get_stats()
        if self.dedup_index is not None:
            self.dedup_index.close()

        corpus_manifest = shard_builder.finalize(
            extra_metadata={
                "target_tokens": target_tokens,
                "subsample_rate": subsample_rate,
                "dedup_backend": (
                    "dataset.dedup_index.DedupIndex (sqlite)" if dedup_stats else
                    "dataset.production_processor.Deduplicator (in-memory)"
                ),
                "dedup": dedup_stats or {},
                "processor_stats": processor_stats,
                "ingested_sources": dict(per_source_docs),
                "configured_datasets": [
                    asdict(DATASETS[n]) for n in (dataset_names or []) if n in DATASETS
                ],
            }
        )
        return corpus_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build production 3B/20B shards from real code sources")
    parser.add_argument("--dataset", type=str, default="", help="Comma-separated HF datasets (e.g. the-stack-v2,starcoderdata)")
    parser.add_argument("--local_jsonl_dir", type=str, default=None, help="Directory of JSONL shards from fetch_real_corpus.py")
    parser.add_argument("--local_repos", type=str, default=None, help="Directory of local git repositories")
    parser.add_argument("--emit_jsonl_dir", type=str, default=None, help="Optional directory to also write filtered JSONL shards")
    parser.add_argument("--heldout_dir", type=str, default="data/corpus_v1/heldout")
    parser.add_argument("--output", type=str, default="data/corpus_v1/shards_3b")
    parser.add_argument("--tokenizer", type=str, default="tokenizer")
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--target_tokens", type=int, default=18_000_000_000)
    parser.add_argument("--subsample_rate", type=float, default=1.0)
    parser.add_argument("--languages", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--dedup_index",
        type=str,
        default=None,
        help="SQLite dedup index path; required in practice for billion-token ingestion",
    )
    args = parser.parse_args()

    from model.tokenizer import CodeTokenizer

    tok_path = Path(args.tokenizer)
    tokenizer = CodeTokenizer.load(str(tok_path)) if (tok_path / "tokenizer.json").exists() else CodeTokenizer()

    dataset_names = [d.strip() for d in args.dataset.split(",") if d.strip()]
    languages = [l.strip() for l in args.languages.split(",") if l.strip()] if args.languages else None

    builder = DatasetBuilder(
        args.output,
        tokenizer,
        seq_length=args.seq_length,
        heldout_dir=args.heldout_dir,
        dedup_index_path=args.dedup_index,
    )
    manifest = builder.build(
        dataset_names=dataset_names,
        local_jsonl_dir=args.local_jsonl_dir,
        local_repos=args.local_repos,
        emit_jsonl_dir=args.emit_jsonl_dir,
        target_tokens=args.target_tokens,
        max_samples_per_dataset=args.max_samples,
        languages=languages,
        subsample_rate=args.subsample_rate,
    )
    logger.info(
        "Built %s packed tokens across %s documents (manifest sha256=%s)",
        f"{manifest['total_packed_tokens']:,}",
        f"{manifest['total_documents']:,}",
        manifest["corpus_manifest_sha256"][:16],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
