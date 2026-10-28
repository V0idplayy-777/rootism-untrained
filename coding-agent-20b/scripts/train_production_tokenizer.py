"""
Production tokenizer training (streaming BPE, real corpus, gated).

What "production" means here, concretely:

1. **Real corpus, streamed** — never capped by RAM, tail-truncated or silently
   sampled down. `--data_dir` walks a checked-out corpus (e.g. The Stack smol,
   StarCoderData, or a local repo cache) and streams documents one at a time.
2. **Quality + licence gates before training** — min/max lines, non-empty,
   per-language caps so Python doesn't drown out Rust, and an optional SPDX
   allow-list read from the file header.
3. **Deterministic** — files sorted, content-hash dedup, tie-break on pair
   frequency then lexicographic pair.
4. **Exact vocabulary** — the trained vocab is padded with reserved specials to
   exactly `--vocab_size`, so ids line up with the model embedding.
5. **Held-out validation with a competitiveness gate** — measures round-trip,
   unk rate, chars/token, per-language fertility, whole-identifier rate and
   unified-diff cost, then benchmarks against installed baselines (bytes,
   word-punct, and tiktoken when available). Fails the run unless the candidate
   wins on code and diffs. Untrained/undertrained vocabularies cannot pass.

Usage:
    python scripts/train_production_tokenizer.py --data_dir /path/to/corpus \
        --output_dir tokenizer --vocab_size 128256 --benchmark-gate
    python scripts/train_production_tokenizer.py --dataset bigcode/the-stack-smol \
        --output_dir tokenizer --max_samples 200000 --benchmark-gate
"""

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from model.fast_bpe import HAS_TOKENIZERS, FastBPETrainer, verify_parity
from model.tokenizer import CodeTokenizer, EfficientBPETrainer, TokenizerValidator

logger = logging.getLogger(__name__)


EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sh": "shell",
    ".diff": "diff",
    ".patch": "diff",
}

PERMISSIVE_SPDX = (
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "MPL-2.0",
    "Unlicense",
    "CC0-1.0",
)


def _spdx_ok(content: str) -> bool:
    """Accept files with a permissive SPDX header, or with none (opt-in flag)."""

    head = content[:2000].lower()

    for token in ("spdx-license-identifier:", "license:"):

        idx = head.find(token)

        if idx >= 0:

            line = head[idx : idx + 120]

            return any(lic.lower() in line for lic in PERMISSIVE_SPDX)

    return True


def _iter_shard(path: Path) -> Iterator[tuple[str, str]]:
    """Stream (language, document) pairs out of a JSONL shard."""

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():

        line = line.strip()

        if not line:

            continue

        try:

            row = json.loads(line)

        except json.JSONDecodeError:

            continue

        content = row.get("content") or row.get("text") or ""

        lang = row.get("language") or "unknown"

        if content:

            yield str(lang).lower(), content


def iter_corpus(
    data_dir: str,
    max_chars: int | None = None,
    per_language_cap: int | None = None,
    require_spdx: bool = False,
    seen_hashes: set[str] | None = None,
) -> Iterator[tuple[str, str]]:
    """Stream (language, document) pairs; max_chars=None means unlimited.

    Accepts both a source tree (extension -> language) and JSONL shards
    produced by `scripts/sample_tokenizer_corpus.py`.

    `per_language_cap` bounds how many documents a single language can
    contribute (default 100k), which keeps the merge distribution from being
    dominated by one ecosystem in mixed repos.
    """

    total = 0

    cap = per_language_cap if per_language_cap is not None else 100_000

    counts: Counter = Counter()

    seen = seen_hashes if seen_hashes is not None else set()

    def emit(lang: str, content: str):
        nonlocal total

        lines = content.split("\n")

        if not (10 <= len(lines) <= 5000):

            return False

        if require_spdx and not _spdx_ok(content):

            return False

        digest = hashlib.sha256(content.encode("utf-8", "ignore")).hexdigest()[:32]

        if digest in seen:

            return False

        seen.add(digest)

        counts[lang] += 1

        total += len(content)

        return True

    # Group shards by language and stream them round-robin. Without this,
    # an alphabetical shard walk plus a char cap would cut the corpus off
    # partway through the alphabet (python after php) and silently bias the
    # vocabulary toward whichever languages sort first.

    by_lang: dict[str, list[Path]] = defaultdict(list)

    for filepath in sorted(Path(data_dir).rglob("*")):

        if filepath.is_file() and filepath.suffix == ".jsonl":

            lang_of_shard = filepath.stem.split("-")[0]

            by_lang[lang_of_shard].append(filepath)

    def _shard_docs(shards: list[Path]):

        for shard in shards:

            for lang, content in _iter_shard(shard):

                if counts[lang] < cap and emit(lang, content):

                    yield lang, content

    languages = sorted(by_lang)

    shard_iterators = {

        lang: iter(_shard_docs(by_lang[lang])) for lang in languages

    }

    active = [lang for lang in languages]

    while active:

        for lang in list(active):

            try:

                doc_lang, content = next(shard_iterators[lang])

            except StopIteration:

                active.remove(lang)

                continue

            yield doc_lang, content

            if max_chars is not None and total >= max_chars:

                logger.info(f"Reached max corpus size: {total:,} chars")

                return

    for filepath in sorted(Path(data_dir).rglob("*")):

        if not filepath.is_file() or filepath.suffix == ".jsonl":

            continue

        lang = EXTENSIONS.get(filepath.suffix)

        if lang is None:

            continue

        if counts[lang] >= cap:

            continue

        try:

            content = filepath.read_text(encoding="utf-8", errors="ignore")

        except Exception as e:

            logger.warning(f"Error reading {filepath}: {e}")

            continue

        if not emit(lang, content):

            continue

        yield lang, content

        if max_chars is not None and total >= max_chars:

            logger.info(f"Reached max corpus size: {total:,} chars")

            return


class ProductionTokenizerTrainer:
    """Trainer facade over EfficientBPETrainer with a real-corpus contract."""

    def __init__(
        self,
        vocab_size: int = 128256,
        min_frequency: int = 2,
        max_corpus_size: int | None = None,
        num_workers: int | None = None,
        min_chars_per_token: float = 2.5,
        per_language_cap: int | None = None,
        require_spdx: bool = False,
        holdout_docs: int = 200,
        engine: str = "auto",
    ):

        self.vocab_size = vocab_size

        self.min_frequency = min_frequency

        self.max_corpus_size = max_corpus_size

        self.num_workers = num_workers

        self.min_chars_per_token = min_chars_per_token

        self.per_language_cap = per_language_cap

        self.require_spdx = require_spdx

        self.holdout_docs = holdout_docs

        self.engine = engine

        use_rust = engine == "rust" or (engine == "auto" and HAS_TOKENIZERS)

        if use_rust and not HAS_TOKENIZERS:
            raise RuntimeError(
                "--engine rust requires the `tokenizers` package "
                "(pip install tokenizers>=0.19)"
            )

        self._engine_name = "rust" if use_rust else "python"

        trainer_cls = FastBPETrainer if use_rust else EfficientBPETrainer

        self._trainer = trainer_cls(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            pad_to_vocab_size=True,
        )

    def load_corpus(self, data_dir: str) -> Iterator[tuple[str, str]]:

        return iter_corpus(
            data_dir,
            max_chars=self.max_corpus_size,
            per_language_cap=self.per_language_cap,
            require_spdx=self.require_spdx,
        )

    def train(self, data_dir: str, output_dir: str, benchmark_gate: bool = False) -> CodeTokenizer:

        logger.info(f"Training tokenizer vocab_size={self.vocab_size} (streaming)")

        held_out: list = []

        seen = [0]

        manifest_hasher = hashlib.sha256()

        langs: Counter = Counter()

        chars = [0]

        def stream():

            for i, (lang, doc) in enumerate(self.load_corpus(data_dir)):

                seen[0] += 1

                chars[0] += len(doc)

                langs[lang] += 1

                manifest_hasher.update(lang.encode())

                manifest_hasher.update(
                    hashlib.sha256(doc.encode("utf-8", "ignore")).digest()
                )

                if i % 97 == 0 and len(held_out) < self.holdout_docs:

                    held_out.append(doc)

                yield doc

        tokenizer = self._trainer.train_from_iterator(stream())

        if seen[0] == 0:

            raise RuntimeError(
                f"no usable documents under {data_dir!r} (quality filter: 10-5000 lines/file)"
            )

        if len(tokenizer) != self.vocab_size:

            raise RuntimeError(
                f"vocabulary alignment failed: built {len(tokenizer)} != target {self.vocab_size}"
            )

        parity = None

        if self._engine_name == "rust" and held_out:
            # Prove the GPT-2-alphabet -> repo-alphabet conversion survived:
            # both tokenizers must segment held-out documents identically.
            parity = verify_parity(
                self._trainer.rust_tokenizer, tokenizer, held_out[:100]
            )

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        tokenizer.save(str(output_dir))

        report = TokenizerValidator.validate(
            tokenizer, held_out[:100] or ["def f():\n    return 1\n"]
        )

        TokenizerValidator.assert_production_ready(
            report, min_chars_per_token=self.min_chars_per_token
        )

        manifest = {
            "data_dir": str(data_dir),
            "documents": seen[0],
            "chars": chars[0],
            "languages": dict(sorted(langs.items())),
            "corpus_sha256": manifest_hasher.hexdigest(),
            "vocab_size": len(tokenizer),
            "merges": len(tokenizer.merges),
            "min_frequency": self.min_frequency,
            "holdout_documents": len(held_out),
            "engine": self._engine_name,
            "rust_python_parity": parity,
            "validation": {k: v for k, v in report.items() if k != "per_sample"},
        }

        benchmark = None

        if benchmark_gate:

            from evaluation.tokenizer_bench import TokenizerBenchmark

            bench = TokenizerBenchmark()

            benchmark = bench.run(tokenizer)

            try:

                TokenizerBenchmark.assert_competitive(benchmark)

            except AssertionError as e:

                manifest["benchmark"] = benchmark

                manifest["benchmark_passed"] = False

                with open(Path(output_dir) / "manifest.json", "w") as f:

                    json.dump(manifest, f, indent=2)

                raise RuntimeError(str(e)) from e

            manifest["benchmark_passed"] = True

        with open(Path(output_dir) / "validation.json", "w") as f:

            json.dump({k: v for k, v in report.items() if k != "per_sample"}, f, indent=2)

        with open(Path(output_dir) / "manifest.json", "w") as f:

            json.dump(manifest, f, indent=2)

        if benchmark is not None:

            with open(Path(output_dir) / "benchmark.json", "w") as f:

                json.dump(benchmark, f, indent=2)

        logger.info(
            "Tokenizer saved to %s | vocab=%d | chars/token=%.2f | gold %s",
            output_dir,
            len(tokenizer),
            report["chars_per_token"],
            report["diff"]["chars_per_token"],
        )

        return tokenizer


def train_from_huggingface(
    dataset_name: str,
    output_dir: str,
    vocab_size: int = 128256,
    max_samples: int = 100000,
    benchmark_gate: bool = False,
):

    from datasets import load_dataset

    logger.info(f"Loading dataset: {dataset_name}")

    dataset = load_dataset(dataset_name, split="train", streaming=True)

    def stream():

        for i, sample in enumerate(dataset):

            if max_samples is not None and i >= max_samples:

                return

            content = sample.get("content") or sample.get("text") or sample.get("code")

            if content:

                yield content

    trainer = EfficientBPETrainer(vocab_size=vocab_size, pad_to_vocab_size=True)

    tokenizer = trainer.train_from_iterator(stream())

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(output_dir))

    if benchmark_gate:

        from evaluation.tokenizer_bench import TokenizerBenchmark

        benchmark = TokenizerBenchmark().run(tokenizer)

        TokenizerBenchmark.assert_competitive(benchmark)

        with open(Path(output_dir) / "benchmark.json", "w") as f:

            json.dump(benchmark, f, indent=2)

    return tokenizer


def main():

    parser = argparse.ArgumentParser(description="Train production tokenizer")

    parser.add_argument("--data_dir", type=str, help="Directory with code files")

    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name")

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )

    parser.add_argument("--vocab_size", type=int, default=128256)

    parser.add_argument("--min_frequency", type=int, default=2)

    parser.add_argument("--max_samples", type=int, default=100000)

    parser.add_argument("--min_chars_per_token", type=float, default=2.5)

    parser.add_argument(
        "--max_corpus_chars",
        type=int,
        default=None,
        help="Optional cap in chars (default: unlimited streaming)",
    )

    parser.add_argument(
        "--per_language_cap",
        type=int,
        default=100_000,
        help="Max documents per language (balance guard)",
    )

    parser.add_argument(
        "--require_spdx",
        action="store_true",
        help="Only accept files whose header declares a permissive SPDX licence",
    )

    parser.add_argument(
        "--benchmark_gate",
        action="store_true",
        help="Fail unless the trained tokenizer beats installed baselines on code/diffs",
    )

    parser.add_argument(
        "--engine",
        choices=["auto", "rust", "python"],
        default="auto",
        help=(
            "BPE trainer backend: `rust` (tokenizers, fast, exact-parity to "
            "CodeTokenizer), `python` (EfficientBPETrainer), or `auto`"
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.data_dir:

        trainer = ProductionTokenizerTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            max_corpus_size=args.max_corpus_chars,
            min_chars_per_token=args.min_chars_per_token,
            per_language_cap=args.per_language_cap,
            require_spdx=args.require_spdx,
            engine=args.engine,
        )

        tokenizer = trainer.train(
            args.data_dir, args.output_dir, benchmark_gate=args.benchmark_gate
        )

    elif args.dataset:

        tokenizer = train_from_huggingface(
            args.dataset,
            args.output_dir,
            vocab_size=args.vocab_size,
            max_samples=args.max_samples,
            benchmark_gate=args.benchmark_gate,
        )

    else:

        parser.error("Must specify either --data_dir or --dataset")

    logger.info(f"Done. Vocabulary size: {tokenizer.vocab_size:,}")


if __name__ == "__main__":

    main()
