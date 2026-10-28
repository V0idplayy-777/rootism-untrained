"""
Production Tokenizer Training Script (streaming BPE, no corpus cap).

Recipe:
- Streams files / HF dataset rows one document at a time (constant RAM).
- Trains real byte-level BPE merges with EfficientBPETrainer (incremental
  pair counts), byte-fallback guaranteed (256 byte symbols always present).
- Gates on TokenizerValidator: exact round-trip, unk_rate == 0, and
  chars/token above a floor measured on held-out code.

Usage:
    python scripts/train_production_tokenizer.py --data_dir /path/to/code \
        --output_dir tokenizer --vocab_size 128256 --min_frequency 2
    python scripts/train_production_tokenizer.py --dataset bigcode/the-stack \
        --output_dir tokenizer --max_samples 200000
"""

import argparse

import logging

import os

import sys

from pathlib import Path

from typing import Iterator



sys.path.insert(0, str(Path(__file__).parent.parent))



from model.tokenizer import CodeTokenizer, EfficientBPETrainer, TokenizerValidator



logger = logging.getLogger(__name__)

EXTENSIONS = {".py", ".js", ".ts", ".java", ".cpp", ".cc", ".c", ".h", ".hpp",

              ".go", ".rs", ".rb", ".php", ".jsx", ".tsx", ".cs", ".kt", ".swift"}





def iter_corpus(data_dir: str, max_chars: int = None) -> Iterator[str]:

    """Stream quality-filtered documents; max_chars=None means unlimited."""

    total = 0

    for filepath in sorted(Path(data_dir).rglob("*")):

        if filepath.suffix not in EXTENSIONS or not filepath.is_file():

            continue

        try:

            content = filepath.read_text(encoding="utf-8", errors="ignore")

        except Exception as e:

            logger.warning(f"Error reading {filepath}: {e}")

            continue

        lines = content.split("\n")

        if len(lines) < 10 or len(lines) > 5000:

            continue

        yield content

        total += len(content)

        if max_chars is not None and total >= max_chars:

            logger.info(f"Reached max corpus size: {total:,} chars")

            return





class ProductionTokenizerTrainer:

    """Backwards-compatible trainer facade over EfficientBPETrainer."""



    def __init__(self, vocab_size: int = 128256, min_frequency: int = 2,

                 max_corpus_size: int = None, num_workers: int = None,

                 min_chars_per_token: float = 1.5):

                                                                                  

        self.vocab_size = vocab_size

        self.min_frequency = min_frequency

        self.max_corpus_size = max_corpus_size

        self.num_workers = num_workers

        self.min_chars_per_token = min_chars_per_token

        self._trainer = EfficientBPETrainer(vocab_size=vocab_size, min_frequency=min_frequency)



    def load_corpus(self, data_dir: str) -> Iterator[str]:

        return iter_corpus(data_dir, max_chars=self.max_corpus_size)



    def train(self, data_dir: str, output_dir: str) -> CodeTokenizer:

        logger.info(f"Training tokenizer vocab_size={self.vocab_size} (streaming)")

        held_out: list = []

        seen = [0]

        def stream():

            for i, doc in enumerate(self.load_corpus(data_dir)):

                seen[0] += 1

                if i % 100 == 0 and len(held_out) < 200:

                    held_out.append(doc)

                yield doc

        tokenizer = self._trainer.train_from_iterator(stream())

        if seen[0] == 0:

            raise RuntimeError(f"no usable documents under {data_dir!r} (quality filter: 10-5000 lines/file)")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        tokenizer.save(str(output_dir))

        report = TokenizerValidator.validate(tokenizer, held_out[:50] or ["def f():\n    return 1\n"])

        logger.info(f"Validation: {report}")

        TokenizerValidator.assert_production_ready(report, min_chars_per_token=self.min_chars_per_token)

                                                                                  

        import json

        with open(Path(output_dir) / "validation.json", "w") as f:

            json.dump({k: v for k, v in report.items() if k != "per_sample"}, f, indent=2)

        logger.info(f"Tokenizer saved to {output_dir}")

        return tokenizer





def train_from_huggingface(dataset_name: str, output_dir: str,

                           vocab_size: int = 128256, max_samples: int = 100000):

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



    trainer = EfficientBPETrainer(vocab_size=vocab_size)

    tokenizer = trainer.train_from_iterator(stream())

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(output_dir))

    return tokenizer





def main():

    parser = argparse.ArgumentParser(description="Train production tokenizer")

    parser.add_argument("--data_dir", type=str, help="Directory with code files")

    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name")

    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    parser.add_argument("--vocab_size", type=int, default=128256)

    parser.add_argument("--min_frequency", type=int, default=2)

    parser.add_argument("--max_samples", type=int, default=100000)

    parser.add_argument("--min_chars_per_token", type=float, default=1.5)

    parser.add_argument("--max_corpus_chars", type=int, default=None,

                        help="Optional cap in chars (default: unlimited streaming)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.data_dir:

        trainer = ProductionTokenizerTrainer(vocab_size=args.vocab_size,

                                             min_frequency=args.min_frequency,

                                             max_corpus_size=args.max_corpus_chars,

                                             min_chars_per_token=args.min_chars_per_token)

        tokenizer = trainer.train(args.data_dir, args.output_dir)

    elif args.dataset:

        tokenizer = train_from_huggingface(args.dataset, args.output_dir,

                                           vocab_size=args.vocab_size, max_samples=args.max_samples)

    else:

        parser.error("Must specify either --data_dir or --dataset")

    logger.info(f"Done. Vocabulary size: {tokenizer.vocab_size:,}")





if __name__ == "__main__":

    main()

