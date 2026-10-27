#!/usr/bin/env python3
"""
Generate and Process the Complete Training Dataset.

This script:
1. Generates millions of code samples, instruction pairs, and FIM data
2. Tokenizes everything with the code-optimized BPE tokenizer
3. Splits into train/val/test
4. Creates binary shards for efficient training

Usage:
    python scripts/generate_dataset.py --output_dir data/processed --target_samples 500000
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.tokenizer import CodeTokenizer, BPETrainer
from dataset.generator import DatasetGenerator
from dataset.processor import DatasetConfig, DatasetProcessor


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def generate_sample_corpus_for_tokenizer(num_samples: int = 50000) -> list:
    """Generate a sample corpus for training the BPE tokenizer."""
    from dataset.code_templates import PythonCodeGenerator

    gen = PythonCodeGenerator()
    texts = []

    complexities = ["beginner", "intermediate", "advanced", "expert"]
    domains = ["algorithms", "data", "web", "systems", "testing"]

    for i in range(num_samples):
        complexity = complexities[i % len(complexities)]
        domain = domains[i % len(domains)]
        try:
            sample = gen.generate(complexity, domain)
            texts.append(sample.code)
            if sample.description:
                texts.append(sample.description)
        except Exception:
            pass

    # Add some extra common code patterns
    extra_patterns = [
        "def __init__(self, ",
        "if __name__ == '__main__':",
        "from typing import List, Dict, Optional, Tuple",
        "import numpy as np",
        "import torch",
        "import torch.nn as nn",
        "class Config:",
        "async def fetch_data(",
        "try:\n    result = process()\nexcept Exception as e:\n    logger.error(str(e))",
        "@app.route('/api/v1/",
        "SELECT * FROM users WHERE",
        "const response = await fetch(",
        "console.log(JSON.stringify(",
        "func main() {",
        "pub fn new(",
        "fn main() {",
        "package main\nimport (",
        "#include <iostream>",
        "template<typename T>",
        "docker-compose up -d",
        "kubectl apply -f",
        "git commit -m",
        "npm install",
        "pip install -r requirements.txt",
    ]
    texts.extend(extra_patterns * 100)

    return texts


def main():
    parser = argparse.ArgumentParser(description="Generate training dataset")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--tokenizer_dir", type=str, default="data/tokenizer")
    parser.add_argument("--target_samples", type=int, default=500000)
    parser.add_argument("--vocab_size", type=int, default=128256)
    parser.add_argument("--seq_length", type=int, default=8192)
    parser.add_argument("--train_tokenizer", action="store_true", default=True)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_tokenization", action="store_true")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    start_time = time.time()

    # ========================================
    # STEP 1: Train the Tokenizer
    # ========================================
    if args.train_tokenizer and not args.skip_generation:
        logger.info("=" * 60)
        logger.info("STEP 1: Training BPE Tokenizer")
        logger.info("=" * 60)

        # Generate corpus for tokenizer training
        logger.info("Generating corpus for tokenizer training...")
        corpus = generate_sample_corpus_for_tokenizer(num_samples=50000)
        logger.info(f"Generated {len(corpus)} texts for tokenizer training")

        # Train BPE
        logger.info(f"Training BPE tokenizer (vocab_size={args.vocab_size})...")
        trainer = BPETrainer(
            vocab_size=args.vocab_size,
            min_frequency=2,
        )
        tokenizer = trainer.train(corpus, verbose=True)

        # Save tokenizer
        os.makedirs(args.tokenizer_dir, exist_ok=True)
        tokenizer.save(args.tokenizer_dir)
        logger.info(f"Tokenizer saved to {args.tokenizer_dir}")
        logger.info(f"Vocabulary size: {tokenizer.vocab_size}")

        # Test tokenizer
        test_text = "def hello_world():\n    print('Hello, World!')"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        logger.info(f"Tokenizer test: {len(tokens)} tokens for '{test_text[:30]}...'")
    else:
        logger.info("Loading existing tokenizer...")
        tokenizer = CodeTokenizer.load(args.tokenizer_dir)

    # ========================================
    # STEP 2: Generate Raw Dataset
    # ========================================
    if not args.skip_generation:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 2: Generating Raw Dataset")
        logger.info("=" * 60)

        generator = DatasetGenerator(output_dir=args.raw_dir)
        gen_stats = generator.generate_full_dataset(
            target_samples=args.target_samples,
            output_dir=args.raw_dir,
        )

        logger.info(f"Generation stats:")
        for k, v in gen_stats.items():
            logger.info(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    else:
        logger.info("Skipping dataset generation (using existing data)")

    # ========================================
    # STEP 3: Tokenize and Shard
    # ========================================
    if not args.skip_tokenization:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 3: Tokenizing and Sharding")
        logger.info("=" * 60)

        os.makedirs(args.output_dir, exist_ok=True)

        config = DatasetConfig(
            max_seq_length=args.seq_length,
            train_split=0.95,
            val_split=0.04,
            test_split=0.01,
            min_sequence_length=32,
        )

        generator = DatasetGenerator()
        proc_stats = generator.tokenize_and_shard(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            config=config,
        )

        logger.info(f"Processing stats:")
        for k, v in proc_stats.items():
            logger.info(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    else:
        logger.info("Skipping tokenization (using existing shards)")

    # ========================================
    # SUMMARY
    # ========================================
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"DATASET GENERATION COMPLETE ({elapsed:.1f}s)")
    logger.info("=" * 60)
    logger.info(f"  Tokenizer: {args.tokenizer_dir}")
    logger.info(f"  Raw data:  {args.raw_dir}")
    logger.info(f"  Shards:    {args.output_dir}")
    logger.info(f"  Vocab:     {tokenizer.vocab_size:,}")
    logger.info("")
    logger.info("Ready for training! Run:")
    logger.info(f"  deepspeed training/train.py --config configs/training_config.json")


if __name__ == "__main__":
    main()
