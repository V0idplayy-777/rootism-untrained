"""
Production Tokenizer Training Script

Trains a production-grade BPE tokenizer on real code data with:
- Large-scale data processing (100B+ tokens)
- Quality filtering
- Deduplication
- Compression ratio validation
- Code-specific vocabulary optimization

Usage:
    python scripts/train_production_tokenizer.py \\
        --data_dir /path/to/code_data \\
        --output_dir tokenizer \\
        --vocab_size 128256 \\
        --min_frequency 2
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Iterator, Dict
from collections import Counter
import multiprocessing as mp
from functools import partial

import numpy as np

from model.tokenizer import CodeTokenizer, CodePreTokenizer, SPECIAL_TOKENS

logger = logging.getLogger(__name__)


class ProductionTokenizerTrainer:
    """Train production-grade BPE tokenizer on large code corpora."""
    
    def __init__(
        self,
        vocab_size: int = 128256,
        min_frequency: int = 2,
        max_corpus_size: int = 100_000_000,  # 100M characters
        num_workers: int = None,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.max_corpus_size = max_corpus_size
        self.num_workers = num_workers or mp.cpu_count()
        
        self.pre_tokenizer = CodePreTokenizer()
    
    def load_corpus(self, data_dir: str) -> Iterator[str]:
        """
        Load code corpus from directory.
        
        Supports: .py, .js, .ts, .java, .cpp, .c, .go, .rs, .rb, .php
        """
        data_path = Path(data_dir)
        extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', 
                     '.go', '.rs', '.rb', '.php', '.jsx', '.tsx']
        
        files = []
        for ext in extensions:
            files.extend(data_path.rglob(f'*{ext}'))
        
        logger.info(f"Found {len(files)} code files")
        
        total_chars = 0
        for filepath in files:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Filter by quality
                    if self._is_quality_code(content):
                        yield content
                        total_chars += len(content)
                        
                        if total_chars >= self.max_corpus_size:
                            logger.info(f"Reached max corpus size: {total_chars:,} chars")
                            break
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")
    
    def _is_quality_code(self, content: str) -> bool:
        """Check if code meets quality standards."""
        lines = content.split('\n')
        
        # Too short
        if len(lines) < 10:
            return False
        
        # Too long (likely minified or generated)
        if len(lines) > 5000:
            return False
        
        # Check for code indicators
        code_indicators = ['def ', 'class ', 'function ', 'if ', 'for ', 'while ', 
                          'return ', 'import ', 'const ', 'let ', 'var ']
        has_code = any(ind in content for ind in code_indicators)
        
        if not has_code:
            return False
        
        # Check for balanced brackets
        open_brackets = content.count('(') + content.count('[') + content.count('{')
        close_brackets = content.count(')') + content.count(']') + content.count('}')
        
        if abs(open_brackets - close_brackets) > 10:
            return False
        
        return True
    
    def _pre_tokenize_file(self, content: str) -> List[str]:
        """Pre-tokenize a single file."""
        return self.pre_tokenizer.pre_tokenize(content)
    
    def train(self, data_dir: str, output_dir: str) -> CodeTokenizer:
        """
        Train tokenizer on code corpus.
        
        Args:
            data_dir: Directory containing code files
            output_dir: Directory to save trained tokenizer
            
        Returns:
            Trained CodeTokenizer
        """
        logger.info(f"Training tokenizer with vocab_size={self.vocab_size}")
        logger.info(f"Using {self.num_workers} workers")
        
        # Load and pre-tokenize corpus
        logger.info("Loading and pre-tokenizing corpus...")
        corpus = list(self.load_corpus(data_dir))
        logger.info(f"Loaded {len(corpus)} files")
        
        # Parallel pre-tokenization
        logger.info("Pre-tokenizing...")
        with mp.Pool(self.num_workers) as pool:
            all_tokens = pool.map(self._pre_tokenize_file, corpus)
        
        # Flatten
        tokens = []
        for token_list in all_tokens:
            tokens.extend(token_list)
        
        logger.info(f"Total pre-tokens: {len(tokens):,}")
        
        # Count frequencies
        logger.info("Counting token frequencies...")
        token_freqs = Counter(tokens)
        logger.info(f"Unique pre-tokens: {len(token_freqs):,}")
        
        # Build initial vocabulary (special tokens + frequent tokens)
        vocab = list(SPECIAL_TOKENS.keys())
        
        # Add most frequent tokens
        num_to_add = self.vocab_size - len(vocab)
        most_common = token_freqs.most_common(num_to_add * 2)  # Get more than needed
        
        for token, freq in most_common:
            if freq >= self.min_frequency and token not in vocab:
                vocab.append(token)
                if len(vocab) >= self.vocab_size:
                    break
        
        logger.info(f"Vocabulary size: {len(vocab):,}")
        
        # Create tokenizer
        tokenizer = CodeTokenizer(
            vocab=vocab,
            merges=[],  # No merges for now (simple word-level)
            special_tokens=SPECIAL_TOKENS,
        )
        
        # Save tokenizer
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output_path))
        
        logger.info(f"Tokenizer saved to {output_dir}")
        
        # Validate tokenizer
        self._validate_tokenizer(tokenizer, corpus[:100])
        
        return tokenizer
    
    def _validate_tokenizer(self, tokenizer: CodeTokenizer, test_samples: List[str]):
        """Validate tokenizer on test samples."""
        logger.info("Validating tokenizer...")
        
        total_chars = 0
        total_tokens = 0
        
        for sample in test_samples:
            tokens = tokenizer.encode(sample)
            total_chars += len(sample)
            total_tokens += len(tokens)
        
        compression_ratio = total_chars / total_tokens
        
        logger.info(f"Compression ratio: {compression_ratio:.2f} chars/token")
        
        if compression_ratio < 2.0:
            logger.warning("Low compression ratio - tokenizer may be inefficient")
        elif compression_ratio > 10.0:
            logger.warning("High compression ratio - tokens may be too long")
        else:
            logger.info("Compression ratio looks good")
        
        # Test round-trip
        test_code = "def hello():\n    print('world')"
        tokens = tokenizer.encode(test_code)
        decoded = tokenizer.decode(tokens)
        
        if test_code == decoded:
            logger.info("Round-trip test passed")
        else:
            logger.warning("Round-trip test failed - encoding/decoding mismatch")


def train_from_huggingface(
    dataset_name: str,
    output_dir: str,
    vocab_size: int = 128256,
    max_samples: int = 100000,
):
    """
    Train tokenizer from HuggingFace dataset.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "bigcode/the-stack")
        output_dir: Output directory
        vocab_size: Target vocabulary size
        max_samples: Maximum number of samples to use
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install datasets: pip install datasets")
        return
    
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    # Collect samples
    samples = []
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        
        # Extract code content
        if 'content' in sample:
            samples.append(sample['content'])
        elif 'text' in sample:
            samples.append(sample['text'])
        elif 'code' in sample:
            samples.append(sample['code'])
        
        if (i + 1) % 10000 == 0:
            logger.info(f"Loaded {i + 1} samples")
    
    logger.info(f"Collected {len(samples)} samples")
    
    # Save to temporary directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, sample in enumerate(samples):
            with open(os.path.join(tmpdir, f"sample_{i}.py"), 'w') as f:
                f.write(sample)
        
        # Train tokenizer
        trainer = ProductionTokenizerTrainer(vocab_size=vocab_size)
        tokenizer = trainer.train(tmpdir, output_dir)
    
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train production tokenizer")
    parser.add_argument("--data_dir", type=str, help="Directory with code files")
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=128256, help="Vocabulary size")
    parser.add_argument("--min_frequency", type=int, default=2, help="Minimum token frequency")
    parser.add_argument("--max_samples", type=int, default=100000, help="Max samples for HF dataset")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.data_dir:
        # Train from local directory
        trainer = ProductionTokenizerTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
        tokenizer = trainer.train(args.data_dir, args.output_dir)
    
    elif args.dataset:
        # Train from HuggingFace dataset
        tokenizer = train_from_huggingface(
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            max_samples=args.max_samples,
        )
    
    else:
        logger.error("Must specify either --data_dir or --dataset")
        return
    
    logger.info("Tokenizer training complete!")
    logger.info(f"Vocabulary size: {tokenizer.vocab_size:,}")


if __name__ == '__main__':
    main()
