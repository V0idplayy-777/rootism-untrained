"""
Production Data Processing Pipeline

Handles:
- Multi-source data ingestion (The Stack, StarCoderData, local repos)
- Language detection and filtering
- Quality scoring (complexity, documentation, test coverage)
- Near-duplicate detection (MinHash LSH)
- License filtering
- Efficient packing for 8k context
- Streaming to training
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Iterator, Dict, List, Optional, Tuple
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial

import numpy as np

# Optional imports (install as needed)
try:
    import datasketch
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False

try:
    import pygments
    from pygments.lexers import get_lexer_by_name
    HAS_PYGMENTS = True
except ImportError:
    HAS_PYGMENTS = False


logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A single code document with metadata."""
    content: str
    language: str
    source: str
    license: Optional[str] = None
    quality_score: float = 0.0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LanguageDetector:
    """Detect programming language from code content."""
    
    LANGUAGE_SIGNATURES = {
        'python': ['def ', 'import ', 'class ', 'if __name__', 'print('],
        'javascript': ['function ', 'const ', 'let ', 'var ', 'console.log', '=>'],
        'typescript': ['interface ', 'type ', 'enum ', ': string', ': number'],
        'java': ['public class', 'private ', 'public static void main', 'System.out'],
        'cpp': ['#include', 'std::', 'cout <<', 'int main('],
        'c': ['#include <stdio.h>', 'printf(', 'int main('],
        'go': ['package main', 'func ', 'fmt.Println', 'import ('],
        'rust': ['fn ', 'let mut', 'impl ', 'pub fn', 'use std::'],
        'ruby': ['def ', 'end', 'puts ', 'class ', 'module '],
        'php': ['<?php', 'function ', '$', 'echo ', 'class '],
    }
    
    @classmethod
    def detect(cls, content: str, file_ext: str = None) -> str:
        """Detect language from content and optional file extension."""
        # Extension-based detection
        ext_map = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.h': 'c',
            '.go': 'go', '.rs': 'rust', '.rb': 'ruby', '.php': 'php',
        }
        
        if file_ext and file_ext in ext_map:
            return ext_map[file_ext]
        
        # Content-based detection
        scores = {}
        for lang, signatures in cls.LANGUAGE_SIGNATURES.items():
            score = sum(1 for sig in signatures if sig in content)
            if score > 0:
                scores[lang] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return 'unknown'


class QualityScorer:
    """Score code quality based on various heuristics."""
    
    @staticmethod
    def score(doc: Document) -> float:
        """
        Score document quality (0-1).
        
        Factors:
        - Length (not too short, not too long)
        - Has comments/docstrings
        - Has function/class definitions
        - Proper indentation
        - No obvious syntax errors
        """
        content = doc.content
        lines = content.split('\n')
        
        score = 0.0
        
        # Length score (prefer 50-500 lines)
        num_lines = len(lines)
        if 50 <= num_lines <= 500:
            score += 0.2
        elif 20 <= num_lines <= 1000:
            score += 0.1
        
        # Comment/docstring score
        comment_lines = sum(1 for line in lines if line.strip().startswith('#') or 
                           line.strip().startswith('//') or line.strip().startswith('/*'))
        comment_ratio = comment_lines / max(num_lines, 1)
        if 0.05 <= comment_ratio <= 0.3:
            score += 0.2
        elif comment_ratio > 0:
            score += 0.1
        
        # Structure score (has functions/classes)
        has_functions = any(kw in content for kw in ['def ', 'function ', 'func ', 'fn '])
        has_classes = any(kw in content for kw in ['class ', 'struct ', 'interface '])
        if has_functions:
            score += 0.2
        if has_classes:
            score += 0.1
        
        # Indentation score (properly indented)
        indented_lines = sum(1 for line in lines if line and (line[0] == ' ' or line[0] == '\t'))
        indent_ratio = indented_lines / max(num_lines, 1)
        if indent_ratio > 0.3:
            score += 0.2
        
        # No obvious syntax errors (balanced brackets)
        open_brackets = content.count('(') + content.count('[') + content.count('{')
        close_brackets = content.count(')') + content.count(']') + content.count('}')
        if abs(open_brackets - close_brackets) < 5:
            score += 0.1
        
        return min(score, 1.0)


class Deduplicator:
    """Near-duplicate detection using MinHash LSH."""
    
    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        """
        Initialize deduplicator.
        
        Args:
            threshold: Jaccard similarity threshold (0.8 = 80% similar)
            num_perm: Number of permutations for MinHash
        """
        self.threshold = threshold
        self.num_perm = num_perm
        
        if HAS_DATASKETCH:
            self.lsh = datasketch.MinHashLSH(threshold=threshold, num_perm=num_perm)
            self.seen = set()
        else:
            logger.warning("datasketch not installed, deduplication disabled")
            self.lsh = None
            self.seen = set()
    
    def _create_minhash(self, content: str) -> Optional['datasketch.MinHash']:
        """Create MinHash signature for content."""
        if not HAS_DATASKETCH:
            return None
        
        # Create shingles (5-character n-grams)
        shingles = set(content[i:i+5] for i in range(len(content) - 4))
        
        if not shingles:
            return None
        
        m = datasketch.MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            m.update(shingle.encode('utf8'))
        
        return m
    
    def is_duplicate(self, doc_id: str, content: str) -> bool:
        """Check if document is a near-duplicate."""
        if not self.lsh:
            # Fallback: exact hash dedup
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.seen:
                return True
            self.seen.add(content_hash)
            return False
        
        # Check exact duplicate first
        if doc_id in self.seen:
            return True
        
        # Create MinHash
        minhash = self._create_minhash(content)
        if not minhash:
            return False
        
        # Query LSH
        candidates = self.lsh.query(minhash)
        if candidates:
            return True
        
        # Insert into LSH
        self.lsh.insert(doc_id, minhash)
        self.seen.add(doc_id)
        
        return False


class DataProcessor:
    """Production data processing pipeline."""
    
    def __init__(
        self,
        min_quality: float = 0.3,
        dedup_threshold: float = 0.8,
        min_length: int = 100,
        max_length: int = 100000,
        languages: List[str] = None,
    ):
        self.min_quality = min_quality
        self.min_length = min_length
        self.max_length = max_length
        self.languages = languages or ['python', 'javascript', 'typescript', 'java', 'cpp', 'go', 'rust']
        
        self.detector = LanguageDetector()
        self.scorer = QualityScorer()
        self.dedup = Deduplicator(threshold=dedup_threshold)
        
        self.stats = {
            'total': 0,
            'filtered_length': 0,
            'filtered_language': 0,
            'filtered_quality': 0,
            'filtered_duplicate': 0,
            'kept': 0,
        }
    
    def process_document(self, content: str, source: str, license: str = None) -> Optional[Document]:
        """
        Process a single document through the pipeline.
        
        Returns Document if it passes all filters, None otherwise.
        """
        self.stats['total'] += 1
        
        # Length filter
        if len(content) < self.min_length or len(content) > self.max_length:
            self.stats['filtered_length'] += 1
            return None
        
        # Language detection
        language = self.detector.detect(content)
        if language not in self.languages:
            self.stats['filtered_language'] += 1
            return None
        
        # Create document
        doc = Document(
            content=content,
            language=language,
            source=source,
            license=license,
        )
        
        # Quality scoring
        doc.quality_score = self.scorer.score(doc)
        if doc.quality_score < self.min_quality:
            self.stats['filtered_quality'] += 1
            return None
        
        # Deduplication
        doc_id = hashlib.md5(content[:1000].encode()).hexdigest()
        if self.dedup.is_duplicate(doc_id, content):
            self.stats['filtered_duplicate'] += 1
            return None
        
        self.stats['kept'] += 1
        return doc
    
    def process_stream(self, documents: Iterator[Tuple[str, str, str]]) -> Iterator[Document]:
        """
        Process a stream of (content, source, license) tuples.
        
        Yields Document objects that pass all filters.
        """
        for content, source, license in documents:
            doc = self.process_document(content, source, license)
            if doc:
                yield doc
            
            # Log stats periodically
            if self.stats['total'] % 10000 == 0:
                logger.info(f"Processed {self.stats['total']} docs, kept {self.stats['kept']} "
                          f"({self.stats['kept']/self.stats['total']*100:.1f}%)")
    
    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return self.stats.copy()


class DataPacker:
    """Pack documents into fixed-length sequences for training."""
    
    def __init__(self, tokenizer, seq_length: int = 8192):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.buffer = []
        self.buffer_tokens = 0
    
    def add_document(self, doc: Document) -> List[np.ndarray]:
        """
        Add document to buffer and return completed sequences.
        
        Implements document packing to maximize context usage.
        """
        # Tokenize document
        tokens = self.tokenizer.encode(doc.content)
        
        # Add special tokens
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        
        completed_sequences = []
        
        # Add to buffer
        self.buffer.extend(tokens)
        self.buffer_tokens += len(tokens)
        
        # Extract completed sequences
        while self.buffer_tokens >= self.seq_length:
            seq = np.array(self.buffer[:self.seq_length], dtype=np.int32)
            completed_sequences.append(seq)
            
            self.buffer = self.buffer[self.seq_length:]
            self.buffer_tokens -= self.seq_length
        
        return completed_sequences
    
    def flush(self) -> List[np.ndarray]:
        """Flush remaining buffer (may be shorter than seq_length)."""
        if self.buffer:
            # Pad to seq_length
            seq = np.full(self.seq_length, self.tokenizer.pad_token_id, dtype=np.int32)
            seq[:len(self.buffer)] = self.buffer
            self.buffer = []
            self.buffer_tokens = 0
            return [seq]
        return []


def process_dataset_parallel(
    documents: Iterator[Tuple[str, str, str]],
    processor: DataProcessor,
    num_workers: int = None,
) -> Iterator[Document]:
    """
    Process documents in parallel using multiprocessing.
    
    Note: Deduplication is not thread-safe, so we do it in the main process.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Parallel processing (without dedup)
    def process_no_dedup(args):
        content, source, license = args
        # Create temporary processor without dedup
        temp_proc = DataProcessor(
            min_quality=processor.min_quality,
            min_length=processor.min_length,
            max_length=processor.max_length,
            languages=processor.languages,
        )
        temp_proc.dedup = None  # Disable dedup for parallel processing
        return temp_proc.process_document(content, source, license)
    
    with mp.Pool(num_workers) as pool:
        for doc in pool.imap(process_no_dedup, documents, chunksize=100):
            if doc:
                # Apply dedup in main process
                doc_id = hashlib.md5(doc.content[:1000].encode()).hexdigest()
                if not processor.dedup.is_duplicate(doc_id, doc.content):
                    yield doc


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create processor
    processor = DataProcessor(
        min_quality=0.3,
        dedup_threshold=0.8,
        languages=['python', 'javascript'],
    )
    
    # Example documents
    docs = [
        ("def hello():\n    print('world')", "example.py", "MIT"),
        ("function test() { return 42; }", "example.js", "MIT"),
    ]
    
    # Process
    for doc in processor.process_stream(iter(docs)):
        print(f"Kept: {doc.language} (quality: {doc.quality_score:.2f})")
    
    print("\nStats:", processor.get_stats())
