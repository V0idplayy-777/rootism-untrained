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





PERMISSIVE_LICENSES = {

    "mit", "apache-2.0", "apache 2.0", "bsd-2-clause", "bsd-3-clause", "bsd",

    "isc", "mpl-2.0", "unlicense", "cc0-1.0", "lgpl-2.1", "lgpl-3.0",

}

NON_PERMISSIVE_MARKERS = ("gpl-3.0", "gpl-2.0", "agpl", "proprietary", "all rights reserved", "no license")





class LicenseFilter:

    """SPDX-style license gate. Unknown license => rejected unless allow_unknown."""



    def __init__(self, allow_unknown: bool = False):

        self.allow_unknown = allow_unknown



    def allows(self, license_str) -> bool:

        if license_str is None or not str(license_str).strip():

            return self.allow_unknown

        norm = str(license_str).strip().lower()

        if any(m in norm for m in NON_PERMISSIVE_MARKERS) and norm not in PERMISSIVE_LICENSES:

            return False

        if norm in PERMISSIVE_LICENSES:

            return True

                                                                                   

        return any(tok in norm for tok in PERMISSIVE_LICENSES)





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

                                   

        ext_map = {

            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',

            '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.h': 'c',

            '.go': 'go', '.rs': 'rust', '.rb': 'ruby', '.php': 'php',

        }

        

        if file_ext and file_ext in ext_map:

            return ext_map[file_ext]

        

                                                            

        first = content[:200]

        if first.startswith('#!'):

            for name, lang in (('python', 'python'), ('node', 'javascript'),

                               ('ruby', 'ruby'), ('bash', 'shell'), ('sh', 'shell')):

                if name in first.split('\n')[0]:

                    return lang



                                                           

        scores = {}

        for lang, signatures in cls.LANGUAGE_SIGNATURES.items():

            score = sum(1 for sig in signatures if sig in content)

            if score > 0:

                scores[lang] = score

        if scores:

                                                                           

                                                          

            return max(scores, key=scores.get)



                                                                     

        if HAS_PYGMENTS:

            try:

                from pygments.lexers import guess_lexer

                from pygments.util import ClassNotFound

                try:

                    lexer = guess_lexer(content[:4000])

                    alias = (lexer.aliases[0] if lexer.aliases else 'unknown')

                    alias_map = {'py': 'python', 'python3': 'python', 'js': 'javascript',

                                 'ts': 'typescript', 'c++': 'cpp', 'rs': 'rust', 'rb': 'ruby',

                                 'golang': 'go', 'sh': 'shell', 'bash': 'shell'}

                    return alias_map.get(alias, alias)

                except ClassNotFound:

                    pass

            except Exception:

                pass

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



                                                                       

        if doc.language == 'python':

            import ast as _ast

            try:

                _ast.parse(content)

            except SyntaxError:

                return 0.0

        if any(m in content[:500] for m in ('<auto-generated', 'Code generated by', 'DO NOT EDIT')):

            return 0.05

                                         

        if lines:

            avg_line = sum(len(l) for l in lines) / len(lines)

            if avg_line > 300:                 

                return 0.05

            uniq = len(set(lines)) / max(len(lines), 1)

            if uniq < 0.3:

                return 0.05



        score = 0.0

        

                                            

        num_lines = len(lines)

        if 50 <= num_lines <= 500:

            score += 0.2

        elif 20 <= num_lines <= 1000:

            score += 0.1

        

                                 

        comment_lines = sum(1 for line in lines if line.strip().startswith('#') or 

                           line.strip().startswith('//') or line.strip().startswith('/*'))

        comment_ratio = comment_lines / max(num_lines, 1)

        if 0.05 <= comment_ratio <= 0.3:

            score += 0.2

        elif comment_ratio > 0:

            score += 0.1

        

                                                 

        has_functions = any(kw in content for kw in ['def ', 'function ', 'func ', 'fn '])

        has_classes = any(kw in content for kw in ['class ', 'struct ', 'interface '])

        if has_functions:

            score += 0.2

        if has_classes:

            score += 0.1

        

                                               

        indented_lines = sum(1 for line in lines if line and (line[0] == ' ' or line[0] == '\t'))

        indent_ratio = indented_lines / max(num_lines, 1)

        if indent_ratio > 0.3:

            score += 0.2

        

                                                      

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

        self._exact = set()

        self._sigs = {}

        self._buckets = {}

        

        if HAS_DATASKETCH:

            self.lsh = datasketch.MinHashLSH(threshold=threshold, num_perm=num_perm)

            self.seen = set()

        else:

            logger.warning("datasketch not installed, deduplication disabled")

            self.lsh = None

            self.seen = set()

    

    @staticmethod

    def _builtin_signature(content: str, num_perm: int = 128) -> np.ndarray:

        """Dependency-free MinHash signature (universal hashing over shingles)."""

        import hashlib as _hl

        shingles = {content[i:i+5] for i in range(max(len(content) - 4, 0))} or {content}

        sig = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)

        for sh in shingles:

            h = int.from_bytes(_hl.sha256(sh.encode('utf8', 'ignore')).digest()[:8], 'little')

                                           

            x = h

            for i in range(num_perm):

                x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF

                y = x

                y = (y ^ (y >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF

                y = (y ^ (y >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF

                y = y ^ (y >> 31)

                if y < sig[i]:

                    sig[i] = y

                x = y

        return sig



    def _create_minhash(self, content: str) -> Optional['datasketch.MinHash']:

        """Create MinHash signature for content (datasketch when available)."""

        if not HAS_DATASKETCH:

            return None

        

                                               

        shingles = set(content[i:i+5] for i in range(len(content) - 4))

        

        if not shingles:

            return None

        

        m = datasketch.MinHash(num_perm=self.num_perm)

        for shingle in shingles:

            m.update(shingle.encode('utf8'))

        

        return m

    

    def is_duplicate(self, doc_id: str, content: str) -> bool:

        """Check if document is a near-duplicate (exact + MinHash LSH)."""

        content_hash = hashlib.sha256(content.encode('utf8', 'ignore')).hexdigest()

        if content_hash in getattr(self, '_exact', set()):

            return True

        if not self.lsh:

                                                                                    

            sig = self._builtin_signature(content, self.num_perm)

            bands, rows = 16, self.num_perm // 16

            for b in range(bands):

                key = (b, tuple(sig[b * rows:(b + 1) * rows].tolist()))

                for other in self._buckets.get(key, []):

                    sim = float(np.mean(sig == self._sigs[other]))

                    if sim >= self.threshold:

                        return True

            self._sigs[doc_id] = sig

            for b in range(bands):

                key = (b, tuple(sig[b * rows:(b + 1) * rows].tolist()))

                self._buckets.setdefault(key, []).append(doc_id)

            self._exact.add(content_hash)

            self.seen.add(doc_id)

            return False

        self._exact.add(content_hash)

        

                                     

        if doc_id in self.seen:

            return True

        

                        

        minhash = self._create_minhash(content)

        if not minhash:

            return False

        

                   

        candidates = self.lsh.query(minhash)

        if candidates:

            return True

        

                         

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

        self.license_filter = LicenseFilter(allow_unknown=False)

        

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

        

                       

        if len(content) < self.min_length or len(content) > self.max_length:

            self.stats['filtered_length'] += 1

            return None

        

                                                                    

        if not self.license_filter.allows(license):

            self.stats['filtered_license'] = self.stats.get('filtered_license', 0) + 1

            return None



                            

        language = self.detector.detect(content)

        if language not in self.languages:

            self.stats['filtered_language'] += 1

            return None

        

                         

        doc = Document(

            content=content,

            language=language,

            source=source,

            license=license,

        )

        

                         

        doc.quality_score = self.scorer.score(doc)

        if doc.quality_score < self.min_quality:

            self.stats['filtered_quality'] += 1

            return None

        

                       

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

                                                                        

        tokens = self.tokenizer.encode(doc.content, add_special_tokens=False)

        

        completed_sequences = []

        

                       

        self.buffer.extend(tokens)

        self.buffer_tokens += len(tokens)

        

                                     

        while self.buffer_tokens >= self.seq_length:

            seq = np.array(self.buffer[:self.seq_length], dtype=np.int32)

            completed_sequences.append(seq)

            

            self.buffer = self.buffer[self.seq_length:]

            self.buffer_tokens -= self.seq_length

        

        return completed_sequences

    

    def flush(self) -> List[np.ndarray]:

        """Flush remaining buffer (may be shorter than seq_length)."""

        if self.buffer:

                               

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

    

                                         

    def process_no_dedup(args):

        content, source, license = args

                                                  

        temp_proc = DataProcessor(

            min_quality=processor.min_quality,

            min_length=processor.min_length,

            max_length=processor.max_length,

            languages=processor.languages,

        )

        temp_proc.dedup = None                                         

        return temp_proc.process_document(content, source, license)

    

    with mp.Pool(num_workers) as pool:

        for doc in pool.imap(process_no_dedup, documents, chunksize=100):

            if doc:

                                             

                doc_id = hashlib.md5(doc.content[:1000].encode()).hexdigest()

                if not processor.dedup.is_duplicate(doc_id, doc.content):

                    yield doc





if __name__ == '__main__':

                   

    logging.basicConfig(level=logging.INFO)

    

                      

    processor = DataProcessor(

        min_quality=0.3,

        dedup_threshold=0.8,

        languages=['python', 'javascript'],

    )

    

                       

    docs = [

        ("def hello():\n    print('world')", "example.py", "MIT"),

        ("function test() { return 42; }", "example.js", "MIT"),

    ]

    

             

    for doc in processor.process_stream(iter(docs)):

        print(f"Kept: {doc.language} (quality: {doc.quality_score:.2f})")

    

    print("\nStats:", processor.get_stats())

