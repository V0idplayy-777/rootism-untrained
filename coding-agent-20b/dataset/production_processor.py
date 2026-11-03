"""Production Data Processing & Shard Building Pipeline (Corpus v1).

Sole production path for building tokenized training/validation shards for
``training/train_3b.py`` and ``training/train.py``:
- Fail-closed SPDX permissive license gate (no GPL/LGPL/AGPL/SSPL/unknown)
- Language detection across 20 code/config/doc/diff languages
- Hard quality & structural complexity filters (size, line count, minified,
  vendored, generated, syntax check, AST/construct complexity floor)
- Document-level exact SHA-256 + MinHash LSH near-duplicate removal
- Held-out isolation registry (guarantees 0 overlap between train/val and held-out)
- Sequence packing into uint32 binary shards + SHA-256 manifests
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import multiprocessing as mp
import os
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import datasketch

    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False

try:
    import pygments  # noqa: F401

    HAS_PYGMENTS = True
except ImportError:
    HAS_PYGMENTS = False

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A single code or agent-slice document with provenance metadata."""

    content: str
    language: str
    source: str
    license: str | None = None
    quality_score: float = 0.0
    path: str = ""
    slice_name: str = "the-stack-code"
    is_multi_file: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8", "ignore")).hexdigest()

    def to_jsonl_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "content": self.content,
            "language": self.language,
            "license": self.license or "",
            "source": self.source,
            "path": self.path or self.source,
            "quality_score": round(float(self.quality_score), 4),
            "slice": self.slice_name,
            "sha256": self.sha256(),
        }
        if self.is_multi_file:
            row["is_multi_file"] = True
        if self.metadata:
            row["metadata"] = self.metadata
        return row


# Strictly permissive SPDX licenses only (LGPL/GPL/AGPL/SSPL/BSL-1.1 excluded).
CANONICAL_PERMISSIVE_SPDX: dict[str, str] = {
    "mit": "MIT",
    "mit license": "MIT",
    "apache-2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd 2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd 3-clause": "BSD-3-Clause",
    "bsd": "BSD-3-Clause",
    "bsd license": "BSD-3-Clause",
    "0bsd": "0BSD",
    "isc": "ISC",
    "isc license": "ISC",
    "mpl-2.0": "MPL-2.0",
    "unlicense": "Unlicense",
    "the unlicense": "Unlicense",
    "cc0-1.0": "CC0-1.0",
    "zlib": "Zlib",
    "bsl-1.0": "BSL-1.0",
    "psf-2.0": "PSF-2.0",
    "python-2.0": "PSF-2.0",
}

PERMISSIVE_LICENSES: set[str] = set(CANONICAL_PERMISSIVE_SPDX.keys())

NON_PERMISSIVE_MARKERS: tuple[str, ...] = (
    "agpl",
    "lgpl",
    "gpl-3",
    "gpl-2",
    "gplv2",
    "gplv3",
    "gnu general public",
    "gnu lesser",
    "sspl",
    "busl",
    "bsl-1.1",
    "elastic",
    "commons clause",
    "proprietary",
    "commercial",
    "all rights reserved",
    "no license",
    "non-commercial",
    "cc-by-nc",
    "cc-by-sa",
)


class LicenseFilter:
    """Fail-closed SPDX permissive license gate."""

    def __init__(self, allow_unknown: bool = False):
        self.allow_unknown = allow_unknown

    def canonicalize(self, license_str: Any) -> str | None:
        if license_str is None:
            return "UNKNOWN" if self.allow_unknown else None
        if isinstance(license_str, (list, tuple)):
            for entry in license_str:
                got = self.canonicalize(entry)
                if got is None:
                    return None
                return got
            return "UNKNOWN" if self.allow_unknown else None

        raw = str(license_str).strip().strip("`\"'")
        if not raw:
            return "UNKNOWN" if self.allow_unknown else None

        lowered = raw.lower()
        if any(marker in lowered for marker in NON_PERMISSIVE_MARKERS):
            return None

        if " with " in lowered:
            return None

        if " and " in lowered:
            parts = [p.strip(" ()") for p in re.split(r"\s+and\s+", raw, flags=re.I)]
            canon = [self.canonicalize(p) for p in parts]
            if all(canon):
                return canon[0]
            return None

        if " or " in lowered:
            parts = [p.strip(" ()") for p in re.split(r"\s+or\s+", raw, flags=re.I)]
            for p in parts:
                got = self.canonicalize(p)
                if got:
                    return got
            return None

        cleaned = lowered.strip(" ()")
        if cleaned in CANONICAL_PERMISSIVE_SPDX:
            return CANONICAL_PERMISSIVE_SPDX[cleaned]
        return None

    def allows(self, license_str: Any) -> bool:
        return self.canonicalize(license_str) is not None


class LanguageDetector:
    """Detect programming language from file extension, shebang, or content."""

    EXT_MAP: dict[str, str] = {
        ".py": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".hh": "cpp",
        ".c": "c",
        ".h": "c",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".swift": "swift",
        ".scala": "scala",
        ".sql": "sql",
        ".sh": "shell",
        ".bash": "shell",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
        ".rst": "markdown",
        ".diff": "diff",
        ".patch": "diff",
    }

    LANGUAGE_SIGNATURES: dict[str, list[str]] = {
        "python": ["def ", "import ", "class ", "if __name__", "print("],
        "javascript": ["function ", "const ", "let ", "var ", "console.log", "=>"],
        "typescript": ["interface ", "type ", "enum ", ": string", ": number", "export interface"],
        "java": ["public class", "private ", "public static void main", "System.out"],
        "cpp": ["#include", "std::", "cout <<", "namespace ", "template<"],
        "c": ["#include <stdio.h>", "#include <stdlib.h>", "printf(", "int main("],
        "go": ["package main", "package ", "func ", "fmt.", "import ("],
        "rust": ["fn ", "let mut", "impl ", "pub fn", "use std::", "pub struct"],
        "ruby": ["def ", "end", "puts ", "class ", "module "],
        "php": ["<?php", "function ", "$", "echo ", "namespace "],
        "csharp": ["using System", "namespace ", "public class ", "async Task"],
        "kotlin": ["fun ", "val ", "data class ", "package "],
        "swift": ["import Foundation", "func ", "struct ", "guard let "],
        "sql": ["SELECT ", "CREATE TABLE", "INSERT INTO", "ALTER TABLE"],
        "shell": ["#!/usr/bin/env bash", "#!/bin/sh", "set -e"],
        "diff": ["diff --git ", "--- a/", "+++ b/", "@@ -"],
    }

    @classmethod
    def detect(cls, content: str, file_ext: str | None = None) -> str:
        if file_ext:
            ext = file_ext.lower() if file_ext.startswith(".") else Path(file_ext).suffix.lower()
            if ext in cls.EXT_MAP:
                return cls.EXT_MAP[ext]

        first = content[:200]
        if first.startswith("#!"):
            first_line = first.split("\n", 1)[0]
            for name, lang in (
                ("python", "python"),
                ("node", "javascript"),
                ("ruby", "ruby"),
                ("bash", "shell"),
                ("sh", "shell"),
            ):
                if name in first_line:
                    return lang

        scores: dict[str, int] = {}
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

                lexer = guess_lexer(content[:4000])
                alias = lexer.aliases[0] if lexer.aliases else "unknown"
                alias_map = {
                    "py": "python",
                    "python3": "python",
                    "js": "javascript",
                    "ts": "typescript",
                    "c++": "cpp",
                    "rs": "rust",
                    "rb": "ruby",
                    "golang": "go",
                    "sh": "shell",
                    "bash": "shell",
                }
                return alias_map.get(alias, alias)
            except (ClassNotFound, Exception):
                pass

        return "unknown"


_GENERATED_MARKERS = (
    "<auto-generated",
    "@generated",
    "code generated by",
    "do not edit",
    "generated automatically",
    "this file was automatically generated",
    "autogenerated by",
)

_VENDORED_PATH_PARTS = frozenset(
    {
        "node_modules",
        "vendor",
        "vendors",
        "third_party",
        "thirdparty",
        "external",
        "dist",
        "build",
        ".venv",
        "venv",
        "__pycache__",
        "target",
    }
)

_MINIFIED_OR_LOCK_RE = re.compile(
    r"(\.min\.(js|css|ts)$|\.bundle\.js$|\.pb\.go$|_pb2\.py$|\.gen\.(py|go|ts|rs)$|"
    r"(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|Cargo\.lock|poetry\.lock|go\.sum)$)",
    re.I,
)

_COMPLEXITY_KEYWORDS = re.compile(
    r"\b(def|class|async|await|return|yield|if|elif|else|for|while|try|except|with|"
    r"function|const|let|interface|type|export|switch|case|catch|"
    r"func|struct|impl|enum|match|pub|trait|go|defer|select|"
    r"SELECT|FROM|WHERE|JOIN|GROUP BY|CREATE|INSERT|UPDATE)\b"
)


def classify_real_slice(path: str, language: str, content: str) -> str:
    """Route a real permissive repository file into its Corpus v1 backbone slice."""
    lower_path = (path or "").lower()
    if language == "diff" or content.startswith("diff --git ") or ("\n--- a/" in content and "\n+++ b/" in content):
        return "github-prs"
    if lower_path.startswith("issue:") or lower_path.endswith(".issue.md"):
        return "github-issues"
    if lower_path.startswith("pr:") or "/pull/" in lower_path:
        return "github-prs"

    parts = [p for p in re.split(r"[/\\:]", lower_path) if p]
    base = parts[-1] if parts else ""
    if (
        any(p in {"test", "tests", "__tests__", "spec", "specs", "testing"} for p in parts[:-1])
        or base.startswith("test_")
        or base.endswith(("_test.py", "_test.go", "_test.rs", ".test.ts", ".test.tsx", ".test.js", ".spec.ts", ".spec.js", "test.java", "tests.cs"))
    ):
        return "tests"

    if language == "markdown" or base.endswith((".md", ".rst")) or any(p in {"doc", "docs", "documentation"} for p in parts[:-1]):
        return "docs-and-comments"

    return "the-stack-code"


def ast_pattern_signature(content: str, language: str) -> str:
    """Compute a structural skeleton hash (identifier-agnostic) for diversity gates."""
    if language == "python":
        try:
            tree = ast.parse(content)
            node_types = [type(node).__name__ for node in ast.walk(tree)]
            if len(node_types) >= 3:
                digest = hashlib.sha1(",".join(node_types[:256]).encode("utf-8")).hexdigest()[:16]
                return f"py:{digest}"
        except SyntaxError:
            pass
    tokens = _COMPLEXITY_KEYWORDS.findall(content[:8000])
    braces = re.findall(r"[{}\[\]()]", content[:4000])
    skeleton = f"{language}:{' '.join(tokens[:128])}:{''.join(braces[:128])}"
    digest = hashlib.sha1(skeleton.encode("utf-8")).hexdigest()[:16]
    return f"{language[:4]}:{digest}"


class QualityScorer:
    """Quality scoring and hard structural filters for real code."""

    @staticmethod
    def complexity_count(content: str, language: str = "unknown") -> int:
        if language == "python":
            try:
                tree = ast.parse(content)
                structural = (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.If,
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.Try,
                    ast.With,
                    ast.Match,
                    ast.ListComp,
                    ast.DictComp,
                    ast.Return,
                )
                return sum(1 for n in ast.walk(tree) if isinstance(n, structural))
            except SyntaxError:
                return 0
        if language in {"markdown", "yaml", "toml", "diff"}:
            return max(1, content.count("\n") // 5)
        return len(_COMPLEXITY_KEYWORDS.findall(content))

    @classmethod
    def passes_hard_filters(
        cls,
        content: str,
        language: str = "unknown",
        path: str = "",
        *,
        min_lines: int = 4,
        max_lines: int = 6000,
        max_bytes: int = 512_000,
        max_line_length: int = 1000,
        min_complexity: int = 1,
    ) -> tuple[bool, str]:
        if not content:
            return False, "empty"
        byte_len = len(content.encode("utf-8", "ignore"))
        if byte_len > max_bytes:
            return False, "too_large"

        if path:
            rel_path = path.split(":", 1)[-1] if ":" in path else path
            parts = {p.lower() for p in Path(rel_path).parts}
            if parts & _VENDORED_PATH_PARTS:
                return False, "vendored_dir"
            if _MINIFIED_OR_LOCK_RE.search(Path(rel_path).name):
                return False, "minified_or_generated_name"

        head_lower = content[:800].lower()
        if any(marker in head_lower for marker in _GENERATED_MARKERS):
            return False, "generated_header"

        lines = content.split("\n")
        num_lines = len(lines)
        if num_lines < min_lines or num_lines > max_lines:
            return False, "line_count"

        longest = max((len(line) for line in lines), default=0)
        avg_len = len(content) / max(num_lines, 1)
        if longest > max_line_length or avg_len > 220:
            return False, "minified_line_length"

        alnum = sum(1 for ch in content[:4000] if ch.isalnum())
        if alnum / max(len(content[:4000]), 1) < 0.15:
            return False, "low_alnum_ratio"

        if language == "python":
            try:
                ast.parse(content)
            except SyntaxError:
                return False, "python_syntax_error"

        if min_complexity > 0 and cls.complexity_count(content, language) < min_complexity:
            return False, "below_complexity_floor"

        return True, "ok"

    @classmethod
    def score(cls, doc: Document) -> float:
        """Score document quality in ``[0.0, 1.0]``."""
        content = doc.content
        lines = content.split("\n")

        if doc.language == "python":
            try:
                ast.parse(content)
            except SyntaxError:
                return 0.0

        head_lower = content[:600].lower()
        if any(m in head_lower for m in _GENERATED_MARKERS):
            return 0.05

        if lines:
            avg_line = sum(len(line) for line in lines) / len(lines)
            max_line = max(len(line) for line in lines)
            if avg_line > 240 or max_line > 1000:
                return 0.05
            uniq = len(set(lines)) / max(len(lines), 1)
            if len(lines) >= 20 and uniq < 0.30:
                return 0.05

        score = 0.0
        num_lines = len(lines)
        if 25 <= num_lines <= 800:
            score += 0.22
        elif 8 <= num_lines <= 2500:
            score += 0.12

        comment_lines = sum(
            1
            for line in lines
            if line.strip().startswith(("#", "//", "/*", "*", '"""', "'''"))
        )
        comment_ratio = comment_lines / max(num_lines, 1)
        if 0.04 <= comment_ratio <= 0.40:
            score += 0.18
        elif comment_ratio > 0:
            score += 0.10

        complexity = cls.complexity_count(content, doc.language)
        if complexity >= 6:
            score += 0.25
        elif complexity >= 2:
            score += 0.18
        elif complexity >= 1:
            score += 0.10

        has_types_or_classes = any(
            kw in content for kw in ("class ", "struct ", "interface ", "impl ", "->", ": ")
        )
        if has_types_or_classes:
            score += 0.10

        indented_lines = sum(1 for line in lines if line and line[0] in (" ", "\t"))
        if indented_lines / max(num_lines, 1) > 0.25 or doc.language in {"markdown", "diff", "sql"}:
            score += 0.15

        open_b = content.count("(") + content.count("[") + content.count("{")
        close_b = content.count(")") + content.count("]") + content.count("}")
        if abs(open_b - close_b) <= 5:
            score += 0.10

        return round(min(score, 1.0), 4)


class Deduplicator:
    """Document-level exact SHA-256 + MinHash LSH near-duplicate detector."""

    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self._exact: set[str] = set()
        self._sigs: dict[str, np.ndarray] = {}
        self._buckets: dict[tuple[int, tuple[int, ...]], list[str]] = {}
        self.seen: set[str] = set()
        self.total_checked = 0
        self.exact_duplicates = 0
        self.near_duplicates = 0

        if HAS_DATASKETCH:
            self.lsh = datasketch.MinHashLSH(threshold=threshold, num_perm=num_perm)
        else:
            self.lsh = None

    @staticmethod
    def _normalized_text(content: str) -> str:
        return re.sub(r"\s+", " ", content.strip())

    @staticmethod
    def _builtin_signature(content: str, num_perm: int = 128) -> np.ndarray:
        shingles = {content[i : i + 5] for i in range(max(len(content) - 4, 0))} or {content}
        if len(shingles) > 4096:
            shingles = set(sorted(shingles)[:4096])
        sig = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        for sh in shingles:
            h = int.from_bytes(
                hashlib.sha256(sh.encode("utf8", "ignore")).digest()[:8], "little"
            )
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

    def _create_minhash(self, content: str) -> Optional["datasketch.MinHash"]:
        if not HAS_DATASKETCH:
            return None
        norm = self._normalized_text(content)
        if len(norm) < 5:
            return None
        step = max(1, len(norm) // 4096)
        shingles = {norm[i : i + 5] for i in range(0, len(norm) - 4, step)}
        if not shingles:
            return None
        m = datasketch.MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            m.update(shingle.encode("utf8", "ignore"))
        return m

    def is_duplicate(self, doc_id: str, content: str) -> bool:
        self.total_checked += 1
        norm = self._normalized_text(content)
        content_hash = hashlib.sha256(norm.encode("utf8", "ignore")).hexdigest()
        if content_hash in self._exact or doc_id in self.seen:
            self.exact_duplicates += 1
            return True

        unique_key = f"{doc_id}:{content_hash[:12]}"
        if not self.lsh:
            sig = self._builtin_signature(norm, self.num_perm)
            bands, rows = 16, max(1, self.num_perm // 16)
            for b in range(bands):
                key = (b, tuple(sig[b * rows : (b + 1) * rows].tolist()))
                for other in self._buckets.get(key, []):
                    sim = float(np.mean(sig == self._sigs[other]))
                    if sim >= self.threshold:
                        self.near_duplicates += 1
                        return True
            self._sigs[unique_key] = sig
            for b in range(bands):
                key = (b, tuple(sig[b * rows : (b + 1) * rows].tolist()))
                self._buckets.setdefault(key, []).append(unique_key)
            self._exact.add(content_hash)
            self.seen.add(doc_id)
            return False

        minhash = self._create_minhash(norm)
        if minhash is not None:
            if self.lsh.query(minhash):
                self.near_duplicates += 1
                return True
            self.lsh.insert(unique_key, minhash)

        self._exact.add(content_hash)
        self.seen.add(doc_id)
        return False

    def stats(self) -> dict[str, Any]:
        removed = self.exact_duplicates + self.near_duplicates
        return {
            "checked": self.total_checked,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "unique_kept": self.total_checked - removed,
            "dedup_rate": round(removed / max(self.total_checked, 1), 6),
        }


class HeldoutRegistry:
    """Tracks held-out content hashes and IDs so they can NEVER enter training."""

    def __init__(self) -> None:
        self.sha256_set: set[str] = set()
        self.norm_sha256_set: set[str] = set()
        self.paths_and_ids: set[str] = set()
        self.blocked_attempts: int = 0

    @staticmethod
    def _norm_hash(text: str) -> str:
        norm = re.sub(r"\s+", " ", text.strip())
        return hashlib.sha256(norm.encode("utf-8", "ignore")).hexdigest()

    def register_text(self, content: str, identifier: str = "") -> None:
        if content:
            self.sha256_set.add(hashlib.sha256(content.encode("utf-8", "ignore")).hexdigest())
            self.norm_sha256_set.add(self._norm_hash(content))
        if identifier:
            self.paths_and_ids.add(identifier)

    def load_directory(self, heldout_dir: str | Path) -> int:
        root = Path(heldout_dir)
        if not root.exists():
            return 0
        before = len(self.sha256_set)
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.name.endswith("manifest.json"):
                try:
                    meta = json.loads(path.read_text(encoding="utf-8"))
                    for h in meta.get("heldout_sha256", []):
                        self.sha256_set.add(str(h))
                    for p in meta.get("heldout_paths", []):
                        self.paths_and_ids.add(str(p))
                except Exception:
                    pass
                continue
            if path.suffix == ".jsonl":
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                            self.register_text(
                                row.get("content", ""),
                                str(row.get("path") or row.get("id") or ""),
                            )
                        except json.JSONDecodeError:
                            continue
            else:
                try:
                    content = path.read_text(encoding="utf-8")
                    self.register_text(content, str(path.relative_to(root)))
                except UnicodeDecodeError:
                    continue
        return len(self.sha256_set) - before

    def is_heldout(self, content: str, identifier: str = "") -> bool:
        if identifier and identifier in self.paths_and_ids:
            self.blocked_attempts += 1
            return True
        if content:
            raw_h = hashlib.sha256(content.encode("utf-8", "ignore")).hexdigest()
            if raw_h in self.sha256_set or self._norm_hash(content) in self.norm_sha256_set:
                self.blocked_attempts += 1
                return True
        return False

    def summary(self) -> dict[str, Any]:
        return {
            "heldout_hashes": len(self.sha256_set),
            "heldout_identifiers": len(self.paths_and_ids),
            "blocked_contamination_attempts": self.blocked_attempts,
        }


class DataProcessor:
    """Production document filter (license + language + quality + MinHash dedup + held-out)."""

    def __init__(
        self,
        min_quality: float = 0.3,
        dedup_threshold: float = 0.8,
        min_length: int = 100,
        max_length: int = 200_000,
        languages: list[str] | None = None,
        min_lines: int = 4,
        max_lines: int = 6000,
        min_complexity: int = 1,
        heldout: HeldoutRegistry | None = None,
    ):
        self.min_quality = min_quality
        self.min_length = min_length
        self.max_length = max_length
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.min_complexity = min_complexity
        self.languages = languages or [
            "python",
            "javascript",
            "typescript",
            "java",
            "cpp",
            "c",
            "go",
            "rust",
            "ruby",
            "php",
            "csharp",
            "kotlin",
            "swift",
            "scala",
            "sql",
            "shell",
            "yaml",
            "toml",
            "markdown",
            "diff",
        ]

        self.detector = LanguageDetector()
        self.scorer = QualityScorer()
        self.dedup = Deduplicator(threshold=dedup_threshold)
        self.license_filter = LicenseFilter(allow_unknown=False)
        self.heldout = heldout or HeldoutRegistry()

        self.stats: dict[str, int] = {
            "total": 0,
            "filtered_length": 0,
            "filtered_license": 0,
            "filtered_language": 0,
            "filtered_hard_quality": 0,
            "filtered_quality": 0,
            "filtered_heldout": 0,
            "filtered_duplicate": 0,
            "kept": 0,
        }

    def process_document(
        self,
        content: str,
        source: str,
        license: str | None = None,
        *,
        path: str | None = None,
        language_hint: str | None = None,
        slice_hint: str | None = None,
    ) -> Document | None:
        self.stats["total"] += 1

        if len(content) < self.min_length or len(content) > self.max_length:
            self.stats["filtered_length"] += 1
            return None

        canonical_license = self.license_filter.canonicalize(license)
        if canonical_license is None:
            self.stats["filtered_license"] += 1
            return None

        doc_path = path or source
        if self.heldout.is_heldout(content, doc_path):
            self.stats["filtered_heldout"] += 1
            return None

        language = (language_hint or "").lower().strip()
        if not language or language == "unknown":
            language = self.detector.detect(content, doc_path)
        if language not in self.languages:
            self.stats["filtered_language"] += 1
            return None

        eff_min_lines = 2 if self.min_quality == 0.0 else self.min_lines
        eff_complexity = 0 if self.min_quality == 0.0 else self.min_complexity
        ok, _reason = self.scorer.passes_hard_filters(
            content,
            language,
            doc_path,
            min_lines=eff_min_lines,
            max_lines=self.max_lines,
            max_bytes=self.max_length,
            min_complexity=eff_complexity,
        )
        if not ok:
            self.stats["filtered_hard_quality"] += 1
            return None

        resolved_slice = slice_hint or classify_real_slice(doc_path, language, content)
        doc = Document(
            content=content,
            language=language,
            source=source,
            license=canonical_license,
            path=doc_path,
            slice_name=resolved_slice,
        )
        doc.quality_score = self.scorer.score(doc)
        if doc.quality_score < self.min_quality:
            self.stats["filtered_quality"] += 1
            return None

        if self.dedup is not None:
            doc_id = hashlib.sha256(f"{doc_path}:{content[:512]}".encode("utf-8", "ignore")).hexdigest()[:24]
            if self.dedup.is_duplicate(doc_id, content):
                self.stats["filtered_duplicate"] += 1
                return None

        self.stats["kept"] += 1
        return doc

    def process_stream(
        self, documents: Iterator[tuple[str, str, str]]
    ) -> Iterator[Document]:
        for content, source, license_id in documents:
            doc = self.process_document(content, source, license_id)
            if doc is not None:
                yield doc
            if self.stats["total"] % 10000 == 0:
                logger.info(
                    "Processed %d docs, kept %d (%.1f%%)",
                    self.stats["total"],
                    self.stats["kept"],
                    self.stats["kept"] / max(self.stats["total"], 1) * 100,
                )

    def get_stats(self) -> dict[str, Any]:
        out = dict(self.stats)
        if self.dedup is not None:
            out["dedup"] = self.dedup.stats()
        out["heldout"] = self.heldout.summary()
        return out


class DataPacker:
    """Pack documents into fixed-length sequences with BOS/EOS boundaries."""

    def __init__(self, tokenizer, seq_length: int = 4096):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.buffer: list[int] = []
        self.buffer_tokens = 0

    def add_document(self, doc: Document) -> list[np.ndarray]:
        tokens = self.tokenizer.encode(doc.content, add_special_tokens=True)
        completed: list[np.ndarray] = []
        self.buffer.extend(tokens)
        self.buffer_tokens += len(tokens)
        while self.buffer_tokens >= self.seq_length:
            seq = np.array(self.buffer[: self.seq_length], dtype=np.uint32)
            completed.append(seq)
            self.buffer = self.buffer[self.seq_length :]
            self.buffer_tokens -= self.seq_length
        return completed

    def flush(self) -> list[np.ndarray]:
        if self.buffer:
            pad_id = getattr(self.tokenizer, "pad_token_id", 0)
            seq = np.full(self.seq_length, pad_id, dtype=np.uint32)
            seq[: len(self.buffer)] = self.buffer
            self.buffer = []
            self.buffer_tokens = 0
            return [seq]
        return []


class ProductionShardBuilder:
    """Sole production shard writer for 3B/20B pretraining.

    Writes uint32 binary shards (``train_00000.bin``, ``val_00000.bin``),
    ``train_manifest.json``, ``val_manifest.json``, and a comprehensive
    ``corpus_manifest.json`` with SHA-256 checksums, source weights, language
    and license breakdowns, AST diversity counts, and held-out isolation proof.
    """

    def __init__(
        self,
        output_dir: str | Path,
        tokenizer,
        *,
        seq_length: int = 4096,
        shard_size_tokens: int = 50_000_000,
        val_fraction: float = 0.02,
        seed: int = 42,
        heldout: HeldoutRegistry | None = None,
    ):
        if seq_length < 2:
            raise ValueError("seq_length must be >= 2")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        # Keep shards an exact multiple of seq_length so every sequence is aligned.
        aligned = max(seq_length, (shard_size_tokens // seq_length) * seq_length)
        self.shard_size_tokens = aligned
        self.val_fraction = val_fraction
        self.seed = seed
        self.heldout = heldout or HeldoutRegistry()

        self._buffers: dict[str, list[int]] = {"train": [], "val": []}
        self._shards: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
        self._source_tokens: Counter[str] = Counter()
        self._source_docs: Counter[str] = Counter()
        self._lang_tokens: Counter[str] = Counter()
        self._lang_docs: Counter[str] = Counter()
        self._license_docs: Counter[str] = Counter()
        self._unique_paths: set[str] = set()
        self._ast_patterns: set[str] = set()
        self._doc_hashes: dict[str, set[str]] = {"train": set(), "val": set()}
        self._multi_file_docs = 0
        self._single_file_docs = 0

    def _choose_split(self, doc_sha: str, index: int) -> str:
        if self.val_fraction <= 0:
            return "train"
        h = int(hashlib.sha256(f"{self.seed}:{doc_sha}:{index}".encode()).hexdigest()[:8], 16)
        return "val" if (h % 10_000) < int(self.val_fraction * 10_000) else "train"

    def _write_shard(self, split: str, tokens: Sequence[int]) -> None:
        idx = len(self._shards[split])
        name = f"{split}_{idx:05d}.bin"
        path = self.output_dir / name
        tmp = path.with_suffix(".bin.tmp")
        arr = np.asarray(tokens, dtype=np.uint32)
        arr.tofile(tmp)
        os.replace(tmp, path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        self._shards[split].append(
            {
                "shard_id": idx,
                "path": name,
                "tokens": int(arr.size),
                "size_bytes": int(arr.nbytes),
                "sha256": sha,
            }
        )

    def add_document(self, doc: Document, *, force_split: str | None = None) -> int:
        if not doc.license:
            raise ValueError(f"refusing unlicensed document from {doc.source!r}")
        if self.heldout.is_heldout(doc.content, doc.path):
            raise ValueError(f"held-out document {doc.path!r} attempted to enter shards")

        doc_sha = doc.sha256()

        # Never let one document land in both splits: a synthesized slice can
        # legitimately repeat content across ids, and a train/val hash overlap is
        # leakage regardless of intent.
        if doc_sha in self._doc_hashes["train"]:
            split = "train"
        elif doc_sha in self._doc_hashes["val"]:
            split = "val"
        else:
            total_docs_so_far = self._multi_file_docs + self._single_file_docs
            split = force_split or self._choose_split(doc_sha, total_docs_so_far)

        tokens = self.tokenizer.encode(doc.content, add_special_tokens=True)
        if not tokens:
            return 0

        self._doc_hashes[split].add(doc_sha)
        self._source_tokens[doc.slice_name] += len(tokens)
        self._source_docs[doc.slice_name] += 1
        self._lang_tokens[doc.language] += len(tokens)
        self._lang_docs[doc.language] += 1
        self._license_docs[doc.license] += 1
        self._unique_paths.add(doc.path or f"{doc.source}:{doc_sha[:12]}")
        self._ast_patterns.add(ast_pattern_signature(doc.content, doc.language))
        if doc.is_multi_file or doc.content.count("### ") >= 2 or doc.content.count("diff --git ") >= 2:
            self._multi_file_docs += 1
        else:
            self._single_file_docs += 1

        buf = self._buffers[split]
        buf.extend(tokens)
        while len(buf) >= self.shard_size_tokens:
            chunk = buf[: self.shard_size_tokens]
            del buf[: self.shard_size_tokens]
            self._write_shard(split, chunk)
        return len(tokens)

    def finalize(self, extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        pad_id = int(getattr(self.tokenizer, "eos_token_id", getattr(self.tokenizer, "pad_token_id", 2)))
        # Ensure val has at least one full sequence if any val tokens exist or borrow last sequence if empty
        if not self._buffers["val"] and not self._shards["val"] and len(self._buffers["train"]) >= 2 * self.seq_length:
            borrowed = self._buffers["train"][-self.seq_length :]
            del self._buffers["train"][-self.seq_length :]
            self._buffers["val"].extend(borrowed)

        for split in ("train", "val"):
            buf = self._buffers[split]
            if not buf:
                continue
            rem = len(buf) % self.seq_length
            if rem != 0:
                # Pad final partial window with EOS/PAD so total_tokens is an exact multiple of seq_length
                buf.extend([pad_id] * (self.seq_length - rem))
            self._write_shard(split, buf)
            buf.clear()

        split_manifests: dict[str, dict[str, Any]] = {}
        for split in ("train", "val"):
            shards = self._shards[split]
            total_tokens = sum(s["tokens"] for s in shards)
            manifest = {
                "split": split,
                "seq_length": self.seq_length,
                "total_tokens": total_tokens,
                "num_sequences": total_tokens // self.seq_length,
                "num_shards": len(shards),
                "shards": shards,
            }
            manifest_path = self.output_dir / f"{split}_manifest.json"
            manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            manifest_path.write_bytes(manifest_bytes)
            manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
            split_manifests[split] = manifest

        raw_stream_tokens = sum(self._source_tokens.values())
        realized_weights = {
            name: round(toks / max(raw_stream_tokens, 1), 6)
            for name, toks in sorted(self._source_tokens.items())
        }
        train_val_overlap = len(self._doc_hashes["train"] & self._doc_hashes["val"])
        heldout_overlap = len(
            (self._doc_hashes["train"] | self._doc_hashes["val"]) & self.heldout.sha256_set
        )
        total_docs = self._multi_file_docs + self._single_file_docs

        overall_hasher = hashlib.sha256()
        for split in ("train", "val"):
            for s in self._shards[split]:
                overall_hasher.update(f"{split}:{s['path']}:{s['sha256']}".encode("utf-8"))

        corpus_manifest: dict[str, Any] = {
            "corpus_version": "corpus-v1",
            "seq_length": self.seq_length,
            "seed": self.seed,
            "corpus_sha256": overall_hasher.hexdigest(),
            "splits": split_manifests,
            "total_documents": total_docs,
            "total_stream_tokens": raw_stream_tokens,
            "total_packed_tokens": sum(m["total_tokens"] for m in split_manifests.values()),
            "sources": {
                name: {
                    "documents": self._source_docs[name],
                    "tokens": self._source_tokens[name],
                    "realized_weight": realized_weights[name],
                }
                for name in sorted(self._source_tokens)
            },
            "languages": {
                lang: {
                    "documents": self._lang_docs[lang],
                    "tokens": self._lang_tokens[lang],
                    "token_share": round(self._lang_tokens[lang] / max(raw_stream_tokens, 1), 6),
                }
                for lang in sorted(self._lang_tokens)
            },
            "licenses": dict(sorted(self._license_docs.items())),
            "diversity": {
                "unique_files": len(self._unique_paths),
                "unique_ast_patterns": len(self._ast_patterns),
                "multi_file_documents": self._multi_file_docs,
                "single_file_documents": self._single_file_docs,
                "multi_file_ratio": round(self._multi_file_docs / max(total_docs, 1), 6),
            },
            "isolation": {
                **self.heldout.summary(),
                "heldout_overlap_documents": heldout_overlap,
                "train_val_overlap_documents": train_val_overlap,
                "duplicate_documents": max(
                    0,
                    total_docs
                    - len(self._doc_hashes["train"] | self._doc_hashes["val"]),
                ),
            },
        }
        if extra_metadata:
            corpus_manifest.update(extra_metadata)

        corpus_manifest_path = self.output_dir / "corpus_manifest.json"
        payload = json.dumps(corpus_manifest, indent=2, sort_keys=True).encode("utf-8")
        corpus_manifest_path.write_bytes(payload)
        corpus_manifest["corpus_manifest_sha256"] = hashlib.sha256(payload).hexdigest()
        return corpus_manifest


def process_dataset_parallel(
    documents: Iterator[tuple[str, str, str]],
    processor: DataProcessor,
    num_workers: int | None = None,
) -> Iterator[Document]:
    """Process documents in parallel using multiprocessing (dedup in main process)."""
    if num_workers is None:
        num_workers = mp.cpu_count()

    def process_no_dedup(args: tuple[str, str, str]) -> Document | None:
        content, source, license_id = args
        temp_proc = DataProcessor(
            min_quality=processor.min_quality,
            min_length=processor.min_length,
            max_length=processor.max_length,
            languages=processor.languages,
            min_lines=processor.min_lines,
            max_lines=processor.max_lines,
            min_complexity=processor.min_complexity,
        )
        temp_proc.dedup = None
        return temp_proc.process_document(content, source, license_id)

    with mp.Pool(num_workers) as pool:
        for doc in pool.imap(process_no_dedup, documents, chunksize=100):
            if doc is not None:
                doc_id = hashlib.sha256(f"{doc.path}:{doc.content[:512]}".encode("utf-8", "ignore")).hexdigest()[:24]
                if not processor.heldout.is_heldout(doc.content, doc.path) and not processor.dedup.is_duplicate(doc_id, doc.content):
                    yield doc
