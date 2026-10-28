"""Repository context: indexing, retrieval and hierarchical repo maps.

Motivation: an 8k (32k YaRN) window cannot hold a real repository. SWE-bench
style work therefore needs *retrieval* rather than "paste the file", and it
needs a repository-level summary so the model knows what it does not know.

This module is dependency-free (no tree-sitter, no network) and deterministic:

- ``RepoIndex`` chunks Python by AST (top-level def/class, plus methods of large
  classes) and other languages by blank-line blocks, then builds a BM25 index
  over identifier-aware tokens.
- ``symbols`` gives exact-name lookup ("where is ``parse_tool_call`` defined?")
  and ``imports``/``callers`` give one-hop graph expansion so a retrieved chunk
  drags in its dependencies.
- ``repo_map`` renders a budgeted hierarchical outline (repo -> file -> symbols
  with signatures) which is what makes "hold the codebase in mind" tractable.
- ``ContextAssembler`` packs repo map + retrieved chunks into an explicit token
  budget, and can be run in ablation modes (``none`` / ``flat`` /
  ``hierarchical``) so retrieval quality is measured, not assumed.

The assembler fences evidence with the reserved context tokens so the model can
tell retrieved code (read-only) from the code it is asked to write.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CODE_EXTENSIONS = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".sql": "sql",
    ".sh": "shell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".md": "markdown",
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    "target",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "vendor",
    "third_party",
    ".idea",
}

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")
_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "is",
    "it",
    "for",
    "on",
    "with",
    "this",
    "that",
    "be",
    "as",
    "at",
    "by",
    "from",
    "self",
    "return",
    "def",
    "class",
    "import",
}


def tokenize_for_index(text: str) -> list[str]:
    """Identifier-aware tokens: `getUserById` -> get, user, by, id."""

    out: list[str] = []

    for tok in _IDENT.findall(text):

        lowered = tok.lower()

        if lowered in _STOP:

            continue

        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):

            for part in tok.split("_"):
                if part:
                    out.extend(p.lower() for p in _CAMEL.split(part) if p)

            out.append(lowered)

        else:

            out.append(lowered)

    return out


@dataclass
class Chunk:
    path: str
    start_line: int
    end_line: int
    kind: str
    text: str
    language: str = "text"
    name: str | None = None
    score: float = 0.0

    @property
    def header(self) -> str:
        label = f" {self.kind} {self.name}" if self.name else ""
        return f"{self.path}:{self.start_line}-{self.end_line}{label}"


@dataclass
class Symbol:
    name: str
    kind: str
    path: str
    line: int
    signature: str
    end_line: int


@dataclass
class RepoStats:
    files: int = 0
    chunks: int = 0
    symbols: int = 0
    languages: dict[str, int] = field(default_factory=dict)
    skipped: int = 0


def _py_chunks(path: str, text: str, max_lines: int = 120) -> list[Chunk]:
    """AST chunking: module header + one chunk per top-level def/class."""

    lines = text.splitlines()

    chunks: list[Chunk] = []

    try:

        tree = ast.parse(text)

    except SyntaxError:

        return chunk_by_blocks(path, text, "python")

    header_end = 0

    for node in tree.body:

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):

            break

        end = getattr(node, "end_lineno", None)

        if end:

            header_end = end

    header_text = "\n".join(lines[:header_end]).strip()

    if header_text:

        chunks.append(
            Chunk(path, 1, header_end, "module", header_text, "python", "header")
        )

    for node in tree.body:

        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):

            continue

        start = node.lineno

        end = getattr(node, "end_lineno", start)

        kind = "class" if isinstance(node, ast.ClassDef) else "function"

        segment = "\n".join(lines[start - 1 : end])

        if end - start + 1 <= max_lines:

            chunks.append(
                Chunk(path, start, end, kind, segment, "python", node.name)
            )

            continue

        head = "\n".join(lines[start - 1 : start - 1 + max_lines])

        chunks.append(Chunk(path, start, start + max_lines - 1, kind, head, "python", node.name))

        if isinstance(node, ast.ClassDef):

            for sub in node.body:

                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):

                    s_end = getattr(sub, "end_lineno", sub.lineno)

                    if s_end - sub.lineno + 1 > 400:

                        continue

                    chunks.append(
                        Chunk(
                            path,
                            sub.lineno,
                            s_end,
                            "method",
                            "\n".join(lines[sub.lineno - 1 : s_end]),
                            "python",
                            f"{node.name}.{sub.name}",
                        )
                    )

    return chunks


def chunk_by_blocks(
    path: str, text: str, language: str = "text", window: int = 40
) -> list[Chunk]:
    """Language-agnostic chunking by blank-line-separated blocks."""

    lines = text.splitlines()

    chunks: list[Chunk] = []

    start = 0

    for i in range(1, len(lines)):

        block_break = not lines[i].strip() and lines[i - 1].strip()

        too_long = i - start >= window

        if block_break or too_long:

            segment = "\n".join(lines[start:i]).strip("\n")

            if segment.strip():

                chunks.append(
                    Chunk(path, start + 1, i, "block", segment, language, None)
                )

            start = i

    tail = "\n".join(lines[start:]).strip("\n")

    if tail.strip():

        chunks.append(Chunk(path, start + 1, len(lines), "block", tail, language, None))

    return chunks


def extract_symbols(path: str, text: str, language: str) -> list[Symbol]:
    """Signatures for the repo map (Python via AST, others via regex)."""

    if language != "python":

        pat = re.compile(
            r"^\s*(?:export\s+)?(?:pub\s+)?(?:async\s+)?"
            r"(?:def|fn|func|function|class|interface|struct|type|const|let)\s+"
            r"([A-Za-z_][A-Za-z0-9_]*)"
        )

        out = []

        for i, line in enumerate(text.splitlines(), start=1):

            m = pat.match(line)

            if m:

                out.append(
                    Symbol(m.group(1), "symbol", path, i, line.strip()[:160], i)
                )

        return out

    try:

        tree = ast.parse(text)

    except SyntaxError:

        return []

    out: list[Symbol] = []

    for node in ast.walk(tree):

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):

            args = ""

            if not isinstance(node, ast.ClassDef):

                try:

                    args = ast.unparse(node.args)

                except Exception:

                    args = "..."

                signature = f"def {node.name}({args})"

            else:

                bases = [ast.unparse(b) for b in node.bases] if node.bases else []

                signature = f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")

            out.append(
                Symbol(
                    node.name,
                    "class" if isinstance(node, ast.ClassDef) else "function",
                    path,
                    node.lineno,
                    signature[:200],
                    getattr(node, "end_lineno", node.lineno),
                )
            )

    return out


def extract_imports(path: str, text: str, language: str) -> set[str]:
    """Module-level imports, normalised to dotted names."""

    imports: set[str] = set()

    if language == "python":

        try:

            tree = ast.parse(text)

        except SyntaxError:

            return imports

        for node in ast.walk(tree):

            if isinstance(node, ast.Import):

                for a in node.names:

                    imports.add(a.name)

            elif isinstance(node, ast.ImportFrom):

                if node.module:

                    imports.add(node.module)

        return imports

    for m in re.finditer(
        r"""(?:import\s+(?:[^'"]*from\s+)?['"]([^'"]+)['"]|require\(['"]([^'"]+)['"]\))""",
        text,
    ):

        imports.add(m.group(1) or m.group(2))

    return imports


class RepoIndex:
    """In-memory BM25 index over repository chunks + symbol/import graph."""

    def __init__(
        self,
        root: str,
        max_file_bytes: int = 400_000,
        chunk_lines: int = 40,
        max_files: int = 5000,
    ):

        self.root = Path(root)

        self.max_file_bytes = max_file_bytes

        self.chunk_lines = chunk_lines

        self.max_files = max_files

        self.chunks: list[Chunk] = []

        self.symbols: list[Symbol] = []

        self.imports: dict[str, set[str]] = {}

        self.file_language: dict[str, str] = {}

        self.stats = RepoStats()

        self._doc_tokens: list[Counter] = []

        self._df: Counter = Counter()

        self._built = False

    @classmethod
    def build(cls, root: str, **kwargs) -> "RepoIndex":

        index = cls(root, **kwargs)

        index.add_directory(root)

        index.finalise()

        return index

    def add_directory(self, directory: str) -> "RepoIndex":

        base = Path(directory).resolve()

        for path in sorted(base.rglob("*")):

            if len(self.file_language) >= self.max_files:

                break

            if not path.is_file():

                continue

            if any(part in SKIP_DIRS for part in path.parts):

                self.stats.skipped += 1

                continue

            if CODE_EXTENSIONS.get(path.suffix) is None:

                continue

            try:

                rel = str(path.relative_to(base))

            except ValueError:

                rel = str(path)

            self.add_file(rel, path)

        return self

    def add_file(self, rel_path: str, path: Path | None = None) -> "RepoIndex":

        language = CODE_EXTENSIONS.get(Path(rel_path).suffix, "text")

        full = path or (self.root / rel_path)

        try:

            size = full.stat().st_size

            if size > self.max_file_bytes:

                self.stats.skipped += 1

                return self

            text = full.read_text(encoding="utf-8", errors="ignore")

        except OSError:

            self.stats.skipped += 1

            return self

        if not text.strip():

            return self

        self.file_language[rel_path] = language

        self.stats.files += 1

        self.stats.languages[language] = self.stats.languages.get(language, 0) + 1

        if language == "python":

            chunks = _py_chunks(rel_path, text)

            self.symbols.extend(extract_symbols(rel_path, text, language))

        else:

            chunks = chunk_by_blocks(rel_path, text, language, self.chunk_lines)

            self.symbols.extend(extract_symbols(rel_path, text, language))

        self.chunks.extend(chunks)

        self.imports[rel_path] = extract_imports(rel_path, text, language)

        self._built = False

        return self

    def finalise(self) -> "RepoIndex":
        """Build BM25 statistics (idempotent)."""

        if self._built:

            return self

        self._doc_tokens = [Counter(tokenize_for_index(c.text)) for c in self.chunks]

        self._df = Counter()

        for counts in self._doc_tokens:

            self._df.update(counts.keys())

        self.stats.chunks = len(self.chunks)

        self.stats.symbols = len(self.symbols)

        self._built = True

        return self

    def _bm25(self, query: str, k: int) -> list[tuple[float, int]]:
        """Okapi BM25 (k1=1.5, b=0.75) over chunk token counts."""

        self.finalise()

        q_tokens = tokenize_for_index(query)

        if not q_tokens or not self.chunks:

            return []

        n = len(self.chunks)

        avgdl = sum(sum(c.values()) for c in self._doc_tokens) / max(n, 1)

        k1, b = 1.5, 0.75

        scores: list[tuple[float, int]] = []

        for i, counts in enumerate(self._doc_tokens):

            dl = sum(counts.values())

            score = 0.0

            for q in set(q_tokens):

                tf = counts.get(q, 0)

                if not tf:

                    continue

                idf = math.log(1 + (n - self._df[q] + 0.5) / (self._df[q] + 0.5))

                score += idf * (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * dl / max(avgdl, 1))
                )

            if score > 0:

                scores.append((score, i))

        scores.sort(key=lambda x: (-x[0], self.chunks[x[1]].path, self.chunks[x[1]].start_line))

        return scores[:k]

    def search(
        self, query: str, k: int = 8, path_glob: str | None = None
    ) -> list[Chunk]:
        """BM25 retrieval with an optional path filter."""

        wanted = None

        if path_glob:

            import fnmatch

            wanted = lambda p: fnmatch.fnmatch(p, path_glob)  # noqa: E731

        results = []

        for score, i in self._bm25(query, k * 4):

            chunk = self.chunks[i]

            if wanted and not wanted(chunk.path):

                continue

            clone = Chunk(
                chunk.path,
                chunk.start_line,
                chunk.end_line,
                chunk.kind,
                chunk.text,
                chunk.language,
                chunk.name,
                score,
            )

            results.append(clone)

            if len(results) >= k:

                break

        return results

    def symbol_lookup(self, name: str, limit: int = 10) -> list[Symbol]:
        """Exact then suffix match on symbol names."""

        exact = [s for s in self.symbols if s.name == name]

        if exact:

            return exact[:limit]

        suffix = [s for s in self.symbols if s.name.endswith(f".{name}")]

        return suffix[:limit]

    def defined_in(self, path: str) -> list[Symbol]:

        return [s for s in self.symbols if s.path == path]

    def related_files(self, path: str) -> set[str]:
        """One-hop graph expansion: importers, imports, shared top-level names."""

        out: set[str] = set()

        module = path[: -len(Path(path).suffix)].replace("/", ".")

        for other, imports in self.imports.items():

            if other == path:

                continue

            if any(module.endswith(i) or i.endswith(module) for i in imports):

                out.add(other)

        for imp in self.imports.get(path, set()):

            for other in self.file_language:

                if other == path:

                    continue

                other_module = other[: -len(Path(other).suffix)].replace("/", ".")

                if other_module.endswith(imp):

                    out.add(other)

        return out

    def file_outline(self, path: str) -> str:
        """One-line-per-symbol outline for a file."""

        syms = self.defined_in(path)

        if not syms:

            return f"{path}"

        lines = [f"{path}"]

        for s in sorted(syms, key=lambda s: (s.line, s.name)):

            depth = "  " if "." in s.name else ""

            lines.append(f"{depth}{s.signature}  # L{s.line}")

        return "\n".join(lines)

    def repo_map(
        self,
        max_tokens: int = 1200,
        max_files: int = 200,
        tokenizer=None,
        focus_paths: list[str] | None = None,
    ) -> str:
        """Budgeted hierarchical outline: directory -> file -> signatures.

        Focus paths (retrieved files) get their full outline; other files get a
        single-line summary. Stops when the token budget is exhausted.
        """

        def count(text: str) -> int:

            if tokenizer is not None:

                return len(tokenizer.encode(text, add_special_tokens=False))

            return max(1, len(text) // 4)

        budget = max_tokens

        focus = list(focus_paths or [])

        by_dir: dict[str, list[str]] = defaultdict(list)

        for path in sorted(self.file_language):

            by_dir[str(Path(path).parent)].append(path)

        lines = ["<|repo_map|>"]

        spent = count("\n".join(lines))

        ordered_files = focus + [
            p for p in sorted(self.file_language) if p not in focus
        ]

        emitted: set[str] = set()

        for path in ordered_files[:max_files]:

            if emitted and spent >= budget:

                break

            outline = self.file_outline(path) if path in focus else f"{path}"

            cost = count(outline)

            if spent + cost > budget:

                if path in focus:

                    outline = f"{path}  # outline omitted (budget)"

                else:

                    continue

            if path in focus:

                lines.append(outline)

            else:

                lines.append(f"{path}  [{self.file_language.get(path, '?')}, "
                             f"{sum(1 for s in self.symbols if s.path == path)} symbols]")

            emitted.add(path)

            spent += count(lines[-1])

        if len(self.file_language) > len(emitted):

            lines.append(f"... {len(self.file_language) - len(emitted)} more files not shown")

        lines.append("<|endoftext|>")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "root": str(self.root),
            "stats": {
                "files": self.stats.files,
                "chunks": self.stats.chunks,
                "symbols": self.stats.symbols,
                "languages": self.stats.languages,
                "skipped": self.stats.skipped,
            },
            "files": sorted(self.file_language),
        }


@dataclass
class ContextPack:
    """Assembled, budgeted context with provenance."""

    text: str
    tokens: int
    mode: str
    chunks: list[Chunk] = field(default_factory=list)
    map_included: bool = False
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "tokens": self.tokens,
            "map_included": self.map_included,
            "truncated": self.truncated,
            "provenance": [c.header for c in self.chunks],
        }


class ContextAssembler:
    """Pack repo map + retrieved chunks into a token budget.

    Modes exist so retrieval can be ablated in evaluation:

    - ``none``: no repository context at all (the "local change" baseline),
    - ``flat``: BM25 chunks only, no repo map,
    - ``hierarchical``: repo map + chunks + graph expansion (default).
    """

    def __init__(self, index: RepoIndex, tokenizer=None):

        self.index = index

        self.tokenizer = tokenizer

    def _count(self, text: str) -> int:

        if self.tokenizer is not None:

            return len(self.tokenizer.encode(text, add_special_tokens=False))

        return max(1, len(text) // 4)

    def build(
        self,
        task: str,
        budget_tokens: int = 6000,
        k: int = 8,
        mode: str = "hierarchical",
        focus_paths: list[str] | None = None,
        map_fraction: float = 0.25,
    ) -> ContextPack:

        assert mode in {"none", "flat", "hierarchical"}, f"unknown mode {mode!r}"

        if mode == "none" or not task.strip():

            return ContextPack(text="", tokens=0, mode=mode)

        hits = self.index.search(task, k=k)

        focus = list(focus_paths or [])

        for chunk in hits:

            if chunk.path not in focus:

                focus.append(chunk.path)

        pieces: list[str] = []

        used_chunks: list[Chunk] = []

        map_included = False

        truncated = False

        spent = 0

        if mode == "hierarchical":

            map_text = self.index.repo_map(
                max_tokens=int(budget_tokens * map_fraction),
                tokenizer=self.tokenizer,
                focus_paths=focus[:5],
            )

            cost = self._count(map_text)

            if spent + cost <= budget_tokens:

                pieces.append(map_text)

                spent += cost

                map_included = True

        seen_ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for chunk in hits:

            chunk_cost = self._count(chunk.text) + 8

            if spent + chunk_cost > budget_tokens:

                truncated = True

                continue

            overlap = any(
                not (chunk.end_line < s or chunk.start_line > e)
                for s, e in seen_ranges[chunk.path]
            )

            if overlap:

                continue

            header = f"<|file_chunk|> {chunk.header}"

            block = f"{header}\n{chunk.text}\n<|endoftext|>"

            pieces.append(block)

            used_chunks.append(chunk)

            seen_ranges[chunk.path].append((chunk.start_line, chunk.end_line))

            spent += chunk_cost

        if mode == "hierarchical" and used_chunks:

            related: list[str] = []

            for chunk in used_chunks[:4]:

                for rel in sorted(self.index.related_files(chunk.path)):

                    if rel not in focus and rel not in related:

                        related.append(rel)

            for rel in related[:3]:

                outline = self.index.file_outline(rel)

                cost = self._count(outline)

                if spent + cost > budget_tokens:

                    truncated = True

                    break

                pieces.append(f"<|retrieval|> {outline}")

                spent += cost

        if not pieces:

            return ContextPack(text="", tokens=0, mode=mode, truncated=truncated)

        text = "\n".join(pieces)

        return ContextPack(
            text=text,
            tokens=spent,
            mode=mode,
            chunks=used_chunks,
            map_included=map_included,
            truncated=truncated,
        )

    def build_task_prompt(
        self,
        problem_statement: str,
        budget_tokens: int = 6000,
        mode: str = "hierarchical",
    ) -> str:
        """Context block ready to be spliced into an agent prompt."""

        pack = self.build(problem_statement, budget_tokens=budget_tokens, mode=mode)

        if not pack.text:

            return ""

        return (
            "Repository context (retrieved evidence, read-only; do not edit "
            "files that are not part of the fix):\n" + pack.text + "\n"
        )
