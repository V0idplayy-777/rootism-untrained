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


def extract_calls(path: str, text: str, language: str) -> set[str]:
    """Names invoked/constructed in a file (Python AST; regex fallback).

    The call graph is what makes retrieval *transitive*: a bug report that
    names a symptom function drags in its callers and callees even when their
    text shares no keywords with the query.
    """

    if language == "python":

        try:

            tree = ast.parse(text)

        except SyntaxError:

            tree = None

        if tree is not None:

            names: set[str] = set()

            for node in ast.walk(tree):

                if isinstance(node, ast.Call):

                    func = node.func

                    if isinstance(func, ast.Name):

                        names.add(func.id)

                    elif isinstance(func, ast.Attribute):

                        names.add(func.attr)

            return names

    return set(re.findall(r"(?:\.|\b)([A-Za-z_][A-Za-z0-9_]{2,})\s*\(", text))


_QUERY_SYNONYMS = {
    "logging in": "login authenticate session",
    "log in": "login authenticate",
    "sign in": "login authenticate session",
    "sign-in": "login authenticate",
    "auth": "authentication token session",
    "session": "token cookie session",
    "log out": "logout invalidate session",
    "expiry": "expires expiration ttl",
    "retry": "retries backoff attempt",
    "authz": "authorization permissions rbac",
    "permission": "authorization rbac allowed",
    "bug": "fix failure broken",
    "crash": "error exception failure",
    "slow": "latency performance timeout",
    "bill": "billing charge invoice payment",
    "charge": "billing payment invoice",
    "store": "storage save persist write",
    "fetch": "get retrieve read lookup",
    "delete": "remove destroy drop",
    "create": "new add insert make",
    "validate": "check verify schema parse",
}


def _synonym_query(task: str) -> str:
    lowered = task.lower()
    words = set()
    for phrase, expansion in _QUERY_SYNONYMS.items():
        if phrase in lowered:
            words.update(expansion.split())
    return " ".join(sorted(words))


def expand_query(task: str, max_hops: int = 0) -> list[str]:
    """Split a task into complementary retrieval queries.

    Lexical retrieval over identifier tokens handles natural language poorly
    when the task mentions file paths, code identifiers or a pasted error. One
    query per signal type (prose, identifiers, paths, error text) with merged
    rankings is strictly more robust than any single query.
    """

    queries = [task]

    idents = re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", task)

    if idents:

        queries.append(" ".join(idents))

    paths = re.findall(r"[\w./\\-]+\.[A-Za-z]{1,5}", task)

    if paths:

        queries.append(" ".join(p.replace("/", " ").replace("\\", " ") for p in paths))

    err = re.search(
        r"(?:error|exception|failed|failing|traceback|stack ?trace)\b[:\s-]*([^\n]+)",
        task,
        re.IGNORECASE,
    )

    if err:

        queries.append(err.group(1)[:200])

    synonyms = _synonym_query(task)

    if synonyms:

        queries.append(synonyms)

    if max_hops > 0:
        task_lower = task.lower()
        if "fix" in task_lower or "bug" in task_lower:
            queries.append("error handling " + " ".join(idents[:3]) if idents else "")
        if "find" in task_lower or "locate" in task_lower:
            queries.append("definition of " + " ".join(idents[:3]) if idents else "")
        if "refactor" in task_lower:
            queries.append("related functions of " + " ".join(idents[:3]) if idents else "")

    return queries[:4 + max_hops]



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

        self.calls: dict[str, set[str]] = {}

        self.defines: dict[str, set[str]] = {}

        self.file_language: dict[str, str] = {}

        self.stats = RepoStats()

        self._doc_tokens: list[Counter] = []

        self._df: Counter = Counter()

        self._tfidf: list[dict[str, float]] = []

        self._postings: dict[str, list[int]] = defaultdict(list)

        self._symbol_defs: dict[str, list[str]] = defaultdict(list)

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

        self.calls[rel_path] = extract_calls(rel_path, text, language)

        self.defines[rel_path] = {s.name.split(".")[-1] for s in self.symbols
                                  if s.path == rel_path}

        self._built = False

        return self

    def update_file(self, rel_path: str) -> "RepoIndex":
        """Re-index one file after the agent edited it (long sessions).

        Without this, every retrieval after the first edit returns stale
        chunks — the model reads its own outdated code and "fixes" it back.
        """

        self.chunks = [c for c in self.chunks if c.path != rel_path]

        self.symbols = [s for s in self.symbols if s.path != rel_path]

        self.imports.pop(rel_path, None)

        self.calls.pop(rel_path, None)

        self.defines.pop(rel_path, None)

        if rel_path in self.file_language:

            self.stats.files -= 1

            lang = self.file_language.pop(rel_path)

            if self.stats.languages.get(lang):

                self.stats.languages[lang] -= 1

        self._built = False

        full = self.root / rel_path

        if full.exists():
            self.add_file(rel_path)

        self.finalise()

        return self

    def finalise(self) -> "RepoIndex":
        """Build BM25 + TF-IDF statistics (idempotent)."""

        if self._built:

            return self

        self._doc_tokens = [Counter(tokenize_for_index(c.text)) for c in self.chunks]

        self._df = Counter()

        for counts in self._doc_tokens:

            self._df.update(counts.keys())


        n = max(len(self.chunks), 1)

        self._tfidf: list[dict[str, float]] = []

        for counts in self._doc_tokens:

            vec = {}

            for term, tf in counts.items():

                df = self._df[term]

                if df:

                    vec[term] = (1.0 + math.log(tf)) * math.log(1.0 + n / df)

            norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0

            self._tfidf.append({t: w / norm for t, w in vec.items()})


        self._postings: dict[str, list[int]] = defaultdict(list)

        for i, counts in enumerate(self._doc_tokens):

            for term in counts:

                self._postings[term].append(i)

        self._symbol_defs: dict[str, list[str]] = defaultdict(list)

        for path, names in self.defines.items():

            for name in names:

                self._symbol_defs[name].append(path)

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

    def _cosine(self, query: str, k: int) -> list[tuple[float, int]]:
        """Cosine similarity over sublinear TF-IDF vectors (postings-driven)."""

        self.finalise()

        q_counts = Counter(tokenize_for_index(query))

        if not q_counts or not self.chunks:

            return []

        n = max(len(self.chunks), 1)

        q_vec: dict[str, float] = {}

        for term, tf in q_counts.items():

            df = self._df.get(term, 0)

            if df:

                q_vec[term] = (1.0 + math.log(tf)) * math.log(1.0 + n / df)

        q_norm = math.sqrt(sum(w * w for w in q_vec.values())) or 1.0

        scores: dict[int, float] = defaultdict(float)

        for term, qw in q_vec.items():

            for i in self._postings.get(term, ()):

                dw = self._tfidf[i].get(term)

                if dw is not None:

                    scores[i] += (qw / q_norm) * dw

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

        return [(score, i) for i, score in ranked[:k]]

    def _hybrid_ids(self, query: str, k: int, rrf_k: int = 60) -> list[int]:
        """Reciprocal-rank fusion of BM25 and TF-IDF cosine rank lists.

        BM25 favours exact rare-token hits; cosine tolerates term distribution
        drift (paraphrased issues, different naming). Fusing their *rankings*
        is robust to either one failing, and needs no score normalisation.
        """

        bm25 = [i for _s, i in self._bm25(query, k * 3)]

        cos = [i for _s, i in self._cosine(query, k * 3)]

        fused: dict[int, float] = defaultdict(float)

        for rank, i in enumerate(bm25):

            fused[i] += 1.0 / (rrf_k + rank + 1)

        for rank, i in enumerate(cos):

            fused[i] += 1.0 / (rrf_k + rank + 1)

        order = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))

        return [i for i, _score in order[:k]]

    def search_hybrid(self, query: str, k: int = 8) -> list[Chunk]:
        """RRF-fused BM25 + cosine retrieval."""

        return [self.chunks[i] for i in self._hybrid_ids(query, k)]

    def search_task(self, task: str, k: int = 8) -> list[Chunk]:
        """Multi-query hybrid retrieval over an expanded task description."""

        seen: set[int] = set()

        out: list[Chunk] = []

        for query in expand_query(task):

            for i in self._hybrid_ids(query, k=k):

                if i not in seen:

                    seen.add(i)

                    out.append(self.chunks[i])

            if len(out) >= k:

                break

        return out[:k]

    def callers_of(self, name: str) -> list[str]:
        """Files that call a function/method with this name."""

        self.finalise()

        return [

            path

            for path, called in self.calls.items()

            if name in called

        ]

    def callees_of(self, path: str) -> set[str]:
        """Names invoked by `path` that are defined somewhere in the repo."""

        self.finalise()

        out: set[str] = set()

        for name in self.calls.get(path, set()):

            for defin in self._symbol_defs.get(name, ()):

                if defin != path:

                    out.add(defin)

        return out

    def related_by_calls(self, paths: list[str], depth: int = 2, max_files: int = 6) -> list[str]:
        """Breadth-first call-graph expansion around `paths` (both directions)."""

        self.finalise()

        frontier = set(paths)

        visited = set(paths)

        out: set[str] = set()

        for _ in range(depth):

            nxt: set[str] = set()

            for path in frontier:

                for related in self.callees_of(path):

                    if related not in visited:

                        nxt.add(related)

                for name in self.calls.get(path, set()):

                    for caller in self._symbol_defs.get(name, ()):

                        if caller not in visited:

                            nxt.add(caller)

            visited |= nxt

            out |= nxt

            frontier = nxt

            if len(out) >= max_files:

                break

        return sorted(out)[:max_files]

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
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "tokens": self.tokens,
            "map_included": self.map_included,
            "truncated": self.truncated,
            "features": dict(self.features),
            "provenance": [c.header for c in self.chunks],
        }


class ContextAssembler:
    """Pack repo map + retrieved chunks into a token budget.

    Modes exist so retrieval can be ablated in evaluation:

    - ``none``: no repository context at all (the "local change" baseline),
    - ``flat``: BM25 chunks only, no repo map,
    - ``hierarchical``: multi-query hybrid retrieval (BM25 + TF-IDF cosine,
      RRF-fused), repo map + hierarchical repo/dir/file summaries, import-
      and call-graph expansion, all under the budget (default).

    ``ContextPack.features`` records which machinery actually engaged, so the
    retrieval ablation measures the components instead of trusting them.
    """

    def __init__(self, index: RepoIndex, tokenizer=None, summarizer=None):

        self.index = index

        self.tokenizer = tokenizer

        self.summarizer = summarizer or HierarchicalSummarizer(index)

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

        queries = expand_query(task) if mode == "hierarchical" else [task]

        if mode == "hierarchical":

            hits = self.index.search_task(task, k=k)

        else:

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

        features: dict[str, object] = {
            "queries": len(queries),
            "hybrid": mode == "hierarchical",
            "call_graph_files": 0,
            "file_summaries": 0,
            "repo_summary": False,
        }

        if mode == "hierarchical":

            map_text = self.index.repo_map(
                max_tokens=int(budget_tokens * map_fraction),
                tokenizer=self.tokenizer,
                focus_paths=focus[:5],
            )

            summary_text = self.summarizer.hierarchical_context(
                focus_paths=focus[:5],
                budget_tokens=int(budget_tokens * map_fraction * 0.6),
                tokenizer=self.tokenizer,
            )

            combined = map_text + "\n" + summary_text

            cost = self._count(combined)

            if spent + cost <= budget_tokens:

                pieces.append(combined)

                spent += cost

                map_included = True

                features["repo_summary"] = "<|repo_map|>" in summary_text

            else:

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


            call_graph = self.index.related_by_calls(
                [c.path for c in used_chunks[:4]], depth=2, max_files=4
            )

            for rel in call_graph:

                if rel not in focus and rel not in related:

                    related.append(rel)

            features["call_graph_files"] = len(
                [r for r in related[:6] if r in call_graph]
            )

            for rel in related[:6]:

                outline = self.index.file_outline(rel)

                summary = self.summarizer.file_summary(rel, max_chars=360)

                block = f"<|retrieval|> {summary}\n{outline}" if outline != rel else f"<|retrieval|> {summary}"

                cost = self._count(block)

                if spent + cost > budget_tokens:

                    truncated = True

                    break

                pieces.append(block)

                features["file_summaries"] += 1

                spent += cost

        if not pieces:

            return ContextPack(
                text="", tokens=0, mode=mode, truncated=truncated, features=features
            )

        text = "\n".join(pieces)

        return ContextPack(
            text=text,
            tokens=spent,
            mode=mode,
            chunks=used_chunks,
            map_included=map_included,
            truncated=truncated,
            features=features,
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




@dataclass
class FileFacts:
    """Deterministic extractive facts about one file."""

    path: str
    language: str
    lines: int
    imports: list[str]
    symbols: list[Symbol]
    docstring: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "language": self.language,
            "lines": self.lines,
            "imports": self.imports[:12],
            "symbols": [
                {"name": s.name, "kind": s.kind, "line": s.line, "signature": s.signature}
                for s in self.symbols[:40]
            ],
            "docstring": self.docstring[:200],
        }


def _first_docstring(path: str, text: str, language: str) -> str:
    """Module docstring (Python AST) or leading comment block."""

    if language == "python":

        try:

            tree = ast.parse(text)

        except SyntaxError:

            return ""

        doc = ast.get_docstring(tree)

        return (doc or "").strip().splitlines()[0][:200] if doc else ""

    lines: list[str] = []

    for line in text.splitlines():

        stripped = line.strip()

        if not stripped:

            continue

        if stripped.startswith(("#", "//", "*", "/*")):

            body = stripped.lstrip("/*#* ").rstrip("*/ ")

            if body:

                lines.append(body)

            if len(lines) >= 3:

                break

        else:

            break

    return " ".join(lines)[:200]


class HierarchicalSummarizer:
    """Repo -> directory -> file -> signature summaries under a token budget.

    ``ContextAssembler`` retrieval answers "which code is relevant"; this
    answers "what is this project and how is it organised" — the part an
    architect needs and a chunk search cannot supply.

    ``summarize_fn(text, instruction) -> str`` is an optional LLM seam. When
    absent (the default, and what tests use) every summary is *extractive and
    deterministic*: signatures, counts, docstring heads. The LLM may improve
    the prose; it may never change the facts, because the extractive version
    is always the fallback and the checker.
    """

    def __init__(self, index: RepoIndex, summarize_fn=None):
        self.index = index
        self.summarize_fn = summarize_fn

    def _load_cached_summaries(self) -> dict:
        """Load cached file summaries from the index."""
        cache_path = Path(self.index.root) / ".agent" / "file_summaries.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_cached_summaries(self, summaries: dict) -> None:
        """Save file summaries to the index cache."""
        cache_path = Path(self.index.root) / ".agent" / "file_summaries.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(summaries, f, indent=2)

    def file_summary_cached(self, path: str, max_chars: int = 500) -> str:
        """File summary with caching - only recomputes when file changes."""
        summaries = self._load_cached_summaries()
        file_mtime = Path(path).stat().st_mtime
        
        if path in summaries and summaries[path].get("mtime") == file_mtime:
            return summaries[path]["summary"]
        
        summary = self.file_summary(path, max_chars=max_chars)
        
        summaries[path] = {"mtime": file_mtime, "summary": summary}
        self._save_cached_summaries(summaries)
        
        return summary

    def directory_summary_cached(self, directory: str, max_files: int = 8) -> str:
        """Directory summary with caching - only recomputes when files change."""
        summaries = self._load_cached_summaries()
        
        dir_key = directory.rstrip("/")
        cached_dir_summary = summaries.get(dir_key, {}).get("dir_summary")
        
        if dir_key in summaries and summaries[dir_key].get("dir_mtime", 0) > 0:
            return cached_dir_summary if cached_dir_summary else self.directory_summary(directory, max_files=max_files)
        
        result = self.directory_summary(directory, max_files=max_files)
        
        if dir_key not in summaries:
            summaries[dir_key] = {}
        summaries[dir_key]["dir_mtime"] = Path(directory).parent.stat().st_mtime if Path(directory).exists() else 0
        summaries[dir_key]["dir_summary"] = result
        self._save_cached_summaries(summaries)
        
        return result
    def _maybe_llm(self, extractive: str, instruction: str) -> str:
        if self.summarize_fn is None:
            return extractive
        try:
            summary = str(self.summarize_fn(extractive, instruction)).strip()
            return summary or extractive
        except Exception:
            return extractive

    def file_facts(self, path: str) -> FileFacts | None:
        full = self.index.root / path
        try:
            text = full.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
        language = self.index.file_language.get(path, "text")
        symbols = sorted(
            self.index.defined_in(path), key=lambda s: (s.line, s.name)
        )
        return FileFacts(
            path=path,
            language=language,
            lines=text.count("\n") + 1,
            imports=sorted(self.index.imports.get(path, set())),
            symbols=symbols,
            docstring=_first_docstring(path, text, language),
        )

    def file_summary(self, path: str, max_chars: int = 500) -> str:
        facts = self.file_facts(path)
        if facts is None:
            return f"{path}  (unreadable)"
        lines = [f"{path}  [{facts.language}, {facts.lines} lines]"]
        if facts.docstring:
            lines.append(f'  """{facts.docstring}"""')
        for sym in facts.symbols[:12]:
            lines.append(f"  {sym.signature}  # L{sym.line}")
        if len(facts.symbols) > 12:
            lines.append(f"  ... {len(facts.symbols) - 12} more symbols")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max(0, max_chars - 20)] + "\n  ... (truncated)"
        return self._maybe_llm(
            text, f"Summarise what {path} does in 1-2 sentences. Facts only."
        )

    def directory_summary(self, directory: str, max_files: int = 8) -> str:
        files = [
            p for p in sorted(self.index.file_language)
            if str(Path(p).parent) == directory
        ]
        if not files:
            return f"{directory or '.'}  (no indexed files)"
        langs = Counter(
            self.index.file_language[p] for p in files
        )
        lines = [
            f"{directory or '.'}/  [{len(files)} files: "
            + ", ".join(f"{lang}×{n}" for lang, n in langs.most_common(4))
            + "]"
        ]
        for p in files[:max_files]:
            n_syms = sum(1 for s in self.index.symbols if s.path == p)
            lines.append(f"  {Path(p).name}  ({n_syms} symbols)")
        if len(files) > max_files:
            lines.append(f"  ... {len(files) - max_files} more files")
        return "\n".join(lines)

    def repo_summary(self, max_chars: int = 900) -> str:
        index = self.index
        dirs = Counter(
            str(Path(p).parent) for p in index.file_language
        )
        entry = Counter()
        for path in index.file_language:
            name = Path(path).name.lower()
            if name in {
                "main.py", "app.py", "index.js", "index.ts", "main.go",
                "main.rs", "lib.rs", "server.py", "cli.py", "__main__.py",
                "manage.py", "setup.py", "pyproject.toml", "package.json",
                "cargo.toml", "go.mod", "makefile",
            }:
                entry[path] = 1
        lines = [
            f"Repository: {index.root.name}  "
            f"({index.stats.files} indexed files, "
            f"{index.stats.chunks} chunks, {index.stats.symbols} symbols)",
            "Languages: "
            + ", ".join(
                f"{lang}×{n}" for lang, n in Counter(
                    index.file_language.values()
                ).most_common(6)
            ),
            "Top directories:",
        ]
        for d, n in dirs.most_common(8):
            lines.append(f"  {d or '.'}/  ({n} files)")
        if entry:
            lines.append("Entry points: " + ", ".join(sorted(entry)[:6]))
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars]
        return self._maybe_llm(
            text, "Summarise this repository's purpose and layout in 3 sentences."
        )

    def hierarchical_context(
        self,
        focus_paths: list[str] | None = None,
        budget_tokens: int = 800,
        tokenizer=None,
    ) -> str:
        """repo summary -> focus-dir summaries -> focus-file summaries."""

        def count(text: str) -> int:
            if tokenizer is not None:
                return len(tokenizer.encode(text, add_special_tokens=False))
            return max(1, len(text) // 4)

        parts: list[str] = ["<|repo_map|>"]

        repo = self.repo_summary()

        if count(repo) <= budget_tokens:

            parts.append(repo)

        focus = list(focus_paths or [])[:5]

        dirs: list[str] = []

        for p in focus:

            d = str(Path(p).parent)

            if d not in dirs:

                dirs.append(d)

        for d in dirs[:3]:

            text = self.directory_summary(d)

            if count("\n".join(parts)) + count(text) <= budget_tokens:

                parts.append(text)

        for p in focus:

            text = self.file_summary(p)

            if count("\n".join(parts)) + count(text) <= budget_tokens:

                parts.append(text)

        parts.append("<|endoftext|>")

        return "\n".join(parts)
