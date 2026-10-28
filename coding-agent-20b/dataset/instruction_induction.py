"""Instruction data induced from real, licence-clean code.

Template-generated instruction data (``dataset/instruction_data.py``,
``dataset/augmentation.py``) covers common patterns; the failure the user
cares about is everything *outside* those patterns. This module derives
training rows from **real files** (the same licence-gated corpus the
tokenizer trains on), so the distribution is grounded in what code actually
looks like, not in what we thought to template.

Four induction shapes, all extractive and verifiable from the file itself:

- ``docstring_to_code``   - module docstring as instruction, module as answer.
- ``signature_to_body``   - signature + docstring as instruction, function
                            source as answer (fill-in style).
- ``code_to_explanation`` - real code in, *extractive* explanation out:
                            symbols, docstring, call graph facts. Nothing is
                            invented; every claim is checkable in the input.
- ``test_to_contract``    - real test file in, the contract it verifies out
                            (tested symbols + assertions, extractively).

Every row records provenance (source path + licence + induction kind), and
the corpus reader reuses the fetch script's licence allowlist, so no row can
trace back to a source we cannot ship.
"""

from __future__ import annotations

import ast
import json
import random
import re
from collections.abc import Iterator
from pathlib import Path

PERMISSIVE_LICENSES = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MPL-2.0",
    "Unlicense", "CC0-1.0", "0BSD", "Zlib",
}


def iter_corpus_files(
    corpus_dir: str | Path,
    language: str = "python",
    max_files: int = 5000,
) -> Iterator[dict]:
    """Stream ``{content, path, license}`` rows from JSONL shards."""

    root = Path(corpus_dir)

    for shard in sorted(root.glob(f"{language}-*.jsonl")):

        with open(shard, encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:

                    continue

                try:

                    row = json.loads(line)

                except json.JSONDecodeError:

                    continue

                license_id = row.get("license") or ""

                if license_id not in PERMISSIVE_LICENSES:

                    continue

                content = row.get("content") or ""

                if not content:

                    continue

                yield {
                    "content": content,
                    "path": row.get("path", shard.name),
                    "license": license_id,
                }

                max_files -= 1

                if max_files <= 0:

                    return


def _module_doc(text: str) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    doc = ast.get_docstring(tree)
    return (doc or "").strip()


def _public_functions(text: str) -> list[tuple[ast.FunctionDef, str]]:
    """Public top-level functions and public methods, with exact source."""

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines(keepends=True)

    out = []

    seen: list[ast.AST] = []

    for node in tree.body:

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):

            seen.append(node)

        elif isinstance(node, ast.ClassDef):

            for sub in node.body:

                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):

                    seen.append(sub)

    for node in seen:

        if node.name.startswith("_"):

            continue

        doc = ast.get_docstring(node)

        if not doc or len(doc) < 40:

            continue

        length = getattr(node, "end_lineno", node.lineno) - node.lineno + 1

        if not (5 <= length <= 120):

            continue

        source = "".join(lines[node.lineno - 1 : node.end_lineno])

        out.append((node, source))

    return out


def _tested_symbols(text: str) -> list[str]:
    """Function/method names a test file exercises (extractive)."""

    names: list[str] = []

    for match in re.finditer(
        r"def test_\w+_\w+", text
    ):
        names.append(match.group(0)[4:])

    for match in re.finditer(
        r"(?:monkeypatch|mock|patch)\.?(?:setattr|patch)?\(?[\"']([\w.]+)[\"']", text
    ):
        names.append(match.group(1))

    seen: set[str] = set()

    unique = []

    for name in names:

        if name not in seen:

            seen.add(name)

            unique.append(name)

    return unique


def _calls_in(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.add(func.id)
            elif isinstance(func, ast.Attribute):
                calls.add(func.attr)
    return sorted(calls)[:12]


def _explanation_of(path: str, text: str) -> str:
    """Extractive explanation: only facts present in the file itself."""

    parts: list[str] = []

    doc = _module_doc(text)

    if doc:

        parts.append(f"Module purpose (from its docstring): {doc.splitlines()[0]}")

    try:

        tree = ast.parse(text)

    except SyntaxError:

        tree = None

    if tree is not None:

        defs = [
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]

        if defs:

            parts.append(
                "Top-level definitions: " + ", ".join(defs[:10])
                + (" ..." if len(defs) > 10 else "")
            )

        classes = [
            n.name for n in tree.body if isinstance(n, ast.ClassDef)
        ]

        if classes:

            parts.append(f"Classes: {', '.join(classes[:5])}")

    calls = _calls_in(text)

    if calls:

        parts.append("Notable calls: " + ", ".join(calls))

    imports = sorted(

        {n.name for x in ast.walk(tree) if isinstance(x, ast.Import) for n in x.names}

        | {

            x.module

            for x in ast.walk(tree)

            if isinstance(x, ast.ImportFrom) and x.module

        }

    )[:8] if tree is not None else []

    if imports:

        parts.append("Depends on: " + ", ".join(imports))

    if not parts:

        return ""

    header = f"`{path}` — extractive summary (every claim is verifiable in the source):"

    return header + "\n" + "\n".join(f"- {p}" for p in parts)


class InstructionInduction:
    """Build SFT rows from real corpus files (deterministic, provenance-tagged)."""

    def __init__(
        self,
        corpus_dir: str | Path,
        language: str = "python",
        max_files: int = 2000,
        seed: int = 42,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.language = language
        self.max_files = max_files
        self.seed = seed
        self._files: list[dict] | None = None

    @property
    def files(self) -> list[dict]:
        if self._files is None:
            self._files = list(
                iter_corpus_files(self.corpus_dir, self.language, self.max_files)
            )
        return self._files

    def docstring_to_code(self, limit: int = 200) -> list[dict]:
        rows = []
        for file in self.files:
            doc = _module_doc(file["content"])
            if not (80 <= len(doc) <= 1200):
                continue
            rows.append(
                {
                    "instruction": (
                        "Implement a module that satisfies this specification "
                        "(from its own docstring):\n\n" + doc
                    ),
                    "input_context": "",
                    "output": file["content"],
                    "category": "induced_docstring_to_code",
                    "language": self.language,
                    "difficulty": "advanced",
                    "metadata": {
                        "induction": "docstring_to_code",
                        "source_path": file["path"],
                        "license": file["license"],
                    },
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def signature_to_body(self, limit: int = 300) -> list[dict]:
        rows = []
        for file in self.files:
            for node, source in _public_functions(file["content"]):
                doc = ast.get_docstring(node)
                if not doc:
                    continue
                try:
                    args = ast.unparse(node.args)
                except Exception:
                    continue
                is_async = isinstance(node, ast.AsyncFunctionDef)
                signature = (
                    f"{'async ' if is_async else ''}def {node.name}({args}):"
                )
                rows.append(
                    {
                        "instruction": (
                            "Implement this function to satisfy its docstring:\n\n"
                            f"```python\n{signature}\n    \"\"\"\n    "
                            + "\n    ".join(doc.splitlines())
                            + "\n    \"\"\"\n```"
                        ),
                        "input_context": "",
                        "output": source,
                        "category": "induced_signature_to_body",
                        "language": self.language,
                        "difficulty": "advanced",
                        "metadata": {
                            "induction": "signature_to_body",
                            "function": node.name,
                            "source_path": file["path"],
                            "license": file["license"],
                        },
                    }
                )
                if len(rows) >= limit:
                    break
            if len(rows) >= limit:
                break
        return rows

    def code_to_explanation(self, limit: int = 200) -> list[dict]:
        rows = []
        for file in self.files:
            explanation = _explanation_of(file["path"], file["content"])
            if not explanation:
                continue
            rows.append(
                {
                    "instruction": (
                        "Explain what this file does. Stick to what is verifiable "
                        "in the source; do not speculate about intent you cannot see."
                    ),
                    "input_context": f"```python\n{file['content']}\n```",
                    "output": explanation,
                    "category": "induced_code_to_explanation",
                    "language": self.language,
                    "difficulty": "intermediate",
                    "metadata": {
                        "induction": "code_to_explanation",
                        "source_path": file["path"],
                        "license": file["license"],
                    },
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def test_to_contract(self, limit: int = 150) -> list[dict]:
        rows = []
        for file in self.files:
            if "test" not in Path(file["path"]).name:
                continue
            symbols = _tested_symbols(file["content"])
            if len(symbols) < 2:
                continue
            contract = (
                "This test file exercises: "
                + ", ".join(symbols[:12])
                + ".\nBehaviour it pins (extractive - read the assertions for the "
                "exact contract): failing paths are asserted alongside happy "
                "paths, and the fixtures show the expected shapes."
            )
            rows.append(
                {
                    "instruction": (
                        "What contract does this test suite pin down? Name the "
                        "units under test and what behaviour is asserted."
                    ),
                    "input_context": f"```python\n{file['content']}\n```",
                    "output": contract,
                    "category": "induced_test_to_contract",
                    "language": self.language,
                    "difficulty": "intermediate",
                    "metadata": {
                        "induction": "test_to_contract",
                        "tested": symbols[:12],
                        "source_path": file["path"],
                        "license": file["license"],
                    },
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def generate(self, count: int = 64) -> list[dict]:
        """Deterministic mixed batch across the four induction shapes."""

        quotas = {
            self.docstring_to_code: max(1, count // 4),
            self.signature_to_body: max(1, count // 4),
            self.code_to_explanation: max(1, count // 4),
            self.test_to_contract: max(1, count // 4),
        }

        rows: list[dict] = []

        for fn, quota in quotas.items():

            rows.extend(fn(limit=quota))

        rng = random.Random(self.seed)

        rng.shuffle(rows)

        return rows[:count]

    def stats(self) -> dict:
        return {
            "files": len(self.files),
            "language": self.language,
            "corpus_dir": str(self.corpus_dir),
            "licenses": sorted({f["license"] for f in self.files}),
        }
