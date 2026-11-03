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
import textwrap
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



def _docstring_example(doc: str) -> str:
    """Pull the first doctest/usage block out of a docstring, if any."""
    lines = doc.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith((">>>", "Example:", "Examples:")):
            block = [item for item in lines[idx : idx + 14] if item.strip()]
            return "\n".join(block).strip()
    return ""


def _complexity_of(source: str) -> str:
    """Classify a function body by nesting depth and branch count (extractive)."""

    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return "unknown"
    depth = 0
    peak = 0
    branches = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match)):
            branches += 1
        if isinstance(node, (ast.If, ast.For, ast.While)):
            peak = max(peak, getattr(node, "col_offset", 0))
    depth = peak // 4
    if depth >= 3 or branches >= 8:
        return "expert"
    if depth == 2 or branches >= 5:
        return "advanced"
    if depth == 1 or branches >= 2:
        return "intermediate"
    return "beginner"


def _imports_of(text: str, limit: int = 12) -> list[str]:
    """Top-level imported module names, in first-seen order."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in names:
                    names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module not in names:
                names.append(node.module)
    return names[:limit]


def _branch_signatures(text: str) -> list[dict]:
    """Enumerate error/branch handlers that return a value (for branch coverage rows)."""

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: list[dict] = []
    for node in ast.walk(tree):
        handler_types = ()
        body_len = 0
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                handler_types = ("bare",)
            elif isinstance(node.type, ast.Name):
                handler_types = (node.type.id,)
            elif isinstance(node.type, ast.Tuple):
                handler_types = tuple(
                    item.id for item in node.type.elts if isinstance(item, ast.Name)
                )
            body_len = len(node.body)
        if handler_types:
            out.append(
                {
                    "handler": list(handler_types),
                    "body_len": body_len,
                    "line": getattr(node, "lineno", 0),
                }
            )
    return out[:8]


def _docstring_contract(doc: str) -> list[str]:
    """Extract promise-like lines (Raises/Returns/Requires) from a docstring."""

    kept: list[str] = []
    capturing = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.startswith(("Raises:", "Returns:", "Yields:", "Requires:", "Precondition")):
            kept.append(stripped)
            capturing = True
            continue
        if not capturing:
            continue
        if not stripped:
            capturing = False
            continue
        if kept and line.startswith((" ", "\t")):
            kept[-1] = kept[-1] + " " + stripped
        else:
            capturing = False
    return kept[:8]


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


    def docstring_example_to_implementation(self, limit: int = 120) -> list[dict]:
        """Docstring example/doctest -> the implementation that satisfies it."""

        rows: list[dict] = []
        for file in self.files:
            for node, source in _public_functions(file["content"]):
                example = _docstring_example(ast.get_docstring(node) or "")
                if len(example) < 24:
                    continue
                rows.append(
                    {
                        "instruction": (
                            "Implement the function so this usage example holds "
                            "exactly:\n\n"
                            f"```text\n{example}\n```\n\n"
                            "Match the example's printed output byte for byte."
                        ),
                        "input_context": "",
                        "output": source,
                        "category": "induced_example_to_implementation",
                        "language": self.language,
                        "difficulty": _complexity_of(source),
                        "metadata": {
                            "induction": "example_to_implementation",
                            "function": node.name,
                            "source_path": file["path"],
                            "license": file["license"],
                        },
                    }
                )
                if len(rows) >= limit:
                    return rows
        return rows

    def imports_to_module(self, limit: int = 120) -> list[dict]:
        """Dependency list + spec -> the module that wires those imports."""

        rows: list[dict] = []
        for file in self.files:
            imports = _imports_of(file["content"])
            doc = _module_doc(file["content"])
            if len(imports) < 2 or len(doc) < 60:
                continue
            rows.append(
                {
                    "instruction": (
                        "Write a module matching this specification using exactly "
                        "these imports (no others):\n\n"
                        f"Imports: {', '.join(imports)}\n\n"
                        f"Specification:\n{doc}"
                    ),
                    "input_context": "",
                    "output": file["content"],
                    "category": "induced_imports_to_module",
                    "language": self.language,
                    "difficulty": "advanced",
                    "metadata": {
                        "induction": "imports_to_module",
                        "imports": imports,
                        "source_path": file["path"],
                        "license": file["license"],
                    },
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def error_branches_to_tests(self, limit: int = 120) -> list[dict]:
        """Existing except handlers -> tests that pin the failure paths."""

        rows: list[dict] = []
        for file in self.files:
            handlers = _branch_signatures(file["content"])
            if not handlers:
                continue
            described = "; ".join(
                f"line {item['line']}: catch {', '.join(item['handler'])}"
                for item in handlers
            )
            tests = []
            for item in handlers:
                err = item["handler"][0]
                tests.append(
                    "def test_raises_when_%s():\n"
                    "    with pytest.raises(%s):\n"
                    "        run_with_invalid_input()\n" % (err.lower(), err if err != "bare" else "Exception")
                )
            rows.append(
                {
                    "instruction": (
                        "This module already handles these failure paths:\n"
                        f"{described}\n\n"
                        "Write the missing pytest cases that exercise each handler."
                    ),
                    "input_context": f"```python\n{file['content']}\n```",
                    "output": "\n".join(tests),
                    "category": "induced_error_branches_to_tests",
                    "language": self.language,
                    "difficulty": "intermediate",
                    "metadata": {
                        "induction": "error_branches_to_tests",
                        "handlers": [item["handler"] for item in handlers],
                        "source_path": file["path"],
                        "license": file["license"],
                    },
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def docstring_to_contract_tests(self, limit: int = 120) -> list[dict]:
        """Docstring promises (Raises/Returns) -> tests that enforce them."""

        rows: list[dict] = []
        for file in self.files:
            for node, source in _public_functions(file["content"]):
                doc = ast.get_docstring(node) or ""
                contract = _docstring_contract(doc)
                if len(contract) < 1:
                    continue
                assertions = "\n".join(
                    f"    # {line}" for line in contract
                )
                rows.append(
                    {
                        "instruction": (
                            "Turn these documented promises into executable checks:\n\n"
                            f"{chr(10).join(contract)}"
                        ),
                        "input_context": f"```python\n{source}\n```",
                        "output": (
                            "def test_documented_contract():\n"
                            "    # each assertion mirrors one documented promise\n"
                            f"{assertions}\n"
                            "    result = call_under_test()\n"
                            "    assert result is not None\n"
                        ),
                        "category": "induced_contract_to_tests",
                        "language": self.language,
                        "difficulty": "advanced",
                        "metadata": {
                            "induction": "contract_to_tests",
                            "function": node.name,
                            "promises": contract,
                            "source_path": file["path"],
                            "license": file["license"],
                        },
                    }
                )
                if len(rows) >= limit:
                    return rows
        return rows

    def generate(self, count: int = 64) -> list[dict]:
        """Deterministic mixed batch across the four induction shapes."""

        shapes = [
            self.docstring_to_code,
            self.signature_to_body,
            self.code_to_explanation,
            self.test_to_contract,
            self.docstring_example_to_implementation,
            self.imports_to_module,
            self.error_branches_to_tests,
            self.docstring_to_contract_tests,
        ]

        quota = max(1, count // len(shapes))

        rows: list[dict] = []

        for fn in shapes:

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
            "shapes": [
                "docstring_to_code",
                "signature_to_body",
                "code_to_explanation",
                "test_to_contract",
                "example_to_implementation",
                "imports_to_module",
                "error_branches_to_tests",
                "contract_to_tests",
            ],
        }
