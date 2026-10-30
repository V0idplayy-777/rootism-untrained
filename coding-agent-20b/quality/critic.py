"""Deterministic code review ("critic" step) before results are shown.

The critic is intentionally *not* a model: it is a rule engine over the AST and
the text, so it gives the same answer twice, runs in CI, and cannot hallucinate
a finding. It catches the failure modes that actually show up in generated
projects (swallowed exceptions, mutable default args, hard-coded secrets,
unbounded requests, unsafe YAML/eval/shell, missing tests, XSS sinks).

Use it as a gate::

    review = CodeCritic().review("./app")
    review.ok                     # score >= threshold and no critical findings
    print(review.render())        # grouped, with file:line and a fix per finding
    review.fix_prompt()           # ready to feed back to a model

``CriticLoop`` optionally applies a fixer and re-reviews, bounded by rounds.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

SEVERITY_WEIGHT = {"critical": 25, "high": 12, "medium": 5, "low": 2, "nit": 1}
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "nit": 4}
SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".agent",
    "dist",
    "build",
    "coverage",
    ".next",
}
CODE_EXT = {".py", ".js", ".mjs", ".jsx", ".ts", ".tsx"}
SECRET_NAME = re.compile(r"(secret|token|password|passwd|api[_-]?key|private[_-]?key|access[_-]?key)", re.I)
PLACEHOLDER = re.compile(r"(example|dummy|placeholder|changeme|your[-_]|xxx+|test|local|fake|<.*>)", re.I)
SQL_KEYWORD = re.compile(r"\b(SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM|DROP|ALTER)\b", re.I)


@dataclass
class Finding:
    rule: str
    severity: str
    path: str
    line: int
    message: str
    fix: str = ""
    snippet: str = ""
    category: str = "correctness"

    @property
    def location(self) -> str:
        return f"{self.path}:{self.line}" if self.line else self.path

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "path": self.path,
            "line": self.line,
            "message": self.message,
            "fix": self.fix,
            "snippet": self.snippet,
            "category": self.category,
        }

    def render(self) -> str:
        head = f"[{self.severity}] {self.location} {self.message}"
        return head + (f"\n    fix: {self.fix}" if self.fix else "")


@dataclass
class Review:
    findings: list[Finding] = field(default_factory=list)
    files_reviewed: int = 0
    score: int = 100
    threshold: int = 80
    skipped: list[str] = field(default_factory=list)

    @property
    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "critical"]

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity in {"critical", "high"}]

    @property
    def ok(self) -> bool:
        return self.score >= self.threshold and not self.critical

    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f.severity == severity)

    def top(self, n: int = 5) -> list[Finding]:
        return sorted(self.findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.path, f.line))[:n]

    def by_file(self) -> dict[str, list[Finding]]:
        grouped: dict[str, list[Finding]] = {}
        for finding in self.findings:
            grouped.setdefault(finding.path, []).append(finding)
        return grouped

    def summary(self) -> str:
        if not self.findings:
            return f"clean ({self.files_reviewed} files) - score {self.score}"
        parts = [f"{self.count(s)} {s}" for s in ("critical", "high", "medium", "low", "nit") if self.count(s)]
        return f"score {self.score}/100 - " + ", ".join(parts) + f" ({self.files_reviewed} files)"

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "ok": self.ok,
            "threshold": self.threshold,
            "files_reviewed": self.files_reviewed,
            "summary": self.summary(),
            "counts": {s: self.count(s) for s in SEVERITY_WEIGHT},
            "findings": [f.to_dict() for f in self.findings],
            "skipped": self.skipped,
        }

    def render(self, limit: int = 40) -> str:
        lines = [f"Review: {self.summary()}"]
        for finding in sorted(self.findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.path, f.line))[:limit]:
            lines.append(finding.render())
        if len(self.findings) > limit:
            lines.append(f"... {len(self.findings) - limit} more")
        return "\n".join(lines)

    def fix_prompt(self, limit: int = 8) -> str:
        """Prompt asking for a minimal patch for the worst findings."""

        rows = []
        for finding in self.top(limit):
            rows.append(
                f"- {finding.location} [{finding.severity}/{finding.rule}] {finding.message}\n"
                f"  suggested fix: {finding.fix}"
            )
        return (
            "Review the following findings from a static analysis of this project and "
            "produce one unified diff that fixes them without changing behaviour or "
            "renaming public interfaces. Keep the diff minimal, then re-run the tests.\n\n"
            + "\n".join(rows)
        )


class CodeCritic:
    def __init__(self, threshold: int = 80, max_findings_per_rule: int = 25):
        self.threshold = threshold
        self.max_findings_per_rule = max_findings_per_rule

    def review(
        self,
        root: str | Path,
        paths: Iterable[str] | None = None,
        include_tests: bool = False,
    ) -> Review:
        root = Path(root)
        files = self._files(root, paths, include_tests)
        findings: list[Finding] = []
        for rel in files:
            path = root / rel
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            findings.extend(self._review_python(rel, text) if path.suffix == ".py" else self._review_js(rel, text))
        findings.extend(self._project_rules(root, files))
        findings = self._dedupe(findings)
        score = max(0, 100 - sum(SEVERITY_WEIGHT[f.severity] for f in findings))
        return Review(findings=findings, files_reviewed=len(files), score=score, threshold=self.threshold)

    def _files(self, root: Path, paths: Iterable[str] | None, include_tests: bool) -> list[str]:
        if paths is not None:
            candidates = []
            for item in paths:
                path = root / item
                if path.is_dir():
                    candidates.extend(p for p in path.rglob("*") if p.is_file())
                elif path.is_file():
                    candidates.append(path)
        else:
            candidates = [p for p in root.rglob("*") if p.is_file()]
        out: list[str] = []
        for path in candidates:
            rel = path.relative_to(root)
            if any(part in SKIP_DIRS for part in rel.parts[:-1]):
                continue
            if path.suffix not in CODE_EXT:
                continue
            if not include_tests and _is_test_file(rel):
                continue
            out.append(str(rel))
        return sorted(out)

    def _review_python(self, rel: str, text: str) -> list[Finding]:
        findings: list[Finding] = []
        lines = text.splitlines()
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            return [
                Finding(
                    rule="syntax-error",
                    severity="critical",
                    path=rel,
                    line=int(exc.lineno or 0),
                    message=f"file does not parse: {exc.msg}",
                    fix="fix the syntax error before anything else",
                    category="correctness",
                )
            ]

        for node in ast.walk(tree):
            line = getattr(node, "lineno", 0)

            if isinstance(node, ast.ExceptHandler) and node.type is None:
                findings.append(self._f("bare-except", "high", rel, line, "bare `except:` catches everything, including KeyboardInterrupt", "catch the specific exception type(s) you can handle", "correctness"))
            if isinstance(node, ast.ExceptHandler) and len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                findings.append(self._f("except-pass", "high", rel, line, "exception is silently swallowed (`except ...: pass`)", "log it, re-raise it, or handle it explicitly", "correctness"))
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name in {"eval", "exec"}:
                    findings.append(self._f("eval-exec", "critical", rel, line, f"`{name}()` executes arbitrary code", "use a parser (ast.literal_eval, json.loads) or an explicit dispatch table", "security"))
                if name and name.endswith(("system", "popen")):
                    findings.append(self._f("os-system", "high", rel, line, f"`{name}()` runs a shell command", "use subprocess.run([...], shell=False) with a fixed argument list", "security"))
                if name and (name.endswith("Popen") or name == "subprocess.run"):
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            findings.append(self._f("shell-true", "high", rel, line, "`shell=True` allows command injection", "pass a list of arguments and keep shell=False", "security"))
                if name in {"requests.get", "requests.post", "requests.put", "requests.delete", "requests.request", "requests.head", "httpx.get", "httpx.post"}:
                    if not any(kw.arg == "timeout" for kw in node.keywords):
                        findings.append(self._f("http-no-timeout", "medium", rel, line, "HTTP call without a timeout can hang forever", "pass timeout=<seconds>", "reliability"))
                if name in {"hashlib.md5", "hashlib.sha1", "md5", "sha1"}:
                    findings.append(self._f("weak-hash", "high", rel, line, "MD5/SHA-1 are broken for passwords and signatures", "use hashlib.sha256 or a password KDF (scrypt/argon2)", "security"))
                if name in {"yaml.load"}:
                    if not any(kw.arg == "Loader" for kw in node.keywords):
                        findings.append(self._f("yaml-load", "critical", rel, line, "`yaml.load` without SafeLoader can execute code", "use yaml.safe_load()", "security"))
                if name in {"pickle.load", "pickle.loads", "cPickle.loads"}:
                    findings.append(self._f("pickle", "high", rel, line, "unpickling untrusted data is remote code execution", "use JSON or another data-only format", "security"))
                if name in {"random.randint", "random.choice", "random.random", "random.randrange", "random.sample"}:
                    snippet = lines[line - 1] if 0 <= line - 1 < len(lines) else ""
                    if SECRET_NAME.search(snippet):
                        findings.append(self._f("insecure-random", "high", rel, line, "`random` is not cryptographically secure", "use `secrets` for tokens, keys and passwords", "security"))
                if name and kwargs_contains(node, "verify", False):
                    findings.append(self._f("tls-disabled", "critical", rel, line, "TLS verification is disabled (verify=False)", "remove verify=False and fix the certificate/proxy trust instead", "security"))
                if name and name.startswith("subprocess") and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.JoinedStr) or (
                        isinstance(first, ast.BinOp) and isinstance(first.op, (ast.Add, ast.Mod))
                    ):
                        findings.append(self._f("subprocess-interpolation", "high", rel, line, "shell command built from interpolated strings", "pass a list and avoid shell evaluation", "security"))

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        findings.append(self._f("mutable-default", "high", rel, line, f"mutable default argument in `{node.name}` is shared across calls", "default to None and create the value inside the function", "correctness"))
                arg_count = len(node.args.args) + len(node.args.kwonlyargs)
                if arg_count > 6:
                    findings.append(self._f("too-many-args", "low", rel, line, f"`{node.name}` takes {arg_count} arguments", "group related arguments into a dataclass/dict", "maintainability"))
                end = getattr(node, "end_lineno", line) or line
                if end - line > 80:
                    findings.append(self._f("long-function", "medium", rel, line, f"`{node.name}` is {end - line} lines long", "extract cohesive steps into named helpers", "maintainability"))
                if (
                    not node.name.startswith("_")
                    and not ast.get_docstring(node)
                    and len([n for n in node.body if not isinstance(n, ast.Pass)]) > 3
                ):
                    findings.append(self._f("missing-docstring", "nit", rel, line, f"public function `{node.name}` has no docstring", "state the contract: inputs, output, failures", "docs"))

            if isinstance(node, ast.Compare):
                for op, comparator in zip(node.ops, node.comparators):
                    if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(comparator, ast.Constant) and comparator.value is None:
                        findings.append(self._f("none-comparison", "nit", rel, line, "comparison to None with ==/!=", "use `is None` / `is not None`", "style"))

            if isinstance(node, ast.Assert):
                findings.append(self._f("assert-validation", "low", rel, line, "`assert` is stripped under `python -O`; do not use it for validation", "raise an explicit exception instead", "correctness"))

            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and SECRET_NAME.search(target.id) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        value = node.value.value
                        if len(value) >= 8 and not PLACEHOLDER.search(value):
                            findings.append(self._f("hardcoded-secret", "critical", rel, line, f"`{target.id}` looks like a hard-coded credential", "read it from the environment and never commit it", "security"))
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and isinstance(node.value, ast.Constant):
                if SECRET_NAME.search(node.target.id) and isinstance(node.value.value, str) and len(node.value.value) >= 8 and not PLACEHOLDER.search(node.value.value):
                    findings.append(self._f("hardcoded-secret", "critical", rel, line, f"`{node.target.id}` looks like a hard-coded credential", "read it from the environment and never commit it", "security"))

            if isinstance(node, ast.Call) and _call_name(node.func) == "execute" and node.args:
                first = node.args[0]
                if isinstance(first, (ast.JoinedStr, ast.BinOp)) and SQL_KEYWORD.search(ast.dump(first)):
                    findings.append(self._f("sql-injection", "critical", rel, line, "SQL built by string formatting/concatenation", "pass parameters: execute('... WHERE id = ?', (value,))", "security"))

        findings.extend(self._regex_python(rel, text))
        findings.extend(self._unused_imports(rel, text, tree))
        return findings

    def _regex_python(self, rel: str, text: str) -> list[Finding]:
        findings: list[Finding] = []
        for index, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if re.match(r"#\s*(TODO|FIXME|XXX)", stripped, re.I):
                findings.append(self._f("todo-comment", "low", rel, index, "unfinished work marker left in the code", "finish it or file an issue and remove the marker", "maintainability"))
            if re.match(r"print\(", stripped) and "__main__" not in text:
                findings.append(self._f("debug-print", "low", rel, index, "`print()` in library code", "use the `logging` module", "maintainability"))
            if re.search(r"app\.run\([^)]*debug\s*=\s*True", line):
                findings.append(self._f("flask-debug", "high", rel, index, "Flask debug mode enabled (remote code execution console)", "keep debug=False outside local development", "security"))
            if re.search(r"allow_origins\s*=\s*\[\s*[\"']\*[\"']\s*\]", line) and "allow_credentials=True" in text:
                findings.append(self._f("cors-wildcard", "high", rel, index, "wildcard CORS together with credentials", "list explicit origins or drop allow_credentials", "security"))
            if re.search(r"except\s+Exception\s*:\s*$", line) and index < len(text.splitlines()):
                nxt = text.splitlines()[index].strip()
                if nxt == "pass":
                    findings.append(self._f("except-pass", "high", rel, index, "`except Exception: pass` hides every failure", "log and re-raise, or catch a specific error", "correctness"))
            if re.search(r"open\([^)]*\)\s*$", line) and "with " not in line and "Path(" not in line and "=" in line:
                findings.append(self._f("open-no-context", "medium", rel, index, "file opened without a `with` block", "use `with open(...) as f:` so it always closes", "reliability"))
        return findings

    def _unused_imports(self, rel: str, text: str, tree: ast.AST) -> list[Finding]:
        if Path(rel).name == "__init__.py":
            return []
        imported: dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        imported[alias.asname] = node.lineno
                    elif "." not in alias.name:
                        imported[alias.name] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    imported[alias.asname or alias.name] = node.lineno
        if not imported:
            return []
        used = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        } | {
            node.value.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        }
        findings = []
        for name, line in imported.items():
            if name in used or name == "annotations":
                continue
            if len(re.findall(rf"\b{re.escape(name)}\b", text)) > 1:
                continue
            findings.append(
                self._f("unused-import", "low", rel, line, f"`{name}` is imported but never used", "remove the import", "maintainability")
            )
        return findings

    def _review_js(self, rel: str, text: str) -> list[Finding]:
        findings: list[Finding] = []
        for index, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if re.match(r"//\s*(TODO|FIXME|XXX)", stripped, re.I):
                findings.append(self._f("todo-comment", "low", rel, index, "unfinished work marker left in the code", "finish it or file an issue", "maintainability"))
            if re.search(r"\bconsole\.(log|debug)\(", line):
                findings.append(self._f("console-log", "low", rel, index, "`console.log` left in shipped code", "remove it or use a logger with levels", "maintainability"))
            if re.search(r"\beval\s*\(", line):
                findings.append(self._f("eval", "critical", rel, index, "`eval()` executes arbitrary code", "parse data with JSON.parse or an explicit parser", "security"))
            if re.search(r"dangerouslySetInnerHTML", line):
                findings.append(self._f("dangerous-html", "high", rel, index, "raw HTML injection point (XSS)", "sanitise with DOMPurify or render text nodes", "security"))
            elif re.search(r"\.innerHTML\s*=", line) and "${" in line:
                findings.append(self._f("xss-innerhtml", "high", rel, index, "template string written to innerHTML (XSS)", "use textContent or sanitise the value", "security"))
            if re.search(r"child_process.*exec\(|\bexecSync\(", line):
                findings.append(self._f("shell-exec", "high", rel, index, "shell command execution", "use execFile/spawn with an argument array", "security"))
            if re.search(r"process\.env\.[A-Z_]*SECRET[^;]*\|\|\s*[\"'][^\"']{8,}", line, re.I):
                findings.append(self._f("secret-fallback", "critical", rel, index, "hard-coded secret used as a fallback", "fail fast when the env var is missing", "security"))
            if re.search(r"(?<![=!<>])==(?!=)", line) and not re.search(r"typeof|== null", line):
                findings.append(self._f("loose-equality", "nit", rel, index, "loose equality (==) does type coercion", "use === / !==", "style"))
            if re.search(r"\bcatch\s*(\([^)]*\))?\s*\{\s*\}", line):
                findings.append(self._f("empty-catch", "high", rel, index, "empty catch block swallows the error", "handle it, log it, or rethrow", "correctness"))
            if re.search(r"verify\s*:\s*false|rejectUnauthorized\s*:\s*false", line):
                findings.append(self._f("tls-disabled", "critical", rel, index, "TLS verification disabled", "remove the flag and fix certificate trust", "security"))
            if re.search(r":\s*any\b", line) and rel.endswith((".ts", ".tsx")):
                findings.append(self._f("any-type", "nit", rel, index, "explicit `any` disables type checking here", "use a precise type or `unknown` + narrowing", "maintainability"))
        return findings

    def _project_rules(self, root: Path, files: list[str]) -> list[Finding]:
        findings: list[Finding] = []
        names = set(files)
        source = [f for f in files if not _is_test_file(Path(f))]
        has_tests = any(_is_test_file(Path(f)) for f in files) or (root / "tests").is_dir()
        has_public_defs = False
        for rel in source:
            if Path(rel).suffix != ".py":
                continue
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            if any(
                isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_")
                for node in tree.body
            ):
                has_public_defs = True
                break
        if has_public_defs and not has_tests:
            findings.append(
                self._f(
                    "no-tests",
                    "high",
                    "(project)",
                    0,
                    "source modules exist but there are no tests",
                    "generate starter tests (agent testgen) and add assertions for the main paths",
                    "testing",
                )
            )
        readme = root / "README.md"
        if source and not readme.exists():
            findings.append(self._f("no-readme", "medium", "(project)", 0, "no README", "add purpose, setup, run and test commands", "docs"))

        gitignore = root / ".gitignore"
        gitignore_text = gitignore.read_text(encoding="utf-8", errors="replace") if gitignore.exists() else ""
        if ".env" not in gitignore_text:
            findings.append(
                self._f(
                    "env-not-ignored",
                    "critical",
                    ".gitignore" if gitignore.exists() else "(project)",
                    0,
                    ".env is not git-ignored, so secrets can be committed",
                    "add `.env` (and `.env.*`, keeping `!.env.example`) to .gitignore",
                    "security",
                )
            )

        requirements = [f for f in files if Path(f).name.startswith("requirements") and f.count("/") == 0]
        for rel in requirements:
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
            unpinned = [
                line
                for line in text.splitlines()
                if line.strip() and not line.strip().startswith(("#", "-")) and "==" not in line and not line.strip().startswith(("git+", "http"))
            ]
            if unpinned:
                findings.append(
                    self._f(
                        "unpinned-deps",
                        "low",
                        rel,
                        0,
                        f"{len(unpinned)} dependency line(s) are not pinned",
                        "pin versions (==) so builds are reproducible",
                        "reliability",
                    )
                )
        if "Dockerfile" not in names and not (root / "Dockerfile").exists() and source:
            findings.append(
                self._f("no-dockerfile", "nit", "(project)", 0, "no Dockerfile", "run `agent deploy` to generate one", "packaging")
            )
        return findings

    def _f(self, rule, severity, path, line, message, fix, category) -> Finding:
        return Finding(rule=rule, severity=severity, path=path, line=line, message=message, fix=fix, category=category)

    def _dedupe(self, findings: list[Finding]) -> list[Finding]:
        seen: set[tuple[str, str, int]] = set()
        out: list[Finding] = []
        per_rule: dict[str, int] = {}
        for finding in findings:
            key = (finding.rule, finding.path, finding.line)
            if key in seen:
                continue
            per_rule[finding.rule] = per_rule.get(finding.rule, 0) + 1
            if per_rule[finding.rule] > self.max_findings_per_rule:
                continue
            seen.add(key)
            out.append(finding)
        return out


class CriticLoop:
    """Review, ask a fixer to address findings, re-review - bounded.

    ``fixer`` receives ``(prompt, review)`` and returns True when it changed
    something. Without a fixer the loop is a single review, which is still the
    useful default (and what CI runs).
    """

    def __init__(self, critic: CodeCritic | None = None, fixer: Callable[[str, Review], bool] | None = None, max_rounds: int = 2):
        self.critic = critic or CodeCritic()
        self.fixer = fixer
        self.max_rounds = max_rounds

    def run(self, root: str | Path, paths: Iterable[str] | None = None) -> dict:
        history = []
        review = self.critic.review(root, paths)
        history.append(review)
        rounds = 0
        while self.fixer and not review.ok and rounds < self.max_rounds:
            rounds += 1
            changed = False
            try:
                changed = bool(self.fixer(review.fix_prompt(), review))
            except Exception:  # noqa: BLE001 - a broken fixer must not hide the review
                changed = False
            if not changed:
                break
            review = self.critic.review(root, paths)
            history.append(review)
        return {
            "ok": review.ok,
            "rounds": rounds,
            "review": review,
            "history": [r.to_dict() for r in history],
        }


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _call_name(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    return ""


def kwargs_contains(node: ast.Call, name: str, value) -> bool:
    for kw in node.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant) and kw.value.value == value:
            return True
    return False


def _is_test_file(rel: Path) -> bool:
    name = rel.name
    return (
        name.startswith("test_")
        or name.endswith(("_test.py", ".test.js", ".test.ts", ".test.mjs", ".spec.js", ".spec.ts"))
        or "tests" in rel.parts
    )
