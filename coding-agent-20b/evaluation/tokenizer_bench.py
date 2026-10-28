"""Tokenizer competitiveness benchmark.

A tokenizer is only "code-tuned" if it beats real baselines on real code, per
language, on the formats an agent actually emits and consumes. This module
provides:

- a deterministic, license-clean sample corpus (no external download needed for
  the gate; `--corpus-dir` swaps in held-out real files),
- baselines: raw bytes (the floor), whitespace/punctuation splitting, and
  ``tiktoken`` encodings when the package is installed,
- per-language fertility, identifier cost, diff cost and an effective-context
  estimate (how many bytes of code fit in an 8k/32k window),
- a gate: ``assert_competitive`` fails unless the trained tokenizer wins on the
  corpora that matter (code + diffs), not just on averaged prose.

Usage::

    from evaluation.tokenizer_bench import TokenizerBenchmark
    report = TokenizerBenchmark().run(my_tokenizer)
    report["gate"]["passed"]

CLI::

    python scripts/evaluate_tokenizer.py --tokenizer tokenizer/ [--json out.json]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Sample corpus: short, recognisable, per-format. Kept small so the gate runs
# in CI in well under a second, and extended at runtime from real repositories
# when `extra_documents`/`corpus_dir` is supplied.
# ---------------------------------------------------------------------------

SAMPLE_CORPUS: dict[str, list[str]] = {
    "python": [
        "from dataclasses import dataclass\n\n\n@dataclass\nclass RetryPolicy:\n    attempts: int = 3\n    backoff: float = 0.5\n\n    def delay_for(self, attempt: int) -> float:\n        return self.backoff * (2 ** attempt)\n",
        "def batch(items: list[dict], size: int) -> list[list[dict]]:\n    return [items[i : i + size] for i in range(0, len(items), size)]\n",
        "async def fetch_all(client, urls):\n    import asyncio\n\n    return await asyncio.gather(*(client.get(u) for u in urls))\n",
        "class LRUCache:\n    def __init__(self, capacity: int = 128):\n        self.capacity = capacity\n        self._store: dict[str, object] = {}\n",
    ],
    "typescript": [
        "export interface UserRecord {\n  id: string;\n  email: string;\n  createdAt: Date;\n}\n\nexport async function getUser(id: string): Promise<UserRecord | null> {\n  const res = await fetch(`/api/users/${id}`);\n  if (!res.ok) return null;\n  return (await res.json()) as UserRecord;\n}\n",
        "type Handler<T> = (event: T) => Promise<void>;\n\nexport const debounce = <A extends unknown[]>(fn: (...args: A) => void, ms = 100) => {\n  let timer: ReturnType<typeof setTimeout> | undefined;\n  return (...args: A) => {\n    if (timer) clearTimeout(timer);\n    timer = setTimeout(() => fn(...args), ms);\n  };\n};\n",
    ],
    "rust": [
        "use std::collections::HashMap;\n\npub struct Registry<K, V> {\n    entries: HashMap<K, V>,\n}\n\nimpl<K: std::hash::Hash + Eq, V> Registry<K, V> {\n    pub fn insert(&mut self, key: K, value: V) -> Option<V> {\n        self.entries.insert(key, value)\n    }\n}\n",
        "fn parse_port(input: &str) -> Result<u16, std::num::ParseIntError> {\n    input.trim().parse::<u16>()\n}\n",
    ],
    "go": [
        "package server\n\nimport (\n\t\"net/http\"\n\t\"time\"\n)\n\nfunc NewServer(addr string) *http.Server {\n\treturn &http.Server{\n\t\tAddr:         addr,\n\t\tReadTimeout:  5 * time.Second,\n\t\tWriteTimeout: 10 * time.Second,\n\t}\n}\n",
        "func Filter[T any](items []T, keep func(T) bool) []T {\n\tout := make([]T, 0, len(items))\n\tfor _, item := range items {\n\t\tif keep(item) {\n\t\t\tout = append(out, item)\n\t\t}\n\t}\n\treturn out\n}\n",
    ],
    "sql": [
        "SELECT u.id, u.email, COUNT(o.id) AS order_count\nFROM users AS u\nLEFT JOIN orders AS o ON o.user_id = u.id\nWHERE u.created_at >= NOW() - INTERVAL '30 days'\nGROUP BY u.id, u.email\nHAVING COUNT(o.id) > 3\nORDER BY order_count DESC\nLIMIT 50;\n",
    ],
    "yaml": [
        "version: '3.9'\nservices:\n  api:\n    image: ghcr.io/example/api:1.4.2\n    environment:\n      - DATABASE_URL=${DATABASE_URL}\n    ports:\n      - '8080:8080'\n    healthcheck:\n      test: ['CMD', 'curl', '-f', 'http://localhost:8080/healthz']\n      interval: 30s\n",
    ],
    "diff": [
        "--- a/src/auth/session.py\n+++ b/src/auth/session.py\n@@ -18,6 +18,10 @@ def rotate_token(session, now):\n-    if session.expires_at < now:\n-        return None\n+    if session.expires_at <= now:\n+        session.revoked_reason = 'expired'\n+        return None\n+    return session.token\n",
        "--- a/pkg/handler.go\n+++ b/pkg/handler.go\n@@ -44,3 +44,3 @@ func Handle(w http.ResponseWriter, r *http.Request) {\n-\tw.WriteHeader(200)\n+\tw.WriteHeader(http.StatusOK)\n",
    ],
    "identifiers": [
        "getUserAccountByEmailAndTenantId",
        "parseContentSecurityPolicyDirectives",
        "XMLHttpRequestUploadProgressHandler",
        "deserialize_avro_schema_registry_document",
        "http2_settings_max_concurrent_streams",
        "AWS4HmacSha256SigningKeyDerivation",
    ],
    "prose": [
        "The deployment failed because the migration job exceeded its timeout while waiting for the primary database to accept connections.",
        "Refactor the retry logic so that transient network errors are retried with exponential backoff, but permanent validation errors are not.",
    ],
}

# Corpora that decide the gate. Prose is reported but never required: a code
# tokenizer that trades prose compression for code/patch compression is a win.
GATE_CORPORA = ("python", "typescript", "rust", "go", "diff", "identifiers")

# Absolute fertility floors (chars/token). These are a *floor*, not a target:
# they sit above byte fallback (~1.0) and above what a tokenizer trained on a
# narrow corpus reaches (~1.6-1.9), so they prove the vocabulary is trained
# without pretending to measure quality. The quality bar is beating a real
# baseline (bytes, or tiktoken when installed) on every gated corpus; a full
# production run should be calibrated on held-out real code.
FERTILITY_FLOORS = {
    "python": 1.7,
    "typescript": 1.6,
    "rust": 1.45,
    "go": 1.5,
    "diff": 1.35,
    "identifiers": 1.5,
}

# Baselines that represent a real competing vocabulary. Word/punctuation
# splitting is reported for context but is not a fixed-vocabulary tokenizer, so
# it is not used to decide the gate.
REAL_BASELINE_PREFIXES = ("tiktoken:", "hf:", "bytes")

_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\s+|.", re.DOTALL
)


class Encoder(Protocol):
    name: str
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...


@dataclass
class SimpleEncoder:
    """Small encoder adapter (any object with encode/vocab_size)."""

    name: str
    vocab_size: int
    fn: Any

    def encode(self, text: str) -> list[int]:
        return self.fn(text)


class ByteEncoder:
    """Identity byte encoding: the unavoidable floor for any tokenizer."""

    name = "bytes"
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class WordPunctEncoder:
    """Whitespace + punctuation splitting, a reasonable cheap baseline."""

    name = "word-punct"
    vocab_size = 0

    def encode(self, text: str) -> list[int]:
        return [i for i, _ in enumerate(_TOKEN_RE.findall(text))]


class TiktokenEncoder:
    """Wraps an OpenAI/tiktoken encoding when available (real baseline)."""

    def __init__(self, encoding_name: str = "cl100k_base"):

        import tiktoken

        self.encoding = tiktoken.get_encoding(encoding_name)

        self.name = f"tiktoken:{encoding_name}"

        self.vocab_size = self.encoding.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, disallowed_special=())


def available_baselines(include_tiktoken: bool = True) -> list[Encoder]:
    """Baselines that are installed/available in this environment."""

    baselines: list[Encoder] = [ByteEncoder(), WordPunctEncoder()]

    if include_tiktoken:

        try:

            baselines.append(TiktokenEncoder("cl100k_base"))

        except Exception:

            pass

    return baselines


@dataclass
class CorpusReport:
    corpus: str
    chars: int
    tokens: int
    chars_per_token: float
    context_bytes_at: dict[int, int] = field(default_factory=dict)


def _summarise(name: str, encoder: Encoder, docs: list[str], windows=(8192, 32768)) -> CorpusReport:

    chars = tokens = 0

    for doc in docs:

        chars += len(doc)

        tokens += max(len(encoder.encode(doc)), 1)

    cpt = chars / max(tokens, 1)

    return CorpusReport(
        corpus=name,
        chars=chars,
        tokens=tokens,
        chars_per_token=cpt,
        context_bytes_at={w: int(w * cpt) for w in windows},
    )


class TokenizerBenchmark:
    """Score a tokenizer per corpus and against baselines."""

    def __init__(
        self,
        corpus: dict[str, list[str]] | None = None,
        extra_documents: dict[str, list[str]] | None = None,
        seed: int = 42,
    ):

        self.corpus = {k: list(v) for k, v in (corpus or SAMPLE_CORPUS).items()}

        if extra_documents:

            for name, docs in extra_documents.items():

                self.corpus.setdefault(name, []).extend(docs)

        self.seed = seed

    @classmethod
    def from_directory(
        cls, corpus_dir: str, max_files_per_language: int = 40, max_bytes: int = 200_000
    ) -> "TokenizerBenchmark":
        """Build a held-out corpus from real files (extension -> language)."""

        from pathlib import Path

        ext_lang = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".rs": "rust",
            ".go": "go",
            ".sql": "sql",
            ".yaml": "yaml",
            ".yml": "yaml",
        }

        docs: dict[str, list[str]] = {}

        root = Path(corpus_dir)

        for path in sorted(root.rglob("*")):

            lang = ext_lang.get(path.suffix)

            if lang is None or not path.is_file():

                continue

            if len(docs.setdefault(lang, [])) >= max_files_per_language:

                continue

            try:

                text = path.read_text(encoding="utf-8", errors="ignore")[:max_bytes]

            except OSError:

                continue

            if text.strip():

                docs[lang].append(text)

        corpus = dict(SAMPLE_CORPUS)

        corpus.update({k: v for k, v in docs.items() if v})

        return cls(corpus=corpus)

    def run(self, tokenizer, baselines: list[Encoder] | None = None) -> dict[str, Any]:
        """Return a JSON-serialisable competitiveness report."""

        encoder = SimpleEncoder(
            name=getattr(tokenizer, "name", "candidate"),
            vocab_size=len(tokenizer),
            fn=lambda text: tokenizer.encode(text, add_special_tokens=False),
        )

        base = baselines if baselines is not None else available_baselines()

        candidate = {
            name: _summarise(name, encoder, docs) for name, docs in self.corpus.items()
        }

        baseline_reports = {
            b.name: {name: _summarise(name, b, docs) for name, docs in self.corpus.items()}
            for b in base
        }

        real = {
            name: reports
            for name, reports in baseline_reports.items()
            if name.startswith(REAL_BASELINE_PREFIXES)
        } or baseline_reports

        reference = {
            name: reports
            for name, reports in baseline_reports.items()
            if not name.startswith(REAL_BASELINE_PREFIXES)
        }

        # Compare against the strongest *real* installed baseline per corpus.
        deltas: dict[str, dict[str, float]] = {}

        wins = 0

        gate_total = 0

        failures: list[str] = []

        for corpus in self.corpus:

            cand = candidate[corpus].chars_per_token

            strongest = max(
                (r[corpus].chars_per_token, name) for name, r in real.items()
            )

            deltas[corpus] = {
                "baseline": strongest[1],
                "candidate_cpt": cand,
                "baseline_cpt": strongest[0],
                "relative": cand / max(strongest[0], 1e-9),
            }

            if corpus in GATE_CORPORA:

                gate_total += 1

                if cand > strongest[0]:

                    wins += 1

                else:

                    failures.append(
                        f"{corpus}: {cand:.2f} <= {strongest[1]} {strongest[0]:.2f}"
                    )

                floor = FERTILITY_FLOORS.get(corpus)

                if floor is not None and cand < floor:

                    failures.append(
                        f"{corpus}: chars/token {cand:.2f} < floor {floor} "
                        "(vocabulary is not actually trained)"
                    )

        gate = {
            "corpora": gate_total,
            "wins": wins,
            "win_rate": wins / max(gate_total, 1),
            "failures": failures,
            "passed": wins == gate_total and gate_total > 0 and not failures,
        }

        return {
            "tokenizer": getattr(tokenizer, "name", "candidate"),
            "vocab_size": len(tokenizer),
            "candidate": {k: v.__dict__ for k, v in candidate.items()},
            "baselines": {
                name: {k: v.__dict__ for k, v in reports.items()}
                for name, reports in baseline_reports.items()
            },
            "gate_baselines": sorted(real),
            "reference_baselines": sorted(reference),
            "delta": deltas,
            "gate": gate,
        }

    @staticmethod
    def assert_competitive(
        report: dict[str, Any],
        min_relative: float = 1.0,
        required_corpora: tuple[str, ...] = GATE_CORPORA,
    ) -> None:
        """Fail unless the candidate beats every baseline on required corpora."""

        failures = []

        for corpus in required_corpora:

            delta = report["delta"].get(corpus)

            if delta is None:

                failures.append(f"{corpus}: missing")

            elif delta["relative"] < min_relative:

                failures.append(
                    f"{corpus}: {delta['candidate_cpt']:.2f} vs "
                    f"{delta['baseline']} {delta['baseline_cpt']:.2f} "
                    f"(ratio {delta['relative']:.2f} < {min_relative})"
                )

        assert not failures, "tokenizer is not competitive: " + "; ".join(failures)
        assert report["vocab_size"] > 0, "empty vocabulary"
        assert report["gate"]["passed"], (
            "tokenizer failed its own gate: "
            + "; ".join(report["gate"].get("failures", []))
        )


def save_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w") as f:

        json.dump(report, f, indent=2)


if __name__ == "__main__":  # pragma: no cover - manual smoke
    from model.tokenizer import CodeTokenizer

    bench = TokenizerBenchmark()

    print(json.dumps(bench.run(CodeTokenizer())["gate"], indent=2))
