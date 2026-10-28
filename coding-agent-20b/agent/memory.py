"""Cross-session memory for a repository.

Session state (``agent/session.py``) lives and dies with one task. But a
coding agent that works on the same repo every day keeps re-learning the same
things: which test command actually works, which linter is enforced, that
``tests/test_auth.py`` needs a running redis, that the maintainers reject
``pytest.mark.skip``.

``MemoryStore`` is a small, auditable, JSON-file store keyed by repo:

- ``remember(key, value, kind)`` — facts, lessons (what failed and why),
  patterns (conventions observed in the code).
- ``recall(query)`` — keyword-overlap scoring; deliberately simple, no
  embeddings, no network.
- ``render()`` — the compact block injected into a *new* session's prompt.
- Atomic writes, size-capped, dedup by key: a memory that grows unboundedly
  is a memory that eventually gets ignored.

Privacy/robustness contract: memory is per-repo-path, stored inside the
workspace (``.agent/memory.json``) so it is visible in diffs and deletable
with the repo; nothing is sent anywhere.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

MAX_MEMORIES = 200

KINDS = ("fact", "lesson", "pattern", "command")


@dataclass
class MemoryEntry:
    key: str
    value: str
    kind: str = "fact"
    uses: int = 0
    updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "kind": self.kind,
            "uses": self.uses,
            "updated": self.updated,
        }


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)}


class MemoryStore:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.entries: dict[str, MemoryEntry] = {}
        self.load()

    # -- persistence ------------------------------------------------------

    def load(self) -> "MemoryStore":
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for row in data.get("memories", []):
                    entry = MemoryEntry(
                        key=row["key"],
                        value=row.get("value", ""),
                        kind=row.get("kind", "fact"),
                        uses=int(row.get("uses", 0)),
                        updated=float(row.get("updated", 0.0)),
                    )
                    self.entries[entry.key] = entry
            except (json.JSONDecodeError, OSError, KeyError, ValueError):
                # corrupt memory must never brick the agent
                self.entries = {}
        return self

    def save(self) -> "MemoryStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"memories": [e.to_dict() for e in self.entries.values()]},
                f,
                indent=2,
                ensure_ascii=False,
            )
        os.replace(tmp, self.path)
        return self

    # -- API ---------------------------------------------------------------

    def remember(self, key: str, value: str, kind: str = "fact") -> MemoryEntry:
        key = key.strip()
        value = str(value).strip()
        if not key or not value:
            raise ValueError("memory key and value must be non-empty")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        existing = self.entries.get(key)
        if existing is not None:
            existing.value = value
            existing.kind = kind
            existing.updated = time.time()
            return existing
        entry = MemoryEntry(key=key, value=value, kind=kind)
        self.entries[key] = entry
        if len(self.entries) > MAX_MEMORIES:
            # evict least-useful: fewest uses, then oldest
            worst = min(
                self.entries.values(), key=lambda e: (e.uses, e.updated)
            )
            self.entries.pop(worst.key, None)
        return entry

    def forget(self, key: str) -> bool:
        return self.entries.pop(key, None) is not None

    def get(self, key: str) -> MemoryEntry | None:
        return self.entries.get(key)

    def recall(self, query: str, k: int = 8) -> list[MemoryEntry]:
        q = _tokens(query)
        if not q:
            return []
        scored = []
        for entry in self.entries.values():
            overlap = len(q & _tokens(entry.key) | (q & _tokens(entry.value)))
            if overlap:
                scored.append((overlap + min(entry.uses, 5), entry))
        scored.sort(key=lambda kv: (-kv[0], kv[1].key))
        out = [e for _s, e in scored[:k]]
        for entry in out:
            entry.uses += 1
        return out

    def render(self, k: int = 8) -> str:
        """Compact prompt block for a new session on this repo."""

        if not self.entries:
            return ""

        ordered = sorted(
            self.entries.values(), key=lambda e: (-min(e.uses, 99), -e.updated)
        )[:k]

        lines = ["Memory from previous sessions on this repository:"]

        for entry in ordered:

            lines.append(f"- [{entry.kind}] {entry.key}: {entry.value}")

        lines.append(
            "Treat these as hints with history, not ground truth: verify "
            "before relying on them."
        )

        return "\n".join(lines)

    def learn_from_session(self, state) -> int:
        """Distil a finished session into memories. Returns count stored.

        Deliberately conservative: only pinned facts with evidence (tests,
        commands) and failure lessons with a concrete signature become
        memories — hallucinated 'facts' must not outlive a session.
        """

        stored = 0

        for fact in getattr(state, "facts", [])[:20]:

            key = f"fact:{_key(fact)}"

            if self.entries.get(key) is None:

                self.remember(key, fact, "fact")

                stored += 1

        for record in getattr(state, "failures", [])[-10:]:

            signature = getattr(record, "signature", "")

            recovery = getattr(record, "recovery", "")

            if not signature or not recovery:

                continue

            key = f"lesson:{signature[:80]}"

            value = f"{getattr(record, 'error', '')[:160]} -> resolved by {recovery}"

            if self.entries.get(key) is None:

                self.remember(key, value, "lesson")

                stored += 1

        for name in getattr(state, "tests_passed", [])[:10]:

            key = f"test:{name}"

            if self.entries.get(key) is None:

                self.remember(key, "passing; do not regress", "pattern")

                stored += 1

        if stored:

            self.save()

        return stored

    def __len__(self) -> int:
        return len(self.entries)


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]
