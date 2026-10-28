"""File-backed scratch space that outlives the context window.

Deep algorithmic reasoning on an 8k–32k window fails in a specific way: the
model's *own intermediate work* (hypotheses, invariants, partial traces,
per-file findings) crowds out the task, gets compacted away, and step N+1
silently re-derives — or contradicts — step N.

``ScratchWorkspace`` moves that work to disk and keeps only a tiny *index* in
the prompt:

- ``write(topic, text)`` appends a timestamped entry to
  ``.agent/scratch/<topic>.md`` and updates ``index.json``.
- ``read(topic)`` reloads it on demand (one tool call, bounded chars).
- ``render_index()`` is the ≤N-char block injected into every prompt:
  topic, entry count, size, one-line summary. The model can *see* what it
  wrote without paying for it.
- ``search(query)`` finds topics by keyword when the name is forgotten.

Everything is plain markdown + JSON in the workspace: diffable, greppable,
deletable, and safe to commit or ignore. Entries are append-mostly (an audit
trail of the reasoning), with an explicit ``replace`` mode for corrections.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

MAX_TOPIC_CHARS = 400_000
INDEX_VERSION = 1


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")
    return slug[:48] or "notes"


def _summary_of(text: str, chars: int = 110) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#-*> ").strip()
        if stripped:
            return stripped[:chars]
    return ""


class ScratchWorkspace:
    def __init__(self, root: str | os.PathLike = ".agent/scratch"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.index: dict = self._load_index()

    # -- index -------------------------------------------------------------

    def _load_index(self) -> dict:
        if self.index_path.is_file():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                if isinstance(data.get("topics"), dict):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {"version": INDEX_VERSION, "topics": {}}

    def _save_index(self) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.index_path)

    def _topic_path(self, topic: str) -> Path:
        return self.root / f"{_slug(topic)}.md"

    # -- API ----------------------------------------------------------------

    def write(self, topic: str, text: str, mode: str = "append") -> dict:
        """Append (default) or replace a scratch entry. Returns a receipt."""

        if mode not in {"append", "replace"}:
            raise ValueError("mode must be 'append' or 'replace'")
        text = str(text).strip()
        if not text:
            raise ValueError("scratch text must be non-empty")

        path = self._topic_path(topic)
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""

        if mode == "append":
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            body = existing + f"\n## [{stamp}]\n{text}\n" if existing else f"# {topic}\n\n## entry 1\n{text}\n"
        else:
            body = f"# {topic}\n\n{text}\n"

        if len(body) > MAX_TOPIC_CHARS:
            # keep the newest 3/4 of the budget (scratch grows downward)
            cut = len(body) - (MAX_TOPIC_CHARS * 3) // 4
            body = body[cut:]
            body = "# …older entries pruned\n" + body[body.find("\n##"):]

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, path)

        entries = len(re.findall(r"^## ", body, re.M))
        record = self.index["topics"].get(_slug(topic), {})
        self.index["topics"][_slug(topic)] = {
            "title": str(record.get("title") or topic),
            "entries": entries,
            "chars": len(body),
            "updated": time.time(),
            "summary": _summary_of(text),
        }
        self._save_index()

        return {
            "topic": _slug(topic),
            "entries": entries,
            "chars": len(body),
            "mode": mode,
            "note": (
                "written to scratch; the index (not the content) is in your "
                "prompt — use scratch_read when you need it back"
            ),
        }

    def read(self, topic: str, max_chars: int = 6000) -> str:
        path = self._topic_path(topic)
        if not path.is_file():
            raise KeyError(f"no scratch topic named {topic!r}")
        text = path.read_text(encoding="utf-8")
        if len(text) > max_chars:
            # newest entries are at the end; keep those
            text = "…earlier entries omitted…\n" + text[-max_chars:]
        return text

    def list_topics(self) -> list[dict]:
        rows = []
        for slug, record in sorted(
            self.index["topics"].items(), key=lambda kv: -kv[1].get("updated", 0)
        ):
            rows.append(
                {
                    "topic": slug,
                    "title": record.get("title", slug),
                    "entries": record.get("entries", 0),
                    "chars": record.get("chars", 0),
                    "summary": record.get("summary", ""),
                }
            )
        return rows

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query)}
        if not q:
            return []
        scored = []
        for row in self.list_topics():
            hay = {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", row["title"] + " " + row["summary"])}
            overlap = len(q & hay)
            if overlap:
                scored.append((overlap, row))
        scored.sort(key=lambda kv: -kv[0])
        return [row for _s, row in scored[:k]]

    def render_index(self, max_chars: int = 600) -> str:
        """The block injected into every prompt: what scratch exists."""

        rows = self.list_topics()
        if not rows:
            return ""
        lines = ["<|scratchpad|> index (your disk notes; scratch_read to load):"]
        used = len(lines[0])
        for row in rows:
            line = (
                f"- {row['topic']} ({row['entries']} entries, "
                f"{row['chars']} chars): {row['summary']}"
            )
            if used + len(line) > max_chars:
                lines.append(f"- … {len(rows)} topics total (scratch_list)")
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"topics": self.list_topics()}

    def __len__(self) -> int:
        return len(self.index["topics"])
