"""Training rows for the loop that actually matters: fail -> diagnose -> fix -> green.

Two sources feed this generator:

1. **Curated repair pairs** - small, real bug/fix pairs covering the failures
   generated code actually produces (off-by-one, mutable default, missing await,
   swallowed exception, wrong SQL parameterisation, stale cache, unclosed file,
   timezone bug, None handling, test that asserts the wrong thing). Each pair
   carries a realistic failure message, so the model learns to read evidence.
2. **Recorded verification transcripts** - ``VerificationLoop.to_trace()`` from a
   real run (test command, failure signal, fix, re-run, green). These are the
   highest-value rows because nothing about them is guessed.

Rows are SFT (``prompt -> corrected code + reasoning``) and DPO
(``chosen`` = green fix, ``rejected`` = the plausible-but-failing patch). Every
row is deterministic for a given seed and records its provenance.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Iterable

#: (id, task, buggy code, fixed code, failure output, diagnosis, rejected patch)
CURATED: list[tuple[str, str, str, str, str, str, str]] = [
    (
        "off-by-one",
        "Implement `last_n(items, n)` returning the last n items in order.",
        "def last_n(items, n):\n    return items[len(items) - n - 1:]\n",
        "def last_n(items, n):\n    if n <= 0:\n        return []\n    return items[-n:]\n",
        "FAILED tests/test_utils.py::test_last_n - AssertionError: assert ['b', 'c'] == ['a', 'b', 'c']",
        "The slice start was off by one and n<=0 was not handled.",
        "def last_n(items, n):\n    return items[len(items) - n:]\n",
    ),
    (
        "mutable-default",
        "Write `add_tag(tag, tags=None)` that appends to a per-call list.",
        "def add_tag(tag, tags=[]):\n    tags.append(tag)\n    return tags\n",
        "def add_tag(tag, tags=None):\n    tags = list(tags or [])\n    tags.append(tag)\n    return tags\n",
        "FAILED tests/test_tags.py::test_add_tag_isolated - AssertionError: assert ['a', 'b'] == ['b']",
        "The default list is shared across calls, so state leaked between tests.",
        "def add_tag(tag, tags=[]):\n    return tags + [tag]\n",
    ),
    (
        "missing-await",
        "`fetch_user` must await the async client and return the parsed user.",
        "async def fetch_user(client, user_id):\n    response = client.get(f\"/users/{user_id}\")\n    return response.json()\n",
        "async def fetch_user(client, user_id):\n    response = await client.get(f\"/users/{user_id}\")\n    return response.json()\n",
        "FAILED tests/test_client.py::test_fetch_user - AttributeError: 'coroutine' object has no attribute 'json'",
        "The coroutine was never awaited, so `.json()` was called on a coroutine object.",
        "async def fetch_user(client, user_id):\n    response = client.get(f\"/users/{user_id}\")\n    await response\n    return response.json()\n",
    ),
    (
        "swallowed-exception",
        "`parse_port` must raise `ValueError` on non-numeric input, with a readable message.",
        "def parse_port(value):\n    try:\n        return int(value)\n    except Exception:\n        return 8000\n",
        "def parse_port(value):\n    try:\n        port = int(value)\n    except (TypeError, ValueError) as exc:\n        raise ValueError(f\"invalid port: {value!r}\") from exc\n    if not 1 <= port <= 65535:\n        raise ValueError(f\"port out of range: {port}\")\n    return port\n",
        "FAILED tests/test_config.py::test_bad_port - Failed: DID NOT RAISE <class 'ValueError'>",
        "A broad except swallowed the error and silently substituted a default.",
        "def parse_port(value):\n    try:\n        return int(value)\n    except Exception:\n        raise\n",
    ),
    (
        "sql-parameterisation",
        "`find_user` must look a user up by email without SQL injection.",
        "def find_user(conn, email):\n    return conn.execute(f\"SELECT * FROM users WHERE email = '{email}'\").fetchone()\n",
        "def find_user(conn, email):\n    return conn.execute(\"SELECT * FROM users WHERE email = ?\", (email,)).fetchone()\n",
        "FAILED tests/test_store.py::test_quote_in_email - sqlite3.OperationalError: near \"o\": syntax error",
        "The email was interpolated into SQL: a quote breaks the query, and a crafted value is an injection.",
        "def find_user(conn, email):\n    return conn.execute(\"SELECT * FROM users WHERE email = '\" + email.replace(\"'\", \"''\") + \"'\").fetchone()\n",
    ),
    (
        "unclosed-file",
        "`read_config` must return the file contents and always close the handle.",
        "def read_config(path):\n    handle = open(path)\n    data = handle.read()\n    return data\n",
        "def read_config(path):\n    with open(path, encoding=\"utf-8\") as handle:\n        return handle.read()\n",
        "FAILED tests/test_config.py::test_handle_closed - ResourceWarning: unclosed file <_io.TextIOWrapper ...>",
        "The handle leaked: on Windows the file stays locked, and the descriptor is exhausted under load.",
        "def read_config(path):\n    handle = open(path)\n    data = handle.read()\n    handle.close()\n    return data\n",
    ),
    (
        "timezone-bug",
        "`expires_at` must be a timezone-aware UTC timestamp one hour from now.",
        "from datetime import datetime, timedelta\n\n\ndef expires_at():\n    return datetime.now() + timedelta(hours=1)\n",
        "from datetime import datetime, timedelta, timezone\n\n\ndef expires_at():\n    return datetime.now(timezone.utc) + timedelta(hours=1)\n",
        "FAILED tests/test_tokens.py::test_expiry_is_aware - TypeError: can't subtract offset-naive and offset-aware datetimes",
        "`datetime.now()` returns a naive local time; comparisons with UTC-aware values fail.",
        "from datetime import datetime, timedelta\n\n\ndef expires_at():\n    return datetime.utcnow() + timedelta(hours=1)\n",
    ),
    (
        "none-handling",
        "`display_name(user)` must fall back to the email local-part when name is missing.",
        "def display_name(user):\n    return user[\"name\"].strip()\n",
        "def display_name(user):\n    name = (user.get(\"name\") or \"\").strip()\n    if name:\n        return name\n    return (user.get(\"email\") or \"anonymous\").split(\"@\")[0]\n",
        "FAILED tests/test_users.py::test_missing_name - KeyError: 'name'",
        "The code indexed a key that may be absent instead of using .get with a fallback.",
        "def display_name(user):\n    return user.get(\"name\") or \"anonymous\"\n",
    ),
    (
        "pagination",
        "`page(items, number, size)` must return the requested 1-based page and clamp out-of-range pages.",
        "def page(items, number, size):\n    start = number * size\n    return items[start:start + size]\n",
        "def page(items, number, size):\n    if size <= 0:\n        raise ValueError(\"size must be positive\")\n    number = max(1, number)\n    start = (number - 1) * size\n    return items[start:start + size]\n",
        "FAILED tests/test_pagination.py::test_first_page - AssertionError: assert ['b', 'c'] == ['a', 'b']",
        "Pages were treated as 0-based and size<=0 was unguarded, so page 1 skipped the first row.",
        "def page(items, number, size):\n    start = max(0, number - 1) * size\n    return items[start:start + size]\n",
    ),
    (
        "cache-key",
        "`cached_price(symbol)` must not serve one symbol's price for another.",
        "CACHE = {}\n\n\n\ndef cached_price(symbol, fetch):\n    if \"price\" not in CACHE:\n        CACHE[\"price\"] = fetch(symbol)\n    return CACHE[\"price\"]\n",
        "CACHE = {}\n\n\n\ndef cached_price(symbol, fetch):\n    key = str(symbol).upper()\n    if key not in CACHE:\n        CACHE[key] = fetch(key)\n    return CACHE[key]\n",
        "FAILED tests/test_prices.py::test_distinct_symbols - AssertionError: assert 10 == 20",
        "A single global key meant every symbol shared the first cached value.",
        "CACHE = {}\n\n\n\ndef cached_price(symbol, fetch):\n    return CACHE.setdefault(symbol, fetch(symbol))\n",
    ),
    (
        "test-wrong-assertion",
        "Fix the failing `total` test without weakening it.",
        "def test_total():\n    assert total([1, 2, 3]) == 7\n",
        "def test_total():\n    assert total([1, 2, 3]) == 6\n    assert total([]) == 0\n",
        "FAILED tests/test_mathx.py::test_total - AssertionError: assert 6 == 7",
        "The assertion was wrong, not the implementation: 1+2+3 is 6. Fix the test and add an empty case.",
        "def test_total():\n    assert total([1, 2, 3]) == 6 or True\n",
    ),
    (
        "race-condition",
        "`increment` must be safe when called from several threads.",
        "COUNTER = 0\n\n\n\ndef increment():\n    global COUNTER\n    COUNTER = COUNTER + 1\n    return COUNTER\n",
        "import threading\n\nCOUNTER = 0\n_LOCK = threading.Lock()\n\n\ndef increment():\n    global COUNTER\n    with _LOCK:\n        COUNTER += 1\n        return COUNTER\n",
        "FAILED tests/test_counter.py::test_threaded - AssertionError: assert 93 == 100",
        "Read-modify-write on a global is not atomic; lost updates show up only under load.",
        "def increment():\n    global COUNTER\n    COUNTER += 1\n    return COUNTER\n",
    ),
    (
        "unicode-path",
        "`load_names` must read UTF-8 files and not crash on non-ASCII names.",
        "def load_names(path):\n    return [line for line in open(path).read().splitlines()]\n",
        "from pathlib import Path\n\n\n\ndef load_names(path):\n    text = Path(path).read_text(encoding=\"utf-8\")\n    return text.splitlines()\n",
        "FAILED tests/test_io.py::test_unicode - UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3",
        "The platform default encoding was used; on some systems that is ASCII.",
        "def load_names(path):\n    return [line for line in open(path, errors=\"ignore\").read().splitlines()]\n",
    ),
    (
        "config-fallback",
        "`database_url` must fail loudly when neither env var nor config provides a URL.",
        "def database_url(env):\n    return env.get(\"DATABASE_URL\") or \"sqlite:///dev.db\"\n",
        "def database_url(env):\n    url = env.get(\"DATABASE_URL\") or env.get(\"PG_URL\")\n    if not url:\n        raise RuntimeError(\"DATABASE_URL is required (no silent sqlite fallback in production)\")\n    return url\n",
        "FAILED tests/test_settings.py::test_missing_url - Failed: DID NOT RAISE <class 'RuntimeError'>",
        "A silent fallback hid a production misconfiguration instead of failing at boot.",
        "def database_url(env):\n    url = env.get(\"DATABASE_URL\")\n    return url if url else None\n",
    ),
    (
        "resource-limit",
        "`read_all` must stream instead of loading a huge response into memory at once.",
        "def read_all(response):\n    return response.text.splitlines()\n",
        "def read_all(response):\n    for line in response.iter_lines(decode_unicode=True):\n        if line:\n            yield line\n",
        "FAILED tests/test_stream.py::test_memory_bounded - MemoryError",
        "The whole response was materialised; iterating lines keeps memory bounded.",
        "def read_all(response):\n    return [line for line in response.text.splitlines() if line]\n",
    )
    (
        "memory-leak",
        "Fix `process_batch` that leaked file handles on exception.",
        "def process_batch(paths):\n    handles = [open(p) for p in paths]\n    data = [h.read() for h in handles]\n    return data\n",
        "def process_batch(paths):\n    data = []\n    for p in paths:\n        with open(p, encoding=\"utf-8\") as h:\n            data.append(h.read())\n    return data\n",
        "FAILED tests/test_batch.py::test_leak - ResourceWarning: unclosed file",
        "File handles were opened without context managers and never closed on exception path.",
        "def process_batch(paths):\n    return [open(p).read() for p in paths]\n",
    ),
    (
        "auth-bypass",
        "Fix API that allowed unauthenticated access to /admin.",
        "app.get(\"/admin\", lambda: {\"secret\": True})\n",
        "@app.get(\"/admin\")\ndef admin(user=Depends(current_user)):\n    return {\"secret\": True}\n",
        "FAILED tests/test_authz.py::test_admin_requires_auth - AssertionError: assert 200 == 401",
        "Endpoint lacked dependency injection for auth; any user could fetch admin data.",
        "app.get(\"/admin\", lambda user=Depends(current_user): {\"secret\": True})\n",
    ),
    (
        "n-plus-one",
        "Fix `list_orders` that issued N+1 queries.",
        "def list_orders(ids):\n    return [db.get_order(i) for i in ids]\n",
        "def list_orders(ids):\n    if not ids:\n        return []\n    rows = db.get_orders_bulk(ids)\n    return [dict(r) for r in rows]\n",
        "FAILED tests/test_perf.py::test_query_count - AssertionError: assert 11 == 1",
        "Loop issued one query per id; bulk query dedupes and uses single roundtrip.",
        "def list_orders(ids):\n    return list(map(db.get_order, ids))\n",
    ),
    (
        "race-upload",
        "Fix file upload that truncated on concurrent writes.",
        "def save_upload(path, data):\n    open(path, \"w\").write(data)\n",
        "def save_upload(path, data):\n    tmp = Path(path).with_suffix(\".tmp\")\n    tmp.write_text(data, encoding=\"utf-8\")\n    tmp.replace(path)\n",
        "FAILED tests/test_upload.py::test_concurrent - AssertionError: file corrupted",
        "Direct write is not atomic; use temp file + rename for crash safety.",
        "def save_upload(path, data):\n    Path(path).write_text(data)\n",
    ),
    (
        "cors-misconfig",
        "Fix CORS that allowed any origin to read sensitive data.",
        "app.add_middleware(CORSMiddleware, allow_origins=[\"*\"], allow_credentials=True)\n",
        "app.add_middleware(CORSMiddleware, allow_origins=[\"https://app.example.com\"], allow_credentials=True)\n",
        "FAILED tests/test_cors.py::test_no_wildcard_with_creds - AssertionError: wildcard + creds is unsafe",
        "Wildcard with credentials violates spec and browsers will reject it; enumerate allowed origins.",
        "app.add_middleware(CORSMiddleware, allow_origins=[\"*\"])\n",
    ),
    (
        "date-parsing",
        "Fix `parse_date` that mis-handled ISO dates with timezone.",
        "def parse_date(s):\n    return datetime.strptime(s, \"%Y-%m-%d\")\n",
        "def parse_date(s):\n    from datetime import timezone\n    dt = datetime.fromisoformat(s.replace(\"Z\", \"+00:00\"))\n    return dt.astimezone(timezone.utc)\n",
        "FAILED tests/test_date.py::test_tz - ValueError: unconverted data remains: T10:00:00Z",
        "strptime with Y-m-d ignores time and tz; fromisoformat handles ISO8601 with tz.",
        "def parse_date(s):\n    return datetime.fromisoformat(s)\n",
    ),
    (
        "retry-storm",
        "Fix retry loop that retried non-idempotent POST without backoff.",
        "for i in range(3):\n    requests.post(url, json=payload)\n",
        "import time\nfor i in range(3):\n    resp = requests.post(url, json=payload)\n    if resp.ok:\n        break\n    time.sleep(2 ** i)\n",
        "FAILED tests/test_retry.py::test_no_storm - AssertionError: 3 posts sent despite 201 on first",
        "Loop ignored success and had no backoff; check response and sleep exponentially.",
        "for i in range(3):\n    time.sleep(1)\n    requests.post(url, json=payload)\n",
    ),
    (
        "secrets-in-log",
        "Fix logger that leaked API keys.",
        "logger.info(f\"calling {url} with key={api_key}\")\n",
        "logger.info(f\"calling {url} with key=***REDACTED***\")\n",
        "FAILED tests/test_security.py::test_no_secret_in_log - AssertionError: 'sk-' in log",
        "Logging the key leaks it to log aggregators; redact before logging.",
        "logger.info(f\"calling {url}\")\n",
    ),
    (
        "validation-bypass",
        "Fix endpoint that trusted client-supplied is_admin flag.",
        "def create_user(data):\n    return User(**data)\n",
        "def create_user(data):\n    allowed = {k: v for k, v in data.items() if k in {\"email\", \"password\", \"name\"}}\n    return User(**allowed)\n",
        "FAILED tests/test_validation.py::test_cannot_set_admin - AssertionError: is_admin True leaked",
        "Unvalidated **data allowed mass assignment; whitelist fields.",
        "def create_user(data):\n    data.pop(\"is_admin\", None)\n    return User(**data)\n",
    ),
    (
        "timeout-missing",
        "Fix HTTP client that hung without timeout.",
        "resp = httpx.get(url)\n",
        "resp = httpx.get(url, timeout=10)\n",
        "FAILED tests/test_http.py::test_timeout - Timeout not set, test hung",
        "No timeout means hangs forever on slow host; always set timeout.",
        "resp = httpx.get(url, timeout=5)\n",
    ),
,
]


@dataclass
class RepairExample:
    bug_id: str
    task: str
    buggy: str
    fixed: str
    failure: str
    diagnosis: str
    rejected: str
    source: str = "curated"

    def prompt(self) -> str:
        return (
            f"Task: {self.task}\n\n"
            f"Current code:\n```python\n{self.buggy}```\n\n"
            f"Test output:\n{self.failure}\n\n"
            "Make the smallest change that makes the behaviour correct. Do not weaken or delete tests."
        )

    def answer(self) -> str:
        return (
            f"Diagnosis: {self.diagnosis}\n\n"
            f"```python\n{self.fixed}```\n\n"
            "Verify by re-running the failing test, then the full suite."
        )

    def to_sft_row(self) -> dict:
        return {
            "id": self.bug_id,
            "prompt": self.prompt(),
            "response": self.answer(),
            "metadata": {
                "source": self.source,
                "kind": "repair",
                "has_test_evidence": True,
                "verified": self.source == "recorded",
                "verified_by": "test-suite" if self.source == "recorded" else "curated-pair",
            },
        }

    def to_dpo_row(self) -> dict:
        return {
            "id": self.bug_id,
            "prompt": self.prompt(),
            "chosen": self.answer(),
            "rejected": f"```python\n{self.rejected}```",
            "metadata": {
                "source": self.source,
                "kind": "repair-dpo",
                "rejected_reason": "plausible patch that still fails the test (or weakens it)",
            },
        }


@dataclass
class RepairTraceGenerator:
    """Curated pairs + recorded verification transcripts -> SFT/DPO rows."""

    seed: int = 42
    include_curated: bool = True
    examples: list[RepairExample] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.include_curated and not self.examples:
            for bug_id, task, buggy, fixed, failure, diagnosis, rejected in CURATED:
                self.examples.append(
                    RepairExample(
                        bug_id=bug_id,
                        task=task,
                        buggy=buggy,
                        fixed=fixed,
                        failure=failure,
                        diagnosis=diagnosis,
                        rejected=rejected,
                    )
                )

    def add_trace(self, trace: dict, *, bug_id: str | None = None) -> int:
        """Record a real VerificationLoop transcript (skips runs that never went green)."""

        runs = trace.get("runs") or []
        if not trace.get("green") or len(runs) < 2:
            return 0
        first = runs[0]
        last = runs[-1]
        failure = first.get("output_tail", "")[-1200:]
        fixed = last.get("fixed_code") or ""
        buggy = first.get("original_code") or ""
        if not failure:
            return 0
        digest = hashlib.sha256((failure + str(trace.get("command", ""))).encode()).hexdigest()[:10]
        example = RepairExample(
            bug_id=bug_id or f"trace-{digest}",
            task=f"Make `{trace.get('command', 'the test suite')}` pass.",
            buggy=buggy,
            fixed=fixed,
            failure=failure,
            diagnosis=(trace.get("attempts") or [{}])[-1].get("diagnosis", "fix derived from the test output"),
            rejected=buggy,
            source="recorded",
        )
        self.examples.append(example)
        return 1

    def to_sft_rows(self) -> list[dict]:
        return [example.to_sft_row() for example in self.examples]

    def to_dpo_rows(self) -> list[dict]:
        return [example.to_dpo_row() for example in self.examples]

    def split(self, heldout_ratio: float = 0.15) -> tuple[list[dict], list[dict]]:
        """Deterministic train/held-out split, keyed by bug id (never by order)."""

        rows = self.to_sft_rows()
        rows.sort(key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())
        cut = max(1, int(len(rows) * heldout_ratio)) if len(rows) > 3 else 0
        return rows[cut:], rows[:cut]

    def stats(self) -> dict:
        sources: dict[str, int] = {}
        for example in self.examples:
            sources[example.source] = sources.get(example.source, 0) + 1
        return {
            "examples": len(self.examples),
            "by_source": sources,
            "sft_rows": len(self.to_sft_rows()),
            "dpo_rows": len(self.to_dpo_rows()),
            "bug_ids": [example.bug_id for example in self.examples],
        }

    def sample(self, n: int) -> list[RepairExample]:
        rng = random.Random(self.seed)
        pool = list(self.examples)
        rng.shuffle(pool)
        return pool[:n]


def generate_repair_rows(traces: Iterable[dict] | None = None, seed: int = 42) -> dict:
    generator = RepairTraceGenerator(seed=seed)
    for trace in traces or []:
        generator.add_trace(trace)
    train, heldout = generator.split()
    return {
        "sft": train,
        "sft_heldout": heldout,
        "dpo": generator.to_dpo_rows(),
        "stats": generator.stats(),
    }
