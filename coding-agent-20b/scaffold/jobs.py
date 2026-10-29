"""Background-job / scheduled-task examples for every starter.

Each template gets a working jobs module that can be run as:
  python -m <module>.jobs --once          (run once)
  python -m <module>.jobs --schedule 60   (every 60s)

No extra dependency: uses stdlib threading + sqlite. The job writes a log table
so 'full-stack includes more than request/response' is demonstrable.
"""

from __future__ import annotations

PY_JOBS = '''"""Background jobs: simple, no external queue required.

Usage:
  python -m {module}.jobs --once
  python -m {module}.jobs --schedule 60   # every 60 seconds
  python -m {module}.jobs --worker        # same as schedule 30

Jobs here are intentionally small and testable: they use the same `store` that
the web app uses, so they work with SQLite or Postgres (DATABASE_URL).

Add real work by editing `run_scheduled_task` and adding callers.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from .store import get_connection

SCHEMA_JOBS = """
CREATE TABLE IF NOT EXISTS job_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL
);
"""


def ensure_schema() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA_JOBS)


def log_run(job: str, status: str, detail: str = "") -> None:
    ensure_schema()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO job_runs (job, status, detail, created_at) VALUES (?, ?, ?, ?)",
            (job, status, detail, datetime.now(timezone.utc).isoformat()),
        )


def run_scheduled_task(detail: str = "") -> dict:
    """Example job: could send emails, clean up, recalc metrics, etc.

    This version counts items and logs the count. Replace with your logic;
    keep it idempotent so retries are safe.
    """
    try:
        with get_connection() as conn:
            # example query: count rows (works even if table is empty)
            try:
                row = conn.execute("SELECT COUNT(*) as c FROM items").fetchone()
                count = row[0] if row else 0
            except Exception:
                count = 0
        log_run("scheduled_task", "ok", f"count={{count}} {{detail}}".strip())
        return {{"ok": True, "count": count}}
    except Exception as exc:  # noqa: BLE001
        log_run("scheduled_task", "error", str(exc))
        return {{"ok": False, "error": str(exc)}}


def run_worker(interval: int = 30, once: bool = False) -> None:
    print(f"[jobs] starting worker (interval={{interval}}s, once={{once}})")
    ensure_schema()
    if once:
        result = run_scheduled_task("once")
        print(f"[jobs] result: {{result}}")
        return
    try:
        while True:
            result = run_scheduled_task("tick")
            print(f"[jobs] {{datetime.now(timezone.utc).isoformat()}} scheduled_task -> {{result}}")
            time.sleep(max(1, interval))
    except KeyboardInterrupt:
        print("[jobs] stopped")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Background jobs for {display_name}")
    parser.add_argument("--once", action="store_true", help="run once and exit")
    parser.add_argument("--schedule", type=int, default=0, help="run every N seconds")
    parser.add_argument("--worker", action="store_true", help="run forever every 30s")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.once:
        run_worker(once=True)
        return 0
    if args.schedule:
        run_worker(interval=args.schedule)
        return 0
    if args.worker:
        run_worker(interval=30)
        return 0
    build_parser().print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''


NODE_JOBS = '''// Background job example (Node.js)
// Usage: node jobs/worker.js --once
//        node jobs/worker.js --schedule 60000

import { setTimeout as sleep } from "node:timers/promises";
import path from "node:path";
import fs from "node:fs";

const LOG = path.join(process.cwd(), "data", "job_runs.log");

function ensureLog() {
  const dir = path.dirname(LOG);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  if (!fs.existsSync(LOG)) fs.writeFileSync(LOG, "");
}

function logRun(job, status, detail = "") {
  ensureLog();
  const line = JSON.stringify({ job, status, detail, ts: new Date().toISOString() }) + "\\n";
  fs.appendFileSync(LOG, line);
}

export async function runScheduledTask(detail = "") {
  try {
    // Example: count items from a JSON file or DB; keep it idempotent
    const dataPath = path.join(process.cwd(), "data", "items.json");
    let count = 0;
    if (fs.existsSync(dataPath)) {
      try {
        const raw = fs.readFileSync(dataPath, "utf-8");
        count = JSON.parse(raw).length;
      } catch {}
    }
    logRun("scheduled_task", "ok", `count=${count} ${detail}`.trim());
    return { ok: true, count };
  } catch (error) {
    logRun("scheduled_task", "error", String(error));
    return { ok: false, error: String(error) };
  }
}

async function runWorker(intervalMs = 30000, once = false) {
  console.log(`[jobs] starting (interval=${intervalMs}ms, once=${once})`);
  ensureLog();
  if (once) {
    console.log("[jobs] result:", await runScheduledTask("once"));
    return;
  }
  while (true) {
    const result = await runScheduledTask("tick");
    console.log(`[jobs] ${new Date().toISOString()} ->`, result);
    await sleep(intervalMs);
  }
}

const args = process.argv.slice(2);
if (args.includes("--once")) {
  runWorker(0, true);
} else if (args.includes("--worker")) {
  runWorker(30000, false);
} else {
  const idx = args.indexOf("--schedule");
  if (idx !== -1 && args[idx + 1]) {
    runWorker(parseInt(args[idx + 1], 10), false);
  } else {
    console.log("Usage: node jobs/worker.js --once | --worker | --schedule <ms>");
    process.exit(2);
  }
}
'''


FLASK_JOBS_CONTENT = PY_JOBS.replace("from .store import get_connection", "from store import get_connection").replace("python -m {module}.jobs", "python jobs.py")

def flask_jobs() -> dict[str, str]:
    content = PY_JOBS.format(module="app", display_name="Flask App").replace("from .store import get_connection", "from store import get_connection")
    content = content.replace("from .store", "from store")
    return {
        "jobs.py": content,
        "tests/test_jobs.py": """"\"\"Jobs contract.\"\"\"
from __future__ import annotations
import os, tempfile
os.environ["SQLITE_PATH"] = os.path.join(tempfile.mkdtemp(), "jobs_test.db")
import jobs
def test_job_runs_and_logs():
    jobs.ensure_schema()
    result = jobs.run_scheduled_task("test")
    assert result["ok"] is True
""",
        "Procfile": "web: python -m flask --app app run --host 0.0.0.0 --port $PORT\nworker: python jobs.py --worker\n",
    }

FLASK_JOBS = FLASK_JOBS_CONTENT



def python_jobs(module: str, display_name: str) -> dict[str, str]:
    content = PY_JOBS.format(module=module, display_name=display_name)
    return {
        f"{module}/jobs.py": content,
        f"tests/test_jobs.py": f'''"""Jobs contract: ensure scheduled task logs and succeeds."""

from __future__ import annotations

import os
import tempfile

os.environ["SQLITE_PATH"] = os.path.join(tempfile.mkdtemp(), "jobs_test.db")

from {module}.jobs import ensure_schema, run_scheduled_task, log_run


def test_job_runs_and_logs():
    ensure_schema()
    result = run_scheduled_task("test")
    assert result["ok"] is True
    # second run should also succeed (idempotent)
    result2 = run_scheduled_task("second")
    assert result2["ok"] is True
''',
        "Procfile": f"web: uvicorn {module}.main:app --host 0.0.0.0 --port $PORT\nworker: python -m {module}.jobs --worker\n",
    }


def node_jobs() -> dict[str, str]:
    return {
        "jobs/worker.js": NODE_JOBS,
        "tests/jobs.test.js": '''import assert from "node:assert/strict";
import { runScheduledTask } from "../jobs/worker.js";

const result = await runScheduledTask("test");
assert.equal(result.ok, true);
console.log("jobs test passed");
''',
        "Procfile": "web: npm start\nworker: node jobs/worker.js --worker\n",
    }
