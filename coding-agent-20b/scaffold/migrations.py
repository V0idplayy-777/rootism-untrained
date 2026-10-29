"""Managed Postgres migration generation + SQLite→Postgres switch.

Provides:
- generate_migration(module, description) -> {path: content}
- switch_command helpers
- CLI wiring via BackendProvisioner

All migrations are plain SQL files plus a Python runner that works with both
sqlite3 and psycopg (via the existing get_connection shim).
"""

from __future__ import annotations

import re
import textwrap
import time
from pathlib import Path


MIGRATION_RUNNER = '''"""Migration runner: apply SQL files in migrations/."""

from __future__ import annotations

import os
from pathlib import Path

from {module}.store import get_connection

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def apply_migrations() -> list[str]:
    applied: list[str] = []
    with get_connection() as conn:
        # track applied migrations if table exists
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS _migrations (id TEXT PRIMARY KEY, applied_at TEXT NOT NULL)")
        except Exception:
            pass
        for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
            mig_id = sql_file.stem
            # check if already applied
            try:
                row = conn.execute("SELECT 1 FROM _migrations WHERE id = ?", (mig_id,)).fetchone()
                if row:
                    continue
            except Exception:
                pass
            sql = sql_file.read_text(encoding="utf-8")
            # split on ; but keep within statements
            for statement in [s.strip() for s in sql.split(";") if s.strip()]:
                conn.execute(statement)
            try:
                conn.execute("INSERT INTO _migrations (id, applied_at) VALUES (?, datetime('now'))", (mig_id,))
            except Exception:
                pass
            applied.append(mig_id)
    return applied


if __name__ == "__main__":
    print(apply_migrations())
'''

MIGRATION_SQL_TEMPLATE = """-- Migration: {name}
-- Generated: {date}
-- Description: {description}

-- Add your DDL here. Example:
-- CREATE TABLE IF NOT EXISTS {module}_extra (
--     id TEXT PRIMARY KEY,
--     created_at TEXT NOT NULL
-- );
"""

SWITCH_GUIDE = """# Switching from SQLite to Postgres

Your project currently uses SQLite (`data/app.db`). To switch to managed Postgres:

1. Provision a Postgres (Neon or Supabase):
   ```
   agent backend --provider neon --name {name}
   # or
   agent backend --provider supabase --name {name}
   ```

2. The command writes `DATABASE_URL` to `.env` (chmod 0600). Verify:
   ```
   cat .env | grep DATABASE_URL
   ```

3. Generate and run migrations:
   ```
   agent db migrate --create "initial schema"
   agent db upgrade
   ```

4. Test locally:
   ```
   python -m {module}.store  # should connect via DATABASE_URL
   python -m pytest -q
   ```

5. Deploy now uses DATABASE_URL automatically (no code change needed; `store.py` already shims `?` -> `%s` for psycopg).

Rollback: remove `DATABASE_URL` from `.env` to revert to SQLite.

Migration files live in `{module}/migrations/` and are tracked in `_migrations` table.
"""


def generate_migration(module: str, name: str, description: str = "") -> dict[str, str]:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "migration").lower()).strip("-")[:40] or "migration"
    timestamp = time.strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{slug}.sql"
    sql_content = MIGRATION_SQL_TEMPLATE.format(
        name=slug,
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        description=description or slug,
        module=module,
    )
    runner = MIGRATION_RUNNER.format(module=module)
    guide = SWITCH_GUIDE.format(name=module, module=module)
    return {
        f"{module}/migrations/{filename}": sql_content,
        f"{module}/migrate.py": runner,
        "MIGRATIONS.md": guide,
    }


def switch_to_postgres_instructions(module: str, name: str) -> str:
    return SWITCH_GUIDE.format(name=name, module=module)


def validate_postgres_url(url: str) -> tuple[bool, str]:
    if not url:
        return False, "empty URL"
    if url.startswith(("postgres://", "postgresql://")) and "@" in url and "/" in url.split("@")[-1]:
        return True, "looks like postgres"
    return False, "not a postgres URL; expected postgres://user:password@host:5432/dbname"
