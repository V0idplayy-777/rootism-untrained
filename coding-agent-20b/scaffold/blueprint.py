"""Render a *real* multi-entity data app from a blueprint.

The six starters in :mod:`scaffold.registry` teach shape. This module builds the
thing users actually ask for: "a task app with login, projects, sharing and a
board". From one :class:`~scaffold.blueprints.AppBlueprint` it renders:

* ``schema.sql`` / ``migrations/0001_init.sql`` - typed columns, enum checks,
  foreign keys with ``ON DELETE`` behaviour and indexes on the columns queries
  actually filter by;
* ``app/db.py``, ``app/store.py`` - parameterised SQL, viewer-scoped queries
  (multi-tenant by default), relation validation, conflict detection, stats and
  CSV export;
* ``app/auth.py``, ``app/auth_routes.py`` - the scrypt + HS256 JWT addon, with
  register/login/me;
* ``app/api.py``, ``app/main.py`` - FastAPI routes with honest status codes,
  per-request validation and readable errors;
* a tasteful single-page UI (separate module), tests that exercise real flows,
  pinned requirements, Docker files and an eject-safe README.

Three knobs exist for one reason: preference data and repair traces need a
*plausible but worse* project. ``style="legacy"`` renders the 2018 UI,
``auth_mode="broken"`` authenticates but forgets to scope rows to the viewer,
and ``entity_mode="incomplete"`` drops relation integrity. Those variants are
never presented as the default path; they exist to be rejected, measured and
repaired.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .blueprints import AppBlueprint, EntitySpec, FieldSpec, get_blueprint, match_blueprint
from .spec import ProjectOptions, find_placeholders, module_name, slugify, titleize

REQUIREMENTS = """fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.4
httpx==0.27.0
pytest==8.2.2
"""

DOCKERFILE = """FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --uid 10001 appuser \\
    && mkdir -p /app/data \\
    && chown -R appuser:appuser /app
USER appuser

ENV APP_ENV=production
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"

CMD ["uvicorn", "@@MODULE@@.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

COMPOSE = """services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      APP_ENV: development
      SQLITE_PATH: /app/data/app.db
    volumes:
      - app-data:/app/data

volumes:
  app-data:
"""

DOCKERIGNORE = """__pycache__/
*.py[cod]
.venv/
venv/
data/
.agent/
.git/
.pytest_cache/
node_modules/
.env
.env.*
!.env.example
"""

CONFIG_PY = '''"""Environment and app metadata for @@TITLE@@."""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "@@TITLE@@"
APP_VERSION = "1.0.0"
FEATURE = "@@FEATURE@@"
AUTH_ENABLED = @@AUTH_BOOL@@
DEMO_USER_ID = "demo"
DEMO_USER_EMAIL = "demo@example.com"
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
SQLITE_PATH = os.environ.get("SQLITE_PATH", str(DATA_DIR / "app.db"))
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Browser origins allowed during local development. Same-origin requests
# (the deployed app) need no entry here.
DEV_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]


def auth_enabled() -> bool:
    """Whether the API requires a bearer token."""
    return AUTH_ENABLED
'''

ERRORS_PY = '''"""Errors that carry a sentence a person can act on."""

from __future__ import annotations


class AppError(Exception):
    """Base class for expected, user-facing failures."""

    status_code = 400


class NotFoundError(AppError):
    """A record or parent record is not visible to this viewer."""

    status_code = 404


class ConflictError(AppError):
    """The request is valid but clashes with existing data."""

    status_code = 409


class ForbiddenError(AppError):
    """The viewer may read the record but not perform this action."""

    status_code = 403


class ValidationProblem(AppError):
    """The payload parsed but does not make sense for this resource."""

    status_code = 422


class DatabaseError(AppError):
    """The database driver is missing or the database cannot be reached."""

    status_code = 500
'''

DB_PY = '''"""Persistence without an ORM: sqlite3 by default, Postgres when configured.

Every statement in the app goes through this module, so there is exactly one
place that rewrites ``?`` placeholders for Postgres and one place that opens a
connection with foreign keys enforced.

Schema (also written to ``schema.sql`` for humans):

    @@SCHEMA@@
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import config
from .errors import DatabaseError

SCHEMA = @@TRIPLE@@@@SCHEMA_BLOCK@@@@TRIPLE@@

_POSTGRES_PREFIXES = ("postgres://", "postgresql://")


class _PostgresShim:
    """Adapt psycopg's API to the sqlite subset used here."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def execute(self, sql: str, params: tuple = ()) -> Any:
        """Run one statement with ``?`` placeholders rewritten for psycopg."""
        return self._conn.execute(sql.replace("?", "%s"), params)

    def executescript(self, script: str) -> None:
        """Run a multi-statement schema the way sqlite's executescript does."""
        for statement in script.split(";"):
            if statement.strip():
                self._conn.execute(statement)

    def commit(self) -> None:
        """Commit is implicit in the surrounding psycopg context manager."""
        self._conn.commit()


@contextmanager
def get_connection() -> Iterator[Any]:
    """Yield a DB-API connection with dict rows and the schema applied."""
    if config.DATABASE_URL.startswith(_POSTGRES_PREFIXES):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise DatabaseError(
                "DATABASE_URL points at Postgres; install psycopg[binary] or unset DATABASE_URL"
            ) from exc
        with psycopg.connect(config.DATABASE_URL, row_factory=dict_row) as conn:
            shim = _PostgresShim(conn)
            shim.executescript(SCHEMA)
            yield shim
        return

    path = Path(config.SQLITE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables and indexes when they are missing (idempotent)."""
    with get_connection():
        pass


def query(sql: str, params: Any = ()) -> list[dict[str, Any]]:
    """Run a SELECT and return every row as a dictionary."""
    with get_connection() as conn:
        cursor = conn.execute(sql, tuple(params))
        return [dict(row) for row in cursor.fetchall()]


def query_one(sql: str, params: Any = ()) -> dict[str, Any] | None:
    """Run a SELECT expected to match at most one row."""
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: Any = ()) -> int:
    """Run one write statement and return the affected row count."""
    with get_connection() as conn:
        cursor = conn.execute(sql, tuple(params))
        return int(cursor.rowcount or 0)


def insert(table: str, row: dict[str, Any]) -> None:
    """Insert one row. Column names come from this codebase, never a request."""
    columns = ", ".join(row)
    placeholders = ", ".join("?" for _ in row)
    statement = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
    execute(statement, list(row.values()))


def update_row(
    table: str,
    row_id: str,
    values: dict[str, Any],
    scope_sql: str,
    scope_params: list[Any],
) -> int:
    """Update one row, constrained by the caller's visibility predicate."""
    if not values:
        return 0
    assignments = ", ".join(f"{column} = ?" for column in values)
    statement = f"UPDATE {table} SET {assignments} WHERE id = ? AND {scope_sql}"
    return execute(statement, [*values.values(), row_id, *scope_params])


def delete_row(table: str, row_id: str, scope_sql: str, scope_params: list[Any]) -> int:
    """Delete one row, constrained by the caller's ownership predicate."""
    statement = f"DELETE FROM {table} WHERE id = ? AND {scope_sql}"
    return execute(statement, [row_id, *scope_params])
'''




def pascal(name: str) -> str:
    """``habit_log`` -> ``HabitLog``."""

    return "".join(part.capitalize() for part in re.split(r"[^a-zA-Z0-9]+", name) if part)


def upper(name: str) -> str:
    """``habit_log`` -> ``HABIT_LOG``."""

    return re.sub(r"[^A-Za-z0-9]+", "_", name).upper()


def sql_type(field_spec: FieldSpec) -> str:
    """The column type used by both sqlite and Postgres."""

    if field_spec.type in {"int", "bool"}:
        return "INTEGER"
    return "TEXT"


def py_type(field_spec: FieldSpec) -> str:
    """The Python annotation used in request/response models."""

    if field_spec.type == "int":
        return "int"
    if field_spec.type == "bool":
        return "bool"
    if field_spec.type == "enum":
        return "Literal[" + ", ".join(json.dumps(choice) for choice in field_spec.choices) + "]"
    return "str"


def header_safe(value: str) -> str:
    """A string usable inside a docstring or SQL identifier context."""

    return re.sub(r"[^A-Za-z0-9 _.:/-]+", "", value or "")


@dataclass
class BlueprintOptions:
    """Everything the generator needs beyond the blueprint itself."""

    name: str = ""
    title: str = ""
    description: str = ""
    port: int = 8000
    accent: str = ""
    auth: bool | None = None
    style: str = "tasteful"
    auth_mode: str = "good"
    entity_mode: str = "complete"
    author: str = "you"
    year: str = "2026"
    license: str = "MIT"
    warnings: list[str] = field(default_factory=list)

    def module(self, blueprint: AppBlueprint) -> str:
        return module_name(self.name or blueprint.id.replace("-", "_"))


@dataclass
class AppProject:
    """A rendered project: path -> content, plus the commands that prove it."""

    blueprint: AppBlueprint
    options: BlueprintOptions
    module: str
    files: dict[str, str] = field(default_factory=dict)
    test_command: str = "python -m pytest -q"
    run_command: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def root_files(self) -> list[str]:
        return sorted(self.files)

    @property
    def prompt(self) -> str:
        return self.blueprint.prompt

    def write(self, root: str | Path, *, force: bool = False) -> list[str]:
        """Write the project to ``root`` and return the paths written."""

        target_root = Path(root)
        target_root.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for rel, content in sorted(self.files.items()):
            path = target_root / rel
            if path.exists() and not force:
                self.warnings.append(f"kept existing {rel}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            written.append(rel)
        return written

    def to_dict(self) -> dict:
        return {
            "blueprint": self.blueprint.to_dict(),
            "module": self.module,
            "root_files": self.root_files,
            "file_count": len(self.files),
            "test_command": self.test_command,
            "run_command": self.run_command,
            "options": {
                "name": self.options.name,
                "title": self.options.title,
                "style": self.options.style,
                "auth_mode": self.options.auth_mode,
                "entity_mode": self.options.entity_mode,
                "port": self.options.port,
            },
            "warnings": list(self.warnings),
        }

    def render_tree(self, max_files: int = 32, max_chars: int = 12000) -> str:
        """A budgeted ``### path`` tree used by the dataset rows."""

        parts: list[str] = []
        budget = max_chars
        for path in sorted(self.files)[:max_files]:
            if budget <= 0:
                break
            content = self.files[path]
            chunk = "### " + path + "\n```\n" + content[: max(600, budget)] + "\n```\n"
            budget -= len(chunk)
            parts.append(chunk)
        if len(self.files) > max_files:
            parts.append("... " + str(len(self.files) - max_files) + " more files\n")
        return "".join(parts)




def _column(spec: FieldSpec) -> str:
    column = "    " + spec.name + " " + sql_type(spec)
    if spec.type == "ref":
        column += " NOT NULL REFERENCES " + _ref_table_placeholder(spec) + "(id) ON DELETE " + spec.on_delete.upper()
    else:
        if spec.required:
            column += " NOT NULL"
        if spec.type == "bool":
            column += " NOT NULL DEFAULT " + ("1" if spec.default else "0")
        elif spec.default not in (None, ""):
            if spec.type == "int":
                column += " DEFAULT " + str(spec.default)
            else:
                column += " DEFAULT '" + str(spec.default) + "'"
    if spec.type == "enum" and spec.choices:
        column += " CHECK (" + spec.name + " IN (" + ", ".join("'" + choice + "'" for choice in spec.choices) + "))"
    if spec.type == "bool":
        column += " CHECK (" + spec.name + " IN (0, 1))"
    return column


def _ref_table_placeholder(spec: FieldSpec) -> str:
    return "__REF_" + spec.ref + "__"


def _schema_for(blueprint: AppBlueprint, auth: bool) -> str:
    tables: list[str] = []
    if auth:
        tables.append(
            "CREATE TABLE IF NOT EXISTS users (\n"
            "    id TEXT PRIMARY KEY,\n"
            "    email TEXT NOT NULL UNIQUE,\n"
            "    password_hash TEXT NOT NULL,\n"
            "    created_at TEXT NOT NULL\n"
            ");\n"
        )
    if blueprint.feature == "team":
        tables.append(
            "CREATE TABLE IF NOT EXISTS shares (\n"
            "    id TEXT PRIMARY KEY,\n"
            "    resource TEXT NOT NULL,\n"
            "    resource_id TEXT NOT NULL,\n"
            "    owner_id TEXT NOT NULL,\n"
            "    email TEXT NOT NULL,\n"
            "    created_at TEXT NOT NULL\n"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_shares_lookup ON shares(resource, resource_id);\n"
            "CREATE INDEX IF NOT EXISTS idx_shares_email ON shares(email);\n"
        )
    for entity in blueprint.entities:
        columns = ["    id TEXT PRIMARY KEY"]
        columns.append(
            "    user_id TEXT NOT NULL DEFAULT 'demo'"
            if not auth
            else "    user_id TEXT NOT NULL"
        )
        for spec in entity.fields:
            columns.append(_column(spec).replace(_ref_table_placeholder(spec), _table_for_ref(blueprint, spec.ref)))
        columns.append("    created_at TEXT NOT NULL")
        columns.append("    updated_at TEXT NOT NULL")
        body = ",\n".join(columns)
        tables.append(
            "CREATE TABLE IF NOT EXISTS " + entity.table + " (\n" + body + "\n);\n"
        )
        tables.append(
            "CREATE INDEX IF NOT EXISTS idx_" + entity.table + "_user ON " + entity.table + "(user_id);\n"
        )
        tables.append(
            "CREATE INDEX IF NOT EXISTS idx_" + entity.table + "_created ON " + entity.table + "(created_at);\n"
        )
        for spec in entity.relations:
            tables.append(
                "CREATE INDEX IF NOT EXISTS idx_"
                + entity.table
                + "_"
                + spec.name
                + " ON "
                + entity.table
                + "("
                + spec.name
                + ");\n"
            )
    return "".join(tables)


def _table_for_ref(blueprint: AppBlueprint, ref: str) -> str:
    entity = blueprint.entity(ref)
    if entity is None:
        raise ValueError(f"blueprint {blueprint.id} references unknown entity {ref!r}")
    return entity.table


def _index_statements(blueprint: AppBlueprint, auth: bool) -> list[str]:
    schema = _schema_for(blueprint, auth)
    return [line.strip() for line in schema.splitlines() if line.strip().upper().startswith(("CREATE INDEX", "CREATE TABLE"))]




def _schema_module(blueprint: AppBlueprint, auth: bool) -> str:
    fields = [spec for entity in blueprint.entities for spec in entity.fields]
    has_enum = any(spec.type == "enum" for spec in fields)
    has_date = any(spec.type == "date" for spec in fields)
    has_timestamp = any(spec.type == "timestamp" for spec in fields)
    lines: list[str] = [
        '"""Request and response models for ' + header_safe(blueprint.title) + '."""',
        "",
        "from __future__ import annotations",
        "",
    ]
    if has_date and has_timestamp:
        lines.append("from datetime import date, datetime")
    elif has_date:
        lines.append("from datetime import date")
    elif has_timestamp:
        lines.append("from datetime import datetime")
    typing_names = "Any, Literal" if has_enum else "Any"
    lines.append("from typing import " + typing_names)
    lines.append("")
    pydantic_names = "BaseModel, ConfigDict, Field"
    if has_date or has_timestamp or blueprint.feature == "team":
        pydantic_names += ", field_validator"
    lines.append("from pydantic import " + pydantic_names)
    lines.extend(["", ""])
    if has_date:
        lines.extend(
            [
                "def _iso_date(value: str | None) -> str | None:",
                '    """Normalise an ISO date, rejecting free text."""',
                '    if value in (None, ""):',
                "        return None",
                "    try:",
                "        date.fromisoformat(value)",
                "    except ValueError as exc:",
                '        raise ValueError("use an ISO date such as 2026-11-01") from exc',
                "    return value",
                "",
                "",
            ]
        )
    if has_timestamp:
        lines.extend(
            [
                "def _iso_datetime(value: str | None) -> str | None:",
                '    """Normalise an ISO timestamp (``Z`` accepted, offset preserved)."""',
                '    if value in (None, ""):',
                "        return None",
                '    text = str(value).replace("Z", "+00:00")',
                "    try:",
                "        datetime.fromisoformat(text)",
                "    except ValueError as exc:",
                '        raise ValueError("use an ISO timestamp such as 2026-11-01T09:00:00+00:00") from exc',
                "    return text",
                "",
                "",
            ]
        )
    for entity in blueprint.entities:
        model = pascal(entity.name)
        lines.append("class " + model + "In(BaseModel):")
        lines.append('    """Fields a client may send when creating a ' + entity.label.lower() + '."""')
        lines.append("")
        lines.append("    model_config = ConfigDict(extra=\"forbid\")")
        lines.append("")
        for spec in entity.fields:
            annotation = py_type(spec)
            if spec.type in {"ref"}:
                annotation = "str"
            if not spec.required:
                annotation = annotation + " | None"
            constraints: list[str] = []
            if spec.type in {"text", "longtext", "ref"}:
                minimum = spec.min_length or (1 if spec.required and spec.type != "longtext" else 0)
                if minimum:
                    constraints.append("min_length=" + str(minimum))
                maximum = spec.max_length or {"text": 500, "longtext": 20000, "ref": 64}[spec.type]
                constraints.append("max_length=" + str(maximum))
            if spec.type == "int":
                if spec.min_value is not None:
                    constraints.append("ge=" + str(spec.min_value))
                if spec.max_value is not None:
                    constraints.append("le=" + str(spec.max_value))
            if spec.required and spec.default in (None, "") and not constraints:
                value = "Field(...)"
            elif constraints:
                value = "Field(" + ", ".join(constraints) + ")"
            elif spec.default not in (None, ""):
                value = repr(spec.default)
            else:
                value = "None"
            lines.append("    " + spec.name + ": " + annotation + " = " + value)
        lines.append("")
        date_fields = [spec for spec in entity.fields if spec.type in {"date", "timestamp"}]
        for spec in date_fields:
            helper = "_iso_date" if spec.type == "date" else "_iso_datetime"
            lines.append("    @field_validator(\"" + spec.name + "\")")
            lines.append("    @classmethod")
            lines.append("    def _check_" + spec.name + "(cls, value: str | None) -> str | None:")
            lines.append('        """Reject values that are not ISO ' + spec.type + 's."""')
            lines.append("        return " + helper + "(value)")
            lines.append("")
        if blueprint.feature == "schedule" and entity.name == _feature_entity(blueprint).name:
            lines.extend(
                [
                    "    @field_validator(\"end_at\")",
                    "    @classmethod",
                    "    def _check_window(cls, value: str | None, info: Any) -> str | None:",
                    '        """Require the window to have a positive length."""',
                    "        start = info.data.get(\"start_at\")",
                    "        if start and value and str(value) <= str(start):",
                    '            raise ValueError("the end time must be after the start time")',
                    "        return value",
                    "",
                ]
            )
        lines.append("")
        lines.append("class " + model + "Patch(BaseModel):")
        lines.append('    """Every field is optional: PATCH changes only what it sends."""')
        lines.append("")
        lines.append("    model_config = ConfigDict(extra=\"forbid\")")
        lines.append("")
        for spec in entity.fields:
            patch_type = py_type(spec) + " | None"
            constraints: list[str] = []
            if spec.type in {"text", "longtext", "ref"}:
                minimum = spec.min_length or (1 if spec.type != "longtext" else 0)
                if minimum:
                    constraints.append("min_length=" + str(minimum))
                maximum = spec.max_length or {"text": 500, "longtext": 20000, "ref": 64}[spec.type]
                constraints.append("max_length=" + str(maximum))
            if spec.type == "int":
                if spec.min_value is not None:
                    constraints.append("ge=" + str(spec.min_value))
                if spec.max_value is not None:
                    constraints.append("le=" + str(spec.max_value))
            if constraints:
                lines.append("    " + spec.name + ": " + patch_type + " = Field(default=None, " + ", ".join(constraints) + ")")
            else:
                lines.append("    " + spec.name + ": " + patch_type + " = None")
        for spec in entity.fields:
            if spec.type not in {"date", "timestamp"}:
                continue
            helper = "_iso_date" if spec.type == "date" else "_iso_datetime"
            lines.append("")
            lines.append("    @field_validator(\"" + spec.name + "\")")
            lines.append("    @classmethod")
            lines.append("    def _patch_" + spec.name + "(cls, value: str | None) -> str | None:")
            lines.append('        """Reject values that are not ISO ' + spec.type + 's."""')
            lines.append("        return " + helper + "(value)")
        lines.append("")
        lines.append("")
        lines.append("class " + model + "Out(BaseModel):")
        lines.append('    """A ' + entity.label.lower() + ' as returned by the API."""')
        lines.append("")
        lines.append("    model_config = ConfigDict(from_attributes=True)")
        lines.append("")
        lines.append("    id: str")
        lines.append("    user_id: str")
        for spec in entity.fields:
            out_type = py_type(spec)
            optional = (not spec.required) or spec.type == "bool"
            if optional and spec.type != "enum":
                out_type = out_type + " | None"
            lines.append("    " + spec.name + ": " + out_type)
        for spec in entity.relations:
            lines.append("    " + spec.name.replace("_id", "") + "_label: str | None = None")
        lines.append("    created_at: str")
        lines.append("    updated_at: str")
        lines.append("")
        lines.append("")
        lines.append("class " + model + "StatsOut(BaseModel):")
        lines.append('    """Aggregates the dashboard renders, computed in SQL."""')
        lines.append("")
        lines.append("    total: int")
        lines.append("    group_field: str")
        lines.append("    by_group: list[dict[str, Any]]")
        lines.append("    trend: list[dict[str, Any]]")
        lines.append("    sum_field: str | None = None")
        lines.append("    sum_value: int | None = None")
        lines.append("")
        lines.append("")
        if blueprint.feature == "board" and entity.name == _feature_entity(blueprint).name:
            status_field = _status_field(entity)
            lines.append("class " + model + "StatusIn(BaseModel):")
            lines.append('    """The one field the board sends when a card moves."""')
            lines.append("")
            lines.append("    model_config = ConfigDict(extra=\"forbid\")")
            lines.append("")
            lines.append("    " + status_field.name + ": " + py_type(status_field))
            lines.append("")
            lines.append("")
    if blueprint.feature == "team":
        lines.extend(
            [
                "class ShareIn(BaseModel):",
                '    """Email address of the teammate to share with."""',
                "",
                "    model_config = ConfigDict(extra=\"forbid\")",
                "",
                "    email: str = Field(min_length=3, max_length=254)",
                "",
                "    @field_validator(\"email\")",
                "    @classmethod",
                "    def _valid_email(cls, value: str) -> str:",
                '        """Require the shape of an email address without a network check."""',
                "        text = value.strip().lower()",
                "        if \"@\" not in text or \".\" not in text.split(\"@\")[-1]:",
                '            raise ValueError("enter a valid email address")',
                "        return text",
                "",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _status_field(entity: EntitySpec) -> FieldSpec:
    field_spec = entity.field("status") or next(
        (item for item in entity.fields if item.type == "enum"), entity.fields[0]
    )
    return field_spec




def _feature_entity(blueprint: AppBlueprint) -> EntitySpec:
    """The entity a blueprint's non-trivial feature acts on."""

    if blueprint.feature == "schedule":
        for entity in blueprint.entities:
            if entity.field("start_at") and entity.field("end_at"):
                return entity
    if blueprint.feature == "board":
        for entity in blueprint.entities:
            if entity.field(blueprint.feature_field or "status"):
                return entity
    return blueprint.entities[0]


def _group_field(blueprint: AppBlueprint, entity: EntitySpec) -> str:
    if entity.field(blueprint.feature_field or ""):
        return blueprint.feature_field
    for spec in entity.fields:
        if spec.type == "enum":
            return spec.name
    return "created_at"


def _sum_field(entity: EntitySpec) -> str:
    for spec in entity.fields:
        if spec.type == "int":
            return spec.name
    return ""


def _search_fields(entity: EntitySpec) -> list[str]:
    text_fields = [spec.name for spec in entity.fields if spec.type in {"text", "longtext"}]
    preferred = [name for name in entity.searchable if name in text_fields]
    if preferred:
        return preferred
    if text_fields:
        return text_fields[:2]
    fallback = [name for name in entity.searchable if entity.field(name)]
    return fallback[:2] or [entity.display]


def _select_and_joins(entity: EntitySpec, blueprint: AppBlueprint) -> tuple[str, str]:
    select = "t.*"
    joins: list[str] = []
    for index, spec in enumerate(entity.relations):
        target = blueprint.entity(spec.ref)
        if target is None:
            continue
        alias = "r" + str(index)
        label = spec.name.replace("_id", "") + "_label"
        select += ", " + alias + "." + target.display + " AS " + label
        joins.append("LEFT JOIN " + target.table + " " + alias + " ON " + alias + ".id = t." + spec.name)
    return select, " ".join(joins)


def _store_module(blueprint: AppBlueprint, auth: bool, options: BlueprintOptions) -> str:
    team = blueprint.feature == "team"
    primary = _feature_entity(blueprint)
    needs_validation = (
        blueprint.feature in {"booking", "schedule"}
        or any(entity.relations for entity in blueprint.entities)
        or any(
            spec.required and spec.default is None and spec.type in {"text", "longtext"}
            for entity in blueprint.entities
            for spec in entity.fields
        )
    )
    lines: list[str] = [
        '"""Viewer-scoped persistence for ' + header_safe(blueprint.title) + '."""',
        "",
        "from __future__ import annotations",
        "",
        "import csv",
        "import io",
        "import uuid",
        "from datetime import datetime, timezone",
        "from typing import Any",
        "",
        "from . import db",
    ]
    if blueprint.feature in {"booking", "schedule"}:
        lines.append("from .errors import ConflictError, NotFoundError, ValidationProblem")
    elif needs_validation:
        lines.append("from .errors import NotFoundError, ValidationProblem")
    else:
        lines.append("from .errors import NotFoundError")
    lines.extend([
        "",
        "",
        "def _now() -> str:",
        '    """UTC timestamp used for created_at and updated_at."""',
        "    return datetime.now(timezone.utc).isoformat()",
        "",
        "",
        "def _like(query: str | None) -> str:",
        '    """SQL LIKE pattern for a search box (user input stays a parameter)."""',
        '    return "%" + (query or "").strip() + "%"',
        "",
        "",
    ])
    for entity in blueprint.entities:
        sortable = ["created_at", "updated_at"] + [
            spec.name for spec in entity.fields if spec.type != "longtext"
        ]
        lines.append("SORTABLE_" + upper(entity.name) + " = frozenset(" + repr(sorted(set(sortable))) + ")")
    lines.append("")

    lines.append("")
    lines.append("def _scope(viewer: dict[str, str], alias: str = \"t\") -> tuple[str, list[Any]]:")
    if team:
        lines.append('    """Rows the viewer may read or write: their own, plus shares."""')
        lines.append("    prefix = alias + \".\" if alias else \"\"")
        lines.append(
            "    predicate = (prefix + \"user_id = ? OR \" + prefix + \"id IN (SELECT resource_id FROM shares WHERE resource = '"
            + primary.table
            + "' AND email = ?)\")"
        )
        lines.append("    return \"(\" + predicate + \")\", [viewer[\"id\"], viewer[\"email\"]]")
    elif auth:
        lines.append('    """Predicate limiting rows to records the viewer owns."""')
        lines.append("    prefix = alias + \".\" if alias else \"\"")
        lines.append("    return prefix + \"user_id = ?\", [viewer[\"id\"]]")
    else:
        lines.append('    """Single-user mode: every row belongs to the demo owner."""')
        lines.append("    return \"1 = 1\", []")
    lines.extend(
        [
            "",
            "",
            "def _owner_scope(alias: str = \"\") -> str:",
            '    """Ownership predicate (sharing grants editing, never deletion)."""',
            "    prefix = alias + \".\" if alias else \"\"",
            "    return prefix + \"user_id = ?\"",
            "",
            "",
            "def _require_parent(viewer: dict[str, str], table: str, parent_id: str, label: str) -> None:",
            '    """Raise NotFoundError when a referenced row is absent or not visible."""',
            "    scope, params = _scope(viewer, \"p\")",
            "    rows = db.query(\"SELECT p.id FROM \" + table + \" p WHERE p.id = ? AND \" + scope, [parent_id, *params])",
            "    if not rows:",
            "        raise NotFoundError(label + \" not found\")",
            "",
            "",
        ]
    )

    if auth:
        lines.extend(
            [
                "def create_user(email: str, password_hash: str) -> dict[str, Any]:",
                '    """Insert a user with a normalised email and return the record."""',
                "    user = {\"id\": str(uuid.uuid4()), \"email\": email.strip().lower(), \"password_hash\": password_hash, \"created_at\": _now()}",
                "    db.insert(\"users\", user)",
                "    return user",
                "",
                "",
                "def get_user(user_id: str) -> dict[str, Any] | None:",
                '    """Fetch one user by id."""',
                "    return db.query_one(\"SELECT * FROM users WHERE id = ?\", [user_id])",
                "",
                "",
                "def get_user_by_email(email: str) -> dict[str, Any] | None:",
                '    """Fetch one user by case-insensitive email."""',
                "    return db.query_one(\"SELECT * FROM users WHERE email = ?\", [(email or \"\").strip().lower()])",
                "",
                "",
            ]
        )

    if blueprint.feature == "schedule":
        booking = _feature_entity(blueprint)
        relation = booking.relations[0] if booking.relations else None
        resource_table = _table_for_ref(blueprint, relation.ref) if relation else booking.table
        resource_column = relation.name if relation else "id"
        lines.extend(
            [
                "def _check_window(instance_id: str | None, resource_id: str | None, start_at: str | None, end_at: str | None) -> None:",
                '    """Reject empty windows, reversed windows and double bookings."""',
                "    if not resource_id or not start_at or not end_at:",
                "        raise ValidationProblem(\"Choose a resource, a start time and an end time.\")",
                "    if str(end_at) <= str(start_at):",
                "        raise ValidationProblem(\"The end time must be after the start time.\")",
                "    rows = db.query(",
                "        \"SELECT id, start_at, end_at FROM "
                + booking.table
                + " WHERE "
                + resource_column
                + " = ? AND start_at < ? AND end_at > ? AND id != ? LIMIT 1\",",
                "        [resource_id, end_at, start_at, instance_id or \"\"],",
                "    )",
                "    if rows:",
                "        raise ConflictError(\"That slot overlaps an existing booking (\" + str(rows[0][\"start_at\"]) + \" to \" + str(rows[0][\"end_at\"]) + \").\")",
                "",
                "",
            ]
        )

    for entity in blueprint.entities:
        model = pascal(entity.name)
        plural = entity.table
        lines.extend(_cleaner(entity, blueprint))
        lines.extend(_read_functions(entity, blueprint))
        lines.extend(_write_functions(entity, blueprint, auth, options))
        lines.extend(_stats_function(entity, blueprint))
        lines.extend(_csv_function(entity, blueprint))

    if team:
        target = primary
        target_model = pascal(target.name)
        lines.extend(
            [
                "def share_" + target.name + "(viewer: dict[str, str], item_id: str, email: str) -> dict[str, Any]:",
                '    """Grant a teammate read/write access to a ' + target.label.lower() + ' the viewer owns."""',
                "    row = db.query_one(\"SELECT id FROM " + target.table + " WHERE id = ? AND user_id = ?\", [item_id, viewer[\"id\"]])",
                "    if row is None:",
                "        raise NotFoundError(\"" + target.label + " not found\")",
                "    target = (email or \"\").strip().lower()",
                "    existing = db.query_one(\"SELECT id FROM shares WHERE resource = '"
                + target.table
                + "' AND resource_id = ? AND email = ?\", [item_id, target])",
                "    if existing is None:",
                "        db.insert(\"shares\", {\"id\": str(uuid.uuid4()), \"resource\": \""
                + target.table
                + "\", \"resource_id\": item_id, \"owner_id\": viewer[\"id\"], \"email\": target, \"created_at\": _now()})",
                "    return {\"" + target.name + "_id\": item_id, \"shared_with\": target}",
                "",
                "",
                "def shared_" + plural + "(viewer: dict[str, str]) -> list[dict[str, Any]]:",
                '    """' + target.label + 's other people have shared with the viewer."""',
                "    statement = (\"SELECT t.* FROM " + target.table + " t JOIN shares s ON s.resource_id = t.id \"",
                "                 \"WHERE s.resource = '" + target.table + "' AND s.email = ? AND s.owner_id != ? \"",
                "                 \"ORDER BY t.updated_at DESC\")",
                "    return db.query(statement, [viewer[\"email\"], viewer[\"id\"]])",
                "",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _cleaner(entity: EntitySpec, blueprint: AppBlueprint) -> list[str]:
    model = pascal(entity.name)
    lines = [
        "def _clean_" + entity.name + "(payload: dict[str, Any], partial: bool = False) -> dict[str, Any]:",
        '    """Copy the fields a ' + entity.label.lower() + ' accepts out of a validated payload."""',
        "    out: dict[str, Any] = {}",
    ]
    for spec in entity.fields:
        key = spec.name
        block: list[str] = []
        if spec.type in {"text", "longtext", "ref", "date", "timestamp"}:
            if spec.required:
                block = [
                    "value = payload.get(\"" + key + "\")",
                    "out[\"" + key + "\"] = str(value).strip() if value is not None else \"\"",
                    "if not out[\"" + key + "\"]:",
                    "    raise ValidationProblem(\"" + spec.display + " is required.\")",
                ]
            else:
                block = [
                    "value = payload.get(\"" + key + "\")",
                    "out[\"" + key + "\"] = str(value).strip() if isinstance(value, str) else value",
                ]
        elif spec.type == "int":
            if spec.required:
                block = [
                    "value = payload.get(\"" + key + "\")",
                    "out[\"" + key + "\"] = int(value) if value is not None else 0",
                ]
            else:
                block = [
                    "value = payload.get(\"" + key + "\")",
                    "out[\"" + key + "\"] = None if value is None else int(value)",
                ]
        elif spec.type == "bool":
            block = ["out[\"" + key + "\"] = 1 if payload.get(\"" + key + "\") else 0"]
        elif spec.type == "enum":
            fallback = spec.default or (spec.choices[0] if spec.choices else "")
            block = ["out[\"" + key + "\"] = str(payload.get(\"" + key + "\") or \"" + str(fallback) + "\")"]
        if not block:
            continue
        lines.append("    if \"" + key + "\" in payload or not partial:")
        lines.extend("        " + line for line in block)
    lines.extend(["    return out", ""])
    _ = model
    return lines


def _read_functions(entity: EntitySpec, blueprint: AppBlueprint) -> list[str]:
    select, joins = _select_and_joins(entity, blueprint)
    relation_args = [spec for spec in entity.relations]
    filter_args = "".join(", " + spec.name + ": str | None = None" for spec in relation_args)
    lines = [
        "def list_" + entity.table + "(viewer: dict[str, str], q: str | None = None, limit: int = 50, offset: int = 0, sort: str = \"-created_at\"" + filter_args + ") -> list[dict[str, Any]]:",
        '    """List ' + entity.table + ' visible to the viewer, newest first by default."""',
        "    column = sort.lstrip(\"-\")",
        "    if column not in SORTABLE_" + upper(entity.name) + ":",
        "        column = \"created_at\"",
        "    direction = \"DESC\" if sort.startswith(\"-\") else \"ASC\"",
        "    where, params = _scope(viewer, \"t\")",
    ]
    searchable = _search_fields(entity)
    if searchable:
        clauses = " OR ".join("t." + name + " LIKE ?" for name in searchable)
        lines.extend(
            [
                "    if q:",
                "        pattern = _like(q)",
                "        where += \" AND (\" + " + repr(clauses) + " + \")\"",
                "        params.extend([pattern] * " + str(len(searchable)) + ")",
            ]
        )
    for spec in relation_args:
        lines.extend(
            [
                "    if " + spec.name + ":",
                "        where += \" AND t." + spec.name + " = ?\"",
                "        params.append(" + spec.name + ")",
            ]
        )
    statement = (
        "\"SELECT " + select + " FROM " + entity.table + " t " + joins + " WHERE \" + where + "
        "\" ORDER BY t.\" + column + \" \" + direction + \" LIMIT ? OFFSET ?\""
    )
    lines.extend(
        [
            "    statement = " + statement,
            "    window = min(max(int(limit), 1), 200)",
            "    return db.query(statement, [*params, window, max(int(offset), 0)])",
            "",
            "",
        ]
    )
    lines.extend(
        [
            "def get_" + entity.name + "(viewer: dict[str, str], item_id: str) -> dict[str, Any] | None:",
            '    """Fetch one ' + entity.label.lower() + ' visible to the viewer."""',
            "    where, params = _scope(viewer, \"t\")",
            "    statement = \"SELECT " + select + " FROM " + entity.table + " t " + joins + " WHERE t.id = ? AND \" + where",
            "    rows = db.query(statement, [item_id, *params])",
            "    return rows[0] if rows else None",
            "",
            "",
        ]
    )
    return lines


def _write_functions(
    entity: EntitySpec, blueprint: AppBlueprint, auth: bool, options: BlueprintOptions
) -> list[str]:
    lines: list[str] = []
    is_schedule = blueprint.feature == "schedule" and entity is _feature_entity(blueprint)
    relation = entity.relations[0] if entity.relations else None
    lines.append("def create_" + entity.name + "(viewer: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:")
    lines.append('    """Create a ' + entity.label.lower() + ' for the viewer after checking its parent."""')
    lines.append("    values = _clean_" + entity.name + "(payload)")
    if relation:
        target = blueprint.entity(relation.ref)
        lines.extend(
            [
                "    parent_id = values.get(\"" + relation.name + "\")",
                "    if not parent_id:",
                "        raise ValidationProblem(\"Choose a " + (target.label if target else relation.ref) + " for this " + entity.label.lower() + ".\")",
            ]
        )
        if options.entity_mode != "incomplete":
            lines.append("    _require_parent(viewer, \"" + (target.table if target else relation.ref) + "\", parent_id, \"" + (target.label if target else relation.ref) + "\")")
    if is_schedule:
        rel_name = relation.name if relation else "id"
        lines.append("    _check_window(None, values.get(\"" + rel_name + "\"), values.get(\"start_at\"), values.get(\"end_at\"))")
    lines.extend(
        [
            "    row = {\"id\": str(uuid.uuid4()), \"user_id\": viewer[\"id\"], **values, \"created_at\": _now(), \"updated_at\": _now()}",
            "    db.insert(\"" + entity.table + "\", row)",
            "    created = get_" + entity.name + "(viewer, row[\"id\"])",
            "    if created is None:",
            "        raise NotFoundError(\"" + entity.label + " not found\")",
            "    return created",
            "",
            "",
            "def update_" + entity.name + "(viewer: dict[str, str], item_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:",
            '    """Update only the supplied fields, keeping relations and windows valid."""',
            "    current = get_" + entity.name + "(viewer, item_id)",
            "    if current is None:",
            "        return None",
            "    values = _clean_" + entity.name + "(payload, partial=True)",
        ]
    )
    if relation:
        target = blueprint.entity(relation.ref)
        lines.extend(
            [
                "    if \"" + relation.name + "\" in values:",
                "        if not values[\"" + relation.name + "\"]:",
                "            raise ValidationProblem(\"Choose a " + (target.label if target else relation.ref) + " for this " + entity.label.lower() + ".\")",
            ]
        )
        if options.entity_mode != "incomplete":
            lines.append("        _require_parent(viewer, \"" + (target.table if target else relation.ref) + "\", values[\"" + relation.name + "\"], \"" + (target.label if target else relation.ref) + "\")")
    if is_schedule:
        rel_name = relation.name if relation else "id"
        lines.extend(
            [
                "    start_at = values.get(\"start_at\", current.get(\"start_at\"))",
                "    end_at = values.get(\"end_at\", current.get(\"end_at\"))",
                "    resource_id = values.get(\"" + rel_name + "\", current.get(\"" + rel_name + "\"))",
                "    _check_window(item_id, resource_id, start_at, end_at)",
            ]
        )
    lines.extend(
        [
            "    values[\"updated_at\"] = _now()",
            "    scope, params = _scope(viewer, \"\")",
            "    changed = db.update_row(\"" + entity.table + "\", item_id, values, scope, params)",
            "    if not changed:",
            "        return None",
            "    return get_" + entity.name + "(viewer, item_id)",
            "",
            "",
            "def delete_" + entity.name + "(viewer: dict[str, str], item_id: str) -> bool:",
            '    """Delete a ' + entity.label.lower() + ' the viewer owns."""',
            "    changed = db.delete_row(\"" + entity.table + "\", item_id, _owner_scope(), [viewer[\"id\"]])",
            "    return changed > 0",
            "",
            "",
        ]
    )
    return lines


def _stats_function(entity: EntitySpec, blueprint: AppBlueprint) -> list[str]:
    group = _group_field(blueprint, entity)
    sum_field = _sum_field(entity)
    lines = [
        "def " + entity.name + "_stats(viewer: dict[str, str]) -> dict[str, Any]:",
        '    """Totals, group counts and a 14-day trend for the dashboard."""',
        "    where, params = _scope(viewer, \"t\")",
        "    total_row = db.query_one(\"SELECT COUNT(*) AS total FROM " + entity.table + " t WHERE \" + where, params)",
        "    group_rows = db.query(",
        "        \"SELECT t." + group + " AS key, COUNT(*) AS count FROM " + entity.table + " t WHERE \" + where",
        "        + \" GROUP BY t." + group + " ORDER BY count DESC\",",
        "        params,",
        "    )",
        "    trend_rows = db.query(",
        "        \"SELECT substr(t.created_at, 1, 10) AS day, COUNT(*) AS count FROM " + entity.table + " t WHERE \" + where",
        "        + \" GROUP BY day ORDER BY day DESC LIMIT 14\",",
        "        params,",
        "    )",
    ]
    if sum_field:
        lines.extend(
            [
                "    sum_row = db.query_one(\"SELECT COALESCE(SUM(t." + sum_field + "), 0) AS total FROM " + entity.table + " t WHERE \" + where, params)",
                "    sum_value = int((sum_row or {}).get(\"total\") or 0)",
            ]
        )
    else:
        lines.append("    sum_value = None")
    lines.extend(
        [
            "    return {",
            "        \"total\": int((total_row or {}).get(\"total\") or 0),",
            "        \"group_field\": \"" + group + "\",",
            "        \"by_group\": group_rows,",
            "        \"trend\": trend_rows,",
            "        \"sum_field\": " + (repr(sum_field) if sum_field else "None") + ",",
            "        \"sum_value\": sum_value,",
            "    }",
            "",
            "",
        ]
    )
    return lines


def _csv_function(entity: EntitySpec, blueprint: AppBlueprint) -> list[str]:
    columns = ["id"] + [spec.name for spec in entity.fields] + ["created_at", "updated_at"]
    lines = [
        "CSV_COLUMNS_" + upper(entity.name) + " = " + repr(columns),
        "",
        "",
        "def export_" + entity.name + "_csv(viewer: dict[str, str]) -> str:",
        '    """Render the viewer\'s ' + entity.table + ' as CSV for a spreadsheet."""',
        "    rows = list_" + entity.table + "(viewer, limit=200)",
        "    buffer = io.StringIO()",
        "    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS_" + upper(entity.name) + ", extrasaction=\"ignore\")",
        "    writer.writeheader()",
        "    writer.writerows(rows)",
        "    return buffer.getvalue()",
        "",
        "",
    ]
    return lines




def _api_module(blueprint: AppBlueprint, auth: bool) -> str:
    lines: list[str] = [
        '"""HTTP API for ' + header_safe(blueprint.title) + '."""',
        "",
        "from __future__ import annotations",
        "",
        "from fastapi import APIRouter, Depends, Query, Response, status",
        "",
    ]
    lines.append("from . import store")
    if not auth:
        lines.insert(len(lines) - 1, "from . import config")
    lines.append("from .errors import NotFoundError")
    for entity in blueprint.entities:
        model = pascal(entity.name)
        names = [model + "In", model + "Patch", model + "Out", model + "StatsOut"]
        if blueprint.feature == "board" and entity is _feature_entity(blueprint):
            names.append(model + "StatusIn")
        lines.append("from .schemas import " + ", ".join(names))
    if blueprint.feature == "team":
        lines.append("from .schemas import ShareIn")
    if auth:
        lines.append("from .auth_routes import current_viewer")
    lines.extend(["", "", 'router = APIRouter(prefix="/api", tags=["api"])', "", ""])
    if not auth:
        lines.extend(
            [
                "DEMO_VIEWER = {\"id\": config.DEMO_USER_ID, \"email\": config.DEMO_USER_EMAIL}",
                "",
                "",
                "def current_viewer() -> dict[str, str]:",
                '    """Single-user mode: every request acts as the demo owner."""',
                "    return dict(DEMO_VIEWER)",
                "",
                "",
            ]
        )
    lines.extend(
        [
            "@router.get(\"/me\")",
            "def me(viewer: dict[str, str] = Depends(current_viewer)) -> dict[str, str]:",
            '    """Return the signed-in viewer."""',
            "    return viewer",
            "",
            "",
        ]
    )
    for entity in blueprint.entities:
        model = pascal(entity.name)
        plural = entity.table
        lines.extend(
            [
                "@router.get(\"/" + plural + "/stats\", response_model=" + model + "StatsOut)",
                "def " + entity.name + "_stats(viewer: dict[str, str] = Depends(current_viewer)) -> dict:",
                '    """Dashboard totals, group counts and a 14-day trend."""',
                "    return store." + entity.name + "_stats(viewer)",
                "",
                "",
                "@router.get(\"/" + plural + "/export.csv\")",
                "def export_" + plural + "(viewer: dict[str, str] = Depends(current_viewer)) -> Response:",
                '    """Download the visible ' + plural + ' as CSV."""',
                "    return Response(",
                "        content=store.export_" + entity.name + "_csv(viewer),",
                "        media_type=\"text/csv\",",
                "        headers={\"Content-Disposition\": 'attachment; filename=\"" + plural + ".csv\"'},",
                "    )",
                "",
                "",
                "@router.get(\"/" + plural + "\", response_model=list[" + model + "Out])",
                "def list_" + plural + "(",
            ]
        )
        params = [
            "    q: str | None = None,",
            "    limit: int = Query(50, ge=1, le=200),",
            "    offset: int = Query(0, ge=0),",
            "    sort: str = \"-created_at\",",
        ]
        for spec in entity.relations:
            params.append("    " + spec.name + ": str | None = None,")
        params.append("    viewer: dict[str, str] = Depends(current_viewer),")
        lines.extend(params)
        lines.append(") -> list[dict]:")
        lines.append('    """List ' + plural + ' visible to the viewer."""')
        call = "    return store.list_" + plural + "(viewer, q=q, limit=limit, offset=offset, sort=sort"
        for spec in entity.relations:
            call += ", " + spec.name + "=" + spec.name
        lines.extend([call + ")", "", ""])
        lines.extend(
            [
                "@router.post(\"/" + plural + "\", response_model=" + model + "Out, status_code=status.HTTP_201_CREATED)",
                "def create_" + entity.name + "(payload: " + model + "In, viewer: dict[str, str] = Depends(current_viewer)) -> dict:",
                '    """Create one ' + entity.label.lower() + ' for the viewer."""',
                "    return store.create_" + entity.name + "(viewer, payload.model_dump(exclude_unset=True))",
                "",
                "",
                "@router.get(\"/" + plural + "/{item_id}\", response_model=" + model + "Out)",
                "def get_" + entity.name + "(item_id: str, viewer: dict[str, str] = Depends(current_viewer)) -> dict:",
                '    """Fetch one ' + entity.label.lower() + ' or return 404."""',
                "    row = store.get_" + entity.name + "(viewer, item_id)",
                "    if row is None:",
                "        raise NotFoundError(\"" + entity.label + " not found\")",
                "    return row",
                "",
                "",
                "@router.patch(\"/" + plural + "/{item_id}\", response_model=" + model + "Out)",
                "def update_" + entity.name + "(",
                "    item_id: str,",
                "    payload: " + model + "Patch,",
                "    viewer: dict[str, str] = Depends(current_viewer),",
                ") -> dict:",
                '    """Patch the supplied fields on one ' + entity.label.lower() + '."""',
                "    row = store.update_" + entity.name + "(viewer, item_id, payload.model_dump(exclude_unset=True))",
                "    if row is None:",
                "        raise NotFoundError(\"" + entity.label + " not found\")",
                "    return row",
                "",
                "",
                "@router.delete(\"/" + plural + "/{item_id}\", status_code=status.HTTP_204_NO_CONTENT)",
                "def delete_" + entity.name + "(item_id: str, viewer: dict[str, str] = Depends(current_viewer)) -> Response:",
                '    """Delete one ' + entity.label.lower() + ' the viewer owns."""',
                "    if not store.delete_" + entity.name + "(viewer, item_id):",
                "        raise NotFoundError(\"" + entity.label + " not found\")",
                "    return Response(status_code=status.HTTP_204_NO_CONTENT)",
                "",
                "",
            ]
        )
        if blueprint.feature == "board" and entity is _feature_entity(blueprint):
            status_field = _status_field(entity)
            lines.extend(
                [
                    "@router.patch(\"/" + plural + "/{item_id}/status\", response_model=" + model + "Out)",
                    "def move_" + entity.name + "(item_id: str, payload: " + model + "StatusIn, viewer: dict[str, str] = Depends(current_viewer)) -> dict:",
                    '    """Move one card to another column."""',
                    "    row = store.update_" + entity.name + "(viewer, item_id, payload.model_dump())",
                    "    if row is None:",
                    "        raise NotFoundError(\"" + entity.label + " not found\")",
                    "    return row",
                    "",
                    "",
                ]
            )
            _ = status_field
    if blueprint.feature == "team":
        primary = _feature_entity(blueprint)
        model = pascal(primary.name)
        lines.extend(
            [
                "@router.get(\"/shared\", response_model=list[" + model + "Out])",
                "def shared_" + primary.table + "(viewer: dict[str, str] = Depends(current_viewer)) -> list[dict]:",
                '    """' + primary.label + 's other people have shared with the viewer."""',
                "    return store.shared_" + primary.table + "(viewer)",
                "",
                "",
                "@router.post(\"/" + primary.table + "/{item_id}/share\")",
                "def share_" + primary.name + "(item_id: str, payload: ShareIn, viewer: dict[str, str] = Depends(current_viewer)) -> dict:",
                '    """Share one item with a teammate by email."""',
                "    return store.share_" + primary.name + "(viewer, item_id, payload.email)",
                "",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


AUTH_ROUTES_PY = '''"""Auth HTTP surface: register, login, me (scrypt + HS256, stdlib only)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from . import store
from .auth import AuthError, create_token, decode_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


class Credentials(BaseModel):
    """Email and password for register and login."""

    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=256)


def current_viewer(authorization: str = Header(default="")) -> dict[str, str]:
    """Resolve a bearer token to a viewer, or raise 401 with a plain reason."""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    try:
        payload = decode_token(token)
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc
    user = store.get_user(str(payload["sub"]))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "account no longer exists")
    return {"id": str(user["id"]), "email": str(user["email"])}


@router.post("/register", status_code=201)
def register(body: Credentials) -> dict:
    """Create an account and return a token for it."""
    if store.get_user_by_email(body.email):
        raise HTTPException(status.HTTP_409_CONFLICT, "email already registered")
    user = store.create_user(body.email, hash_password(body.password))
    return {"id": user["id"], "email": user["email"], "token": create_token(user["id"])}


@router.post("/login")
def login(body: Credentials) -> dict:
    """Exchange credentials for a token."""
    user = store.get_user_by_email(body.email)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
    return {"email": user["email"], "token": create_token(user["id"])}


@router.get("/me")
def me(viewer: dict[str, str] = Depends(current_viewer)) -> dict:
    """Return the signed-in viewer."""
    return viewer
'''

MAIN_PY = '''"""@@TITLE@@ - FastAPI application entry point."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, db
from .api import router
from .errors import AppError, ConflictError, ForbiddenError, NotFoundError, ValidationProblem
@@AUTH_IMPORT@@

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app() -> FastAPI:
    """Build the app: schema, routers, error mapping, static UI and health."""
    db.init_db()
    app = FastAPI(title=config.APP_NAME, version=config.APP_VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.DEV_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.include_router(router)
@@INCLUDE_AUTH@@
    @app.exception_handler(NotFoundError)
    def not_found(_request: Request, exc: NotFoundError) -> JSONResponse:
        """Map a missing or invisible record to 404 with a readable sentence."""
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ConflictError)
    def conflict(_request: Request, exc: ConflictError) -> JSONResponse:
        """Map a data clash (double booking, duplicate) to 409."""
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(ForbiddenError)
    def forbidden(_request: Request, exc: ForbiddenError) -> JSONResponse:
        """Map an action the viewer may not take to 403."""
        return JSONResponse({"detail": str(exc)}, status_code=403)

    @app.exception_handler(ValidationProblem)
    def invalid(_request: Request, exc: ValidationProblem) -> JSONResponse:
        """Map semantic validation failures to 422 with the same sentence."""
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(AppError)
    def app_error(_request: Request, exc: AppError) -> JSONResponse:
        """Last-resort mapping for expected application errors."""
        return JSONResponse({"detail": str(exc)}, status_code=getattr(exc, "status_code", 400))

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Liveness probe used by deploys and the build health check."""
        return {"status": "ok", "app": config.APP_NAME}

    if WEB_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            """Serve the application shell."""
            return FileResponse(WEB_DIR / "index.html")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            """Serve the shell for client-side routes (API routes match first)."""
            _ = path
            return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()
'''

CONFTEST_PY = '''"""Test environment: a private database and a fixed auth secret per run."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("AUTH_SECRET", "test-only-secret")
os.environ.setdefault("SQLITE_PATH", str(Path(tempfile.mkdtemp(prefix="app-test-")) / "app.db"))
'''




FIELD_SAMPLES = {
    "date": "2026-11-01",
    "timestamp": "2026-11-01T09:00:00+00:00",
}


def _sample_value(spec: FieldSpec) -> str:
    if spec.name in {"end_at", "ends_at", "end_time"}:
        return repr("2026-11-01T11:00:00+00:00")
    if spec.name in {"end_on", "ends_on", "end_date"}:
        return repr("2026-11-05")
    if spec.type == "date":
        return repr(FIELD_SAMPLES["date"])
    if spec.type == "timestamp":
        return repr(FIELD_SAMPLES["timestamp"])
    if spec.type == "int":
        if spec.max_value is not None and spec.max_value < 1:
            return str(spec.max_value)
        return str(spec.min_value if spec.min_value not in (None, 0) else 1)
    if spec.type == "bool":
        return "False"
    if spec.type == "enum":
        return repr(spec.default or (spec.choices[0] if spec.choices else ""))
    if spec.type == "ref":
        return repr("missing-parent")
    return repr("Sample " + spec.display.lower())


def _signin_helper(auth: bool) -> list[str]:
    if auth:
        return [
            "def sign_in(email: str | None = None) -> tuple[dict[str, str], str]:",
            '    """Register a fresh user and return (auth headers, email)."""',
            "    address = email or \"user-\" + uuid.uuid4().hex[:8] + \"@example.com\"",
            "    response = client.post(\"/auth/register\", json={\"email\": address, \"password\": PASSWORD})",
            "    assert response.status_code == 201, response.text",
            "    return {\"Authorization\": \"Bearer \" + response.json()[\"token\"]}, address",
            "",
            "",
        ]
    return [
        "def sign_in(email: str | None = None) -> tuple[dict[str, str], str]:",
        '    """Single-user mode: there is no account, only the demo viewer."""',
        "    _ = email",
        "    return {}, \"demo@example.com\"",
        "",
        "",
    ]


def _test_target(entity: EntitySpec) -> tuple[FieldSpec | None, str, str]:
    """Pick a field the CRUD tests can safely write twice."""
    for spec in entity.fields:
        if spec.type in {"text", "longtext"}:
            return spec, "First record", "Updated record"
    for spec in entity.fields:
        if spec.type == "enum" and len(spec.choices) >= 2:
            return spec, spec.choices[0], spec.choices[1]
    return None, "", ""


def _tests_api_module(blueprint: AppBlueprint, auth: bool) -> str:
    module = "__MODULE__"
    lines: list[str] = [
        '"""Behaviour tests for ' + header_safe(blueprint.title) + '."""',
        "",
        "from __future__ import annotations",
        "",
        "import uuid",
        "",
        "from fastapi.testclient import TestClient",
        "",
        "from " + module + ".main import app",
        "",
        "client = TestClient(app)",
        "PASSWORD = \"s3cret-passphrase\"",
        "",
        "",
    ]
    lines.extend(_signin_helper(auth))

    for entity in blueprint.entities:
        lines.append("def payload_" + entity.name + "(**overrides) -> dict:")
        lines.append('    """A valid ' + entity.label.lower() + ' payload; tests override one field at a time."""')
        lines.append("    data = {")
        for spec in entity.fields:
            lines.append("        " + repr(spec.name) + ": " + _sample_value(spec) + ",")
        lines.append("    }")
        lines.append("    data.update(overrides)")
        lines.append("    return data")
        lines.append("")
        lines.append("")

    for entity in blueprint.entities:
        relation_args = "".join(", " + spec.name + "=None" for spec in entity.relations)
        lines.append("def make_" + entity.name + "(headers: dict[str, str]" + relation_args + ", **overrides) -> dict:")
        lines.append('    """Create a ' + entity.label.lower() + ' through the API, creating parents when needed."""')
        lines.append("    data = payload_" + entity.name + "(**overrides)")
        for spec in entity.relations:
            target = blueprint.entity(spec.ref)
            lines.append("    if " + spec.name + " is None:")
            assert target is not None
            lines.append("        " + spec.name + " = make_" + target.name + "(headers)[\"id\"]")
            lines.append("    data[\"" + spec.name + "\"] = " + spec.name)
        lines.append("    response = client.post(\"/api/" + entity.table + "\", json=data, headers=headers)")
        lines.append("    assert response.status_code == 201, response.text")
        lines.append("    return response.json()")
        lines.append("")
        lines.append("")

    primary = blueprint.primary
    lines.extend(
        [
            "def test_healthz():",
            '    """The deploy health check has one job: answer 200 quickly."""',
            "    response = client.get(\"/healthz\")",
            "    assert response.status_code == 200",
            "    assert response.json()[\"status\"] == \"ok\"",
            "",
            "",
        ]
    )
    if auth:
        lines.extend(
            [
                "def test_register_login_and_me():",
                '    """Registration returns a token, login exchanges credentials, /me identifies."""',
                "    headers, email = sign_in()",
                "    me = client.get(\"/api/me\", headers=headers)",
                "    assert me.status_code == 200",
                "    assert me.json()[\"email\"] == email",
                "    login = client.post(\"/auth/login\", json={\"email\": email, \"password\": PASSWORD})",
                "    assert login.status_code == 200",
                "    assert login.json()[\"token\"]",
                "    wrong = client.post(\"/auth/login\", json={\"email\": email, \"password\": \"wrong-password\"})",
                "    assert wrong.status_code == 401",
                "",
                "",
                "def test_requires_token():",
                '    """Protected routes reject anonymous requests instead of leaking data."""',
                "    assert client.get(\"/api/" + primary.table + "\").status_code == 401",
                "",
                "",
                "def test_isolation_between_users():",
                '    """One user can never read, edit or delete another user\'s rows."""',
                "    alice, _ = sign_in()",
                "    bob, _ = sign_in()",
                "    created = make_" + primary.name + "(alice)",
                "    visible = client.get(\"/api/" + primary.table + "\", headers=bob).json()",
                "    assert all(row[\"id\"] != created[\"id\"] for row in visible)",
                "    assert client.get(\"/api/" + primary.table + "/\" + created[\"id\"], headers=bob).status_code == 404",
                "    assert client.patch(\"/api/" + primary.table + "/\" + created[\"id\"], json={}, headers=bob).status_code == 404",
                "    assert client.delete(\"/api/" + primary.table + "/\" + created[\"id\"], headers=bob).status_code == 404",
                "",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "def test_open_access():",
                '    """Single-user mode needs no token and still scopes rows to the demo owner."""',
                "    headers, email = sign_in()",
                "    assert headers == {}",
                "    assert email == \"demo@example.com\"",
                "    assert client.get(\"/api/" + primary.table + "\", headers=headers).status_code == 200",
                "",
                "",
            ]
        )
    target, first_value, second_value = _test_target(primary)
    required = next(
        (spec for spec in primary.fields if spec.required and spec.default is None),
        None,
    )
    lines.append("def test_create_edit_delete_" + primary.name + "():")
    lines.append('    """The core loop: create, see it listed, patch it, delete it."""')
    lines.append("    headers, _ = sign_in()")
    if target is None:
        lines.append("    created = make_" + primary.name + "(headers)")
    else:
        lines.append(
            "    created = make_"
            + primary.name
            + "(headers, "
            + target.name
            + "="
            + repr(first_value)
            + ")"
        )
    lines.extend(
        [
            "    assert created[\"id\"]",
            "    listed = client.get(\"/api/" + primary.table + "\", headers=headers)",
            "    assert listed.status_code == 200",
            "    assert any(row[\"id\"] == created[\"id\"] for row in listed.json())",
        ]
    )
    if target is not None:
        lines.extend(
            [
                "    updated = client.patch(",
                "        \"/api/" + primary.table + "/\" + created[\"id\"],",
                "        json={" + repr(target.name) + ": " + repr(second_value) + "},",
                "        headers=headers,",
                "    )",
                "    assert updated.status_code == 200",
                "    assert updated.json()[" + repr(target.name) + "] == " + repr(second_value),
            ]
        )
    lines.extend(
        [
            "    deleted = client.delete(\"/api/" + primary.table + "/\" + created[\"id\"], headers=headers)",
            "    assert deleted.status_code == 204",
            "    assert client.get(\"/api/" + primary.table + "/\" + created[\"id\"], headers=headers).status_code == 404",
            "",
            "",
            "def test_validation_rejects_bad_input():",
            '    """Missing required fields, unknown fields and blank text are 422."""',
            "    headers, _ = sign_in()",
        ]
    )
    if required is not None:
        lines.extend(
            [
                "    missing = payload_" + primary.name + "()",
                "    missing.pop(" + repr(required.name) + ")",
                "    assert client.post(\"/api/" + primary.table + "\", json=missing, headers=headers).status_code == 422",
            ]
        )
    lines.extend(
        [
            "    unknown = payload_" + primary.name + "()",
            '    unknown["is_admin"] = True',
            "    assert client.post(\"/api/" + primary.table + "\", json=unknown, headers=headers).status_code == 422",
        ]
    )
    if target is not None and target.type in {"text", "longtext"}:
        lines.extend(
            [
                "    blank = payload_" + primary.name + "(" + target.name + "=\"   \")",
                "    assert client.post(\"/api/" + primary.table + "\", json=blank, headers=headers).status_code == 422",
            ]
        )
    lines.append("")
    lines.append("")
    if target is not None and target.type in {"text", "longtext"}:
        lines.extend(
            [
                "def test_search_and_pagination():",
                '    """Search matches the text fields and pagination does not repeat rows."""',
                "    headers, _ = sign_in()",
                "    token = uuid.uuid4().hex[:8]",
                "    for index in range(3):",
                "        make_" + primary.name + "(headers, " + target.name + "=f\"{token} item {index}\")",
                "    found = client.get(\"/api/" + primary.table + "\", params={\"q\": token}, headers=headers).json()",
                "    assert len(found) == 3",
                "    page = client.get(",
                "        \"/api/" + primary.table + "\",",
                "        params={\"q\": token, \"limit\": 1, \"offset\": 1},",
                "        headers=headers,",
                "    ).json()",
                "    assert len(page) == 1",
                "    assert page[0][\"id\"] != found[0][\"id\"]",
                "",
                "",
            ]
        )
    lines.extend(
        [
            "def test_stats_endpoint():",
            '    """Aggregates are computed in SQL and exposed for the dashboard."""',
            "    headers, _ = sign_in()",
            "    make_" + primary.name + "(headers)",
            "    response = client.get(\"/api/" + primary.table + "/stats\", headers=headers)",
            "    assert response.status_code == 200",
            "    data = response.json()",
            '    assert data["total"] >= 1',
            '    assert data["group_field"]',
            '    assert isinstance(data["by_group"], list)',
            "",
            "",
        ]
    )
    for parent in blueprint.entities:
        for child in blueprint.entities:
            relation = next((spec for spec in child.relations if spec.ref == parent.name), None)
            if relation is None:
                continue
            target_display = parent.display
            child_display = child.display
            lines.extend(
                [
                    "def test_" + child.name + "_relation_and_cascade():",
                    '    """A child belongs to its parent, filters by it and cascades on delete."""',
                    "    headers, _ = sign_in()",
                    "    parent = make_" + parent.name + "(headers, " + target_display + "=\"Parent record\")",
                    "    child = make_" + child.name + "(headers, " + relation.name + "=parent[\"id\"], " + child_display + "=\"Child record\")",
                    "    filtered = client.get(",
                    "        \"/api/" + child.table + "\",",
                    "        params={\"" + relation.name + "\": parent[\"id\"]},",
                    "        headers=headers,",
                    "    ).json()",
                    "    assert any(row[\"id\"] == child[\"id\"] for row in filtered)",
                    "    assert filtered[0][\"" + relation.name.replace("_id", "") + "_label\"] == \"Parent record\"",
                    "    orphan = payload_" + child.name + "(" + relation.name + "=\"does-not-exist\")",
                    "    assert client.post(\"/api/" + child.table + "\", json=orphan, headers=headers).status_code == 404",
                    "    assert client.delete(\"/api/" + parent.table + "/\" + parent[\"id\"], headers=headers).status_code == 204",
                    "    assert client.get(\"/api/" + child.table + "/\" + child[\"id\"], headers=headers).status_code == 404",
                    "",
                    "",
                ]
            )
    feature_entity = _feature_entity(blueprint)
    if blueprint.feature == "board":
        status_field = _status_field(feature_entity)
        choices = list(status_field.choices)
        second = choices[1] if len(choices) > 1 else choices[0]
        lines.extend(
            [
                "def test_board_moves_a_card():",
                '    """Moving a card is validated against the column values."""',
                "    headers, _ = sign_in()",
                "    item = make_" + feature_entity.name + "(headers)",
                "    moved = client.patch(",
                "        \"/api/" + feature_entity.table + "/\" + item[\"id\"] + \"/status\",",
                "        json={" + repr(status_field.name) + ": " + repr(second) + "},",
                "        headers=headers,",
                "    )",
                "    assert moved.status_code == 200",
                "    assert moved.json()[" + repr(status_field.name) + "] == " + repr(second),
                "    rejected = client.patch(",
                "        \"/api/" + feature_entity.table + "/\" + item[\"id\"] + \"/status\",",
                "        json={" + repr(status_field.name) + ": \"not-a-column\"},",
                "        headers=headers,",
                "    )",
                "    assert rejected.status_code == 422",
                "",
                "",
            ]
        )
    if blueprint.feature == "schedule":
        relation = feature_entity.relations[0]
        resource = blueprint.entity(relation.ref)
        assert resource is not None
        lines.extend(
            [
                "def test_booking_conflicts_are_rejected():",
                '    """Overlapping windows are 409; adjacent windows and other resources are fine."""',
                "    headers, _ = sign_in()",
                "    resource = make_" + resource.name + "(headers)",
                "    make_" + feature_entity.name + "(",
                "        headers,",
                "        " + relation.name + "=resource[\"id\"],",
                "        start_at=\"2026-11-01T09:00:00+00:00\",",
                "        end_at=\"2026-11-01T10:00:00+00:00\",",
                "    )",
                "    clash = payload_" + feature_entity.name + "(",
                "        " + relation.name + "=resource[\"id\"],",
                "        start_at=\"2026-11-01T09:30:00+00:00\",",
                "        end_at=\"2026-11-01T10:30:00+00:00\",",
                "    )",
                "    response = client.post(\"/api/" + feature_entity.table + "\", json=clash, headers=headers)",
                "    assert response.status_code == 409",
                "    assert \"overlap\" in response.json()[\"detail\"].lower()",
                "    adjacent = payload_" + feature_entity.name + "(",
                "        " + relation.name + "=resource[\"id\"],",
                "        start_at=\"2026-11-01T10:00:00+00:00\",",
                "        end_at=\"2026-11-01T11:00:00+00:00\",",
                "    )",
                "    assert client.post(\"/api/" + feature_entity.table + "\", json=adjacent, headers=headers).status_code == 201",
                "    reversed_window = payload_" + feature_entity.name + "(",
                "        " + relation.name + "=resource[\"id\"],",
                "        start_at=\"2026-11-01T12:00:00+00:00\",",
                "        end_at=\"2026-11-01T11:00:00+00:00\",",
                "    )",
                "    bad = client.post(\"/api/" + feature_entity.table + "\", json=reversed_window, headers=headers)",
                "    assert bad.status_code == 422",
                "",
                "",
            ]
        )
    if blueprint.feature == "team":
        lines.extend(
            [
                "def test_share_flow():",
                '    """Sharing grants read/write to one teammate and nothing to everyone else."""',
                "    alice, _ = sign_in()",
                "    bob, bob_email = sign_in()",
                "    carol, _ = sign_in()",
                "    item = make_" + primary.name + "(alice)",
                "    shared = client.post(",
                "        \"/api/" + primary.table + "/\" + item[\"id\"] + \"/share\",",
                "        json={\"email\": bob_email},",
                "        headers=alice,",
                "    )",
                "    assert shared.status_code == 200",
                "    visible = client.get(\"/api/" + primary.table + "\", headers=bob).json()",
                "    assert any(row[\"id\"] == item[\"id\"] for row in visible)",
                "    shared_list = client.get(\"/api/shared\", headers=bob).json()",
                "    assert any(row[\"id\"] == item[\"id\"] for row in shared_list)",
                "    patched = client.patch(",
                "        \"/api/" + primary.table + "/\" + item[\"id\"],",
                "        json={" + repr(primary.display) + ": \"Team edit\"},",
                "        headers=bob,",
                "    )",
                "    assert patched.status_code == 200",
                "    assert client.delete(\"/api/" + primary.table + "/\" + item[\"id\"], headers=bob).status_code == 404",
                "    others = client.get(\"/api/" + primary.table + "\", headers=carol).json()",
                "    assert all(row[\"id\"] != item[\"id\"] for row in others)",
                "    bad_email = client.post(",
                "        \"/api/" + primary.table + "/\" + item[\"id\"] + \"/share\",",
                "        json={\"email\": \"not-an-email\"},",
                "        headers=alice,",
                "    )",
                "    assert bad_email.status_code == 422",
                "",
                "",
            ]
        )
    if blueprint.feature == "stats":
        sum_field = _sum_field(feature_entity)
        lines.extend(
            [
                "def test_dashboard_totals():",
                '    """The dashboard reads totals, group counts and a numeric sum."""',
                "    headers, _ = sign_in()",
                "    make_" + feature_entity.name + "(headers)",
                "    data = client.get(\"/api/" + feature_entity.table + "/stats\", headers=headers).json()",
                "    assert data[\"total\"] >= 1",
                "    assert data[\"group_field\"]",
                "    assert isinstance(data[\"by_group\"], list)",
                "    assert data[\"sum_field\"] " + ("== " + repr(sum_field) if sum_field else "is None"),
                "",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _tests_schema_module(blueprint: AppBlueprint, auth: bool) -> str:
    checks: list[str] = []
    for entity in blueprint.entities:
        required = [spec.name for spec in entity.fields if spec.required and spec.type != "ref"]
        checks.append(
            "def test_" + entity.table + "_columns():\n"
            "    \"\"\"Required columns are NOT NULL and timestamps exist.\"\"\"\n"
            "    info = columns(\"" + entity.table + "\")\n"
            "    assert \"created_at\" in info and \"updated_at\" in info\n"
            "    for name in " + repr(required) + ":\n"
            "        assert info[name][\"notnull\"] == 1, name\n"
            "\n"
        )
        checks.append(
            "def test_" + entity.table + "_indexes():\n"
            "    \"\"\"The columns queries filter on are indexed.\"\"\"\n"
            "    names = {row[1] for row in connection().execute(\"PRAGMA index_list(" + entity.table + ")\")}\n"
            "    assert \"idx_" + entity.table + "_user\" in names\n"
            "    assert \"idx_" + entity.table + "_created\" in names\n"
            "\n"
        )
        for spec in entity.relations:
            target = blueprint.entity(spec.ref)
            assert target is not None
            checks.append(
                "def test_" + entity.table + "_" + spec.name + "_foreign_key():\n"
                "    \"\"\"The relation is enforced by the database, not just the UI.\"\"\"\n"
                "    keys = list(connection().execute(\"PRAGMA foreign_key_list(" + entity.table + ")\"))\n"
                "    assert any(row[2] == \"" + target.table + "\" and row[3] == \"" + spec.name + "\" for row in keys)\n"
                "    assert any(str(row[6]).upper() == \"CASCADE\" for row in keys)\n"
                "\n"
            )
        if any(spec.type == "enum" for spec in entity.fields):
            checks.append(
                "def test_" + entity.table + "_enum_checks():\n"
                "    \"\"\"Enum columns carry a CHECK constraint so bad data cannot land.\"\"\"\n"
                "    sql = connection().execute(\n"
                "        \"SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?\",\n"
                "        (\"" + entity.table + "\",),\n"
                "    ).fetchone()[0]\n"
                "    assert \"CHECK\" in sql\n"
                "\n"
            )
    if auth:
        checks.append(
            "def test_users_email_is_unique():\n"
            "    \"\"\"Two accounts cannot share an email address.\"\"\"\n"
            "    sql = connection().execute(\n"
            "        \"SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'users'\",\n"
            "    ).fetchone()[0]\n"
            "    assert \"UNIQUE\" in sql\n"
            "\n"
        )
    return (
        '"""Schema contract: keys, checks, indexes and required columns."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "import os\n"
        "import sqlite3\n"
        "from pathlib import Path\n"
        "\n"
        "DB_PATH = Path(os.environ.get(\"SQLITE_PATH\", \"data/app.db\"))\n"
        "\n"
        "\n"
        "def connection():\n"
        "    \"\"\"Open the database the app under test actually wrote to.\"\"\"\n"
        "    assert DB_PATH.exists(), \"run the API tests first (they create the schema)\"\n"
        "    return sqlite3.connect(DB_PATH)\n"
        "\n"
        "\n"
        "def columns(table: str) -> dict:\n"
        "    \"\"\"Column metadata keyed by name.\"\"\"\n"
        "    info = list(connection().execute(\"PRAGMA table_info(\" + table + \")\"))\n"
        "    return {row[1]: {\"type\": row[2], \"notnull\": row[3], \"default\": row[4]} for row in info}\n"
        "\n"
        "\n" + "\n".join(checks)
    ).rstrip() + "\n"




def _readme(blueprint: AppBlueprint, module: str, auth: bool, port: int) -> str:
    feature_entity = _feature_entity(blueprint)
    lines = [
        "# " + header_safe(blueprint.title),
        "",
        blueprint.prompt.strip().rstrip(".") + ".",
        "",
        "Generated by coding-agent-20b from the `" + blueprint.id + "` blueprint. "
        "Every file is ordinary Python/HTML/CSS/JS: there is no runtime dependency "
        "on the agent, and nothing here needs an account to run.",
        "",
        "## What it does",
        "",
    ]
    for entity in blueprint.entities:
        lines.append("- **" + entity.label + "** (`" + entity.table + "`) - " + ", ".join(
            spec.display + (" *" if spec.required and spec.type != "ref" else "") for spec in entity.fields
        ))
    if blueprint.feature == "board":
        status_field = _status_field(feature_entity)
        lines.append(
            "- **Board** - cards move between " + ", ".join(status_field.choices) + " with `PATCH "
            "/api/" + feature_entity.table + "/{id}/status`; the UI renders one column per status."
        )
    elif blueprint.feature == "schedule":
        resource = blueprint.entity(feature_entity.relations[0].ref) if feature_entity.relations else feature_entity
        lines.append(
            "- **Conflict-aware scheduling** - a " + (resource.label.lower() if resource else "resource")
            + " cannot be double booked; overlapping windows return `409` with the clashing times."
        )
    elif blueprint.feature == "team":
        lines.append(
            "- **Sharing** - `POST /api/" + feature_entity.table + "/{id}/share` grants a teammate "
            "read/write access by email. Sharing never grants deletion."
        )
    elif blueprint.feature == "stats":
        lines.append(
            "- **Dashboard** - `/api/" + feature_entity.table + "/stats` returns totals, counts per "
            + _group_field(blueprint, feature_entity) + " and a 14-day creation trend."
        )
    if auth:
        lines.extend(
            [
                "- **Accounts** - scrypt password hashing and HS256 JWTs from the standard library. "
                "Every row is scoped to its owner, so one account cannot see another's data.",
            ]
        )
    else:
        lines.append("- **Single-user mode** - no login; every row belongs to the demo owner.")
    lines.extend(
        [
            "",
            "## Run it",
            "",
            "```bash",
            "python -m venv .venv && source .venv/bin/activate",
            "pip install -r requirements.txt",
            "uvicorn " + module + ".main:app --reload --port " + str(port),
            "```",
            "",
            "Open http://localhost:" + str(port) + " - health at `/healthz`, API docs at `/docs`.",
            "",
            "## Test it",
            "",
            "```bash",
            "python -m pytest -q",
            "```",
            "",
            "The tests hit the real ASGI app: auth, tenant isolation, CRUD, validation, "
            "relations and the feature above. They use a private database per run.",
            "",
            "## Data model",
            "",
            "| Table | Fields | Relations |",
            "| --- | --- | --- |",
        ]
    )
    for entity in blueprint.entities:
        fields = ", ".join(
            "`" + spec.name + "` " + spec.type + ("?" if not spec.required else "") for spec in entity.fields
        )
        relations = ", ".join(
            "`" + spec.name + "` -> `" + (_table_for_ref(blueprint, spec.ref)) + "` (" + spec.on_delete + ")"
            for spec in entity.relations
        ) or "-"
        lines.append("| `" + entity.table + "` | " + fields + " | " + relations + " |")
    lines.extend(
        [
            "",
            "The full DDL is in [`schema.sql`](schema.sql) and [`migrations/0001_init.sql`](migrations/0001_init.sql). "
            "SQLite is the default (`data/app.db`); set `DATABASE_URL=postgresql://...` to run the "
            "same schema on Postgres (install `psycopg[binary]`).",
            "",
            "## API",
            "",
            "| Method | Path | Purpose |",
            "| --- | --- | --- |",
        ]
    )
    if auth:
        lines.extend(
            [
                "| POST | `/auth/register` | create an account, returns a token |",
                "| POST | `/auth/login` | exchange email + password for a token |",
                "| GET | `/api/me` | the signed-in viewer |",
            ]
        )
    for entity in blueprint.entities:
        base = "/api/" + entity.table
        lines.extend(
            [
                "| GET | `" + base + "` | list with `q`, `limit`, `offset`, `sort` and relation filters |",
                "| POST | `" + base + "` | create (unknown fields are rejected) |",
                "| GET | `" + base + "/{id}` | one record |",
                "| PATCH | `" + base + "/{id}` | update the supplied fields |",
                "| DELETE | `" + base + "/{id}` | delete (owner only) |",
                "| GET | `" + base + "/stats` | totals, group counts, trend |",
                "| GET | `" + base + "/export.csv` | CSV export of visible rows |",
            ]
        )
    if blueprint.feature == "board":
        lines.append("| PATCH | `/api/" + feature_entity.table + "/{id}/status` | move a card between columns |")
    if blueprint.feature == "team":
        lines.extend(
            [
                "| POST | `/api/" + feature_entity.table + "/{id}/share` | share with a teammate by email |",
                "| GET | `/api/shared` | items other people shared with you |",
            ]
        )
    lines.extend(
        [
            "",
            "## Deploy it",
            "",
            "```bash",
            "docker compose up --build          # the included Dockerfile + compose file",
            "agent deploy --dry-run             # or let the agent plan a host deploy",
            "agent deploy --provider vercel     # vercel / fly / render / coolify / local",
            "```",
            "",
            "`Dockerfile`, `.dockerignore` and `docker-compose.yml` are plain Docker. "
            "A URL is only reported after `/healthz` answers.",
            "",
        ]
    )
    if auth:
        lines.extend(
            [
                "## Auth",
                "",
                "Passwords are hashed with `hashlib.scrypt` (per-user salt, constant-time compare) and "
                "tokens are HS256 JWTs signed with `AUTH_SECRET`. In production the app refuses to mint "
                "a token without `AUTH_SECRET` - copy `.env.example` to `.env` and set it.",
                "",
                "Tokens live in `localStorage` in the UI and are sent as `Authorization: Bearer <token>`. "
                "There is no refresh-token rotation or revocation list; add Authlib/PyJWT or a hosted "
                "provider when you need those.",
                "",
            ]
        )
    lines.extend(
        [
            "## Eject",
            "",
            "There is nothing to eject from. Delete `agentdeploy.toml` and the project still runs: "
            "it imports no agent module, and the database, auth and UI use only their own libraries. "
            "See `docs/EJECT.md` in the agent repository for the full guarantee.",
            "",
        ]
    )
    return "\n".join(lines)


def _env_example(auth: bool) -> str:
    lines = [
        "# Copy to .env and fill in. Never commit .env.",
        "APP_ENV=development",
        "SQLITE_PATH=data/app.db",
        "# Set this to use Postgres instead (install psycopg[binary]):",
        "# DATABASE_URL=postgresql://user:pass@localhost:5432/app",
    ]
    if auth:
        lines.extend(
            [
                "# Generate one: python -c \"import secrets; print(secrets.token_urlsafe(32))\"",
                "AUTH_SECRET=",
            ]
        )
    return "\n".join(lines) + "\n"


def _agentdeploy_toml(name: str, port: int) -> str:
    from deploy.config import DeployConfig, dump_toml

    config = DeployConfig(name=slugify(name), provider="auto", port=port, health_path="/healthz")
    return dump_toml({"deploy": config.to_mapping()})


class BlueprintGenerator:
    """Render one blueprint into a complete project (never touches the disk)."""

    def __init__(self, blueprint: AppBlueprint, options: BlueprintOptions | None = None):
        self.blueprint = blueprint
        self.options = options or BlueprintOptions()

    def render(self) -> AppProject:
        blueprint = self.blueprint
        options = self.options
        auth = blueprint.auth if options.auth is None else bool(options.auth)
        module = options.module(blueprint)
        title = options.title or blueprint.title
        name = options.name or slugify(blueprint.id)
        port = options.port or 8000
        description = options.description or blueprint.prompt
        accent = options.accent or blueprint.accent

        schema = _schema_for(blueprint, auth)
        project = AppProject(
            blueprint=blueprint,
            options=BlueprintOptions(
                name=name,
                title=title,
                description=description,
                port=port,
                accent=accent,
                auth=auth,
                style=options.style,
                auth_mode=options.auth_mode,
                entity_mode=options.entity_mode,
                author=options.author,
                year=options.year,
                license=options.license,
            ),
            module=module,
            run_command="uvicorn " + module + ".main:app --reload --port " + str(port),
            warnings=list(options.warnings),
        )
        files: dict[str, str] = {
            ".gitignore": _gitignore(),
            ".dockerignore": DOCKERIGNORE,
            ".env.example": _env_example(auth),
            "requirements.txt": REQUIREMENTS,
            "Dockerfile": DOCKERFILE.replace("@@MODULE@@", module),
            "docker-compose.yml": COMPOSE,
            "agentdeploy.toml": _agentdeploy_toml(name, port),
            "README.md": _readme(blueprint, module, auth, port),
            "schema.sql": "-- " + title + " schema (generated from the blueprint)\n\n" + schema,
            "migrations/0001_init.sql": "-- " + title + " initial migration\n\n" + schema,
            module + "/__init__.py": '"""' + title + ' application package."""\n',
            module + "/config.py": CONFIG_PY.replace("@@TITLE@@", title).replace(
                "@@FEATURE@@", blueprint.feature
            ).replace("@@AUTH_BOOL@@", repr(auth)),
            module + "/errors.py": ERRORS_PY,
            module + "/db.py": DB_PY.replace(
                "Schema (also written to ``schema.sql`` for humans):\n\n    @@SCHEMA@@",
                "The full schema is also written to ``schema.sql`` for humans.",
            ).replace("@@SCHEMA_BLOCK@@", schema).replace("@@TRIPLE@@", '"""'),
            module + "/schemas.py": _schema_module(blueprint, auth),
            module + "/store.py": _store_module(blueprint, auth, options),
            module + "/api.py": _api_module(blueprint, auth).replace("__MODULE__", module),
            module + "/main.py": MAIN_PY.replace("@@TITLE@@", title).replace(
                "@@AUTH_IMPORT@@", "from .auth_routes import router as auth_router" if auth else ""
            ).replace(
                "@@INCLUDE_AUTH@@",
                "    app.include_router(auth_router)" if auth else "    # Single-user mode: no auth router.",
            ),
            "tests/conftest.py": CONFTEST_PY,
            "tests/test_api.py": _tests_api_module(blueprint, auth).replace("__MODULE__", module),
            "tests/test_schema.py": _tests_schema_module(blueprint, auth),
        }
        if auth:
            from .addons import PY_AUTH, PY_AUTH_TESTS

            files[module + "/auth.py"] = PY_AUTH
            files[module + "/auth_routes.py"] = AUTH_ROUTES_PY
            files["tests/test_auth.py"] = PY_AUTH_TESTS.format(module=module)

        from .blueprint_web import render_web_assets

        files.update(
            render_web_assets(blueprint=blueprint, module=module, title=title, auth=auth, accent=accent)
        )

        if options.style == "legacy":
            from .blueprint_variants import apply_legacy_ui

            files = apply_legacy_ui(files, title)
            project.warnings.append("legacy UI style: intentionally below the default bar")
        if options.auth_mode == "broken":
            from .blueprint_variants import apply_broken_owner_scope

            files = apply_broken_owner_scope(files, blueprint, auth=auth)
            project.warnings.append("broken auth mode: authentication without authorisation")
        if options.entity_mode == "incomplete":
            from .blueprint_variants import apply_incomplete_entities

            files = apply_incomplete_entities(files, blueprint)
            project.warnings.append("incomplete entity mode: relation integrity removed")

        self._validate(files)
        project.files = files
        return project

    def _validate(self, files: dict[str, str]) -> None:
        leftovers = [path for path, content in files.items() if "@@" in content or "__MODULE__" in content]
        if leftovers:
            raise ValueError("blueprint " + self.blueprint.id + " left placeholders in: " + ", ".join(leftovers[:5]))
        placeholders = find_placeholders(files)
        if placeholders:
            raise ValueError("blueprint left template placeholders: " + ", ".join(placeholders[:5]))
        for path, content in files.items():
            if not path.endswith(".py"):
                continue
            try:
                compile(content, path, "exec")
            except SyntaxError as exc:
                raise ValueError(
                    "generated " + path + " does not compile: line " + str(exc.lineno) + ": " + str(exc.msg)
                ) from exc


def _gitignore() -> str:
    from .addons import gitignore

    return gitignore()


def render_blueprint(blueprint: AppBlueprint, options: BlueprintOptions | None = None) -> AppProject:
    """Convenience wrapper used by builders, tests and the corpus generator."""

    return BlueprintGenerator(blueprint, options).render()


def generate_app(
    root: str | Path,
    request_or_id: str,
    *,
    options: BlueprintOptions | None = None,
    force: bool = False,
) -> AppProject:
    """Resolve a blueprint id (or a prompt) and write the project to ``root``."""

    blueprint = get_blueprint(request_or_id) if request_or_id in {item.id for item in all_blueprints()} else match_blueprint(request_or_id)
    if blueprint is None:
        raise KeyError(
            "no blueprint matches that request; pass a blueprint id or use the starter registry"
        )
    project = render_blueprint(blueprint, options)
    project.write(root, force=force)
    return project


def all_blueprints():
    """Imported lazily to keep the catalogue the single source of truth."""

    from .blueprints import all_blueprints as _all

    return _all()
