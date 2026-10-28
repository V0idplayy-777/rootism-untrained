"""Reusable, security-reviewed building blocks that templates compose.

Auth here is deliberately dependency-free so a generated project runs anywhere:
HS256 JWTs built from ``hmac`` + ``json`` (the same wire format as PyJWT) and
scrypt password hashing from the standard library. That is auditable code, not
"auth by vibes"; tests assert tampered tokens and wrong passwords fail.

Each function returns ``{relative_path: content}`` so templates can merge what
they need, and the Node variant mirrors the Python contract.
"""

from __future__ import annotations

from .spec import ProjectOptions

PY_AUTH = '''"""Authentication: scrypt password hashing + HS256 JWTs (stdlib only).

This is a compact, auditable implementation of the two primitives an app needs:

* ``hash_password`` / ``verify_password`` - scrypt with a per-user salt.
* ``create_token`` / ``decode_token`` - HS256 JWT with ``sub``, ``exp`` and
  constant-time signature comparison.

It is intentionally not a full identity provider: no refresh tokens, no
revocation list, no OAuth. Swap in Authlib/PyJWT or a hosted provider when you
need those; the call sites are one function each.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any

TOKEN_TTL_SECONDS = 60 * 60 * 12
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1


class AuthError(ValueError):
    """Raised for invalid credentials or tokens (never leaks which)."""


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    if not password or len(password) < 8:
        raise AuthError("password must be at least 8 characters")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64encode(salt)}${_b64encode(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, digest_b64 = encoded.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    expected = _b64decode(digest_b64)
    candidate = hashlib.scrypt(
        (password or "").encode("utf-8"),
        salt=_b64decode(salt_b64),
        n=int(n),
        r=int(r),
        p=int(p),
        dklen=len(expected),
    )
    return hmac.compare_digest(candidate, expected)


def _secret() -> str:
    value = os.environ.get("AUTH_SECRET", "")
    if not value:
        if os.environ.get("APP_ENV") == "production":
            raise AuthError("AUTH_SECRET must be set in production")
        value = "dev-only-insecure-secret"
    return value


def create_token(subject: str, *, ttl: int = TOKEN_TTL_SECONDS, extra: dict[str, Any] | None = None) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": str(subject), "iat": now, "exp": now + ttl, **(extra or {})}
    signing_input = f"{_b64encode(json.dumps(header, separators=(',', ':')).encode())}.{_b64encode(json.dumps(payload, separators=(',', ':')).encode())}"
    signature = hmac.new(_secret().encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64encode(signature)}"


def decode_token(token: str) -> dict[str, Any]:
    try:
        header_b64, payload_b64, signature_b64 = (token or "").split(".")
    except ValueError as exc:
        raise AuthError("malformed token") from exc
    signing_input = f"{header_b64}.{payload_b64}"
    expected = hmac.new(_secret().encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _b64decode(signature_b64)):
        raise AuthError("bad signature")
    try:
        header = json.loads(_b64decode(header_b64))
        payload = json.loads(_b64decode(payload_b64))
    except (ValueError, UnicodeDecodeError) as exc:
        raise AuthError("malformed token") from exc
    if header.get("alg") != "HS256":
        raise AuthError("unexpected algorithm")
    if int(payload.get("exp", 0)) < int(time.time()):
        raise AuthError("token expired")
    return payload
'''

PY_AUTH_TESTS = '''"""Auth contract: hashing, token round-trip, tamper and expiry rejection."""

from __future__ import annotations

import time

import pytest

from {module}.auth import (
    AuthError,
    create_token,
    decode_token,
    hash_password,
    verify_password,
)


def test_password_round_trip():
    encoded = hash_password("correct horse battery staple")
    assert encoded.startswith("scrypt$")
    assert verify_password("correct horse battery staple", encoded)
    assert not verify_password("wrong password", encoded)


def test_passwords_are_salted():
    assert hash_password("same password") != hash_password("same password")


def test_short_passwords_rejected():
    with pytest.raises(AuthError):
        hash_password("short")


def test_token_round_trip():
    token = create_token("user-1")
    payload = decode_token(token)
    assert payload["sub"] == "user-1"
    assert payload["exp"] > time.time()


def test_tampered_token_rejected():
    token = create_token("user-1")
    header, payload, signature = token.split(".")
    tampered = f"{{header}}.{{payload}}.{{signature[:-2]}}aa"
    with pytest.raises(AuthError):
        decode_token(tampered)


def test_expired_token_rejected():
    token = create_token("user-1", ttl=-10)
    with pytest.raises(AuthError):
        decode_token(token)


def test_production_requires_secret(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("AUTH_SECRET", raising=False)
    with pytest.raises(AuthError):
        create_token("user-1")
'''

PY_AUTH_ROUTES = '''"""HTTP surface for auth: register, login, and a protected /me."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from . import store
from .auth import AuthError, create_token, decode_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=256)


def current_user(authorization: str = Header(default="")) -> str:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    try:
        return str(decode_token(token)["sub"])
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc


@router.post("/register", status_code=201)
def register(body: Credentials) -> dict:
    if store.get_user_by_email(body.email):
        raise HTTPException(status.HTTP_409_CONFLICT, "email already registered")
    user = store.create_user(body.email, hash_password(body.password))
    return {"id": user["id"], "email": user["email"], "token": create_token(user["id"])}


@router.post("/login")
def login(body: Credentials) -> dict:
    user = store.get_user_by_email(body.email)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
    return {"token": create_token(user["id"])}


@router.get("/me")
def me(user_id: str = Depends(current_user)) -> dict:
    user = store.get_user(user_id)
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    return {"id": user["id"], "email": user["email"]}
'''

PY_STORE = '''"""Tiny persistence layer (stdlib sqlite3, or Postgres via DATABASE_URL).

The SQL is parameterised everywhere: no string formatting reaches the database.
``get_connection`` is the single seam to swap in SQLAlchemy or psycopg later.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    title TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
"""

def sqlite_path() -> str:
    """Resolved per call so tests can point SQLITE_PATH at a temp file."""

    return os.environ.get("SQLITE_PATH", "data/app.db")


def _is_postgres() -> bool:
    return bool(os.environ.get("DATABASE_URL", "").startswith(("postgres://", "postgresql://")))


@contextmanager
def get_connection() -> Iterator[object]:
    """Yield a DB-API connection with ``?`` placeholders rewritten for Postgres."""

    if _is_postgres():  # pragma: no cover - requires a live Postgres
        import psycopg  # type: ignore[import-not-found]

        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            yield _PostgresShim(conn)
        return
    path = Path(sqlite_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


class _PostgresShim:
    """Adapt ``?`` placeholders and rows to the sqlite API used by this file."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql: str, params: tuple = ()):
        return self._conn.execute(sql.replace("?", "%s"), params)

    def executescript(self, script: str) -> None:
        for statement in script.split(";"):
            if statement.strip():
                self._conn.execute(statement)

    def commit(self) -> None:
        self._conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_user(email: str, password_hash: str) -> dict:
    user = {"id": str(uuid.uuid4()), "email": email.lower(), "password_hash": password_hash, "created_at": _now()}
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user["id"], user["email"], user["password_hash"], user["created_at"]),
        )
    return user


def get_user(user_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
    return dict(row) if row else None


def list_items(owner_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM items WHERE owner_id = ? ORDER BY created_at DESC", (owner_id,)).fetchall()
    return [dict(row) for row in rows]


def create_item(owner_id: str, title: str) -> dict:
    item = {"id": str(uuid.uuid4()), "owner_id": owner_id, "title": title, "done": 0, "created_at": _now()}
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO items (id, owner_id, title, done, created_at) VALUES (?, ?, ?, ?, ?)",
            (item["id"], item["owner_id"], item["title"], item["done"], item["created_at"]),
        )
    return item


def toggle_item(owner_id: str, item_id: str) -> dict | None:
    with get_connection() as conn:
        conn.execute("UPDATE items SET done = 1 - done WHERE id = ? AND owner_id = ?", (item_id, owner_id))
        row = conn.execute("SELECT * FROM items WHERE id = ? AND owner_id = ?", (item_id, owner_id)).fetchone()
    return dict(row) if row else None
'''


def python_auth(module: str) -> dict[str, str]:
    """Auth module + routes + contract tests for a FastAPI project."""

    return {
        f"{module}/auth.py": PY_AUTH,
        f"{module}/auth_routes.py": PY_AUTH_ROUTES,
        f"{module}/store.py": PY_STORE,
        "tests/test_auth.py": PY_AUTH_TESTS.format(module=module),
    }


NODE_AUTH = '''// Authentication: scrypt hashing + HS256 JWTs using node:crypto only.
// Mirrors the Python addon's contract: same token shape, same guarantees.
import { createHmac, randomBytes, scryptSync, timingSafeEqual } from "node:crypto";

const TOKEN_TTL_SECONDS = 60 * 60 * 12;

export class AuthError extends Error {}

function b64url(buffer) {
  return Buffer.from(buffer).toString("base64url");
}

export function hashPassword(password) {
  if (!password || password.length < 8) throw new AuthError("password must be at least 8 characters");
  const salt = randomBytes(16);
  const digest = scryptSync(password, salt, 32);
  return `scrypt$${b64url(salt)}$${b64url(digest)}`;
}

export function verifyPassword(password, encoded) {
  const [scheme, saltB64, digestB64] = String(encoded || "").split("$");
  if (scheme !== "scrypt") return false;
  const expected = Buffer.from(digestB64 || "", "base64url");
  const candidate = scryptSync(password || "", Buffer.from(saltB64 || "", "base64url"), expected.length || 32);
  return expected.length === candidate.length && timingSafeEqual(candidate, expected);
}

function secret() {
  const value = process.env.AUTH_SECRET || "";
  if (!value) {
    if (process.env.APP_ENV === "production") throw new AuthError("AUTH_SECRET must be set in production");
    return "dev-only-insecure-secret";
  }
  return value;
}

export function createToken(subject, ttl = TOKEN_TTL_SECONDS, extra = {}) {
  const now = Math.floor(Date.now() / 1000);
  const header = b64url(JSON.stringify({ alg: "HS256", typ: "JWT" }));
  const payload = b64url(JSON.stringify({ sub: String(subject), iat: now, exp: now + ttl, ...extra }));
  const input = `${header}.${payload}`;
  const signature = createHmac("sha256", secret()).update(input).digest("base64url");
  return `${input}.${signature}`;
}

export function decodeToken(token) {
  const parts = String(token || "").split(".");
  if (parts.length !== 3) throw new AuthError("malformed token");
  const [headerB64, payloadB64, signatureB64] = parts;
  const input = `${headerB64}.${payloadB64}`;
  const expected = createHmac("sha256", secret()).update(input).digest("base64url");
  const a = Buffer.from(expected);
  const b = Buffer.from(signatureB64);
  if (a.length !== b.length || !timingSafeEqual(a, b)) throw new AuthError("bad signature");
  const header = JSON.parse(Buffer.from(headerB64, "base64url").toString());
  const payload = JSON.parse(Buffer.from(payloadB64, "base64url").toString());
  if (header.alg !== "HS256") throw new AuthError("unexpected algorithm");
  if (Number(payload.exp || 0) < Math.floor(Date.now() / 1000)) throw new AuthError("token expired");
  return payload;
}
'''

NODE_AUTH_TEST = '''import assert from "node:assert/strict";
import test from "node:test";

import { AuthError, createToken, decodeToken, hashPassword, verifyPassword } from "../src/auth.js";

test("password round trip", () => {
  const encoded = hashPassword("correct horse battery staple");
  assert.ok(encoded.startsWith("scrypt$"));
  assert.equal(verifyPassword("correct horse battery staple", encoded), true);
  assert.equal(verifyPassword("wrong", encoded), false);
});

test("passwords are salted", () => {
  assert.notEqual(hashPassword("same password"), hashPassword("same password"));
});

test("token round trip", () => {
  const payload = decodeToken(createToken("user-1"));
  assert.equal(payload.sub, "user-1");
  assert.ok(payload.exp > Date.now() / 1000);
});

test("tampered token rejected", () => {
  const [h, p, s] = createToken("user-1").split(".");
  assert.throws(() => decodeToken(`${h}.${p}.${s.slice(0, -2)}aa`), AuthError);
});

test("expired token rejected", () => {
  assert.throws(() => decodeToken(createToken("user-1", -10)), AuthError);
});
'''


def node_auth() -> dict[str, str]:
    return {"src/auth.js": NODE_AUTH, "test/auth.test.js": NODE_AUTH_TEST}


def env_example(options: ProjectOptions, include_secret: bool = True) -> str:
    lines = [
        "# Copy to .env and fill in. Never commit .env.",
        "APP_ENV=development",
    ]
    if options.database == "sqlite":
        lines.append("SQLITE_PATH=data/app.db")
    elif options.database == "postgres":
        lines += ["# Postgres (agent backend / Neon / Supabase):", "DATABASE_URL=postgresql://user:pass@localhost:5432/app"]
    if include_secret and options.use_auth:
        lines += ["# Generate one: python -c \"import secrets;print(secrets.token_urlsafe(32))\"", "AUTH_SECRET="]
    return "\n".join(lines) + "\n"


def gitignore() -> str:
    return """# secrets - never commit these
.env
.env.*
!.env.example
# python
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
# node
node_modules/
.next/
dist/
coverage/
# data and agent state
data/
*.sqlite3
*.db
.agent/
# os
.DS_Store
"""


def docker_note() -> str:
    return "Docker files are generated by `agent deploy` (kept out of the scaffold to avoid drift)."
