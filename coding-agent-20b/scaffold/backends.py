"""Managed database provisioning (optional, never required).

The scaffold works with SQLite out of the box. When a project needs Postgres,
this module can either:

* accept a connection string you already have (``DATABASE_URL``),
* or create a free Neon project via its API (``NEON_API_KEY``),
* or create a Supabase project via its API (``SUPABASE_ACCESS_TOKEN`` +
  ``SUPABASE_ORG_ID``) - marked experimental, shapes differ per plan.

Guarantees:

* the connection string is written **only** to ``.env`` (mode 0600) and is
  masked in every result/report;
* ``.env`` is added to ``.gitignore`` if missing;
* ``.env.example`` gets a placeholder, never the real secret;
* with no credentials the provider is ``sqlite`` and nothing is created.

Nothing about the app is tied to a provider: swapping providers means changing
``DATABASE_URL``.
"""

from __future__ import annotations

import os
import re
import secrets
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from deploy.http import FakeHttpClient, HttpClient

NEON_API = "https://console.neon.tech/api/v2"
SUPABASE_API = "https://api.supabase.com/v1"


@dataclass
class BackendResult:
    ok: bool
    provider: str = "sqlite"
    engine: str = "sqlite"
    url: str = ""
    env_var: str = "DATABASE_URL"
    created: bool = False
    detail: str = ""
    error: str = ""
    notes: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def masked_url(self) -> str:
        return mask_url(self.url)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "provider": self.provider,
            "engine": self.engine,
            "url": self.masked_url,
            "env_var": self.env_var,
            "created": self.created,
            "detail": self.detail,
            "error": self.error,
            "notes": list(self.notes),
            "meta": dict(self.meta),
        }

    def render(self) -> str:
        if not self.ok:
            return f"backend provisioning failed ({self.provider}): {self.error}"
        return f"{self.provider} {self.engine} -> {self.masked_url} ({'created' if self.created else 'existing'})"


def mask_url(url: str) -> str:
    """``postgresql://user:pass@host/db`` -> ``postgresql://user:***@host/db``."""

    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "***"
    if parts.password is None:
        return url
    netloc = f"{parts.username or 'user'}:***@{parts.hostname or ''}"
    if parts.port:
        netloc += f":{parts.port}"
    return f"{parts.scheme}://{netloc}{parts.path}"


def validate_postgres_url(url: str) -> tuple[bool, str]:
    if not url:
        return False, "empty connection string"
    if not url.startswith(("postgres://", "postgresql://")):
        return False, "must start with postgres:// or postgresql://"
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        return False, f"unparseable: {exc}"
    if not parts.hostname:
        return False, "no host in the connection string"
    if not parts.path or parts.path == "/":
        return False, "no database name in the path (…/mydb)"
    return True, "looks valid"


def probe_host(url: str, timeout: float = 3.0) -> tuple[bool, str]:
    """TCP reachability check (no driver, no credentials sent)."""

    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port or 5432
    except ValueError as exc:
        return False, str(exc)
    if not host:
        return False, "no host"
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"{host}:{port} reachable"
    except OSError as exc:
        return False, f"cannot reach {host}:{port}: {exc}"


class BackendProvider:
    name = "generic"
    engine = "postgres"

    def available(self, env: Mapping[str, str]) -> tuple[bool, str]:
        return False, "not configured"

    def provision(self, name: str, env: Mapping[str, str], http: HttpClient) -> BackendResult:
        raise NotImplementedError


class PostgresUrlProvider(BackendProvider):
    name = "existing"
    engine = "postgres"

    def available(self, env):
        url = env.get("DATABASE_URL", "")
        if not url:
            return False, "DATABASE_URL is not set"
        ok, detail = validate_postgres_url(url)
        return ok, detail

    def provision(self, name, env, http):
        url = env["DATABASE_URL"]
        ok, detail = validate_postgres_url(url)
        if not ok:
            return BackendResult(ok=False, provider=self.name, error=detail)
        return BackendResult(
            ok=True,
            provider=self.name,
            engine="postgres",
            url=url,
            created=False,
            detail="using the DATABASE_URL from the environment",
        )


class NeonProvider(BackendProvider):
    name = "neon"

    def available(self, env):
        if not env.get("NEON_API_KEY"):
            return False, "NEON_API_KEY is not set"
        return True, "neon API key present"

    def provision(self, name, env, http):
        headers = {
            "Authorization": f"Bearer {env['NEON_API_KEY']}",
            "Accept": "application/json",
        }
        payload = {
            "project": {
                "name": _safe_name(name),
                "region_id": env.get("NEON_REGION", "aws-us-east-1"),
                "pg_version": 16,
            }
        }
        response = http.post(f"{NEON_API}/projects", json_body=payload, headers=headers, timeout=60)
        if not response.ok:
            return BackendResult(
                ok=False,
                provider=self.name,
                error=f"Neon API {response.status}: {response.body[:300]}",
            )
        data = response.json()
        uris = data.get("connection_uris") or []
        url = uris[0].get("connection_uri") if uris else ""
        project = data.get("project") or {}
        if not url and project.get("id"):
            poll = http.get(
                f"{NEON_API}/projects/{project['id']}/connection_uri?database_name=neondb&role_name=neondb_owner",
                headers=headers,
                timeout=60,
            )
            if poll.ok:
                url = poll.json().get("uri", "")
        if not url:
            return BackendResult(
                ok=False,
                provider=self.name,
                error="Neon created the project but returned no connection string",
                meta={"project_id": project.get("id")},
            )
        return BackendResult(
            ok=True,
            provider=self.name,
            engine="postgres",
            url=url,
            created=True,
            detail="created a Neon project (free tier)",
            notes=["Neon suspends idle projects; the first query after idle can be slow."],
            meta={"project_id": project.get("id"), "region": payload["project"]["region_id"]},
        )


class SupabaseProvider(BackendProvider):
    name = "supabase"
    experimental = True

    def available(self, env):
        if not env.get("SUPABASE_ACCESS_TOKEN"):
            return False, "SUPABASE_ACCESS_TOKEN is not set"
        if not env.get("SUPABASE_ORG_ID"):
            return False, "SUPABASE_ORG_ID is not set"
        return True, "supabase token + org present"

    def provision(self, name, env, http):
        headers = {
            "Authorization": f"Bearer {env['SUPABASE_ACCESS_TOKEN']}",
            "Accept": "application/json",
        }
        password = secrets.token_urlsafe(24)
        payload = {
            "name": _safe_name(name),
            "organization_id": env["SUPABASE_ORG_ID"],
            "db_pass": password,
            "region": env.get("SUPABASE_REGION", "us-east-1"),
            "plan": "free",
        }
        response = http.post(f"{SUPABASE_API}/projects", json_body=payload, headers=headers, timeout=90)
        if not response.ok:
            return BackendResult(
                ok=False,
                provider=self.name,
                error=f"Supabase API {response.status}: {response.body[:300]}",
                notes=["Project creation needs a verified org and an access token with project scope."],
            )
        data = response.json()
        ref = data.get("id") or data.get("ref")
        if not ref:
            return BackendResult(ok=False, provider=self.name, error="Supabase returned no project ref")
        url = f"postgresql://postgres:{password}@db.{ref}.supabase.co:5432/postgres"
        return BackendResult(
            ok=True,
            provider=self.name,
            engine="postgres",
            url=url,
            created=True,
            detail="created a Supabase project (free tier); it takes ~2 minutes to accept connections",
            notes=[
                "Experimental: verify the direct connection string in the Supabase dashboard "
                "(pooler hosts differ per region).",
                "The generated database password is only stored in .env - save it if you need it.",
            ],
            meta={"project_ref": ref},
        )


class SqliteProvider(BackendProvider):
    name = "sqlite"
    engine = "sqlite"

    def available(self, env):
        return True, "local SQLite, no account needed"

    def provision(self, name, env, http):
        return BackendResult(
            ok=True,
            provider=self.name,
            engine="sqlite",
            url="",
            env_var="SQLITE_PATH",
            created=False,
            detail="using the local SQLite file (data/app.db)",
            notes=["Zero setup. Switch to Postgres later by setting DATABASE_URL."],
        )


PROVIDERS = {provider.name: provider for provider in (NeonProvider(), SupabaseProvider(), PostgresUrlProvider(), SqliteProvider())}


def available_backends(env: Mapping[str, str] | None = None) -> dict[str, tuple[bool, str]]:
    env = os.environ if env is None else env
    return {name: provider.available(env) for name, provider in PROVIDERS.items()}


class BackendProvisioner:
    """Pick a backend, provision it if needed, write ``.env`` safely."""

    def __init__(self, root: str | Path, http: HttpClient | None = None, env: Mapping[str, str] | None = None):
        self.root = Path(root)
        self.http = http
        self.env = dict(os.environ if env is None else env)

    def _select(self, preferred: str | None) -> BackendProvider:
        if preferred and preferred != "auto":
            if preferred not in PROVIDERS:
                raise KeyError(f"unknown backend {preferred!r}; choose from {', '.join(PROVIDERS)}")
            provider = PROVIDERS[preferred]
            ok, reason = provider.available(self.env)
            if not ok:
                raise ValueError(f"{preferred} is not configured: {reason}")
            return provider
        for candidate in ("existing", "neon", "supabase"):
            provider = PROVIDERS[candidate]
            ok, _ = provider.available(self.env)
            if ok:
                return provider
        return PROVIDERS["sqlite"]

    def ensure(self, *, name: str | None = None, backend: str | None = None, dry_run: bool = False) -> BackendResult:
        project = _safe_name(name or self.root.name or "app")
        try:
            provider = self._select(backend)
        except (KeyError, ValueError) as exc:
            return BackendResult(ok=False, provider=backend or "auto", error=str(exc))

        if dry_run:
            return BackendResult(
                ok=True,
                provider=provider.name,
                engine=provider.engine,
                detail=f"dry run: would use {provider.name}",
                notes=["no .env changes were made"],
            )

        client = self.http or (FakeHttpClient() if _is_offline(provider) else HttpClient())
        result = provider.provision(project, self.env, client)
        if result.ok and provider.engine == "postgres":
            self._write_env(result.url, project)
        elif result.ok:
            self._write_sqlite_env()
            self._ensure_gitignore()
        return result

    def _write_sqlite_env(self) -> None:
        """Make the zero-setup path explicit in ``.env`` (non-secret, 0600)."""

        env_path = self.root / ".env"
        existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        lines = [
            line
            for line in existing.splitlines()
            if not line.startswith(("DATABASE_URL=", "SQLITE_PATH="))
        ]
        lines.append("SQLITE_PATH=data/app.db")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            env_path.chmod(0o600)
        except OSError:
            pass

        example_path = self.root / ".env.example"
        example = example_path.read_text(encoding="utf-8") if example_path.exists() else ""
        if "SQLITE_PATH=" not in example:
            example_path.write_text(
                example + ("\n" if example and not example.endswith("\n") else "") + "SQLITE_PATH=data/app.db\n",
                encoding="utf-8",
            )

    def _ensure_gitignore(self) -> None:
        path = self.root / ".gitignore"
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        lines = {line.strip() for line in text.splitlines()}
        additions = [entry for entry in (".env", ".env.*", "!.env.example") if entry not in lines]
        if additions:
            path.write_text((text + ("\n" if text and not text.endswith("\n") else "") + "\n".join(additions) + "\n"), encoding="utf-8")

    def _write_env(self, url: str, project: str) -> None:
        env_path = self.root / ".env"
        existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        lines = [
            line
            for line in existing.splitlines()
            if not line.startswith(("DATABASE_URL=", "SQLITE_PATH="))
        ]
        lines.append(f"DATABASE_URL={url}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            env_path.chmod(0o600)
        except OSError:
            pass

        example_path = self.root / ".env.example"
        example = example_path.read_text(encoding="utf-8") if example_path.exists() else ""
        if "DATABASE_URL=" not in example:
            example_path.write_text(
                example
                + ("\n" if example and not example.endswith("\n") else "")
                + "DATABASE_URL=postgresql://user:password@host:5432/dbname\n",
                encoding="utf-8",
            )
        self._ensure_gitignore()


def _safe_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", (name or "app").lower()).strip("-")
    return (slug or "app")[:30]


def _is_offline(provider: BackendProvider) -> bool:
    return provider.name in {"sqlite", "existing"}
