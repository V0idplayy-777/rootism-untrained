"""Deployment configuration: on-disk project config plus provider credentials.

Design rules enforced here:

* Secrets are **never** read from the project config file. Provider tokens come
  from the process environment (or an explicit mapping the caller passes in),
  and every rendering path goes through :func:`redact` before it is logged or
  written to ``.agent/deploy/last.json``.
* The config file is optional. ``DeployConfig.discover()`` fills in everything it
  can from :mod:`deploy.detect` so ``agent deploy`` works on an untouched
  project with no config at all.
* No lock-in: the file only records *preferences* (provider, name, health path).
  Nothing in it is required to build or run the generated ``Dockerfile``.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

CONFIG_FILENAME = "agentdeploy.toml"
CONFIG_FILENAME_JSON = "agentdeploy.json"

_SECRET_KEY = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|PRIVATE|CREDENTIAL|AUTH|DSN|DATABASE_URL|CONN)",
    re.IGNORECASE,
)

PROVIDER_ENV: dict[str, tuple[str, ...]] = {
    "local": (),
    "archive": (),
    "github-pages": (),
    "vercel": ("VERCEL_TOKEN", "DEPLOY_TOKEN"),
    "render": ("RENDER_API_KEY", "DEPLOY_TOKEN"),
    "fly": ("FLY_API_TOKEN", "DEPLOY_TOKEN"),
    "coolify": ("COOLIFY_TOKEN", "DEPLOY_TOKEN"),
}

PROVIDER_SETTINGS: dict[str, tuple[str, ...]] = {
    "vercel": ("VERCEL_TEAM_ID", "VERCEL_SCOPE"),
    "render": ("RENDER_OWNER_ID", "RENDER_REGION"),
    "fly": ("FLY_ORG", "FLY_REGION", "FLY_APP"),
    "coolify": ("COOLIFY_URL", "COOLIFY_PROJECT_UUID", "COOLIFY_SERVER_UUID"),
    "github-pages": ("GITHUB_REPOSITORY",),
}


def looks_secret(name: str) -> bool:
    """True when an env/config key name suggests its value is sensitive."""

    return bool(_SECRET_KEY.search(name or ""))


def redact(text: str, secrets: Mapping[str, str] | None = None) -> str:
    """Replace every known secret value in ``text`` with ``***``.

    Also scrubs values of keys that *look* secret when they are long enough to
    be real credentials, so a stray ``print(os.environ)`` cannot leak a token
    into deploy logs.
    """

    out = text or ""
    for value in (secrets or {}).values():
        if value and len(str(value)) >= 4:
            out = out.replace(str(value), "***")
    out = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}", r"\1***", out)
    out = re.sub(r"(?i)((?:token|secret|password|api[_-]?key)\s*[=:]\s*)\S{6,}", r"\1***", out)
    return out


@dataclass
class DeployConfig:
    """Project-level deployment preferences (never holds secrets)."""

    name: str = "app"
    provider: str = "auto"
    region: str = "iad"
    health_path: str = "/"
    port: int | None = None
    static_dir: str | None = None
    repo_url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    build_args: dict[str, str] = field(default_factory=dict)
    settings: dict[str, str] = field(default_factory=dict)
    source: str = "defaults"

    @classmethod
    def load(cls, root: str | Path) -> "DeployConfig":
        """Read ``agentdeploy.toml`` (or ``.json``) if present."""

        root = Path(root)
        toml_path = root / CONFIG_FILENAME
        json_path = root / CONFIG_FILENAME_JSON

        if toml_path.exists():
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            data = data.get("deploy", data)
            return cls._from_mapping(data, source=str(toml_path.name))

        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            data = data.get("deploy", data)
            return cls._from_mapping(data, source=str(json_path.name))

        return cls()

    @classmethod
    def _from_mapping(cls, data: Mapping[str, Any], source: str) -> "DeployConfig":
        known = {
            "name",
            "provider",
            "region",
            "health_path",
            "port",
            "static_dir",
            "repo_url",
            "env",
            "build_args",
            "settings",
        }
        kwargs: dict[str, Any] = {}
        for key in known:
            if key in data and data[key] is not None:
                value = data[key]
                if key == "port":
                    value = int(value)
                if key in {"env", "build_args", "settings"}:
                    value = {str(k): str(v) for k, v in dict(value).items()}
                kwargs[key] = value
        cfg = cls(**kwargs)
        cfg.source = source
        return cfg

    @classmethod
    def discover(
        cls,
        root: str | Path,
        profile=None,
        environ: Mapping[str, str] | None = None,
    ) -> "DeployConfig":
        """Load config and fill unset fields from detection and the environment."""

        cfg = cls.load(root)
        env = dict(os.environ if environ is None else environ)
        root = Path(root)

        if profile is not None:
            if cfg.port is None:
                cfg.port = getattr(profile, "port", None)
            if cfg.static_dir is None:
                cfg.static_dir = getattr(profile, "static_dir", None)
            if cfg.name == "app":
                cfg.name = getattr(profile, "name", None) or root.name
            if cfg.health_path == "/" and getattr(profile, "health_path", None):
                cfg.health_path = profile.health_path
            if not cfg.repo_url:
                cfg.repo_url = getattr(profile, "repo_url", None)

        if cfg.name == "app":
            cfg.name = _slug(root.name or "app")

        for key in ("region", "health_path", "static_dir", "repo_url"):
            env_key = f"AGENT_DEPLOY_{key.upper()}"
            if env.get(env_key):
                setattr(cfg, key, env[env_key])
        if env.get("AGENT_DEPLOY_PROVIDER"):
            cfg.provider = env["AGENT_DEPLOY_PROVIDER"]

        for provider, keys in PROVIDER_SETTINGS.items():
            for key in keys:
                if env.get(key) and key not in cfg.settings:
                    cfg.settings[key] = env[key]
        return cfg

    def to_mapping(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "provider": self.provider,
            "region": self.region,
            "health_path": self.health_path,
        }
        if self.port is not None:
            data["port"] = self.port
        if self.static_dir:
            data["static_dir"] = self.static_dir
        if self.repo_url:
            data["repo_url"] = self.repo_url
        if self.env:
            data["env"] = {k: v for k, v in self.env.items() if not looks_secret(k)}
        if self.build_args:
            data["build_args"] = dict(self.build_args)
        if self.settings:
            data["settings"] = {
                k: v for k, v in self.settings.items() if not looks_secret(k)
            }
        return data

    def save(self, root: str | Path) -> Path:
        """Write ``agentdeploy.toml`` (secret-looking keys are dropped)."""

        path = Path(root) / CONFIG_FILENAME
        path.write_text(dump_toml({"deploy": self.to_mapping()}), encoding="utf-8")
        self.source = path.name
        return path

    def to_dict(self) -> dict[str, Any]:
        data = self.to_mapping()
        data["source"] = self.source
        return data

    def provider_token(self, provider: str | None = None, environ: Mapping[str, str] | None = None) -> str | None:
        env = os.environ if environ is None else environ
        for key in PROVIDER_ENV.get(provider or self.provider, ()):
            value = env.get(key)
            if value:
                return value
        return None

    def runtime_env(self, environ: Mapping[str, str] | None = None) -> dict[str, str]:
        """Env vars to pass into the container: config values + provider token names."""

        env = os.environ if environ is None else environ
        merged = {k: v for k, v in self.env.items() if not looks_secret(k)}
        for key in self.env:
            if looks_secret(key) and env.get(key):
                merged[key] = env[key]
        return merged


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (slug or "app")[:40]


def dump_toml(data: Mapping[str, Any]) -> str:
    """Minimal TOML writer for the flat/nested shapes this module produces."""

    lines: list[str] = []

    def emit(prefix: str, mapping: Mapping[str, Any]) -> None:
        scalars = {k: v for k, v in mapping.items() if not isinstance(v, Mapping)}
        tables = {k: v for k, v in mapping.items() if isinstance(v, Mapping)}
        for key, value in scalars.items():
            lines.append(f"{key} = {_toml_value(value)}")
        for key, value in tables.items():
            header = f"{prefix}.{key}" if prefix else str(key)
            lines.append("")
            lines.append(f"[{header}]")
            emit(header, value)

    emit("", data)
    return "\n".join(lines).strip() + "\n"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return f'"{value}"'
