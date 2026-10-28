"""Project introspection for deployment.

``detect_project(root)`` answers the questions a deployer must answer before it
can build anything:

* what kind of project is this (web app, static site, CLI, worker)?
* how do I install, build, start and test it?
* which port does it listen on, and which path proves it is healthy?
* which environment variables does it need?
* which file should the container entrypoint run?

The detector is deliberately read-only, dependency-free (stdlib only) and
conservative: when it cannot identify a start command it says so in
``profile.notes`` and the deploy engine refuses to guess (an honest failure
beats a container that exits in a loop).
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".next",
    ".nuxt",
    "dist",
    "build",
    "coverage",
    ".agent",
    ".turbo",
    ".idea",
    ".vscode",
    "site-packages",
}

SOURCE_EXT = {
    ".py": "python",
    ".js": "node",
    ".mjs": "node",
    ".cjs": "node",
    ".ts": "node",
    ".tsx": "node",
    ".jsx": "node",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
}

_ENV_PATTERNS = (
    re.compile(r"os\.environ\[\s*[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']\s*\]"),
    re.compile(r"os\.getenv\(\s*[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']"),
    re.compile(r"os\.environ\.get\(\s*[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']"),
    re.compile(r"process\.env\.([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"process\.env\[\s*[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']\s*\]"),
    re.compile(r"getenv\(\s*[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']"),
    re.compile(r"ENV\[\s*[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']\s*\]"),
)

_ENV_NOISE = {
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "PWD",
    "TMPDIR",
    "TERM",
    "CI",
    "HOST",
    "HOSTNAME",
    "PORT",
    "NODE_ENV",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "LANG",
    "LC_ALL",
    "VIRTUAL_ENV",
    "TZ",
    "DEBUG",
}

_PORT_RE = re.compile(r"port\s*[=:]\s*(\d{2,5})", re.IGNORECASE)


@dataclass
class ProjectProfile:
    """What the deployer learned about a project (all fields serialisable)."""

    root: str
    name: str
    kind: str = "unknown"  # static | python-web | node-web | nextjs | worker | cli | unknown
    language: str = "unknown"
    framework: str = "unknown"
    entrypoint: str | None = None
    app_module: str | None = None
    app_factory: bool = False
    app_dir: str = "."
    install_cmd: str | None = None
    build_cmd: str | None = None
    start_cmd: str | None = None
    test_cmd: str | None = None
    port: int = 8000
    health_path: str = "/"
    static_dir: str | None = None
    env_required: list[str] = field(default_factory=list)
    runtime_version: str = "3.11"
    base_image: str | None = None
    package_manager: str | None = None
    has_dockerfile: bool = False
    has_tests: bool = False
    monorepo: bool = False
    subprojects: list[str] = field(default_factory=list)
    repo_url: str | None = None
    file_count: int = 0
    description: str = ""
    features: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_server(self) -> bool:
        return self.kind in {"python-web", "node-web", "nextjs", "static"}

    @property
    def is_worker(self) -> bool:
        return self.kind == "worker"

    @property
    def can_deploy(self) -> bool:
        return self.kind != "unknown" and (
            self.start_cmd is not None or self.static_dir is not None or self.is_worker
        )

    def to_dict(self) -> dict:
        data = {
            "root": self.root,
            "name": self.name,
            "kind": self.kind,
            "language": self.language,
            "framework": self.framework,
            "entrypoint": self.entrypoint,
            "app_module": self.app_module,
            "app_factory": self.app_factory,
            "app_dir": self.app_dir,
            "install_cmd": self.install_cmd,
            "build_cmd": self.build_cmd,
            "start_cmd": self.start_cmd,
            "test_cmd": self.test_cmd,
            "port": self.port,
            "health_path": self.health_path,
            "static_dir": self.static_dir,
            "env_required": list(self.env_required),
            "runtime_version": self.runtime_version,
            "base_image": self.base_image,
            "package_manager": self.package_manager,
            "has_dockerfile": self.has_dockerfile,
            "has_tests": self.has_tests,
            "monorepo": self.monorepo,
            "subprojects": list(self.subprojects),
            "repo_url": self.repo_url,
            "file_count": self.file_count,
            "can_deploy": self.can_deploy,
            "description": self.description,
            "features": list(self.features),
            "notes": list(self.notes),
        }
        return data

    def render(self) -> str:
        """Human-readable summary used by the CLI and the UI."""

        rows = [
            ("Project", self.name),
            ("Kind", self.kind),
            ("Stack", f"{self.language} / {self.framework}"),
            ("Install", self.install_cmd or "-"),
            ("Build", self.build_cmd or "-"),
            ("Start", self.start_cmd or "-"),
            ("Test", self.test_cmd or "-"),
            ("Port", str(self.port)),
            ("Health", self.health_path),
            ("Env", ", ".join(self.env_required) or "-"),
            ("Dockerfile", "present" if self.has_dockerfile else "generated"),
        ]
        width = max(len(k) for k, _ in rows)
        lines = [f"{k.ljust(width)}  {v}" for k, v in rows]
        if self.notes:
            lines.append("")
            lines.extend(f"note: {n}" for n in self.notes)
        return "\n".join(lines)


def iter_project_files(root: str | Path, max_files: int = 5000) -> list[str]:
    """Relative paths of project files, skipping vendored/generated trees."""

    root = Path(root)
    out: list[str] = []
    for path in sorted(root.rglob("*")):
        if len(out) >= max_files:
            break
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in IGNORE_DIRS for part in rel_parts[:-1]):
            continue
        if rel_parts[-1].endswith((".pyc", ".pyo", ".log")):
            continue
        out.append(str(Path(*rel_parts)))
    return out


def _read(path: Path, limit: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _load_package_json(root: Path) -> dict:
    path = root / "package.json"
    if not path.exists():
        return {}
    try:
        return json.loads(_read(path) or "{}")
    except json.JSONDecodeError:
        return {}


def _load_pyproject(root: Path) -> dict:
    path = root / "pyproject.toml"
    if not path.exists():
        return {}
    try:
        return tomllib.loads(_read(path))
    except (tomllib.TOMLDecodeError, OSError):
        return {}


def _python_dependencies(root: Path, files: Iterable[str]) -> set[str]:
    deps: set[str] = set()
    pyproject = _load_pyproject(root)
    project = pyproject.get("project", {}) if isinstance(pyproject, dict) else {}
    for item in list(project.get("dependencies") or []):
        deps.add(_dep_name(item))
    for group in (project.get("optional-dependencies") or {}).values():
        for item in group or []:
            deps.add(_dep_name(item))
    poetry = pyproject.get("tool", {}).get("poetry", {})
    for name in (poetry.get("dependencies") or {}):
        deps.add(name.lower())
    for rel in files:
        if Path(rel).name.startswith("requirements") and rel.count("/") == 0:
            for line in _read(root / rel).splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "-")):
                    continue
                deps.add(_dep_name(line.split(";")[0]))
    setup_cfg = root / "setup.cfg"
    if setup_cfg.exists():
        for line in _read(setup_cfg).splitlines():
            match = re.match(r"\s*install_requires\s*=\s*(.+)", line)
            if match:
                deps.update(_dep_name(d) for d in match.group(1).split(","))
    return {d for d in deps if d}


def _dep_name(spec: str) -> str:
    name = re.split(r"[<>=!~\[\s;]", str(spec).strip(), maxsplit=1)[0]
    return name.lower().replace("_", "-")


def _find_python_entry(root: Path, files: Iterable[str]) -> tuple[str | None, str | None, str]:
    """Return (entrypoint_path, module_for_asgi, kind_hint).

    Hints: ``fastapi`` / ``fastapi-factory`` / ``flask`` / ``flask-factory`` /
    ``python-web`` / ``cli`` / ``unknown``. Factory hints mean the ASGI object is
    built by a function (``create_app()``), so the server needs ``--factory``.
    """

    candidates = [f for f in files if f.endswith(".py") and "/" not in f]
    for name in ("main.py", "app.py", "server.py", "api.py", "run.py", "wsgi.py", "asgi.py"):
        if name in candidates:
            found = _entry_from_text(name, _read(root / name))
            if found is not None:
                return found
    for rel in files:
        if not rel.endswith(".py") or rel.startswith("tests/"):
            continue
        found = _entry_from_text(rel, _read(root / rel))
        if found is not None and (found[2].startswith("fastapi") or found[2] == "flask"):
            return found
    for rel in files:
        if rel.endswith("__main__.py"):
            module = rel[: -len("__main__.py")].replace("/", ".").rstrip(".")
            return rel, module or None, "cli"
    return None, None, "unknown"


def _entry_from_text(rel: str, text: str) -> tuple[str, str, str] | None:
    """Detect an app object (or app factory) in one module's source."""

    stem = rel[:-3].replace("/", ".").replace(".__init__", "")
    if "FastAPI(" in text:
        match = re.search(r"^(\w+)\s*=\s*FastAPI\(", text, re.MULTILINE)
        if match:
            return rel, f"{stem}:{match.group(1)}", "fastapi"
        factory = re.search(r"^def\s+(\w*app\w*)\s*\(", text, re.MULTILINE)
        if factory:
            return rel, f"{stem}:{factory.group(1)}", "fastapi-factory"
        match = re.search(r"^(\w+)\s*=\s*(\w+)\(\s*\)", text, re.MULTILINE)
        if match:
            return rel, f"{stem}:{match.group(1)}", "fastapi"
    if "Flask(" in text:
        match = re.search(r"^(\w+)\s*=\s*Flask\(", text, re.MULTILINE)
        if match:
            return rel, f"{stem}:{match.group(1)}", "flask"
        factory = re.search(r"^def\s+(\w*app\w*)\s*\(", text, re.MULTILINE)
        if factory:
            return rel, f"{stem}:{factory.group(1)}", "flask-factory"
    if "uvicorn.run(" in text or "app.run(" in text:
        return rel, f"{stem}:app", "python-web"
    return None


def _default_port(framework: str, text: str) -> int:
    match = _PORT_RE.search(text or "")
    if match:
        value = int(match.group(1))
        if 1 <= value <= 65535:
            return value
    return {
        "fastapi": 8000,
        "flask": 5000,
        "django": 8000,
        "streamlit": 8501,
        "gradio": 7860,
        "express": 3000,
        "nextjs": 3000,
        "vite": 5173,
        "static": 80,
    }.get(framework, 8000)


def _env_names(root: Path, files: Iterable[str]) -> list[str]:
    names: set[str] = set()
    for rel in files:
        if Path(rel).suffix not in SOURCE_EXT and not rel.endswith((".env.example", ".env.sample", ".env.template")):
            continue
        path = root / rel
        try:
            if path.stat().st_size > 400_000:
                continue
        except OSError:
            continue
        text = _read(path)
        if rel.endswith((".env.example", ".env.sample", ".env.template")):
            for line in text.splitlines():
                match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)=", line)
                if match:
                    names.add(match.group(1))
            continue
        for pattern in _ENV_PATTERNS:
            names.update(pattern.findall(text))
    return sorted(n for n in names if n not in _ENV_NOISE)


def _health_path(root: Path, files: Iterable[str]) -> str:
    for rel in files:
        if not rel.endswith((".py", ".js", ".ts", ".go", ".rb")):
            continue
        text = _read(root / rel)
        for candidate in ("/healthz", "/health", "/health_check", "/api/health", "/status"):
            if candidate in text:
                return candidate
    return "/"


def _git_repo_url(root: Path) -> str | None:
    config = root / ".git" / "config"
    if not config.exists():
        return None
    text = _read(config)
    match = re.search(r"\[remote \"origin\"\][^\[]*?url\s*=\s*(\S+)", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _monorepo_layout(root: Path) -> tuple[bool, list[str]]:
    subprojects: list[str] = []
    pkg = _load_package_json(root)
    workspaces = pkg.get("workspaces")
    if isinstance(workspaces, dict):
        workspaces = workspaces.get("packages")
    for pattern in workspaces or []:
        for match in root.glob(str(pattern).rstrip("/") + "/package.json"):
            subprojects.append(str(match.parent.relative_to(root)))
    if (root / "pnpm-workspace.yaml").exists():
        text = _read(root / "pnpm-workspace.yaml")
        for match in re.finditer(r"-\s*[\"']?([^\"'\n]+)", text):
            for pkg_json in root.glob(match.group(1).strip().rstrip("/") + "/package.json"):
                rel = str(pkg_json.parent.relative_to(root))
                if rel not in subprojects:
                    subprojects.append(rel)
    if (root / "go.work").exists():
        text = _read(root / "go.work")
        for match in re.finditer(r"use\s+\(?([^\n)]+)", text):
            for piece in match.group(1).split():
                if (root / piece).is_dir():
                    subprojects.append(piece.strip("./"))
    return (bool(subprojects), sorted(set(subprojects)))


def detect_project(root: str | Path) -> ProjectProfile:
    """Inspect a workspace and return everything needed to containerise it."""

    root = Path(root).resolve()
    files = iter_project_files(root)
    profile = ProjectProfile(root=str(root), name=_slug(root.name))
    profile.file_count = len(files)
    profile.has_dockerfile = any(Path(f).name.lower().startswith("dockerfile") for f in files)
    profile.has_tests = any(
        f.startswith("tests/") or Path(f).name.startswith("test_") or Path(f).name.endswith((".test.js", ".test.ts", ".spec.ts"))
        for f in files
    )
    profile.monorepo, profile.subprojects = _monorepo_layout(root)
    profile.repo_url = _git_repo_url(root)

    package = _load_package_json(root)
    deps = {
        k.lower()
        for group in ("dependencies", "devDependencies", "peerDependencies")
        for k in (package.get(group) or {})
    }
    scripts = package.get("scripts") or {}
    py_deps = _python_dependencies(root, files)

    # ------------------------------------------------------------- node first
    if package:
        profile.language = "node"
        profile.package_manager = _node_package_manager(root, package)
        build = _node_script(profile.package_manager, "build", scripts)
        if "next" in deps:
            profile.kind = "node-web"
            profile.framework = "nextjs"
            profile.install_cmd = _node_install(profile.package_manager)
            profile.build_cmd = build or f"{_node_run(profile.package_manager)} build"
            profile.start_cmd = _node_script(profile.package_manager, "start", scripts) or f"{_node_run(profile.package_manager)} start"
            profile.port = _default_port("nextjs", json.dumps(scripts))
            profile.runtime_version = _node_version(root, package)
            profile.entrypoint = _find_node_entry(root, files) or "next.config.js"
            profile.notes.append("Next.js detected: Dockerfile installs prod deps and runs `next start`.")
        elif any(d in deps for d in ("express", "fastify", "koa", "hono", "@hapi/hapi")):
            profile.kind = "node-web"
            profile.framework = next(d for d in ("express", "fastify", "koa", "hono", "@hapi/hapi") if d in deps)
            profile.install_cmd = _node_install(profile.package_manager)
            profile.build_cmd = build
            profile.start_cmd = _node_script(profile.package_manager, "start", scripts) or f"node {_find_node_entry(root, files) or 'index.js'}"
            profile.entrypoint = _find_node_entry(root, files)
            profile.port = _default_port(profile.framework, json.dumps(scripts) + _read(root / (profile.entrypoint or "index.js")))
            profile.runtime_version = _node_version(root, package)
        elif any(d in deps for d in ("vite", "react-scripts", "astro", "@sveltejs/kit", "parcel")):
            profile.kind = "static"
            profile.framework = "vite" if "vite" in deps else ("astro" if "astro" in deps else "spa")
            profile.install_cmd = _node_install(profile.package_manager)
            profile.build_cmd = build or f"{_node_run(profile.package_manager)} build"
            profile.static_dir = _guess_static_build_dir(root, profile.framework)
            profile.start_cmd = f"npx serve -s {profile.static_dir} -l {profile.port}"
            profile.port = 80
            profile.runtime_version = _node_version(root, package)
            profile.notes.append("Static build output is served by nginx in the generated image.")
        elif "discord.js" in deps or "telegraf" in deps or "grammy" in deps or "whatsapp-web.js" in deps:
            profile.kind = "worker"
            profile.framework = "discord.js" if "discord.js" in deps else "bot"
            profile.install_cmd = _node_install(profile.package_manager)
            profile.start_cmd = _node_script(profile.package_manager, "start", scripts) or f"node {_find_node_entry(root, files) or 'index.js'}"
            profile.entrypoint = _find_node_entry(root, files)
            profile.port = 0
            profile.health_path = ""
            profile.runtime_version = _node_version(root, package)
            profile.notes.append("Worker/bot: no HTTP port is exposed; health is a process liveness check.")
        elif package.get("bin"):
            profile.kind = "cli"
            profile.framework = "node-cli"
            profile.install_cmd = _node_install(profile.package_manager)
            profile.build_cmd = build
            profile.entrypoint = _find_node_entry(root, files)
            profile.start_cmd = None
            profile.port = 0
            profile.health_path = ""
        else:
            profile.kind = "node-web"
            profile.framework = "node"
            profile.install_cmd = _node_install(profile.package_manager)
            profile.build_cmd = build
            profile.start_cmd = _node_script(profile.package_manager, "start", scripts) or f"node {_find_node_entry(root, files) or 'index.js'}"
            profile.entrypoint = _find_node_entry(root, files)
            profile.port = _default_port("express", json.dumps(scripts))
            profile.runtime_version = _node_version(root, package)
        profile.test_cmd = _node_test(profile.package_manager, scripts)

    # ----------------------------------------------------------- python second
    elif py_deps or any(f.endswith(".py") for f in files):
        profile.language = "python"
        profile.runtime_version = _python_version(root, py_deps)
        entry, module, hint = _find_python_entry(root, files)
        profile.entrypoint = entry
        profile.app_module = module
        if "fastapi" in py_deps or hint.startswith("fastapi"):
            profile.kind = "python-web"
            profile.framework = "fastapi"
            profile.install_cmd = _pip_install(root)
            profile.app_factory = hint.endswith("-factory")
            profile.start_cmd = (
                f"uvicorn {module or 'main:app'} --host 0.0.0.0 --port ${{PORT}}"
                + (" --factory" if profile.app_factory else "")
            )
        elif "flask" in py_deps or hint.startswith("flask"):
            profile.kind = "python-web"
            profile.framework = "flask"
            profile.install_cmd = _pip_install(root)
            profile.app_factory = hint.endswith("-factory")
            if profile.app_factory:
                target = module or "app:create_app"
                profile.start_cmd = f"python -m flask --app '{target}()' run --host 0.0.0.0 --port ${{PORT}}"
            else:
                profile.start_cmd = f"python -m {module.split(':')[0] if module else 'app'}"
            profile.notes.append("Flask app started with `python -m <module>`; ensure `if __name__ == '__main__'` binds 0.0.0.0.")
        elif "django" in py_deps or (root / "manage.py").exists():
            profile.kind = "python-web"
            profile.framework = "django"
            profile.install_cmd = _pip_install(root)
            profile.start_cmd = "python manage.py runserver 0.0.0.0:${PORT}"
            profile.build_cmd = "python manage.py collectstatic --noinput"
            profile.entrypoint = "manage.py"
            profile.notes.append("Django dev server is used for the generated image; use gunicorn in production.")
        elif "streamlit" in py_deps:
            profile.kind = "python-web"
            profile.framework = "streamlit"
            profile.install_cmd = _pip_install(root)
            profile.start_cmd = f"streamlit run {entry or 'app.py'} --server.address 0.0.0.0 --server.port ${{PORT}}"
        elif "gradio" in py_deps:
            profile.kind = "python-web"
            profile.framework = "gradio"
            profile.install_cmd = _pip_install(root)
            profile.start_cmd = f"python {entry or 'app.py'}"
            profile.port = 7860
        elif "discord.py" in py_deps or "discord" in py_deps or "python-telegram-bot" in py_deps or "aiogram" in py_deps:
            profile.kind = "worker"
            profile.framework = "discord.py" if "discord" in py_deps else "bot"
            profile.install_cmd = _pip_install(root)
            profile.start_cmd = f"python {entry or 'bot.py'}"
            profile.port = 0
            profile.health_path = ""
            profile.notes.append("Worker/bot: no HTTP port is exposed; health is a process liveness check.")
        elif entry or any(f == "setup.py" or f == "pyproject.toml" for f in files):
            profile.kind = "cli" if not _has_web_server(root, files) else "python-web"
            profile.framework = "python-cli" if profile.kind == "cli" else "python"
            profile.install_cmd = _pip_install(root)
            profile.start_cmd = None if profile.kind == "cli" else f"python {entry or 'main.py'}"
            profile.port = _default_port(profile.framework, _read(root / entry) if entry else "")
        profile.test_cmd = "python -m pytest -q"

    # ------------------------------------------------------------- static/html
    elif any(f.endswith(".html") for f in files):
        profile.language = "html"
        profile.framework = "static"
        profile.kind = "static"
        for candidate in (".", "public", "docs", "site"):
            index = root / candidate / "index.html"
            if index.exists():
                profile.static_dir = str(Path(candidate)) if candidate != "." else "."
                break
        profile.static_dir = profile.static_dir or "."
        profile.start_cmd = f"npx serve -s {profile.static_dir} -l {profile.port}"
        profile.port = 80
        profile.base_image = "nginx:1.27-alpine"
        profile.notes.append("Static HTML: copied into nginx, no build step required.")

    if profile.kind == "unknown":
        profile.notes.append(
            "Could not identify an app type. Add a start command to agentdeploy.toml "
            "([deploy] start_cmd = \"...\") or a Dockerfile, then re-run."
        )

    if profile.port:
        profile.start_cmd = (profile.start_cmd or "").replace("${PORT}", str(profile.port)) or None
    profile.env_required = _env_names(root, files)
    profile.health_path = profile.health_path if profile.is_worker else _health_path(root, files)
    if profile.kind in {"python-web", "node-web", "nextjs"} and profile.health_path == "/":
        profile.notes.append("No health endpoint found; `/` is used as the health check path.")
    return profile


# --------------------------------------------------------------------- helpers
def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (slug or "app")[:40]


def _node_package_manager(root: Path, package: dict) -> str:
    if (root / "pnpm-lock.yaml").exists():
        return "pnpm"
    if (root / "yarn.lock").exists():
        return "yarn"
    if (root / "bun.lockb").exists():
        return "bun"
    if package.get("packageManager"):
        return str(package["packageManager"]).split("@")[0]
    return "npm"


def _node_install(pm: str | None) -> str:
    return {
        "pnpm": "corepack enable && pnpm install --frozen-lockfile",
        "yarn": "corepack enable && yarn install --immutable",
        "bun": "bun install",
    }.get(pm or "npm", "npm ci")


def _node_run(pm: str | None) -> str:
    return {"pnpm": "pnpm", "yarn": "yarn", "bun": "bun run"}.get(pm or "npm", "npm run")


def _node_script(pm: str | None, name: str, scripts: dict) -> str | None:
    if name not in scripts:
        return None
    return f"{_node_run(pm)} {name}"


def _node_test(pm: str | None, scripts: dict) -> str | None:
    if "test" in scripts:
        return f"{_node_run(pm)} test"
    return None


def _node_version(root: Path, package: dict) -> str:
    engines = package.get("engines") or {}
    match = re.search(r"(\d+)", str(engines.get("node", "")))
    if match:
        return f"{match.group(1)}-bookworm-slim"
    nvmrc = root / ".nvmrc"
    if nvmrc.exists():
        match = re.search(r"(\d+)", _read(nvmrc))
        if match:
            return f"{match.group(1)}-bookworm-slim"
    return "20-bookworm-slim"


def _find_node_entry(root: Path, files: Iterable[str]) -> str | None:
    package = _load_package_json(root)
    main = package.get("main")
    if main and (root / str(main)).exists():
        return str(main)
    for name in ("server.js", "index.js", "app.js", "src/index.js", "src/server.js", "src/index.ts", "src/server.ts", "bot.js"):
        if name in files:
            return name
    return None


def _guess_static_build_dir(root: Path, framework: str) -> str:
    for candidate in ("dist", "build", "out", "public"):
        if (root / candidate).is_dir() or candidate == "dist":
            return candidate
    return "dist"


def _pip_install(root: Path) -> str:
    if (root / "requirements.txt").exists():
        return "pip install --no-cache-dir -r requirements.txt"
    if (root / "pyproject.toml").exists():
        return "pip install --no-cache-dir ."
    return "pip install --no-cache-dir -r requirements.txt"


def _python_version(root: Path, deps: set[str]) -> str:
    pyproject = _load_pyproject(root)
    requires = str((pyproject.get("project") or {}).get("requires-python") or "")
    match = re.search(r"(\d+)\.(\d+)", requires)
    if match:
        return f"{match.group(1)}.{match.group(2)}-slim"
    for name in (".python-version", "runtime.txt"):
        path = root / name
        if path.exists():
            match = re.search(r"(\d+)\.(\d+)", _read(path))
            if match:
                return f"{match.group(1)}.{match.group(2)}-slim"
    return "3.11-slim"


def _has_web_server(root: Path, files: Iterable[str]) -> bool:
    for rel in files:
        if rel.endswith(".py"):
            text = _read(root / rel)
            if re.search(r"(uvicorn|gunicorn|HTTPServer|aiohttp\.web|serve\()", text):
                return True
    return False
