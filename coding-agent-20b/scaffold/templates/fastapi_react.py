"""FastAPI + React starter: typed API, SQLite/Postgres, optional auth, real tests."""

from __future__ import annotations

from ..addons import PY_STORE, env_example, gitignore, python_auth
from ..spec import ProjectOptions, TemplateSpec

MAIN = '''"""{display_name} - FastAPI application.

Run locally:  uvicorn {module}.main:app --reload --port {port}
Health:       GET /healthz
API:          GET/POST /api/items, POST /api/items/{{item_id}}/toggle
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import store
{auth_import}

APP_NAME = "{display_name}"
DEMO_OWNER = "demo"


class ItemIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ItemOut(BaseModel):
    id: str
    owner_id: str
    title: str
    done: int


if AUTH_ENABLED:
    # `current_user` is already a FastAPI dependency: it reads the Authorization
    # header, verifies the token and returns the user id.
    owner_for = current_user
else:

    def owner_for() -> str:
        """Single-user mode: everything belongs to the demo owner."""

        return DEMO_OWNER


def create_app() -> FastAPI:
    app = FastAPI(title=APP_NAME, version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict:
        return {{"status": "ok", "app": APP_NAME}}

    @app.get("/api/items", response_model=list[ItemOut], tags=["items"])
    def list_items(owner_id: str = Depends(owner_for)) -> list[dict]:
        return store.list_items(owner_id)

    @app.post("/api/items", response_model=ItemOut, status_code=201, tags=["items"])
    def create_item(payload: ItemIn, owner_id: str = Depends(owner_for)) -> dict:
        return store.create_item(owner_id, payload.title.strip())

    @app.post("/api/items/{{item_id}}/toggle", response_model=ItemOut, tags=["items"])
    def toggle_item(item_id: str, owner_id: str = Depends(owner_for)) -> dict:
        item = store.toggle_item(owner_id, item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="item not found")
        return item
{auth_routes}
    # Serve the built React app when it exists (single-container deploy).
    web_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
    if web_dist.is_dir():
        app.mount("/assets", StaticFiles(directory=web_dist / "assets"), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(web_dist / "index.html")
    else:

        @app.get("/", include_in_schema=False)
        def placeholder() -> HTMLResponse:
            """Shown until the React app is built (keeps the URL useful)."""

            return HTMLResponse(
                "<!doctype html><title>{display_name}</title>"
                "<main style='font:16px/1.6 system-ui;max-width:38rem;margin:4rem auto;padding:0 1.5rem'>"
                "<h1>{display_name}</h1>"
                "<p>The API is running. The React frontend has not been built yet:</p>"
                "<pre style='background:#f4f4f5;padding:1rem;border-radius:.5rem'>"
                "cd web &amp;&amp; npm install &amp;&amp; npm run build</pre>"
                "<p><a href='/docs'>API docs</a> · <a href='/healthz'>health</a> · "
                "<a href='/api/items'>items</a></p></main>"
            )

    return app


app = create_app()
'''

AUTH_IMPORT = '''from .auth_routes import current_user
from .auth_routes import router as auth_router

AUTH_ENABLED = True
'''

NO_AUTH_IMPORT = '''AUTH_ENABLED = False
'''

AUTH_ROUTES = '''
    app.include_router(auth_router)
'''

APP_INIT = '"""Application package."""\n'

MODELS = '''"""Request/response models (kept separate so routes stay readable)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ItemIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ItemOut(BaseModel):
    id: str
    owner_id: str
    title: str
    done: int
'''

TESTS = '''"""API contract tests - no network, no browser."""

from __future__ import annotations

import os
import uuid

os.environ.setdefault("SQLITE_PATH", "data/test.db")
os.environ.setdefault("APP_ENV", "test")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from {module}.main import app  # noqa: E402

AUTH_ENABLED = {auth_enabled}
PASSWORD = "s3cret-passphrase"
client = TestClient(app)


@pytest.fixture()
def headers():
    """Bearer token when auth is on; empty headers in single-user mode."""

    if not AUTH_ENABLED:
        return {{}}
    email = f"user-{{uuid.uuid4().hex[:8]}}@example.com"
    response = client.post("/auth/register", json={{"email": email, "password": PASSWORD}})
    assert response.status_code == 201, response.text
    return {{"Authorization": f"Bearer {{response.json()['token']}}"}}
'''

TESTS_BODY = '''

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_create_and_list_item(headers):
    created = client.post("/api/items", json={"title": "write the tests first"}, headers=headers)
    assert created.status_code == 201, created.text
    item = created.json()
    assert item["title"] == "write the tests first"
    assert item["done"] == 0

    listed = client.get("/api/items", headers=headers)
    assert listed.status_code == 200
    assert any(row["id"] == item["id"] for row in listed.json())


def test_empty_title_is_rejected(headers):
    assert client.post("/api/items", json={"title": ""}, headers=headers).status_code == 422


def test_toggle_missing_item_is_404(headers):
    assert client.post("/api/items/does-not-exist/toggle", headers=headers).status_code == 404
'''

AUTH_TESTS = '''

def test_register_login_and_me():
    email = f"person-{uuid.uuid4().hex[:8]}@example.com"
    registered = client.post("/auth/register", json={"email": email, "password": PASSWORD})
    assert registered.status_code == 201, registered.text
    token = registered.json()["token"]

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == email

    login = client.post("/auth/login", json={"email": email, "password": PASSWORD})
    assert login.status_code == 200
    assert login.json()["token"]


def test_items_require_auth():
    assert client.get("/api/items").status_code == 401


def test_wrong_password_is_rejected():
    email = f"other-{uuid.uuid4().hex[:8]}@example.com"
    client.post("/auth/register", json={"email": email, "password": PASSWORD})
    response = client.post("/auth/login", json={"email": email, "password": "wrong-password"})
    assert response.status_code == 401
'''

REQUIREMENTS = """fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.4
# dev / test
pytest==8.2.2
httpx==0.27.0
"""

PACKAGE_JSON = """{
  "name": "__NAME__-web",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.1",
    "vite": "^5.4.0"
  }
}
"""

VITE_CONFIG = """import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:__PORT__", "/auth": "http://127.0.0.1:__PORT__" },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
"""

INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{display_name}</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
"""

MAIN_JSX = """import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
"""

APP_JSX = """import { useEffect, useState } from "react";
__AUTH_IMPORT__
const API = "/api";
__AUTH_STATE__
export default function App() {
  const [items, setItems] = useState([]);
  const [title, setTitle] = useState("");
  const [error, setError] = useState("");
__AUTH_HOOKS__
  async function load() {
    const response = await fetch(`${API}/items`__AUTH_HEADER__);
    if (!response.ok) {
      setError(response.status === 401 ? "Sign in to see your items." : "Could not load items.");
      setItems([]);
      return;
    }
    setItems(await response.json());
    setError("");
  }

  useEffect(() => {
    load();
  }, []);

  async function add(event) {
    event.preventDefault();
    if (!title.trim()) return;
    const response = await fetch(`${API}/items`, {
      method: "POST",
      headers: { "Content-Type": "application/json"__AUTH_HEADER_EXTRA__ },
      body: JSON.stringify({ title }),
    });
    if (!response.ok) {
      setError("Could not add that item.");
      return;
    }
    setTitle("");
    load();
  }

  async function toggle(id) {
    await fetch(`${API}/items/${id}/toggle`, { method: "POST"__AUTH_TOGGLE_EXTRA__ });
    load();
  }

  return (
    <main className="shell">
      <header>
        <h1>{display_name}</h1>
        <p className="muted">FastAPI + React starter</p>
__AUTH_UI__
      </header>
      {error && <p className="error">{error}</p>}
      <form onSubmit={add} className="row">
        <input
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          placeholder="Add an item..."
          aria-label="New item"
        />
        <button type="submit">Add</button>
      </form>
      <ul className="list">
        {items.map((item) => (
          <li key={item.id} className={item.done ? "done" : ""}>
            <button className="link" onClick={() => toggle(item.id)}>
              {item.done ? "\u2611" : "\u2610"} {item.title}
            </button>
          </li>
        ))}
        {items.length === 0 && <li className="muted">Nothing yet - add your first item.</li>}
      </ul>
    </main>
  );
}
"""

STYLES = """:root {
  color-scheme: light dark;
  --fg: #14161c;
  --bg: #f6f7fb;
  --card: #fff;
  --line: #e2e5ee;
  --accent: #3b5bfd;
  --muted: #667085;
}
@media (prefers-color-scheme: dark) {
  :root { --fg: #e8ebf3; --bg: #0f1218; --card: #171c24; --line: #262d3a; --accent: #7d97ff; --muted: #98a2b3; }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font: 16px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--fg);
}
.shell { max-width: 640px; margin: 0 auto; padding: 56px 20px; }
h1 { margin: 0 0 4px; font-size: 32px; letter-spacing: -0.02em; }
.muted { color: var(--muted); }
.row { display: flex; gap: 8px; margin: 24px 0 16px; }
input {
  flex: 1; padding: 11px 13px; border-radius: 9px; border: 1px solid var(--line);
  background: var(--card); color: inherit; font: inherit;
}
button {
  padding: 11px 16px; border-radius: 9px; border: 1px solid transparent;
  background: var(--accent); color: #fff; font: inherit; font-weight: 600; cursor: pointer;
}
button:hover { filter: brightness(1.07); }
.list { list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; }
.list li { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.list li.done { opacity: 0.6; text-decoration: line-through; }
.link { background: none; color: inherit; padding: 0; font-weight: 500; text-align: left; }
.error { color: #c0392b; }
.authbar { display: flex; gap: 8px; align-items: center; margin-top: 12px; }
"""

AUTH_UI = """        <div className="authbar">
          <input
            placeholder="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            aria-label="Email"
          />
          <input
            placeholder="password (8+ chars)"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            aria-label="Password"
          />
          <button onClick={() => authenticate("login")} type="button">Sign in</button>
          <button onClick={() => authenticate("register")} type="button" className="ghost">Sign up</button>
        </div>
"""

AUTH_STATE = """const [email, setEmail] = useState("");
const [password, setPassword] = useState("");
const [token, setToken] = useState(() => localStorage.getItem("token") || "");
"""

AUTH_HOOKS = """
  async function authenticate(mode) {
    const response = await fetch(`/auth/${mode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      setError(data.detail || "Could not sign in.");
      return;
    }
    localStorage.setItem("token", data.token);
    setToken(data.token);
    setPassword("");
    load();
  }
"""


def _render(template: str, options: ProjectOptions, **extra) -> str:
    """Render Python templates (they escape literal braces as ``{{``/``}}``)."""

    return template.format(
        module=options.module,
        name=options.name,
        display_name=options.display_name,
        port=options.port,
        description=options.description,
        **extra,
    )


def _tokens(template: str, options: ProjectOptions, **extra) -> str:
    """Render JS/HTML/JSON templates: only ``__TOKEN__`` markers are replaced."""

    values = {
        "NAME": options.name,
        "DISPLAY_NAME": options.display_name,
        "MODULE": options.module,
        "PORT": str(options.port),
        "DESCRIPTION": options.description,
    }
    values.update({k.upper(): v for k, v in extra.items()})
    for key, value in values.items():
        template = template.replace(f"__{key}__", str(value))
    return template


def build(options: ProjectOptions) -> dict[str, str]:
    module = options.module
    auth = options.use_auth
    files: dict[str, str] = {
        f"{module}/__init__.py": APP_INIT,
        f"{module}/main.py": _render(
            MAIN,
            options,
            auth_import=AUTH_IMPORT if auth else NO_AUTH_IMPORT,
            auth_routes=AUTH_ROUTES if auth else "",
        ),
        f"{module}/models.py": MODELS,
        f"{module}/store.py": PY_STORE,
        "tests/test_api.py": _render(TESTS, options, auth_enabled=bool(auth))
        + TESTS_BODY
        + (AUTH_TESTS if auth else ""),
        "tests/conftest.py": (
            '"""Keep test data out of the project database."""\n\n'
            "import os\nimport tempfile\n\n"
            'os.environ.setdefault("SQLITE_PATH", os.path.join(tempfile.mkdtemp(), "test.db"))\n'
        ),
        "requirements.txt": REQUIREMENTS,
        "web/package.json": _tokens(PACKAGE_JSON, options),
        "web/vite.config.js": _tokens(VITE_CONFIG, options),
        "web/index.html": _tokens(INDEX_HTML, options),
        "web/src/main.jsx": MAIN_JSX,
        "web/src/App.jsx": _tokens(
            APP_JSX,
            options,
            auth_import="",
            auth_state=AUTH_STATE if auth else "",
            auth_hooks=AUTH_HOOKS if auth else "",
            auth_header=", { headers: token ? { Authorization: `Bearer ${token}` } : {} }" if auth else "",
            auth_header_extra=", ...(token ? { Authorization: `Bearer ${token}` } : {})" if auth else "",
            auth_toggle_extra=", headers: token ? { Authorization: `Bearer ${token}` } : {}" if auth else "",
            auth_ui=AUTH_UI if auth else "",
        ),
        "web/src/styles.css": STYLES,
        ".env.example": env_example(options),
        ".gitignore": gitignore(),
    }
    if auth:
        files.update(python_auth(module))
    return files


SPEC = TemplateSpec(
    id="fastapi-react",
    name="FastAPI + React",
    tagline="Typed Python API, SQLite/Postgres storage, React frontend",
    description=(
        "A full-stack web app: Python (FastAPI) backend with a real database and JSON API, "
        "and a React frontend built with Vite. Good default when you want a normal API and a "
        "custom interface, with user accounts optional."
    ),
    stack="Python 3.11 · FastAPI · SQLite/Postgres · React 18 (Vite)",
    kind="python-web",
    tags=["web", "api", "full-stack", "react", "python", "database"],
    supports_auth=True,
    supports_database=True,
    default_port=8000,
    build=build,
    good_for=["dashboards", "SaaS MVPs", "internal tools", "apps with logins"],
    run_command=(
        "python -m venv .venv && source .venv/bin/activate\n"
        "pip install -r requirements.txt\n"
        "uvicorn {module}.main:app --reload --port 8000\n"
        "# frontend, in a second terminal:\n"
        "cd web && npm install && npm run dev"
    ),
    test_command="python -m pytest -q",
    preview="web",
    next_steps=[
        "pip install -r requirements.txt && uvicorn {module}.main:app --reload",
        "Run the tests: python -m pytest -q",
        "Add your first feature, then `agent verify` to keep the suite green",
    ],
)
