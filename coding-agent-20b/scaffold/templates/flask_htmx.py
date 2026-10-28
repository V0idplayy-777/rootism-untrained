"""Flask + HTMX starter: server-rendered, one small file, real tests.

No build step, no npm, no client-side state management: forms post to Flask and
HTMX swaps the returned fragment. This is the cheapest way to ship an
interactive internal tool, and it degrades to plain HTML without JavaScript.
"""

from __future__ import annotations

from ..addons import PY_STORE, env_example, gitignore
from ..spec import ProjectOptions, TemplateSpec

APP = '''"""{display_name} - Flask + HTMX app.

Run:   python -m flask --app app run --port {port} --debug
Test:  python -m pytest -q
"""

from __future__ import annotations

import os

from flask import Flask, redirect, render_template, request, url_for

import store

DEMO_OWNER = "demo"


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-only-secret")

    @app.get("/healthz")
    def healthz():
        return {{"status": "ok", "app": "{display_name}"}}

    @app.get("/")
    def index():
        return render_template("index.html", items=store.list_items(DEMO_OWNER))

    @app.post("/items")
    def add_item():
        title = (request.form.get("title") or "").strip()
        if title:
            store.create_item(DEMO_OWNER, title[:200])
        if request.headers.get("HX-Request") == "true":
            return render_template("partials/items.html", items=store.list_items(DEMO_OWNER))
        return redirect(url_for("index"))

    @app.post("/items/<item_id>/toggle")
    def toggle(item_id: str):
        store.toggle_item(DEMO_OWNER, item_id)
        if request.headers.get("HX-Request") == "true":
            return render_template("partials/items.html", items=store.list_items(DEMO_OWNER))
        return redirect(url_for("index"))

    return app


app = create_app()
'''

INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>__DISPLAY_NAME__</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}" />
    <script src="https://unpkg.com/htmx.org@2.0.2" defer></script>
  </head>
  <body>
    <main class="shell">
      <header>
        <h1>__DISPLAY_NAME__</h1>
        <p class="muted">Flask + HTMX starter</p>
      </header>

      <form
        class="row"
        hx-post="/items"
        hx-target="#items"
        hx-swap="innerHTML"
        hx-on::after-request="if(event.detail.successful) this.reset()"
        action="/items"
        method="post"
      >
        <input name="title" placeholder="Add an item..." aria-label="New item" required maxlength="200" />
        <button type="submit">Add</button>
      </form>

      <div id="items">
        {% include "partials/items.html" %}
      </div>
    </main>
  </body>
</html>
"""

PARTIAL = """<ul class="list">
  {% for item in items %}
  <li class="{{ 'done' if item.done else '' }}">
    <form hx-post="/items/{{ item.id }}/toggle" hx-target="#items" hx-swap="innerHTML" method="post">
      <button class="link" type="submit">{{ '☑' if item.done else '☐' }} {{ item.title }}</button>
    </form>
  </li>
  {% else %}
  <li class="muted">Nothing yet - add your first item.</li>
  {% endfor %}
</ul>
"""

STYLE = """* { box-sizing: border-box; }
body {
  margin: 0;
  font: 16px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background: #f6f7fb;
  color: #14161c;
}
.shell { max-width: 620px; margin: 0 auto; padding: 56px 20px; }
h1 { margin: 0 0 4px; font-size: 32px; letter-spacing: -0.02em; }
.muted { color: #667085; }
.row { display: flex; gap: 8px; margin: 24px 0 16px; }
input {
  flex: 1; padding: 11px 13px; border-radius: 9px; border: 1px solid #e2e5ee;
  background: #fff; font: inherit;
}
button {
  padding: 11px 16px; border-radius: 9px; border: 1px solid transparent;
  background: #2563eb; color: #fff; font: inherit; font-weight: 600; cursor: pointer;
}
button.link { background: none; color: inherit; padding: 0; font-weight: 500; text-align: left; }
.list { list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; }
.list li { background: #fff; border: 1px solid #e2e5ee; border-radius: 10px; padding: 12px 14px; }
.list li.done { opacity: 0.6; text-decoration: line-through; }
"""

TESTS = '''"""Route tests with Flask's test client (no server, no browser)."""

from __future__ import annotations

import pytest  # noqa: E402

from app import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Each test gets its own SQLite file, so tests cannot affect each other."""

    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.db"))
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Add an item" in response.data


def test_add_item_renders_fragment_for_htmx(client):
    response = client.post(
        "/items",
        data={"title": "buy milk"},
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    assert b"buy milk" in response.data
    assert b"<html" not in response.data.lower()  # fragment, not a full page


def test_add_item_redirects_without_htmx(client):
    response = client.post("/items", data={"title": "plain form"})
    assert response.status_code == 302


def test_empty_title_is_ignored(client):
    response = client.post("/items", data={"title": "   "}, headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert b"Nothing yet" in response.data


def test_toggle_flips_done(client):
    client.post("/items", data={"title": "toggle me"}, headers={"HX-Request": "true"})
    with client.application.app_context():
        import store

        item = store.list_items("demo")[0]
    response = client.post(f"/items/{item['id']}/toggle", headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert b"done" in response.data
'''

REQUIREMENTS = """Flask==3.0.3
# dev / test
pytest==8.2.2
"""


def _format(text: str, options: ProjectOptions) -> str:
    """Render a Python template (literal braces are escaped as ``{{``/``}}``)."""

    return text.format(
        display_name=options.display_name,
        module=options.module,
        port=options.port,
        name=options.name,
    )


def _tokens(text: str, options: ProjectOptions) -> str:
    """Render a Jinja/HTML template: leave Jinja's braces alone."""

    return text.replace("__DISPLAY_NAME__", options.display_name)


def build(options: ProjectOptions) -> dict[str, str]:
    return {
        "app.py": _format(APP, options),
        "store.py": PY_STORE,
        "templates/index.html": _tokens(INDEX_HTML, options),
        "templates/partials/items.html": PARTIAL,
        "static/style.css": STYLE,
        "tests/test_app.py": TESTS,
        "requirements.txt": REQUIREMENTS,
        ".env.example": env_example(options, include_secret=False),
        ".gitignore": gitignore(),
    }


SPEC = TemplateSpec(
    id="flask-htmx",
    name="Flask + HTMX",
    tagline="Server-rendered interactive page, no build step",
    description=(
        "A small Python web app using Flask templates and HTMX for interactivity. "
        "Best for internal tools and simple forms - no JavaScript build, no npm, and the "
        "page still works without JavaScript."
    ),
    stack="Python 3.11 · Flask · Jinja · HTMX · SQLite",
    kind="python-web",
    tags=["web", "python", "flask", "htmx", "internal-tool", "simple"],
    supports_auth=False,
    supports_database=True,
    default_port=5000,
    build=build,
    good_for=["internal tools", "forms and admin pages", "prototypes in an afternoon"],
    run_command="python -m venv .venv && source .venv/bin/activate\npip install -r requirements.txt\npython -m flask --app app run --port 5000 --debug",
    test_command="python -m pytest -q",
    preview="web",
    next_steps=[
        "pip install -r requirements.txt && python -m flask --app app run --debug",
        "python -m pytest -q",
        "agent deploy (Dockerfile is generated for you)",
    ],
)
