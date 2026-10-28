"""Next.js (App Router) full-stack starter.

Dependency-free business logic (``lib/store.js``, ``src/auth.js``) so the test
suite runs with plain `node --test` - no `npm install` needed to check the
generated project. The API routes and page are thin adapters over those modules.
"""

from __future__ import annotations

from ..addons import NODE_AUTH, NODE_AUTH_TEST, env_example, gitignore
from ..spec import ProjectOptions, TemplateSpec

PACKAGE_JSON = """{
  "name": "__NAME__",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "next dev -p __PORT__",
    "build": "next build",
    "start": "next start -p __PORT__",
    "test": "node --test"
  },
  "dependencies": {
    "next": "14.2.5",
    "react": "18.3.1",
    "react-dom": "18.3.1"
  }
}
"""

NEXT_CONFIG = """/** @type {import('next').NextConfig} */
const nextConfig = {
  // Standalone output keeps the production Docker image small.
  output: "standalone",
  reactStrictMode: true,
};

export default nextConfig;
"""

JS_CONFIG = """{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": { "@/*": ["./*"] }
  }
}
"""

STORE = '''// Storage for this starter: one JSON file, atomic writes, no dependencies.
// Swap this module for Postgres/Prisma/Supabase later - the route handlers only
// use the functions exported at the bottom of this file.
import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const EMPTY = { items: [], users: [] };

// Resolved per call so tests can point DATA_FILE at a temp directory.
export function dataFile() {
  return process.env.DATA_FILE || join(process.cwd(), "data", "store.json");
}

async function readAll() {
  try {
    const raw = await readFile(dataFile(), "utf8");
    const parsed = JSON.parse(raw);
    return { items: parsed.items || [], users: parsed.users || [] };
  } catch (error) {
    if (error.code === "ENOENT") return { ...EMPTY, items: [], users: [] };
    throw error;
  }
}

async function writeAll(data) {
  const target = dataFile();
  await mkdir(dirname(target), { recursive: true });
  const tmp = `${target}.${randomUUID()}.tmp`;
  await writeFile(tmp, JSON.stringify(data, null, 2), "utf8");
  await rename(tmp, target); // atomic on POSIX: readers never see a half file
}

export async function listItems(ownerId) {
  const data = await readAll();
  return data.items.filter((item) => item.ownerId === ownerId).sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export async function createItem(ownerId, title) {
  const text = String(title || "").trim();
  if (!text) throw new Error("title is required");
  if (text.length > 200) throw new Error("title is too long");
  const data = await readAll();
  const item = { id: randomUUID(), ownerId, title: text, done: false, createdAt: new Date().toISOString() };
  data.items.push(item);
  await writeAll(data);
  return item;
}

export async function toggleItem(ownerId, itemId) {
  const data = await readAll();
  const item = data.items.find((row) => row.id === itemId && row.ownerId === ownerId);
  if (!item) return null;
  item.done = !item.done;
  await writeAll(data);
  return item;
}

export async function findUserByEmail(email) {
  const data = await readAll();
  return data.users.find((user) => user.email === String(email || "").toLowerCase()) || null;
}

export async function createUser(email, passwordHash) {
  const data = await readAll();
  const normalized = String(email || "").trim().toLowerCase();
  if (!normalized.includes("@")) throw new Error("a valid email is required");
  if (data.users.some((user) => user.email === normalized)) throw new Error("email already registered");
  const user = { id: randomUUID(), email: normalized, passwordHash, createdAt: new Date().toISOString() };
  data.users.push(user);
  await writeAll(data);
  return user;
}

export async function findUserById(id) {
  const data = await readAll();
  return data.users.find((user) => user.id === id) || null;
}
'''

STORE_TEST = '''import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeEach, test } from "node:test";

import { createItem, createUser, findUserByEmail, listItems, toggleItem } from "../lib/store.js";

beforeEach(async () => {
  const dir = await mkdtemp(join(tmpdir(), "starter-"));
  process.env.DATA_FILE = join(dir, "store.json");
});

test("creating an item persists it for that owner only", async () => {
  const item = await createItem("alice", " first task ");
  assert.equal(item.title, "first task");
  assert.equal(item.done, false);
  assert.equal((await listItems("alice")).length, 1);
  assert.equal((await listItems("bob")).length, 0);
});

test("empty titles are rejected", async () => {
  await assert.rejects(() => createItem("alice", "   "), /title is required/);
});

test("toggling flips done and refuses other owners", async () => {
  const item = await createItem("alice", "task");
  const toggled = await toggleItem("alice", item.id);
  assert.equal(toggled.done, true);
  assert.equal(await toggleItem("bob", item.id), null);
});

test("users are normalised and unique", async () => {
  const user = await createUser("Person@Example.com", "hash");
  assert.equal(user.email, "person@example.com");
  assert.equal((await findUserByEmail("PERSON@example.com")).id, user.id);
  await assert.rejects(() => createUser("person@example.com", "hash"), /already registered/);
});
'''

HEALTH_ROUTE = '''import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json({ status: "ok", app: "__DISPLAY_NAME__" });
}
'''

ITEMS_ROUTE = '''import { NextResponse } from "next/server";

import { createItem, listItems, toggleItem } from "@/lib/store";
__AUTH_IMPORT__

export const dynamic = "force-dynamic";

async function ownerFrom(request) {
__OWNER_BODY__
}

export async function GET(request) {
  const owner = await ownerFrom(request);
  if (!owner) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  return NextResponse.json(await listItems(owner));
}

export async function POST(request) {
  const owner = await ownerFrom(request);
  if (!owner) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
  }
  try {
    const item = await createItem(owner, body?.title);
    return NextResponse.json(item, { status: 201 });
  } catch (error) {
    return NextResponse.json({ error: error.message }, { status: 422 });
  }
}

export async function PATCH(request) {
  const owner = await ownerFrom(request);
  if (!owner) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { id } = await request.json().catch(() => ({}));
  const item = await toggleItem(owner, id);
  if (!item) return NextResponse.json({ error: "not found" }, { status: 404 });
  return NextResponse.json(item);
}
'''

OWNER_DEMO = '''  // Single-user mode: everything belongs to "demo".
  return "demo";'''

OWNER_AUTH = '''  const header = request.headers.get("authorization") || "";
  const [scheme, token] = header.split(" ");
  if (scheme?.toLowerCase() !== "bearer" || !token) return null;
  try {
    return decodeToken(token).sub;
  } catch {
    return null;
  }'''

AUTH_ROUTE = '''import { NextResponse } from "next/server";

import { createToken, decodeToken, hashPassword, verifyPassword } from "@/src/auth";
import { createUser, findUserByEmail, findUserById } from "@/lib/store";

export const dynamic = "force-dynamic";

async function readCredentials(request) {
  try {
    const body = await request.json();
    return { email: String(body?.email || ""), password: String(body?.password || "") };
  } catch {
    return null;
  }
}

export async function POST(request, { params }) {
  const action = params.action;
  const credentials = await readCredentials(request);

  if (action === "register") {
    if (!credentials) return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
    try {
      const user = await createUser(credentials.email, hashPassword(credentials.password));
      return NextResponse.json({ id: user.id, email: user.email, token: createToken(user.id) }, { status: 201 });
    } catch (error) {
      return NextResponse.json({ error: error.message }, { status: 422 });
    }
  }

  if (action === "login") {
    if (!credentials) return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
    const user = await findUserByEmail(credentials.email);
    if (!user || !verifyPassword(credentials.password, user.passwordHash)) {
      return NextResponse.json({ error: "invalid email or password" }, { status: 401 });
    }
    return NextResponse.json({ token: createToken(user.id) });
  }

  if (action === "me") {
    const [scheme, token] = (request.headers.get("authorization") || "").split(" ");
    try {
      const payload = decodeToken(token || "");
      const user = await findUserById(payload.sub);
      if (!user) return NextResponse.json({ error: "not found" }, { status: 404 });
      return NextResponse.json({ id: user.id, email: user.email });
    } catch {
      return NextResponse.json({ error: "unauthorized" }, { status: 401 });
    }
  }

  return NextResponse.json({ error: "unknown action" }, { status: 404 });
}
'''

LAYOUT = '''import "./globals.css";

export const metadata = {
  title: "__DISPLAY_NAME__",
  description: "__DESCRIPTION__",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
'''

PAGE = '''"use client";

import { useEffect, useState } from "react";
__AUTH_STATE__
const API = "/api/items";

export default function Home() {
  const [items, setItems] = useState([]);
  const [title, setTitle] = useState("");
  const [error, setError] = useState("");
__AUTH_HOOKS__
  async function load() {
    const response = await fetch(API__AUTH_HEADER__);
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
    const response = await fetch(API, {
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
    await fetch(API, {
      method: "PATCH",
      headers: { "Content-Type": "application/json"__AUTH_HEADER_EXTRA__ },
      body: JSON.stringify({ id }),
    });
    load();
  }

  return (
    <main className="shell">
      <header>
        <h1>__DISPLAY_NAME__</h1>
        <p className="muted">Next.js full-stack starter</p>
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
              {item.done ? "\\u2611" : "\\u2610"} {item.title}
            </button>
          </li>
        ))}
        {items.length === 0 && <li className="muted">Nothing yet - add your first item.</li>}
      </ul>
    </main>
  );
}
'''

AUTH_STATE = '''const [email, setEmail] = useState("");
const [password, setPassword] = useState("");
const [token, setToken] = useState(() =>
  typeof window === "undefined" ? "" : window.localStorage.getItem("token") || ""
);'''

AUTH_HOOKS = '''
  async function authenticate(mode) {
    const response = await fetch(`/api/auth/${mode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      setError(data.error || "Could not sign in.");
      return;
    }
    window.localStorage.setItem("token", data.token);
    setToken(data.token);
    setPassword("");
    load();
  }
'''

AUTH_UI = '''        <div className="authbar">
          <input placeholder="email" value={email} onChange={(event) => setEmail(event.target.value)} />
          <input
            placeholder="password (8+ chars)"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
          <button type="button" onClick={() => authenticate("login")}>Sign in</button>
          <button type="button" className="ghost" onClick={() => authenticate("register")}>Sign up</button>
        </div>'''

CSS = """:root {
  color-scheme: light dark;
  --fg: #14161c;
  --bg: #f6f7fb;
  --card: #fff;
  --line: #e2e5ee;
  --accent: #111827;
  --muted: #667085;
}
@media (prefers-color-scheme: dark) {
  :root { --fg: #e8ebf3; --bg: #0f1218; --card: #171c24; --line: #262d3a; --accent: #e5e7eb; --muted: #98a2b3; }
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
  background: var(--accent); color: var(--bg); font: inherit; font-weight: 600; cursor: pointer;
}
button.ghost { background: transparent; border-color: var(--line); color: inherit; }
.list { list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; }
.list li { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.list li.done { opacity: 0.6; text-decoration: line-through; }
.link { background: none; color: inherit; padding: 0; font-weight: 500; text-align: left; }
.error { color: #c0392b; }
.authbar { display: flex; gap: 8px; align-items: center; margin-top: 12px; flex-wrap: wrap; }
"""


def _tokens(template: str, options: ProjectOptions) -> str:
    return (
        template.replace("__NAME__", options.name)
        .replace("__DISPLAY_NAME__", options.display_name)
        .replace("__DESCRIPTION__", options.description or options.display_name)
        .replace("__PORT__", str(options.port))
    )


def build(options: ProjectOptions) -> dict[str, str]:
    auth = options.use_auth
    files: dict[str, str] = {
        "package.json": _tokens(PACKAGE_JSON, options),
        "next.config.mjs": NEXT_CONFIG,
        "jsconfig.json": JS_CONFIG,
        "app/layout.js": _tokens(LAYOUT, options),
        "app/page.js": _tokens(PAGE, options).replace("__AUTH_STATE__", AUTH_STATE if auth else "").replace(
            "__AUTH_HOOKS__", AUTH_HOOKS if auth else ""
        ).replace("__AUTH_UI__", AUTH_UI if auth else "").replace(
            "__AUTH_HEADER__", ', { headers: token ? { Authorization: `Bearer ${token}` } : {} }' if auth else ""
        ).replace(
            "__AUTH_HEADER_EXTRA__",
            ", ...(token ? { Authorization: `Bearer ${token}` } : {})" if auth else "",
        ),
        "app/globals.css": CSS,
        "app/api/health/route.js": _tokens(HEALTH_ROUTE, options),
        "app/api/items/route.js": ITEMS_ROUTE.replace("__AUTH_IMPORT__", 'import { decodeToken } from "@/src/auth";' if auth else "").replace(
            "__OWNER_BODY__", OWNER_AUTH if auth else OWNER_DEMO
        ),
        "lib/store.js": STORE,
        "test/store.test.js": STORE_TEST,
        ".env.example": env_example(options),
        ".gitignore": gitignore(),
    }
    if auth:
        files["src/auth.js"] = NODE_AUTH
        files["test/auth.test.js"] = NODE_AUTH_TEST
        files["app/api/auth/[action]/route.js"] = AUTH_ROUTE
    return files


SPEC = TemplateSpec(
    id="nextjs-fullstack",
    name="Next.js full-stack",
    tagline="One app: React UI, API routes and storage, deployed as a single service",
    description=(
        "A single Next.js application that serves both the interface and the JSON API. "
        "Good when you want one deployable unit and a modern React UI without managing a "
        "separate backend, and it is the path of least resistance for Vercel."
    ),
    stack="Next.js 14 (App Router) · React 18 · Node 20+",
    kind="nextjs",
    tags=["web", "full-stack", "react", "next", "javascript", "typescript-ready"],
    supports_auth=True,
    supports_database=True,
    default_port=3000,
    build=build,
    good_for=["single-service apps", "Vercel deploys", "marketing + app in one", "React-first teams"],
    run_command="npm install\nnpm run dev        # http://localhost:3000\nnpm test           # node --test (no install needed)",
    test_command="npm test",
    preview="web",
    next_steps=[
        "npm install && npm run dev",
        "npm test (node --test, storage + auth contracts)",
        "agent deploy --provider vercel (or local Docker)",
    ],
)
