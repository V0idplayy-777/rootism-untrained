"""First-class auth providers: Clerk, Supabase Auth, Auth.js.

Each provider returns files + env wiring that are dropped into a project
when --auth-provider is selected. Unlike the JWT stub, these have working
callbacks, env injection, and a note about required secrets.

The JWT stub remains the default (no external dependency).
"""

from __future__ import annotations

from pathlib import Path


CLERK_BACKEND_PY = '''"""Clerk auth integration (FastAPI).

Requires: CLERK_SECRET_KEY and CLERK_PUBLISHABLE_KEY in .env.
The frontend will use @clerk/clerk-react; backend verifies JWTs via Clerk's JWKS.

Set CLERK_SECRET_KEY from https://dashboard.clerk.com -> API Keys.
"""

from __future__ import annotations

import os

import httpx
from fastapi import Depends, Header, HTTPException, status
from jose import jwt  # python-jose[cryptography]

CLERK_PUBLISHABLE_KEY = os.environ.get("CLERK_PUBLISHABLE_KEY", "")
CLERK_SECRET_KEY = os.environ.get("CLERK_SECRET_KEY", "")


def _clerk_jwks_url() -> str:
    # Clerk's JWKS is at https://<frontend_api_url>/.well-known/jwks.json
    # For simplicity, use CLERK_JWKS_URL if set, otherwise derive.
    if os.environ.get("CLERK_JWKS_URL"):
        return os.environ["CLERK_JWKS_URL"]
    # fallback: clerk needs issuer domain
    return os.environ.get("CLERK_JWKS_URL", "https://clerk.example.com/.well-known/jwks.json")


async def clerk_current_user(authorization: str = Header(default="")) -> str:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    token = authorization.split(" ", 1)[1]
    if not CLERK_SECRET_KEY:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "CLERK_SECRET_KEY not configured")
    # Verify via jwks (simplified: use python-jose with remote jwks)
    try:
        jwks = httpx.get(_clerk_jwks_url(), timeout=10).json()
        # In production, cache jwks and validate iss/aud
        payload = jwt.decode(token, jwks, options={"verify_aud": False})
        return str(payload.get("sub") or payload.get("user_id") or "anonymous")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid Clerk token: {exc}") from exc
'''

CLERK_FRONTEND_JS = '''// Clerk frontend wiring (React)
// npm install @clerk/clerk-react
import { ClerkProvider, SignedIn, SignedOut, SignInButton, UserButton } from "@clerk/clerk-react";

const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

export function AuthShell({ children }) {
  if (!PUBLISHABLE_KEY) {
    throw new Error("Missing Publishable Key: set VITE_CLERK_PUBLISHABLE_KEY");
  }
  return (
    <ClerkProvider publishableKey={PUBLISHABLE_KEY}>
      <header style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem 1rem" }}>
        <SignedOut>
          <SignInButton mode="modal" />
        </SignedOut>
        <SignedIn>
          <UserButton />
        </SignedIn>
      </header>
      {children}
    </ClerkProvider>
  );
}
'''

SUPABASE_AUTH_PY = '''"""Supabase Auth integration (FastAPI).

Requires: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or ANON_KEY for client-side).
Backend verifies JWTs issued by Supabase Auth via JWKS.

Set SUPABASE_URL and SUPABASE_ANON_KEY from your Supabase project dashboard.
"""

from __future__ import annotations

import os

import httpx
from fastapi import Depends, Header, HTTPException, status

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")


async def supabase_current_user(authorization: str = Header(default="")) -> str:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    token = authorization.split(" ", 1)[1]
    if not SUPABASE_URL:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "SUPABASE_URL not configured")
    jwks_url = f"{SUPABASE_URL}/auth/v1/jwks"
    try:
        jwks = httpx.get(jwks_url, timeout=10).json()
        # Use supabase jwt secret if available, else fetch jwks
        from jose import jwt
        payload = jwt.decode(token, jwks, options={"verify_aud": False})
        return str(payload.get("sub") or "anonymous")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid Supabase token: {exc}") from exc


def supabase_client():
    """Return a supabase client for server-side operations."""
    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError("pip install supabase") from exc
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY required")
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
'''

SUPABASE_FRONTEND_JS = '''// Supabase Auth frontend helper
// npm install @supabase/supabase-js
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  console.warn("Supabase env missing: set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY");
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export async function signIn(email, password) {
  const { data, error } = await supabase.auth.signInWithPassword({ email, password });
  if (error) throw error;
  return data;
}

export async function signUp(email, password) {
  const { data, error } = await supabase.auth.signUp({ email, password });
  if (error) throw error;
  return data;
}

export async function signOut() {
  const { error } = await supabase.auth.signOut();
  if (error) throw error;
}

export function useSession() {
  // simple helper; in a real app use onAuthStateChange
  return supabase.auth.getSession();
}
'''

AUTHJS_JS = '''// Auth.js (next-auth) integration for Next.js
// npm install next-auth
// Place at app/api/auth/[...nextauth]/route.js or pages/api/auth/[...nextauth].js

import NextAuth from "next-auth";
import GithubProvider from "next-auth/providers/github";
import GoogleProvider from "next-auth/providers/google";
import CredentialsProvider from "next-auth/providers/credentials";

export const authOptions = {
  providers: [
    GithubProvider({
      clientId: process.env.GITHUB_ID,
      clientSecret: process.env.GITHUB_SECRET,
    }),
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    }),
    // Fallback credentials for local dev
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        // Replace with DB lookup; this is a demo stub that accepts any email with password length >= 8
        if (credentials?.password?.length >= 8) {
          return { id: credentials.email, email: credentials.email };
        }
        return null;
      },
    }),
  ],
  callbacks: {
    async session({ session, token }) {
      if (token) session.user.id = token.sub;
      return session;
    },
  },
  pages: {
    signIn: "/auth/signin",
  },
};

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
'''

AUTHJS_MIDDLEWARE = '''// middleware.js for Auth.js - protect routes
import { withAuth } from "next-auth/middleware";

export default withAuth({
  callbacks: {
    authorized: ({ token }) => !!token,
  },
});

export const config = {
  matcher: ["/dashboard/:path*", "/api/protected/:path*"],
};
'''

PROVIDER_REGISTRY = {
    "clerk": {
        "label": "Clerk",
        "description": "Hosted auth with prebuilt UI, webhooks, and JWT verification",
        "env": ["CLERK_SECRET_KEY", "CLERK_PUBLISHABLE_KEY", "CLERK_JWKS_URL"],
        "files": {
            "app/auth_clerk.py": CLERK_BACKEND_PY,
            "web/src/auth/clerk.jsx": CLERK_FRONTEND_JS,
            "docs/auth-clerk.md": "# Clerk Setup\n\n1. Create project at https://dashboard.clerk.com\n2. Copy `CLERK_PUBLISHABLE_KEY` and `CLERK_SECRET_KEY` to .env\n3. Set `VITE_CLERK_PUBLISHABLE_KEY` for frontend\n4. Backend verifies JWT via JWKS at `CLERK_JWKS_URL`\n",
        },
        "notes": ["Clerk provides hosted UI; backend verifies via JWKS. Add CLERK_WEBHOOK_SECRET to handle user events."],
    },
    "supabase": {
        "label": "Supabase Auth",
        "description": "Supabase Auth with email/password, OAuth, and row-level security",
        "env": ["SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY"],
        "files": {
            "app/auth_supabase.py": SUPABASE_AUTH_PY,
            "web/src/auth/supabase.js": SUPABASE_FRONTEND_JS,
            "docs/auth-supabase.md": "# Supabase Auth Setup\n\n1. Create project at supabase.com\n2. Copy `SUPABASE_URL` and `SUPABASE_ANON_KEY` to .env and VITE_ vars\n3. Enable email auth in Dashboard > Authentication\n4. Use `supabase` client helper for signIn/signUp\n",
        },
        "notes": ["Supabase Auth integrates with Postgres RLS; set SUPABASE_SERVICE_ROLE_KEY only server-side."],
    },
    "authjs": {
        "label": "Auth.js (NextAuth)",
        "description": "Auth.js for Next.js with OAuth providers and credentials",
        "env": ["GITHUB_ID", "GITHUB_SECRET", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "NEXTAUTH_SECRET", "NEXTAUTH_URL"],
        "files": {
            "app/api/auth/[...nextauth]/route.js": AUTHJS_JS,
            "middleware.js": AUTHJS_MIDDLEWARE,
            "docs/auth-authjs.md": "# Auth.js Setup\n\n1. npm install next-auth\n2. Set `NEXTAUTH_SECRET` (openssl rand -base64 32) and `NEXTAUTH_URL`\n3. Configure OAuth: GITHUB_ID/SECRET, GOOGLE_CLIENT_ID/SECRET\n4. Protected routes via middleware.js matcher\n",
        },
        "notes": ["Auth.js works with any DB; add Prisma adapter for persistence. Set NEXTAUTH_SECRET in production."],
    },
}


def get_auth_provider(name: str) -> dict:
    key = (name or "").strip().lower()
    if key not in PROVIDER_REGISTRY:
        raise KeyError(f"unknown auth provider {name!r}; choose from: {', '.join(PROVIDER_REGISTRY)}")
    return PROVIDER_REGISTRY[key]


def list_auth_providers() -> list[dict]:
    return [
        {"id": key, "label": val["label"], "description": val["description"], "env": val["env"]}
        for key, val in PROVIDER_REGISTRY.items()
    ]


def auth_provider_files(provider_id: str, module: str = "app") -> dict[str, str]:
    """Return file map for provider, with module placeholder replaced."""
    provider = get_auth_provider(provider_id)
    out = {}
    for path, content in provider["files"].items():
        # replace generic app/ prefix with module if needed
        if path.startswith("app/") and module != "app":
            path = path.replace("app/", f"{module}/", 1)
        out[path] = content
    return out
