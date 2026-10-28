"""Understand "I want a web app with users and a database" and pick a starter.

The parser is keyword/pattern based on purpose: it must be deterministic,
explainable and testable offline. The output is not just a template id - it is a
:class:`ProjectIntent` (what the user needs) plus a :class:`ScaffoldChoice`
(which starter, which addons, why), so the agent can justify the choice and the
UI can render it in plain English.

Examples::

    parse_intent("build me a to-do app where people can sign up and save tasks")
    # -> wants_web=True, wants_auth=True, wants_db=True, frontend=True ...

    choose_template(intent)
    # -> ScaffoldChoice(template_id="fastapi-react", options=..., reasons=[...])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FRAMEWORKS: dict[str, str] = {
    "next.js": "nextjs-fullstack",
    "nextjs": "nextjs-fullstack",
    "react": "fastapi-react",
    "fastapi": "fastapi-react",
    "flask": "flask-htmx",
    "django": "fastapi-react",
    "express": "nextjs-fullstack",
}

SIGNALS: dict[str, tuple[str, ...]] = {
    "web": ("web app", "website", "web app", "webapp", "app", "dashboard", "saas", "site", "portal", "frontend", "front-end", "ui"),
    "api": ("api", "rest", "endpoint", "backend", "back-end", "service", "microservice", "json"),
    "auth": ("login", "log in", "sign up", "signup", "sign-in", "auth", "users", "accounts", "password", "multi-user", "multi user", "session"),
    "db": ("database", "db", "persist", "save data", "store data", "postgres", "sqlite", "sql", "crud", "records", "tasks", "todos", "to-do"),
    "frontend": ("react", "frontend", "front-end", "ui", "page", "dashboard", "interface", "button", "form"),
    "bot": ("discord", "bot", "telegram", "slack bot", "chatbot", "moderation"),
    "cli": ("cli", "command line", "terminal", "script", "tool", "utility", "batch", "parser"),
    "realtime": ("realtime", "real-time", "websocket", "live updates", "chat", "streaming"),
    "ai": ("llm", "openai", "ai", "agent", "embedding", "prompt", "model"),
    "data_app": (
        "track",
        "tracking",
        "manage",
        "managing",
        "inventory",
        "tickets",
        "bookings",
        "orders",
        "customers",
        "crm",
        "admin panel",
        "spreadsheet",
        "expenses",
        "roster",
        "attendance",
        "invoices",
    ),
    "static": ("landing page", "portfolio", "blog", "docs site", "marketing site", "static site"),
    "next": ("next.js", "nextjs", "vercel", "server components"),
    "python": ("python", "fastapi", "flask", "django", "pandas", "numpy"),
    "node": ("node", "javascript", "typescript", "npm", "express"),
}


@dataclass
class ProjectIntent:
    raw: str = ""
    signals: list[str] = field(default_factory=list)
    wants_web: bool = False
    wants_api: bool = False
    wants_auth: bool = False
    wants_db: bool = False
    wants_frontend: bool = False
    wants_bot: bool = False
    wants_cli: bool = False
    wants_static: bool = False
    frameworks: list[str] = field(default_factory=list)
    wants_realtime: bool = False
    wants_ai: bool = False
    prefers_next: bool = False
    prefers_python: bool = False
    prefers_node: bool = False
    ambiguous: bool = False

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "signals": list(self.signals),
            "wants_web": self.wants_web,
            "wants_api": self.wants_api,
            "wants_auth": self.wants_auth,
            "wants_db": self.wants_db,
            "wants_frontend": self.wants_frontend,
            "wants_bot": self.wants_bot,
            "wants_cli": self.wants_cli,
            "wants_static": self.wants_static,
            "frameworks": list(self.frameworks),
            "wants_realtime": self.wants_realtime,
            "wants_ai": self.wants_ai,
            "prefers_next": self.prefers_next,
            "prefers_python": self.prefers_python,
            "prefers_node": self.prefers_node,
            "ambiguous": self.ambiguous,
        }

    def summary(self) -> str:
        wants = [
            label
            for flag, label in (
                (self.wants_web, "web app"),
                (self.wants_api, "API"),
                (self.wants_auth, "user accounts"),
                (self.wants_db, "database"),
                (self.wants_frontend, "frontend"),
                (self.wants_bot, "chat bot"),
                (self.wants_cli, "command-line tool"),
                (self.wants_static, "static site"),
            )
            if flag
        ]
        if not wants:
            return "unclear intent"
        return ", ".join(wants)


@dataclass
class ScaffoldChoice:
    template_id: str
    options: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "options": dict(self.options),
            "reasons": list(self.reasons),
            "alternatives": list(self.alternatives),
            "questions": list(self.questions),
        }

    def explain(self) -> str:
        lines = [f"Starter: {self.template_id}"]
        lines += [f"  reason: {reason}" for reason in self.reasons]
        if self.questions:
            lines += [f"  question: {question}" for question in self.questions]
        if self.alternatives:
            lines.append("  alternatives: " + ", ".join(self.alternatives))
        return "\n".join(lines)


def _hits(text: str, words: tuple[str, ...]) -> list[str]:
    found = []
    for word in words:
        pattern = r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            found.append(word)
    return found


WEAK_CLI_SIGNALS = frozenset({"tool", "utility", "script", "parser"})


def parse_intent(prose: str) -> ProjectIntent:
    text = (prose or "").lower()
    intent = ProjectIntent(raw=prose or "")
    hits = {name: _hits(text, words) for name, words in SIGNALS.items()}
    for matched in hits.values():
        for word in matched:
            if word not in intent.signals:
                intent.signals.append(word)

    intent.frameworks = [name for name in FRAMEWORKS if _hits(text, (name,))]
    intent.wants_bot = bool(hits["bot"])
    intent.wants_api = bool(hits["api"])
    intent.wants_frontend = bool(hits["frontend"])
    intent.wants_auth = bool(hits["auth"])
    intent.wants_db = bool(hits["db"]) or bool(hits["data_app"])

    data_app = bool(hits["data_app"])
    cli_only_generic = set(hits["cli"]) <= WEAK_CLI_SIGNALS
    intent.wants_cli = bool(hits["cli"]) and not intent.wants_bot and not (
        data_app and cli_only_generic and not hits["web"]
    )
    intent.wants_web = bool(hits["web"]) or intent.wants_frontend or (data_app and not intent.wants_bot)
    intent.wants_static = bool(hits["static"]) and not (
        data_app or intent.wants_auth or intent.wants_api or intent.wants_bot
    )
    intent.wants_realtime = bool(hits["realtime"])
    intent.wants_ai = bool(hits["ai"])
    intent.prefers_next = bool(hits["next"])
    intent.prefers_python = bool(hits["python"])
    intent.prefers_node = bool(hits["node"]) and not intent.prefers_python

    strong = sum(
        [
            intent.wants_web,
            intent.wants_api,
            intent.wants_bot,
            intent.wants_cli,
            intent.wants_static,
        ]
    )
    intent.ambiguous = strong == 0
    return intent


def choose_template(intent: ProjectIntent, available: list[str] | None = None) -> ScaffoldChoice:
    """Pick a starter id + options, with the reasoning that led there."""

    available = available or [
        "fastapi-react",
        "nextjs-fullstack",
        "flask-htmx",
        "cli-tool",
        "discord-bot",
        "static-site",
    ]
    reasons: list[str] = []
    questions: list[str] = []

    if intent.ambiguous:
        reasons.append("no clear app type in the request; defaulting to a full-stack web starter")
        questions.append("Should this be a web app, a command-line tool, or a chat bot?")
        template = "nextjs-fullstack" if intent.prefers_next else "fastapi-react"
    elif intent.wants_bot:
        template = "discord-bot"
        reasons.append("the request mentions a bot/chat integration")
    elif intent.wants_cli:
        template = "cli-tool"
        reasons.append("the request describes a terminal/scripting tool")
    elif "flask" in intent.frameworks:
        template = "flask-htmx"
        reasons.append("Flask was named, so the server-rendered starter uses it")
    elif intent.prefers_next:
        template = "nextjs-fullstack"
        reasons.append("Next.js was named (or preferred) for the frontend")
    elif "django" in intent.frameworks:
        template = "fastapi-react"
        reasons.append("Django was named; there is no Django starter, so the closest full-stack one is used")
        questions.append("Django is not one of the starters - keep FastAPI + React, or scaffold plain Django by hand?")
    elif {"fastapi", "react"} & set(intent.frameworks):
        template = "fastapi-react"
        reasons.append("FastAPI/React were named, so the starter matches that stack")
    elif intent.wants_static:
        template = "static-site"
        reasons.append("a static page needs no server or database to host")
    elif intent.wants_frontend and not intent.wants_api:
        template = "flask-htmx"
        reasons.append("a small interactive UI without a separate API fits server-rendered HTMX")
    else:
        template = "fastapi-react"
        reasons.append("a typed API plus a React frontend fits a full-stack web app")

    if template not in available:
        template = available[0]
        reasons.append("preferred starter is unavailable; using " + template)

    options: dict = {}
    if intent.wants_auth:
        options["use_auth"] = True
        reasons.append("accounts/login were requested, so the auth addon is included")
    if intent.wants_db:
        options["database"] = "sqlite"
        reasons.append("data must persist; starting on SQLite (Postgres is one env var away)")
    if intent.wants_realtime:
        questions.append("Realtime updates are not in the starter - add WebSockets/SSE next?")
    if intent.wants_ai:
        questions.append("Which model provider should the AI feature call, and is there a key?")

    alternatives = [item for item in available if item != template][:2]
    return ScaffoldChoice(
        template_id=template,
        options=options,
        reasons=reasons,
        alternatives=alternatives,
        questions=questions,
    )


def intent_from_request(prose: str, available: list[str] | None = None) -> tuple[ProjectIntent, ScaffoldChoice]:
    intent = parse_intent(prose)
    return intent, choose_template(intent, available)
