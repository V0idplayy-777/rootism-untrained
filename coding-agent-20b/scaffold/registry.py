"""Starter registry: one place that knows which templates exist and how to find them."""

from __future__ import annotations

from .spec import TemplateSpec
from .templates import cli_tool, discord_bot, fastapi_react, flask_htmx, nextjs_fullstack, static_site

TEMPLATES: dict[str, TemplateSpec] = {
    spec.id: spec
    for spec in (
        fastapi_react.SPEC,
        nextjs_fullstack.SPEC,
        flask_htmx.SPEC,
        cli_tool.SPEC,
        discord_bot.SPEC,
        static_site.SPEC,
    )
}

TEMPLATE_ORDER = ["fastapi-react", "nextjs-fullstack", "flask-htmx", "cli-tool", "discord-bot", "static-site"]

ALIASES = {
    "fastapi": "fastapi-react",
    "react": "fastapi-react",
    "python-web": "fastapi-react",
    "next": "nextjs-fullstack",
    "nextjs": "nextjs-fullstack",
    "next.js": "nextjs-fullstack",
    "flask": "flask-htmx",
    "htmx": "flask-htmx",
    "cli": "cli-tool",
    "python-cli": "cli-tool",
    "bot": "discord-bot",
    "discord": "discord-bot",
    "static": "static-site",
    "landing": "static-site",
    "site": "static-site",
}


def get_template(template_id: str) -> TemplateSpec:
    key = (template_id or "").strip().lower()
    key = ALIASES.get(key, key)
    if key not in TEMPLATES:
        raise KeyError(
            f"unknown template {template_id!r}; choose from: {', '.join(TEMPLATE_ORDER)}"
        )
    return TEMPLATES[key]


def list_templates() -> list[TemplateSpec]:
    return [TEMPLATES[template_id] for template_id in TEMPLATE_ORDER if template_id in TEMPLATES]


def catalog() -> list[dict]:
    """JSON-safe catalogue for the UI (plain-English descriptions included)."""

    return [spec.to_dict() for spec in list_templates()]


def template_ids() -> list[str]:
    return [spec.id for spec in list_templates()]
