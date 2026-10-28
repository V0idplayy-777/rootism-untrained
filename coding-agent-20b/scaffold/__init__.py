"""Scaffold: opinionated starters, intent detection, auth/db addons, managed backends.

    from scaffold import ScaffoldGenerator
    ScaffoldGenerator("./my-app").create_from_request(
        "a web app where people sign up and save tasks"
    )

CLI: ``agent init --list`` / ``agent init fastapi-react my-app``.
"""

from __future__ import annotations

from .backends import (
    BackendProvisioner,
    BackendResult,
    available_backends,
    mask_url,
    validate_postgres_url,
)
from .generator import ScaffoldGenerator, ScaffoldResult, plan_for, scaffold_project
from .intent import ProjectIntent, ScaffoldChoice, choose_template, intent_from_request, parse_intent
from .registry import catalog, get_template, list_templates, template_ids
from .spec import ProjectOptions, TemplateSpec

__all__ = [
    "BackendProvisioner",
    "BackendResult",
    "ProjectIntent",
    "ProjectOptions",
    "ScaffoldChoice",
    "ScaffoldGenerator",
    "ScaffoldResult",
    "TemplateSpec",
    "available_backends",
    "catalog",
    "choose_template",
    "get_template",
    "intent_from_request",
    "list_templates",
    "mask_url",
    "parse_intent",
    "plan_for",
    "scaffold_project",
    "template_ids",
    "validate_postgres_url",
]
