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
from .auth_providers import auth_provider_files, get_auth_provider, list_auth_providers
from .backends import (
    BackendProvisioner,
    BackendResult,
    available_backends,
    mask_url,
    validate_postgres_url,
)
from .generator import ScaffoldGenerator, ScaffoldResult, plan_for, scaffold_project
from .intent import ProjectIntent, ScaffoldChoice, choose_template, intent_from_request, parse_intent
from .jobs import node_jobs, python_jobs
from .migrations import generate_migration, switch_to_postgres_instructions
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
    "auth_provider_files",
    "available_backends",
    "catalog",
    "choose_template",
    "generate_migration",
    "get_auth_provider",
    "get_template",
    "intent_from_request",
    "list_auth_providers",
    "list_templates",
    "mask_url",
    "node_jobs",
    "parse_intent",
    "plan_for",
    "python_jobs",
    "scaffold_project",
    "switch_to_postgres_instructions",
    "template_ids",
    "validate_postgres_url",
]
