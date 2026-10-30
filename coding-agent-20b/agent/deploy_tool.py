"""Agent tools for shipping and starting projects (deploy / scaffold / map).

These are *control* tools: they never touch the sandbox directly, they call the
same modules the CLI and UI call, so an agent action and a human action produce
identical, reviewable results.

Payloads are deliberately compact - a deploy summary is a URL, a health verdict
and the notes that matter; the full step log lives in ``.agent/deploy/last.json``
and is only surfaced on failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

MAX_SUMMARY_CHARS = 1400
MAX_FAILURE_CHARS = 2200


def _short(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def deploy_tool(
    args: dict[str, Any] | None,
    workspace: str | Path,
    *,
    emit: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build and deploy the current workspace; returns the live URL when there is one."""

    from deploy import DeployEngine

    args = dict(args or {})
    provider = args.get("provider") or None
    dry_run = bool(args.get("dry_run"))
    publish_metadata = args.get("publish_metadata", True)

    engine = DeployEngine(
        workspace,
        provider=provider,
        dry_run=dry_run,
        emit=emit,
        publish_metadata=bool(publish_metadata),
        timeout=float(args.get("timeout") or 900.0),
    )
    result = engine.run()

    payload: dict[str, Any] = {
        "success": result.ok,
        "provider": result.provider,
        "dry_run": result.dry_run,
        "url": result.url,
        "public": result.public,
        "health": result.health,
        "summary": _short(result.render(), MAX_SUMMARY_CHARS if result.ok else MAX_FAILURE_CHARS),
        "notes": result.notes[:6],
    }
    if result.report_path:
        payload["report"] = result.report_path
    if result.landing:
        payload["landing"] = result.landing
    if result.human_error:
        payload["error"] = result.human_error
    if not result.ok:
        payload["hint"] = (
            "Deploy failed. Read error.plain/next_steps, fix the cause (or retry with "
            "provider='archive' for a portable artifact), then call deploy again."
        )
    return payload


def scaffold_tool(args: dict[str, Any] | None, workspace: str | Path) -> dict[str, Any]:
    """Create a starter project from a template id or a plain-English request."""

    from scaffold import ScaffoldGenerator, list_templates

    args = dict(args or {})
    template = (args.get("template") or "").strip()
    request = (args.get("request") or "").strip()
    name = args.get("name") or None
    forced = bool(args.get("force"))
    dry_run = bool(args.get("dry_run"))

    if request and not template:
        payload = _blueprint_tool(request, workspace, forced, dry_run)
        if payload is not None:
            return payload

    generator = ScaffoldGenerator(workspace, force=forced, dry_run=dry_run)
    if request and not template:
        result = generator.create_from_request(request, name=name)
    elif template:
        result = generator.create(
            template,
            name=name,
            use_auth=args.get("use_auth"),
            database=args.get("database"),
        )
    else:
        return {
            "success": False,
            "error": "provide either template or request",
            "templates": [item["id"] for item in list_templates()],
        }

    payload: dict[str, Any] = {
        "success": result.ok,
        "template": result.template_id,
        "root": str(result.root),
        "files": result.files[:30],
        "file_count": len(result.files),
        "next_steps": result.next_steps[:6],
        "dry_run": result.dry_run,
    }
    if result.warnings:
        payload["warnings"] = result.warnings[:5]
    if result.questions:
        payload["questions"] = result.questions[:5]
    if not result.ok:
        payload["error"] = result.error
        payload["hint"] = (
            "The directory is probably not empty. Scaffolding never overwrites files: "
            "pick an empty directory, or pass force=true to add the starter alongside them."
        )
    return payload



STARTER_KEYWORDS = (
    "flask",
    "django",
    "fastapi",
    "next.js",
    "nextjs",
    "react",
    "vue",
    "svelte",
    "cli",
    "command line",
    "discord",
    "slack bot",
    "static site",
    "landing page",
)


def _blueprint_tool(
    request: str, workspace: str | Path, forced: bool, dry_run: bool
) -> dict[str, Any] | None:
    """Build a full product from the blueprint catalogue when the request fits."""
    from scaffold.blueprint import render_blueprint
    from scaffold.blueprints import match_blueprint

    lowered = request.lower()
    if any(token in lowered for token in STARTER_KEYWORDS):
        return None
    blueprint = match_blueprint(request)
    if blueprint is None:
        return None
    root = Path(workspace)
    if root.exists() and any(root.iterdir()) and not forced and not dry_run:
        return None
    project = render_blueprint(blueprint)
    if not dry_run:
        project.write(root, force=forced)
    return {
        "success": True,
        "template": blueprint.id,
        "blueprint": True,
        "root": str(root),
        "files": sorted(project.files)[:40],
        "file_count": len(project.files),
        "next_steps": [
            "python -m pytest -q",
            "python -m uvicorn " + blueprint.id.replace("-", "_") + ".main:app --reload",
            "agent app \"" + request + "\" --repair (if a later edit breaks a gate)",
        ],
        "dry_run": dry_run,
    }

def build_app_tool(
    args: dict[str, Any] | None,
    workspace: str | Path,
    *,
    emit: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build a whole product and report every gate; optionally repair once."""
    from scaffold.app_builder import (
        NoBlueprint,
        build_app,
        repair_generated,
        resolve_blueprint,
    )

    args = dict(args or {})
    request = (args.get("request") or "").strip()
    blueprint_id = args.get("blueprint") or None
    if not request and not blueprint_id:
        return {"success": False, "error": "provide request or blueprint"}
    try:
        report = build_app(
            workspace,
            request or blueprint_id or "",
            blueprint_id=blueprint_id,
            force=bool(args.get("force")),
        )
    except NoBlueprint as exc:
        return {"success": False, "error": str(exc), "suggestions": exc.suggestions}
    if not report.ok and args.get("repair"):
        if emit:
            emit("gates failed; restoring generated modules")
        blueprint = resolve_blueprint(request or blueprint_id or "", blueprint_id)
        repaired = repair_generated(workspace, blueprint)
        if repaired.restored:
            report = build_app(
                workspace,
                request or blueprint.id,
                blueprint_id=blueprint.id,
                force=True,
                write_project=False,
            )
    return {
        "success": report.ok,
        "blueprint": report.blueprint_id,
        "files": len(report.files),
        "preview": report.preview_command,
        "tests": bool(report.verify and report.verify.ok),
        "critic": {"score": report.critic_score, "ok": report.critic_ok},
        "taste": {"score": report.taste.score if report.taste else 0, "ok": bool(report.taste and report.taste.ok)},
        "deploy_provider": report.deploy_provider,
        "failures": report.failures,
        "summary": report.summary(),
    }


def project_map_tool(args: dict[str, Any] | None, workspace: str | Path) -> dict[str, Any]:
    """Return the project map (tree + summaries + symbols) within a char budget."""

    from agent.project_map import ProjectMap

    args = dict(args or {})
    budget = int(args.get("max_chars") or 2400)
    focus = args.get("focus") or None
    if isinstance(focus, str):
        focus = [focus]
    project_map = ProjectMap.open(workspace)
    return {
        "success": True,
        "map": project_map.render(max_chars=budget, focus=focus),
        "stats": project_map.stats(),
    }


def symbol_search_tool(args: dict[str, Any] | None, workspace: str | Path) -> dict[str, Any]:
    """Find files by symbol or by meaning (hybrid keyword + hashed embeddings)."""

    from agent.semantic_index import SemanticIndex

    args = dict(args or {})
    query = (args.get("query") or args.get("symbol") or "").strip()
    if not query:
        return {"success": False, "error": "query is required"}
    limit = int(args.get("k") or 8)
    mode = (args.get("mode") or "hybrid").lower()

    index = SemanticIndex.open(workspace)
    if mode == "keyword":
        hits = index.keyword_search(query, k=limit)
    elif mode == "semantic":
        hits = index.search(query, k=limit)
    else:
        hits = index.hybrid_search(query, k=limit)
    return {
        "success": True,
        "mode": mode,
        "results": [hit.to_dict() for hit in hits],
        "stats": index.stats(),
    }


def checkpoint_tool(args: dict[str, Any] | None, workspace: str | Path) -> dict[str, Any]:
    """Snapshot the workspace, or roll back to an earlier snapshot."""

    from agent.checkpoint import WorkspaceCheckpoints

    args = dict(args or {})
    action = (args.get("action") or "create").lower()
    store = WorkspaceCheckpoints(workspace)
    if action == "create":
        snapshot = store.create(str(args.get("label") or ""))
        return {"success": True, "snapshot": snapshot.to_dict()}
    if action == "list":
        return {"success": True, "snapshots": [item.to_dict() for item in store.list()]}
    if action in {"rollback", "restore"}:
        outcome = store.restore(args.get("id") or None, delete_extra=bool(args.get("delete_extra", True)))
        return {"success": bool(outcome.get("ok", True)), **outcome}
    if action == "diff":
        return {"success": True, **store.diff(args.get("id") or None)}
    return {"success": False, "error": f"unknown action {action!r}; use create|list|diff|rollback"}


TOOL_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "deploy": deploy_tool,
    "scaffold_project": scaffold_tool,
    "build_app": build_app_tool,
    "project_map": project_map_tool,
    "search_project": symbol_search_tool,
    "checkpoint": checkpoint_tool,
}


def handle_tool(
    name: str,
    args: dict[str, Any] | None,
    workspace: str | Path,
    *,
    emit: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Dispatch a project tool; ``None`` means "not one of mine"."""

    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return None
    try:
        if name in {"deploy", "build_app"}:
            return handler(args, workspace, emit=emit)
        return handler(args, workspace)
    except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the session
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "hint": "This is a bug in the tool, not in the user's project. Report what failed.",
        }


def tool_schemas() -> dict[str, dict[str, Any]]:
    """JSON schemas for the project tools, merged into ``TOOL_SCHEMAS``."""

    return {
        "deploy": {
            "name": "deploy",
            "description": (
                "Ship the current workspace and return a URL. Generates Dockerfile/"
                "compose, picks a provider that is actually usable here, verifies the "
                "health endpoint, writes DEPLOY.md + README link, and falls back to a "
                "portable artifact instead of pretending a deploy worked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "enum": ["auto", "local", "vercel", "github-pages", "fly", "render", "coolify", "archive"],
                        "description": "Target host; 'auto' picks the best usable one",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Show the plan without creating anything",
                        "default": False,
                    },
                    "publish_metadata": {
                        "type": "boolean",
                        "description": "Write landing page, README link and DEPLOY.md",
                        "default": True,
                    },
                },
            },
        },
        "build_app": {
            "name": "build_app",
            "description": (
                "Build a complete product from a plain-English request, then prove it: "
                "tests, code critic, taste gate and deploy plan. Returns the exact failures "
                "when it is not shippable, and can restore generated modules once (repair). "
                "Use this instead of hand-writing a whole app file by file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {"type": "string", "description": "What the user asked for, verbatim"},
                    "blueprint": {"type": "string", "description": "Optional known blueprint id"},
                    "repair": {
                        "type": "boolean",
                        "description": "Restore generated modules once when a gate fails",
                        "default": False,
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Overwrite generated files in a non-empty directory",
                        "default": False,
                    },
                },
            },
        },
        "scaffold_project": {
            "name": "scaffold_project",
            "description": (
                "Create a complete starter project (multi-file app with tests, README "
                "and deploy config) from a plain-English request or a template id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "What the user asked for, verbatim",
                    },
                    "template": {
                        "type": "string",
                        "enum": [
                            "fastapi-react",
                            "nextjs-fullstack",
                            "flask-htmx",
                            "cli-tool",
                            "discord-bot",
                            "static-site",
                        ],
                    },
                    "name": {"type": "string", "description": "Project slug"},
                    "use_auth": {"type": "boolean", "description": "Include login/accounts"},
                    "database": {
                        "type": "string",
                        "enum": ["sqlite", "postgres", "none"],
                        "description": "Storage backend (Postgres is one env var away)",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Add alongside existing files instead of refusing",
                        "default": False,
                    },
                    "dry_run": {"type": "boolean", "default": False},
                },
            },
        },
        "project_map": {
            "name": "project_map",
            "description": (
                "Persistent map of the repo: layout, per-file summary and key symbols. "
                "Cheap way to orient yourself before touching an unfamiliar codebase."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_chars": {"type": "integer", "default": 2400},
                    "focus": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Paths to expand first",
                    },
                },
            },
        },
        "search_project": {
            "name": "search_project",
            "description": (
                "Search the repo by keyword or meaning (local hybrid retrieval: BM25 "
                "plus hashed embeddings, no network)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 8},
                    "mode": {"type": "string", "enum": ["hybrid", "keyword", "semantic"], "default": "hybrid"},
                },
                "required": ["query"],
            },
        },
        "checkpoint": {
            "name": "checkpoint",
            "description": (
                "Snapshot the workspace before risky edits, or roll back to a snapshot. "
                "Cheap insurance for multi-file changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "list", "diff", "rollback"], "default": "create"},
                    "label": {"type": "string"},
                    "id": {"type": "string", "description": "Snapshot id (for diff/rollback)"},
                    "delete_extra": {"type": "boolean", "default": True},
                },
            },
        },
    }


def summarize_for_log(payload: dict[str, Any]) -> str:
    """One-line, log-safe summary of a project tool result."""

    return json.dumps(
        {
            "success": payload.get("success"),
            "url": payload.get("url"),
            "provider": payload.get("provider"),
            "template": payload.get("template"),
        },
        sort_keys=True,
    )
