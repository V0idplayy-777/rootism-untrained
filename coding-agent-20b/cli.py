"""``agent`` - one command for the whole idea -> live app loop.

    agent init --request "a web app where users sign up and save tasks" my-app
    agent deploy --provider local          # or vercel/fly/render/coolify
    agent preview                          # chat UI at :7860
    agent verify                           # run the tests, show the real failures
    agent review                           # static review before shipping

Every command supports ``--json`` for scripting, exits non-zero on failure, and
prints what it will do before doing it. Nothing here needs a GPU; a model is
optional (``--model-url``) and the commands still do real work without one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# --------------------------------------------------------------------- helpers
def _print_json(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _workspace(args) -> Path:
    return Path(getattr(args, "workspace", ".")).resolve()


def _say(message: str = "") -> None:
    print(message)


def _human(command: str, result) -> int:
    """Run a command and humanise a failure (used by deploy/verify paths)."""

    from quality.errors import humanize

    if result.ok:
        return 0
    error = result.error or ""
    human = humanize(error)
    _say(f"\n{human.title}")
    _say(f"  {human.plain}")
    for step in human.next_steps:
        _say(f"  -> {step}")
    return 1


# -------------------------------------------------------------------- commands
def cmd_init(args) -> int:
    from scaffold import ScaffoldGenerator, catalog, get_template

    if args.list:
        if args.json:
            _print_json({"templates": catalog()})
            return 0
        for template in catalog():
            _say(f"{template['id']:<18} {template['name']}")
            _say(f"  {template['tagline']}")
            _say(f"  good for: {', '.join(template['good_for'])}")
            _say(f"  run: {template['run_command'].splitlines()[0]}")
            _say()
        return 0

    root = Path(args.path or args.workspace or ".").resolve()
    name = args.name or root.name
    if args.request:
        generator = ScaffoldGenerator(root, force=args.force, dry_run=args.dry_run)
        result = generator.create_from_request(args.request, name=name)
    elif args.template:
        try:
            get_template(args.template)
        except KeyError as exc:
            _say(str(exc))
            return 2
        database = args.db
        if database == "postgres" and not get_template(args.template).supports_database:
            _say(f"{args.template} does not use a database; ignoring --db postgres")
            database = "none"
        result = ScaffoldGenerator(root, force=args.force, dry_run=args.dry_run).create(
            args.template,
            name,
            description=args.description or "",
            use_auth=args.auth,
            database=database,
        )
    else:
        _say("Nothing to do: pass a template id, or --request \"what you want to build\".")
        _say("Try: agent init --list")
        return 2

    if args.json:
        _print_json(result.to_dict())
    else:
        _say(result.render())
    return 0 if result.ok else 1


def cmd_plan(args) -> int:
    from agent.plan_approval import PlanDrafter, PlanStore

    root = _workspace(args)
    store = PlanStore(root)
    if args.list:
        plans = store.list()
        if args.json:
            _print_json({"plans": [plan.to_dict() for plan in plans]})
            return 0
        if not plans:
            _say("No plans yet. Draft one: agent plan \"build me a web app\"")
            return 0
        for plan in plans:
            _say(f"{plan.id}  {plan.status:<9} {plan.goal[:70]}")
        return 0
    if args.approve:
        plan = store.approve(args.approve)
        _say(f"approved {plan.id}")
        return 0
    if args.reject:
        plan = store.reject(args.reject)
        _say(f"rejected {plan.id}; nothing was written")
        return 0
    if not args.goal:
        _say("Give a goal: agent plan \"build a dashboard for expenses\"")
        return 2
    plan = PlanDrafter(root).draft(args.goal)
    store.save(plan)
    if args.json:
        _print_json(plan.to_dict())
    else:
        _say(plan.render_markdown())
        _say()
        _say(f"Approve with: agent plan --approve {plan.id}")
    return 0


def cmd_map(args) -> int:
    from agent.project_map import ProjectMap

    project = ProjectMap.build(_workspace(args))
    path = project.save()
    if args.json:
        _print_json({"stats": project.stats(), "map": project.to_dict()})
        return 0
    _say(project.render(max_chars=args.chars))
    _say()
    _say(f"saved {path}")
    return 0


def cmd_search(args) -> int:
    from agent.semantic_index import SemanticIndex

    root = _workspace(args)
    index = SemanticIndex.open(root)
    hits = index.hybrid_search(args.query, k=args.k)
    if args.json:
        _print_json({"query": args.query, "hits": [hit.to_dict() for hit in hits], "stats": index.stats()})
        return 0 if hits else 1
    if not hits:
        _say("No matches.")
        return 1
    for hit in hits:
        _say(hit.render())
    return 0


def cmd_checkpoint(args) -> int:
    from agent.checkpoint import WorkspaceCheckpoints

    checkpoints = WorkspaceCheckpoints(_workspace(args))
    action = args.action
    if action == "create":
        snapshot = checkpoints.create(args.label or "manual")
        if args.json:
            _print_json({"snapshot": snapshot.to_dict()})
            return 0
        _say(f"checkpoint {snapshot.id} ({len(snapshot.files)} files, {snapshot.total_bytes / 1024:.0f} KiB)")
        return 0
    if action == "list":
        snapshots = checkpoints.list()
        if args.json:
            _print_json({"snapshots": [snapshot.to_dict() for snapshot in snapshots], "stats": checkpoints.stats()})
            return 0
        if not snapshots:
            _say("No checkpoints yet: agent checkpoint create --label before-refactor")
            return 0
        for snapshot in snapshots:
            _say(f"{snapshot.id}  {snapshot.name}  ({len(snapshot.files)} files)")
        return 0
    if action == "diff":
        diff = checkpoints.diff(args.id)
        if args.json:
            _print_json(diff)
            return 0
        _say(f"since {diff['snapshot']['id']}: {diff['changed']} change(s)")
        for key in ("added", "modified", "removed"):
            for path in diff[key]:
                _say(f"  {key[:3]}: {path}")
        return 0
    if action == "restore":
        try:
            result = checkpoints.restore(args.id)
        except KeyError as exc:
            _say(str(exc))
            return 2
        if args.json:
            _print_json(result)
            return 0
        _say(f"restored {len(result['restored'])} file(s), removed {len(result['deleted'])} added file(s)")
        return 0
    stats = checkpoints.stats()
    _say(json.dumps(stats, indent=2) if args.json else f"snapshots: {stats['snapshots']} · blobs: {stats['blobs']}")
    return 0


def cmd_review(args) -> int:
    from quality import CodeCritic

    review = CodeCritic(threshold=args.threshold).review(_workspace(args), paths=args.paths)
    if args.json:
        _print_json(review.to_dict())
    else:
        _say(review.render())
        _say()
        _say("PASS" if review.ok else f"FAIL (score < {review.threshold} or critical findings)")
    return 0 if review.ok else 1


def cmd_testgen(args) -> int:
    from quality import write_generated_tests

    result = write_generated_tests(_workspace(args), paths=args.paths, force=args.force)
    if args.json:
        _print_json(result)
        return 0
    if not result["written"]:
        _say("No new test files (existing tests kept).")
    for path in result["written"]:
        _say(f"+ {path}")
    for path in result["skipped"]:
        _say(f"= {path} (exists)")
    return 0


def cmd_verify(args) -> int:
    from quality import humanize, run_tests

    root = _workspace(args)
    run = run_tests(root, command=args.command, timeout=args.timeout)
    if args.json:
        _print_json(run.to_dict())
    else:
        _say(run.render())
        if not run.ok:
            human = humanize(run.failure_signal())
            _say()
            _say(human.render())
    return 0 if run.ok else 1


def cmd_backend(args) -> int:
    from scaffold import BackendProvisioner, available_backends

    if args.list:
        _print_json(available_backends()) if args.json else [
            _say(f"{name:<10} {'ready' if ok else 'not configured'}: {reason}")
            for name, (ok, reason) in available_backends().items()
        ]
        return 0
    provisioner = BackendProvisioner(_workspace(args))
    result = provisioner.ensure(name=args.name or _workspace(args).name, backend=args.provider, dry_run=args.dry_run)
    if args.json:
        _print_json(result.to_dict())
    else:
        _say(result.render())
        for note in result.notes:
            _say(f"note: {note}")
    return 0 if result.ok else 1


def cmd_deploy(args) -> int:
    from deploy import DeployEngine

    engine = DeployEngine(
        _workspace(args),
        provider=args.provider,
        dry_run=args.dry_run,
        timeout=args.timeout,
        publish_metadata=not args.no_publish,
        description=args.description or "",
    )
    result = engine.deploy()
    if args.json:
        _print_json(result.to_dict())
    else:
        _say(result.render())
    return 0 if result.ok else 1


def cmd_preview(args) -> int:
    from ui.app import serve

    root = _workspace(args)
    _say(f"Agent UI: http://localhost:{args.port}  (workspace: {root})")
    serve(str(root), host=args.host, port=args.port, model_url=args.model_url)
    return 0


def cmd_run(args) -> int:
    from ui.runtime import AgentRuntime

    runtime = AgentRuntime(_workspace(args), model_url=args.model_url)
    failed = False
    for event in runtime.run(args.task, mode=args.mode, require_approval=args.plan_first):
        kind = event["type"]
        if kind in {"done", "error", "plan_pending"}:
            _say(f"\n[{kind}] {event['message']}")
            if kind == "error":
                failed = True
                for step in (event["data"] or {}).get("next_steps", []):
                    _say(f"  -> {step}")
        elif kind in {"status", "intent", "plan", "scaffold", "tests", "review", "deploy_plan"}:
            _say(f"[{kind}] {event['message']}")
        elif args.verbose and kind != "end":
            _say(f"  [{kind}] {event['message']}")
    return 1 if failed else 0


def cmd_clone(args) -> int:
    from agent.project_map import ProjectMap
    from agent.semantic_index import build_index
    from deploy.detect import detect_project

    target = Path(args.directory or Path(args.url.rstrip("/").split("/")[-1].removesuffix(".git"))).resolve()
    if target.exists() and any(target.iterdir()):
        _say(f"{target} already exists and is not empty")
        return 1
    import subprocess

    _say(f"cloning {args.url} -> {target}")
    completed = subprocess.run(["git", "clone", "--depth", "1", args.url, str(target)], capture_output=True, text=True)
    if completed.returncode != 0:
        from quality.errors import humanize

        human = humanize(completed.stderr)
        _say(human.render())
        return 1
    project = ProjectMap.build(target)
    project.save()
    index = build_index(target)
    profile = detect_project(target)
    if args.json:
        _print_json({"path": str(target), "profile": profile.to_dict(), "map": project.stats(), "index": index.stats()})
        return 0
    _say(profile.render())
    _say()
    _say(project.render(max_chars=2500))
    _say()
    _say(f"semantic index: {index.stats()['documents']} chunks · run `agent search \"...\" --workspace {target}`")
    return 0


def cmd_doctor(args) -> int:
    import shutil

    from deploy.detect import detect_project
    from scaffold import catalog

    root = _workspace(args)
    checks: list[tuple[str, bool, str]] = []
    checks.append(("python", sys.version_info >= (3, 10), sys.version.split()[0]))
    checks.append(("git", bool(shutil.which("git")), shutil.which("git") or "not installed"))
    checks.append(("node", bool(shutil.which("node")), shutil.which("node") or "not installed (JS templates need it)"))
    checks.append(("docker", bool(shutil.which("docker")), shutil.which("docker") or "not installed (local deploys need it)"))
    try:
        import fastapi  # noqa: F401

        checks.append(("fastapi", True, "UI available"))
    except ImportError:
        checks.append(("fastapi", False, "pip install fastapi uvicorn (UI)"))
    try:
        import pytest  # noqa: F401

        checks.append(("pytest", True, "tests available"))
    except ImportError:
        checks.append(("pytest", False, "pip install pytest"))
    profile = detect_project(root)
    checks.append(("project", profile.kind != "unknown", f"{profile.kind} / {profile.framework}"))
    checks.append(("templates", len(catalog()) >= 5, f"{len(catalog())} starters"))
    env_tokens = [key for key in ("VERCEL_TOKEN", "RENDER_API_KEY", "FLY_API_TOKEN", "COOLIFY_TOKEN", "NEON_API_KEY") if os.environ.get(key)]
    checks.append(("deploy tokens", bool(env_tokens), ", ".join(env_tokens) or "none (local/archive still work)"))

    if args.json:
        _print_json({"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks], "profile": profile.to_dict()})
        return 0
    for name, ok, detail in checks:
        _say(f"{'ok  ' if ok else 'warn'} {name:<14} {detail}")
    _say()
    _say("Everything except Docker/model tokens is optional; `agent deploy --dry-run` always works.")
    return 0


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Coding agent: scaffold, test, review, deploy - from one command.",
    )
    parser.add_argument("--version", action="version", version="agent 1.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create a project from a starter or a plain-English request")
    init.add_argument("template", nargs="?", help="template id (see --list)")
    init.add_argument("path", nargs="?", help="target directory")
    init.add_argument("--request", "-r", help="describe the project in plain English")
    init.add_argument("--name", help="project name (default: directory name)")
    init.add_argument("--list", action="store_true", help="list starters")
    init.add_argument("--auth", action="store_true", help="include user accounts (JWT + scrypt)")
    init.add_argument("--db", default="sqlite", choices=["sqlite", "postgres", "none"])
    init.add_argument("--description", default="")
    init.add_argument("--force", action="store_true", help="allow writing into a non-empty directory")
    init.add_argument("--dry-run", action="store_true")
    init.add_argument("--json", action="store_true")
    init.add_argument("--workspace", default=".")
    init.set_defaults(func=cmd_init)

    plan = subparsers.add_parser("plan", help="draft/approve a structured project plan")
    plan.add_argument("goal", nargs="?")
    plan.add_argument("--list", action="store_true")
    plan.add_argument("--approve", metavar="ID")
    plan.add_argument("--reject", metavar="ID")
    plan.add_argument("--save", action="store_true")
    plan.add_argument("--json", action="store_true")
    plan.add_argument("--workspace", default=".")
    plan.set_defaults(func=cmd_plan)

    project_map = subparsers.add_parser("map", help="build/print the persistent project map")
    project_map.add_argument("--json", action="store_true")
    project_map.add_argument("--chars", type=int, default=3000)
    project_map.add_argument("--workspace", default=".")
    project_map.set_defaults(func=cmd_map)

    search = subparsers.add_parser("search", help="local semantic search over the workspace")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=8)
    search.add_argument("--json", action="store_true")
    search.add_argument("--workspace", default=".")
    search.set_defaults(func=cmd_search)

    checkpoint = subparsers.add_parser("checkpoint", help="snapshot / diff / restore the workspace")
    checkpoint.add_argument("action", choices=["create", "list", "diff", "restore"])
    checkpoint.add_argument("--id", help="snapshot id (prefix is enough)")
    checkpoint.add_argument("--label", default="")
    checkpoint.add_argument("--json", action="store_true")
    checkpoint.add_argument("--workspace", default=".")
    checkpoint.set_defaults(func=cmd_checkpoint)

    review = subparsers.add_parser("review", help="static review (secrets, swallowed errors, missing tests)")
    review.add_argument("--paths", nargs="*")
    review.add_argument("--threshold", type=int, default=80)
    review.add_argument("--json", action="store_true")
    review.add_argument("--workspace", default=".")
    review.set_defaults(func=cmd_review)

    testgen = subparsers.add_parser("testgen", help="generate starter tests for untested modules")
    testgen.add_argument("--paths", nargs="*")
    testgen.add_argument("--force", action="store_true")
    testgen.add_argument("--json", action="store_true")
    testgen.add_argument("--workspace", default=".")
    testgen.set_defaults(func=cmd_testgen)

    verify = subparsers.add_parser("verify", help="run the test suite and explain failures")
    verify.add_argument("--command", help="override the test command")
    verify.add_argument("--timeout", type=float, default=600)
    verify.add_argument("--json", action="store_true")
    verify.add_argument("--workspace", default=".")
    verify.set_defaults(func=cmd_verify)

    backend = subparsers.add_parser("backend", help="provision a managed Postgres (Neon/Supabase) or use SQLite")
    backend.add_argument("--provider", help="existing | neon | supabase | sqlite (default: auto)")
    backend.add_argument("--name")
    backend.add_argument("--list", action="store_true")
    backend.add_argument("--dry-run", action="store_true")
    backend.add_argument("--json", action="store_true")
    backend.add_argument("--workspace", default=".")
    backend.set_defaults(func=cmd_backend)

    deploy = subparsers.add_parser("deploy", help="build, push and verify a live URL")
    deploy.add_argument("--provider", help="local | vercel | fly | render | coolify | github-pages | archive")
    deploy.add_argument("--dry-run", action="store_true", help="show the plan, create nothing")
    deploy.add_argument("--timeout", type=float, default=900)
    deploy.add_argument("--no-publish", action="store_true", help="do not write landing page/README/DEPLOY.md")
    deploy.add_argument("--description", default="", help="short description for the landing page")
    deploy.add_argument("--json", action="store_true")
    deploy.add_argument("--workspace", default=".")
    deploy.set_defaults(func=cmd_deploy)

    preview = subparsers.add_parser("preview", help="start the chat UI (and live preview)")
    preview.add_argument("--port", type=int, default=int(os.environ.get("AGENT_UI_PORT", "7860")))
    preview.add_argument("--host", default="0.0.0.0")
    preview.add_argument("--model-url", help="OpenAI-compatible endpoint")
    preview.add_argument("--workspace", default=".")
    preview.set_defaults(func=cmd_preview)

    run = subparsers.add_parser("run", help="run one task (local workflow, or a model when configured)")
    run.add_argument("task")
    run.add_argument("--mode", default="simple", choices=["simple", "power"])
    run.add_argument("--plan-first", action="store_true", help="stop after the plan for approval")
    run.add_argument("--model-url", default=os.environ.get("AGENT_MODEL_URL"))
    run.add_argument("--verbose", action="store_true")
    run.add_argument("--workspace", default=".")
    run.set_defaults(func=cmd_run)

    clone = subparsers.add_parser("clone", help="clone a GitHub repo, index it and summarise the structure")
    clone.add_argument("url")
    clone.add_argument("--directory", "-d")
    clone.add_argument("--json", action="store_true")
    clone.set_defaults(func=cmd_clone, workspace=".")

    doctor = subparsers.add_parser("doctor", help="check the local toolchain and project detection")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--workspace", default=".")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        _say("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
