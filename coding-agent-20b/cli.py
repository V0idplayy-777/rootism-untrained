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
    auth_provider = getattr(args, "auth_provider", None)
    extra = {}
    if auth_provider:
        extra["auth_provider"] = auth_provider
    add_jobs = not getattr(args, "no_jobs", False)
    extra["add_jobs"] = add_jobs
    if args.request:
        generator = ScaffoldGenerator(root, force=args.force, dry_run=args.dry_run)
        if auth_provider:
            from scaffold.intent import intent_from_request
            from scaffold.registry import template_ids
            intent, choice = intent_from_request(args.request, template_ids())
            result = generator.create(
                choice.template_id,
                name=name,
                description=args.request.strip()[:300],
                use_auth=bool(choice.options.get("use_auth") or auth_provider),
                database=choice.options.get("database"),
                extra_options=extra,
            )
            result.reasons = list(choice.reasons)
            result.questions = list(choice.questions)
            result.options["intent"] = intent.to_dict()
        else:
            result = generator.create_from_request(args.request, name=name)
            if add_jobs or auth_provider:
                pass
        if (auth_provider or add_jobs) and result.ok and not result.dry_run:
            if auth_provider and "auth_provider" not in result.options:
                from scaffold.auth_providers import auth_provider_files, get_auth_provider
                try:
                    ap = get_auth_provider(auth_provider)
                    for rel, content in auth_provider_files(auth_provider, result.options.get("module", "app")).items():
                        target = root / rel
                        if not target.exists():
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_text(content, encoding="utf-8")
                            result.files.append(rel)
                    result.warnings.append(f"auth provider {auth_provider} added")
                except Exception:
                    pass
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
            extra_options=extra,
            auth_provider=auth_provider,
            add_jobs=add_jobs,
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


def cmd_ship(args) -> int:
    """Single-command idea -> live app: scaffold -> test -> deploy -> return URL (row 1)."""
    from scaffold import ScaffoldGenerator
    from quality import CodeCritic, run_tests
    from deploy import DeployEngine

    raw_ws = getattr(args, "workspace", ".")
    target_path = getattr(args, "path", None)
    if target_path:
        root = Path(target_path).resolve()
        name = getattr(args, "name", None) or root.name
    else:
        root = Path(raw_ws).resolve()
        name = getattr(args, "name", None) or root.name
    request = getattr(args, "request", "") or getattr(args, "description", "")
    if not request:
        _say("agent ship needs a request: agent ship \"a task app with login\"")
        return 2

    _say(f"→ Ship: {request[:100]}")
    _say(f"  workspace: {root}")
    generator = ScaffoldGenerator(root, force=True, dry_run=False)
    auth_provider = getattr(args, "auth_provider", None)
    extra = {"add_jobs": True}
    if auth_provider:
        extra["auth_provider"] = auth_provider
    result = generator.create_from_request(request, name=name)
    if auth_provider and result.ok:
        try:
            from scaffold.auth_providers import auth_provider_files, get_auth_provider
            ap = get_auth_provider(auth_provider)
            module = result.options.get("module", "app")
            for rel, content in auth_provider_files(auth_provider, module).items():
                target = root / rel
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
        except Exception:
            pass

    if not result.ok:
        _say(f"Scaffold failed: {result.error}")
        if args.json:
            _print_json({"ok": False, "stage": "scaffold", "error": result.error})
        return 1
    _say(f"✓ Scaffolded {result.template_id} ({len(result.files)} files)")

    _say("→ Verifying tests...")
    verify = run_tests(root, timeout=getattr(args, "timeout", 300) or 300)
    _say(verify.render())
    if not verify.ok and not args.force:
        _say("Tests failed — ship blocked by green gate (use --force to bypass)")
        if args.json:
            _print_json({"ok": False, "stage": "verify", "verify": verify.to_dict(), "scaffold": result.to_dict()})
        return 1
    elif not verify.ok:
        _say("Tests failed but --force given, continuing...")

    # 3. Review
    _say("→ Reviewing...")
    critic = CodeCritic(threshold=getattr(args, "threshold", 80) or 80)
    review = critic.review(root)
    _say(review.render())
    if not review.ok and not args.force:
        _say(f"Review failed (score {review.score} < {review.threshold}) — ship blocked (use --force)")
        if args.json:
            _print_json({"ok": False, "stage": "review", "review": review.to_dict(), "verify": verify.to_dict()})
        return 1

    _say("→ Deploying...")
    provider = getattr(args, "provider", None) or "auto"
    engine = DeployEngine(
        root,
        provider=provider,
        dry_run=bool(getattr(args, "dry_run", False)),
        timeout=getattr(args, "timeout", 900) or 900,
        publish_metadata=True,
        description=request,
        force=bool(getattr(args, "force", False)),
        require_green=not bool(getattr(args, "force", False)),
        green_threshold=getattr(args, "threshold", 80) or 80,
    )
    deploy_result = engine.deploy()
    _say(deploy_result.render())
    if args.json:
        _print_json({
            "ok": deploy_result.ok,
            "stage": "deploy",
            "scaffold": result.to_dict(),
            "verify": verify.to_dict(),
            "review": review.to_dict(),
            "deploy": deploy_result.to_dict(),
            "url": getattr(deploy_result, "url", None),
        })
    else:
        if deploy_result.ok and getattr(deploy_result, "url", None):
            _say(f"\n✓ Your app is live at {deploy_result.url}")
            _say(f"  Status dashboard: status/index.html")
            _say(f"  Last deploy: {deploy_result.meta.get('deployed_at', 'now')}")
            _say(f"  Health: {deploy_result.health if hasattr(deploy_result, 'health') else 'checked'}")
            _say("Here's how to change it: edit files, then run `agent ship` again or `agent deploy`.")
        elif deploy_result.ok:
            _say("✓ Ship complete — artifact ready (no public URL; see HOSTING.md)")
        else:
            _say(f"Deploy failed: {deploy_result.error}")
    return 0 if deploy_result.ok else 1


def cmd_app(args) -> int:
    """Local product path: build from a sentence, verify, repair once, show preview."""
    from scaffold.app_builder import (
        NoBlueprint,
        build_app,
        preview_command,
        repair_generated,
        resolve_blueprint,
    )

    say = (lambda *_a, **_k: None) if getattr(args, "json", False) else _say
    root = Path(getattr(args, "path", None) or getattr(args, "workspace", ".")).resolve()
    request = args.request
    if not request:
        say('agent app needs a request: agent app "a team task list with sharing"')
        return 2
    say(f"→ Build: {request[:120]}")
    say(f"  directory: {root}")
    try:
        report = build_app(
            root,
            request,
            blueprint_id=getattr(args, "blueprint", None),
            force=bool(getattr(args, "force", False)),
            taste_threshold=getattr(args, "taste_threshold", 70) or 70,
        )
    except NoBlueprint as exc:
        say(str(exc))
        if getattr(args, "json", False):
            _print_json({"ok": False, "stage": "resolve", "error": str(exc), "suggestions": exc.suggestions})
        return 2

    if not report.ok and getattr(args, "repair", False):
        say("→ Gates failed; restoring generated modules and re-running...")
        blueprint = resolve_blueprint(request, getattr(args, "blueprint", None))
        repair = repair_generated(root, blueprint, taste_threshold=getattr(args, "taste_threshold", 70) or 70)
        say(repair.summary())
        report = build_app(
            root,
            request,
            blueprint_id=blueprint.id,
            force=True,
            taste_threshold=getattr(args, "taste_threshold", 70) or 70,
        )

    if getattr(args, "json", False):
        _print_json(report.to_dict())
    else:
        say(f"  files: {len(report.files)}")
        if report.verify is not None:
            say("  tests: " + ("passing (" + str(len(report.verify.passed)) + ")" if report.verify.ok else "FAILING"))
        say(f"  critic: {report.critic_score}/100" + ("" if report.critic_ok else " (blocked)"))
        if report.taste is not None:
            say(f"  taste: {report.taste.score}/100" + ("" if report.taste.ok else " (below bar)"))
        if report.deploy_provider:
            say(f"  deploy plan: {report.deploy_provider} ({report.deploy_steps} steps)")
        for warning in report.warnings[:3]:
            say("  ! " + warning)

    if not report.ok:
        say("")
        say("✗ Not shipping this: one or more gates failed.")
        for failure in report.failures:
            say("  - " + failure.replace("\n", " ")[:500])
        say("  Next: `agent app ... --repair` restores generated modules; otherwise fix the files above.")
        return 1

    say("")
    say(f"✓ {report.summary()}")
    say(f"  Preview: {report.preview_command}")
    say(f"  Eject check: agent eject --check  (the app has no runtime dependency on this tool)")
    if getattr(args, "deploy", False):
        from deploy import DeployEngine

        result = DeployEngine(root, provider=getattr(args, "provider", None) or "auto").deploy()
        say(result.render())
        if not result.ok:
            return 1
        say("✓ Deployed" + (f": {result.url}" if getattr(result, "url", None) else " (artifact ready)"))
    return 0


def cmd_eject(args) -> int:
    """Eject guarantee: prove project runs with zero agent files (row 6)."""
    from pathlib import Path
    import shutil
    import subprocess
    import tempfile

    root = _workspace(args)
    check = getattr(args, "check", False)
    if not check:
        _say("Usage: agent eject --check  (proves project still runs without agent files)")
        _say("To eject fully: rm -rf .agent landing status agentdeploy.toml DEPLOY.md HOSTING.md")
        return 2

    agent_files = [".agent", "landing", "status", "agentdeploy.toml", "DEPLOY.md", "HOSTING.md"]
    existing = [f for f in agent_files if (root / f).exists()]
    _say(f"Found {len(existing)} agent-owned paths: {', '.join(existing) or 'none'}")
    violations = []
    for path in root.rglob("*.py"):
        if any(part in {".git", ".agent", "__pycache__", "node_modules", ".venv"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            if "import agent" in text or "from agent" in text or "from scaffold" in text:
                if path.relative_to(root).parts[0] not in {"docs", "tests"}:
                    violations.append(str(path.relative_to(root)))
        except Exception:
            continue
    if violations:
        _say(f"Contract violation: {len(violations)} app files import agent runtime: {violations[:5]}")
        _say("Fix: remove agent imports from app code (see docs/PROJECT_CONTRACT.md)")
        if args.json:
            _print_json({"ok": False, "violations": violations})
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        moved = []
        for rel in existing:
            src = root / rel
            dst = tmp_path / rel.replace("/", "_")
            try:
                if src.is_file():
                    shutil.move(str(src), str(dst))
                    moved.append((src, dst, "file"))
                elif src.is_dir():
                    shutil.move(str(src), str(dst))
                    moved.append((src, dst, "dir"))
            except Exception as exc:
                _say(f"Could not move {rel}: {exc}")
        try:
            from quality.verify import run_tests
            run = run_tests(root, timeout=60)
            if not run.ok:
                _say(f"Eject check: tests failed without agent files: {run.failure_signal()[:500]}")
                if args.json:
                    _print_json({"ok": False, "stage": "verify_without_agent", "verify": run.to_dict()})
                return 1
            _say("✓ Eject check passed: tests still pass without agent files")
            _say("  Badge: ✓ No agent runtime required — `agent eject --check` passed")
            if args.json:
                _print_json({"ok": True, "moved": [str(s) for s,_,_ in moved], "verify": run.to_dict(), "badge": "no-agent-runtime"})
            return 0
        finally:
            for src, dst, kind in moved:
                try:
                    shutil.move(str(dst), str(src))
                except Exception:
                    pass


def cmd_health(args) -> int:
    """Project health: broken imports, missing tests, circular deps (row 4)."""
    from agent.health import project_health
    root = _workspace(args)
    report = project_health(root)
    if args.json:
        _print_json(report.to_dict())
    else:
        _say(report.render())
        _say(f"\nStats: {report.stats}")
    return 0 if report.ok else 1


def cmd_onboarding(args) -> int:
    """Repo onboarding: produce arch summary + entry-point map before editing (row 7)."""
    from agent.onboarding import ensure_onboarding
    root = _workspace(args)
    subdir = getattr(args, "subdir", None)
    force = getattr(args, "force", False)
    report = ensure_onboarding(root, subdir=subdir, force=force)
    if args.json:
        _print_json(report.to_dict())
    else:
        _say(report.render())
        _say(f"\nsaved to .agent/onboarding.json + .agent/onboarding.md")
        if subdir:
            _say(f"Constraint: work only inside {subdir}")
    return 0


def cmd_db(args) -> int:
    """Managed Postgres path: switch, migrate, etc. (row 2)."""
    from scaffold.migrations import generate_migration
    action = getattr(args, "action", "status")
    root = _workspace(args)
    if action == "migrate":
        name = getattr(args, "name", "migration") or "migration"
        description = getattr(args, "description", "")
        module = root.name.replace("-", "_")
        try:
            import json as _json
            data = _json.loads((root / ".agent" / "scaffold.json").read_text())
            module = data.get("options", {}).get("module", module)
        except Exception:
            pass
        files = generate_migration(module, name, description)
        for rel, content in files.items():
            target = root / rel
            if target.exists() and not getattr(args, "force", False):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        if args.json:
            _print_json({"ok": True, "files": list(files.keys())})
        else:
            _say(f"Generated {len(files)} file(s):")
            for k in files:
                _say(f"  + {k}")
        return 0
    elif action == "switch":
        from scaffold.backends import BackendProvisioner
        provider = getattr(args, "provider", None) or "postgres"
        if provider == "postgres":
            provider = "existing" if (root / ".env").exists() and "DATABASE_URL" in (root / ".env").read_text() else "neon"
        prov = BackendProvisioner(root)
        result = prov.ensure(name=root.name, backend=provider)
        if args.json:
            _print_json(result.to_dict())
        else:
            _say(result.render())
        return 0 if result.ok else 1
    elif action == "upgrade":
        import subprocess
        module = root.name.replace("-", "_")
        try:
            import json as _json
            data = _json.loads((root / ".agent" / "scaffold.json").read_text())
            module = data.get("options", {}).get("module", module)
        except Exception:
            pass
        runner = root / module / "migrate.py"
        if not runner.exists():
            _say(f"No migrate runner at {runner}; run `agent db migrate` first")
            return 1
        completed = subprocess.run(["python", str(runner)], cwd=str(root), capture_output=True, text=True, timeout=60)
        _say(completed.stdout)
        if completed.stderr:
            _say(completed.stderr)
        if args.json:
            _print_json({"ok": completed.returncode == 0, "stdout": completed.stdout, "stderr": completed.stderr})
        return completed.returncode
    else:
        _say(f"Unknown db action {action}; use migrate|upgrade|switch")
        return 2


def cmd_status(args) -> int:
    """Show public status page info (row 1)."""
    from pathlib import Path
    import json
    root = _workspace(args)
    status_json = root / "status" / "status.json"
    last = root / ".agent" / "deploy" / "last.json"
    data = {}
    if status_json.exists():
        try:
            data["status_page"] = json.loads(status_json.read_text())
        except Exception:
            pass
    if last.exists():
        try:
            data["last_deploy"] = json.loads(last.read_text())
        except Exception:
            pass
    landing = root / "landing" / "index.html"
    data["landing_exists"] = landing.exists()
    data["status_exists"] = status_json.exists()
    if args.json:
        _print_json(data)
    else:
        if status_json.exists():
            _say(f"Status page: {status_json}")
            _say(json.dumps(data.get("status_page", {}), indent=2))
        else:
            _say("No status page yet; run `agent deploy` or `agent ship` to create one")
        if last.exists():
            _say(f"\nLast deploy: {last}")
    return 0


def cmd_contract(args) -> int:
    """Show generated project contract (row 6)."""
    from pathlib import Path
    contract = Path(__file__).parent / "docs" / "PROJECT_CONTRACT.md"
    if not contract.exists():
        contract = Path("docs/PROJECT_CONTRACT.md")
    if contract.exists():
        text = contract.read_text(encoding="utf-8")
        if args.json:
            _print_json({"contract": text})
        else:
            _say(text)
        return 0
    _say("Contract not found")
    return 1


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
        force=getattr(args, "force", False),
        require_green=not getattr(args, "no_require_green", False) and not getattr(args, "force", False),
        green_threshold=getattr(args, "threshold", 80),
    )
    result = engine.deploy()
    if args.json:
        _print_json(result.to_dict())
    else:
        _say(result.render())
        if not result.ok and getattr(result, "human_error", None):
            he = result.human_error
            _say(f"\n{he.get('title','')}")
            _say(f"  {he.get('plain','')}")
            for step in he.get("next_steps", []):
                _say(f"  -> {step}")
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
    init.add_argument("--auth-provider", choices=["clerk", "supabase", "authjs"], help="first-class auth provider (clerk/supabase/authjs)")
    init.add_argument("--no-jobs", action="store_true", help="skip adding background-job example")
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
    deploy.add_argument("--force", action="store_true", help="bypass green gate (deploy even if tests/review fail)")
    deploy.add_argument("--no-require-green", action="store_true", help="disable green-before-live gate")
    deploy.add_argument("--threshold", type=int, default=80, help="review score threshold for green gate")
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

    app = subparsers.add_parser("app", help='build a running product from a sentence (blueprint path)')
    app.add_argument("request", nargs="?", help="describe the app in plain English")
    app.add_argument("path", nargs="?", help="target directory (default: workspace)")
    app.add_argument("--blueprint", help="blueprint id (otherwise matched from the request)")
    app.add_argument("--force", action="store_true", help="overwrite generated files in a non-empty directory")
    app.add_argument("--repair", action="store_true", help="restore generated modules once, then re-verify")
    app.add_argument("--deploy", action="store_true", help="deploy after every gate passes")
    app.add_argument("--provider", help="deploy provider (default: auto)")
    app.add_argument("--taste-threshold", type=int, default=70)
    app.add_argument("--json", action="store_true")
    app.add_argument("--workspace", default=".")
    app.set_defaults(func=cmd_app)

    ship = subparsers.add_parser("ship", help='one-command idea → live app: agent ship "a task app with login"')
    ship.add_argument("request", nargs="?", help="describe the app in plain English")
    ship.add_argument("path", nargs="?", help="target directory (default: workspace)")
    ship.add_argument("--name", help="project name")
    ship.add_argument("--provider", help="deploy provider (default: auto)")
    ship.add_argument("--auth-provider", choices=["clerk", "supabase", "authjs"], help="auth provider")
    ship.add_argument("--threshold", type=int, default=80, help="review threshold for green gate")
    ship.add_argument("--timeout", type=float, default=900)
    ship.add_argument("--dry-run", action="store_true")
    ship.add_argument("--force", action="store_true", help="bypass green gate")
    ship.add_argument("--description", default="", help="short description")
    ship.add_argument("--json", action="store_true")
    ship.add_argument("--workspace", default=".")
    ship.set_defaults(func=cmd_ship)

    eject = subparsers.add_parser("eject", help="prove no lock-in: agent eject --check")
    eject.add_argument("--check", action="store_true", help="verify project still runs with zero agent files")
    eject.add_argument("--json", action="store_true")
    eject.add_argument("--workspace", default=".")
    eject.set_defaults(func=cmd_eject)

    health = subparsers.add_parser("health", help="project health: broken imports, missing tests, circular deps")
    health.add_argument("--json", action="store_true")
    health.add_argument("--workspace", default=".")
    health.set_defaults(func=cmd_health)

    onboarding = subparsers.add_parser("onboarding", help="repo onboarding: arch summary + entry-point map")
    onboarding.add_argument("--subdir", help="constrain work to this subdirectory/package")
    onboarding.add_argument("--force", action="store_true", help="re-generate even if exists")
    onboarding.add_argument("--json", action="store_true")
    onboarding.add_argument("--workspace", default=".")
    onboarding.set_defaults(func=cmd_onboarding)

    db = subparsers.add_parser("db", help="managed Postgres path: agent db migrate|upgrade|switch")
    db.add_argument("action", nargs="?", default="status", choices=["migrate", "upgrade", "switch", "status"], help="db action")
    db.add_argument("--name", help="migration name")
    db.add_argument("--provider", help="postgres provider for switch (neon/supabase/existing)")
    db.add_argument("--description", default="", help="migration description")
    db.add_argument("--force", action="store_true")
    db.add_argument("--json", action="store_true")
    db.add_argument("--workspace", default=".")
    db.set_defaults(func=cmd_db)

    status_cmd = subparsers.add_parser("status", help="show live URL, last deploy time, health (dashboard)")
    status_cmd.add_argument("--json", action="store_true")
    status_cmd.add_argument("--workspace", default=".")
    status_cmd.set_defaults(func=cmd_status)

    contract = subparsers.add_parser("contract", help="show generated project contract (eject guarantee)")
    contract.add_argument("--json", action="store_true")
    contract.add_argument("--workspace", default=".")
    contract.set_defaults(func=cmd_contract)

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
