"""FastAPI app for the chat UI (the "simple mode" surface over the agent).

Design goals, in order:

1. **Nothing raw by default.** Tool calls, stack traces and file dumps are
   collapsed; the chat shows progress, plans, test results, the review score and
   the live URL. Power mode reveals the raw events.
2. **Same backend as the CLI.** Every action calls the same modules the CLI
   calls (scaffold, quality, deploy) - the UI is a surface, not a second
   implementation.
3. **It works without a model.** The local workflow still scaffolds, tests,
   reviews and plans a deploy; the UI states which mode it is in.
4. **One origin.** The preview is proxied through this server at ``/preview/`` so
   the browser never needs to reach a sandbox-local port.

Run: ``python -m ui --port 7860`` (or ``agent preview``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterator

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from deploy import DeployEngine, available_providers
from deploy.config import DeployConfig
from deploy.providers.base import DeployContext
from quality import CodeCritic, humanize, run_tests
from scaffold import ScaffoldGenerator, catalog
from ui.preview import PreviewManager, list_workspace_files, read_workspace_file
from ui.runtime import AgentRuntime

STATIC_DIR = Path(__file__).parent / "static"
MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=8000)
    mode: str = Field(default="simple", pattern="^(simple|power)$")
    require_approval: bool = False


class ProjectRequest(BaseModel):
    template: str = "fastapi-react"
    name: str = ""
    request: str = ""
    dry_run: bool = False


class DeployRequest(BaseModel):
    provider: str | None = None
    dry_run: bool = False
    publish: bool = True


class FileRequest(BaseModel):
    path: str


def _visible(events: Iterator[dict], mode: str) -> Iterator[dict]:
    """Hide tool-level events from the simple stream (server-side, not just CSS).

    Power mode gets everything; simple mode gets progress a non-engineer can act
    on. The client also filters, but doing it here means the simple stream is
    quiet for any client (curl, a terminal, a screenshot test).
    """

    if mode == "power":
        yield from events
        return
    for event in events:
        data = event.get("data") or {}
        if data.get("power_only"):
            continue
        if event.get("type") in {"step", "file"}:
            continue
        yield event


def sse(events: Iterator[dict]) -> Iterator[str]:
    """Encode events as Server-Sent Events (one JSON object per data line)."""

    for event in events:
        yield f"data: {json.dumps(event, default=str)}\n\n"
    yield "data: {\"type\": \"end\"}\n\n"


def create_app(workspace: str | Path = ".", model_url: str | None = None) -> FastAPI:
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    runtime = AgentRuntime(root, model_url=model_url)
    preview = PreviewManager(root)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        preview.stop()

    app = FastAPI(title="Coding agent", version="1.0.0", docs_url="/api/docs", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/assets/{path:path}")
    def asset(path: str) -> Response:
        target = (STATIC_DIR / path).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            return Response(status_code=404)
        return FileResponse(target, media_type=MIME.get(target.suffix.lower(), "application/octet-stream"))

    @app.get("/favicon.ico")
    def favicon() -> Response:
        target = STATIC_DIR / "favicon.svg"
        return FileResponse(target, media_type="image/svg+xml") if target.exists() else Response(status_code=204)

    @app.get("/api/healthz")
    def healthz() -> dict:
        return {"ok": True, "workspace": str(root)}

    @app.get("/api/info")
    def info() -> dict:
        profile = runtime.profile()
        providers = {}
        if profile is not None:
            config = DeployConfig.discover(root, profile)
            context = DeployContext(root=root, profile=profile, config=config, image="", dry_run=True)
            providers = {name: availability.to_dict() for name, availability in available_providers(context).items()}
        return {
            "runtime": runtime.info().to_dict(),
            "project": profile.to_dict() if profile else None,
            "providers": providers,
            "preview": preview.status(),
            "files": len(list_workspace_files(root, max_files=5000)),
            "history": [turn.to_dict() for turn in runtime.history[-50:]],
            "modes": ["simple", "power"],
            "tooling": {
                "git": bool(shutil.which("git")),
                "docker": bool(shutil.which("docker")),
                "node": bool(shutil.which("node")),
            },
        }

    @app.get("/api/templates")
    def templates() -> dict:
        return {"templates": catalog()}

    @app.get("/api/session")
    def session() -> dict:
        return {"history": [turn.to_dict() for turn in runtime.history]}

    @app.post("/api/session/reset")
    def reset_session() -> dict:
        runtime.reset()
        return {"ok": True}

    @app.post("/api/chat")
    def chat(payload: ChatRequest) -> StreamingResponse:
        events = runtime.run(payload.message, mode=payload.mode, require_approval=payload.require_approval)
        return StreamingResponse(
            sse(_visible(events, payload.mode)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/plan/continue")
    def plan_continue(payload: ChatRequest) -> StreamingResponse:
        return StreamingResponse(sse(runtime.continue_plan()), media_type="text/event-stream")

    @app.post("/api/plan/reject")
    def plan_reject() -> dict:
        runtime._pending = None
        return {"ok": True, "detail": "plan rejected; nothing was written"}

    @app.post("/api/projects")
    def create_project(payload: ProjectRequest) -> dict:
        generator = ScaffoldGenerator(root, force=True, dry_run=payload.dry_run)
        if payload.request:
            result = generator.create_from_request(payload.request, name=payload.name or root.name)
        else:
            result = generator.create(payload.template, payload.name or root.name)
        return result.to_dict()

    @app.post("/api/review")
    def review() -> dict:
        review_result = CodeCritic().review(root)
        return review_result.to_dict() | {"rendered": review_result.render(limit=20)}

    @app.post("/api/verify")
    def verify() -> dict:
        run = run_tests(root, timeout=600)
        payload = run.to_dict()
        if not run.ok:
            payload["human"] = humanize(run.failure_signal()).to_dict()
        return payload

    @app.post("/api/deploy")
    def deploy(payload: DeployRequest) -> dict:
        engine = DeployEngine(
            root,
            provider=payload.provider or None,
            dry_run=payload.dry_run,
            publish_metadata=payload.publish,
        )
        result = engine.deploy()
        data = result.to_dict()
        data["rendered"] = result.render()
        # Plain-language for non-engineers (row 8)
        if not result.ok and getattr(result, "human_error", None):
            data["human"] = result.human_error
        elif result.ok and result.url:
            data["plain_success"] = f"Your app is live at {result.url}. Open it, test it, and tell me what to change next."
        return data

    class ShipRequest(BaseModel):
        request: str = Field(default="", max_length=2000)
        provider: str | None = None
        dry_run: bool = False
        force: bool = False

    @app.post("/api/ship")
    def ship(payload: ShipRequest):
        """One-click Ship it for non-engineers (row 1+8): scaffold → verify → review → deploy."""
        from scaffold import ScaffoldGenerator
        from quality import CodeCritic, run_tests

        request = (payload.request or "").strip()
        if not request:
            return JSONResponse(status_code=400, content={"ok": False, "error": "describe your app first"})
        # 1 scaffold
        gen = ScaffoldGenerator(root, force=True, dry_run=False)
        scaffold_result = gen.create_from_request(request, name=root.name)
        if not scaffold_result.ok:
            human = humanize(scaffold_result.error)
            return {"ok": False, "stage": "scaffold", "error": human.title, "human": human.to_dict(), "scaffold": scaffold_result.to_dict()}
        # 2 verify
        verify_result = run_tests(root, timeout=120)
        if not verify_result.ok and not payload.force:
            human = humanize(verify_result.failure_signal())
            return {"ok": False, "stage": "verify", "error": human.title, "human": human.to_dict(), "verify": verify_result.to_dict(), "scaffold": scaffold_result.to_dict()}
        # 3 review
        review_result = CodeCritic(threshold=80).review(root)
        if not review_result.ok and not payload.force:
            return {"ok": False, "stage": "review", "error": f"Review needs work (score {review_result.score})", "human": {"title": "Review needs work", "plain": review_result.summary(), "next_steps": ["Fix the noted findings, then Ship it again"]}, "review": review_result.to_dict()}
        # 4 deploy
        engine = DeployEngine(root, provider=payload.provider or None, dry_run=payload.dry_run, force=payload.force, description=request)
        deploy_result = engine.deploy()
        data = deploy_result.to_dict()
        data["rendered"] = deploy_result.render()
        data["scaffold"] = scaffold_result.to_dict()
        data["verify"] = verify_result.to_dict()
        data["review"] = review_result.to_dict()
        if not deploy_result.ok:
            data["human"] = getattr(deploy_result, "human_error", {"title": "Deploy failed", "plain": deploy_result.error, "next_steps": ["Check DEPLOY.md", "Try `agent deploy --dry-run`"]})
            data["plain"] = f"We couldn't ship it: {data['human'].get('plain','')} Here's what to do: {', '.join(data['human'].get('next_steps',[]))}"
        else:
            url = deploy_result.url or "status/index.html"
            data["plain"] = f"Your app is live at {url}. {data.get('notes', [''])[-1] if data.get('notes') else ''} Tell me what to change next."
            data["plain_success"] = data["plain"]
        return data

    # ----------------------------------------------------------------- preview
    @app.get("/api/preview")
    def preview_status() -> dict:
        return preview.status()

    @app.post("/api/preview/start")
    def preview_start() -> dict:
        return preview.start().to_dict()

    @app.post("/api/preview/stop")
    def preview_stop() -> dict:
        return preview.stop().to_dict()

    @app.api_route("/preview", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    @app.api_route("/preview/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def preview_request(request: Request, path: str = "") -> Response:
        if not preview.status().get("running"):
            started = preview.start()
            if not started.running:
                body = _preview_message(started.detail, started.logs)
                return Response(content=body, media_type="text/html", status_code=503)
        if preview.state.kind == "static":
            result = preview.static_file(path or "index.html")
            if result is None:
                return Response(content=_preview_message("no static directory", []), media_type="text/html", status_code=503)
            status, mime, body = result
            return Response(content=body, media_type=mime, status_code=status)
        body = await request.body()
        status, mime, payload = preview.proxy(
            path or "", method=request.method, body=body or None, headers=dict(request.headers)
        )
        return Response(content=payload, media_type=mime, status_code=status)

    @app.get("/api/files")
    def files() -> dict:
        return {"files": list_workspace_files(root)}

    @app.get("/api/file")
    def file(path: str) -> dict:
        return read_workspace_file(root, path)

    @app.get("/api/plans")
    def plans() -> dict:
        from agent.plan_approval import PlanStore

        return {"plans": [plan.to_dict() for plan in PlanStore(root).list()]}

    @app.post("/api/plans/{plan_id}/approve")
    def approve(plan_id: str) -> dict:
        from agent.plan_approval import PlanStore

        plan = PlanStore(root).approve(plan_id)
        return {"ok": True, "plan": plan.to_dict()}

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover - safety net
        human = humanize(f"{type(exc).__name__}: {exc}")
        return JSONResponse(status_code=500, content={"error": human.title, "human": human.to_dict()})

    return app


def _preview_message(detail: str, logs: list[str]) -> str:
    log_html = "<pre>" + "\n".join(logs[-20:]).replace("<", "&lt;") + "</pre>" if logs else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Preview</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;margin:0;padding:40px;background:#0f1218;color:#e8ebf3}}
code,pre{{font-family:ui-monospace,Menlo,monospace;font-size:12px;background:#161b22;padding:10px;border-radius:8px;overflow:auto}}
h1{{font-size:20px}}</style></head><body>
<h1>Preview is not available yet</h1>
<p>{detail}</p>{log_html}
<p>Ask the agent to start the app, or run it manually and refresh.</p>
</body></html>"""


def serve(workspace: str = ".", host: str = "0.0.0.0", port: int = 7860, model_url: str | None = None) -> None:
    import uvicorn

    uvicorn.run(create_app(workspace, model_url=model_url), host=host, port=port, log_level="info")
