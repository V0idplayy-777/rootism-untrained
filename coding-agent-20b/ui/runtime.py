"""The thing behind the chat box.

Two backends, one event stream:

* **Model backend** - when an OpenAI-compatible server is configured
  (``AGENT_MODEL_URL``), a full :class:`~agent.production_agent.ProductionAgent`
  runs the task with tools, recovery and session state. Steps are surfaced as
  events; the UI decides how much to show.
* **Local workflow** - with no model, the same request still produces real work:
  intent detection -> plan -> scaffold (when the workspace is empty) -> generated
  tests -> test run -> static review -> deploy plan. Nothing is simulated and the
  UI labels it "local workflow (no model)".

Events are plain dicts (``type``, ``message``, ``data``) so the CLI, the SSE
endpoint and tests all consume the same thing.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from deploy import DeployEngine, detect_project
from quality import CodeCritic, humanize, run_tests, write_generated_tests
from scaffold import ScaffoldGenerator, catalog, parse_intent
from scaffold.generator import plan_for

SESSION_FILE = Path(".agent") / "ui_sessions.json"


@dataclass
class Turn:
    role: str
    content: str
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "at": self.at}


@dataclass
class RuntimeInfo:
    mode: str
    model_url: str | None
    workspace: str
    templates: list[dict]
    python: str
    node: str | None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "model_url": self.model_url,
            "workspace": self.workspace,
            "templates": self.templates,
            "python": self.python,
            "node": self.node,
        }


def http_generate_fn(url: str, model: str | None = None, timeout: float = 120.0) -> Callable[[str], str]:
    """Generate with an OpenAI-compatible ``/v1/chat/completions`` endpoint."""

    endpoint = url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = endpoint + "/v1/chat/completions" if endpoint.endswith("/v1") is False else endpoint + "/chat/completions"

    def generate(prompt: str) -> str:
        payload = {
            "model": model or os.environ.get("AGENT_MODEL_NAME", "coding-agent"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            return str(message.get("content") or choices[0].get("text") or "")
        return str(data.get("text") or data.get("generated_text") or "")

    return generate


class AgentRuntime:
    def __init__(
        self,
        root: str | Path,
        *,
        model_url: str | None = None,
        session_file: str | Path | None = None,
        environ: dict | None = None,
    ):
        self.root = Path(root).resolve()
        self.environ = dict(os.environ if environ is None else environ)
        self.model_url = model_url or self.environ.get("AGENT_MODEL_URL") or None
        self.session_file = Path(session_file) if session_file else self.root / SESSION_FILE
        self.history: list[Turn] = self._load_history()

    # ------------------------------------------------------------------- info
    def info(self) -> RuntimeInfo:
        import sys

        node = None
        try:
            import shutil

            node = shutil.which("node")
        except Exception:  # noqa: BLE001
            node = None
        return RuntimeInfo(
            mode="model" if self.model_url else "local",
            model_url=self.model_url,
            workspace=str(self.root),
            templates=catalog(),
            python=sys.version.split()[0],
            node=node,
        )

    def profile(self):
        try:
            return detect_project(self.root)
        except Exception:  # noqa: BLE001 - an odd workspace must not break the UI
            return None

    # ---------------------------------------------------------------- history
    def _load_history(self) -> list[Turn]:
        if not self.session_file.exists():
            return []
        try:
            data = json.loads(self.session_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [Turn(role=item.get("role", "user"), content=item.get("content", ""), at=float(item.get("at", 0))) for item in data]

    def _save_history(self) -> None:
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        payload = [turn.to_dict() for turn in self.history[-200:]]
        self.session_file.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    def remember(self, role: str, content: str) -> None:
        self.history.append(Turn(role=role, content=content))
        self._save_history()

    def reset(self) -> None:
        self.history = []
        self._save_history()

    # ------------------------------------------------------------------- turn
    def run(self, message: str, *, mode: str = "simple", require_approval: bool = False) -> Iterator[dict]:
        """Yield events for one user message (SSE-friendly)."""

        message = (message or "").strip()
        if not message:
            yield _event("error", "Type a request first.", {"plain": "empty message"})
            return
        self.remember("user", message)
        if self.model_url:
            yield from self._run_model(message, mode)
            return
        yield from self._run_local(message, mode, require_approval)

    # ----------------------------------------------------------- model backend
    def _run_model(self, message: str, mode: str) -> Iterator[dict]:
        yield _event("status", "Model connected - planning with tools.", {"mode": "model"})
        try:
            from agent.production_agent import ProductionAgent

            responses: list[str] = []

            def generate(prompt: str) -> str:
                text = http_generate_fn(self.model_url)(prompt)
                responses.append(text)
                return text

            agent = ProductionAgent(
                workspace_path=str(self.root),
                generate_fn=generate,
                max_steps=24,
                state_path=str(self.root / ".agent" / "ui_session_state.json"),
            )
            result = agent.run(message, verbose=False)
        except Exception as exc:  # noqa: BLE001
            human = humanize(f"{type(exc).__name__}: {exc}")
            yield _event("error", human.title, human.to_dict())
            return

        for index, step in enumerate(result.get("steps") or [], 1):
            tool = step.get("tool") or "model"
            result_data = step.get("result") or {}
            ok = not result_data.get("error")
            yield _event(
                "step" if mode == "power" else "progress",
                f"{index}. {tool} {'ok' if ok else 'failed'}",
                {
                    "tool": tool,
                    "arguments": step.get("arguments") or {},
                    "ok": ok,
                    "detail": (str(result_data.get("error") or result_data.get("output") or ""))[:400],
                    "power_only": mode != "power",
                },
            )

        answer = result.get("answer") or "The run ended without a final answer."
        self.remember("assistant", answer)
        yield _event("done", answer, {"ok": bool(result.get("success")), "steps": len(result.get("steps") or [])})

    # ----------------------------------------------------------- local backend
    def _run_local(self, message: str, mode: str, require_approval: bool) -> Iterator[dict]:
        yield _event(
            "status",
            "No model is connected - running the local workflow (scaffold, tests, review, deploy plan).",
            {"mode": "local"},
        )
        intent = parse_intent(message)
        yield _event("intent", f"Read as: {intent.summary()}", intent.to_dict())

        empty = _is_empty(self.root)
        if empty:
            yield from self._scaffold_flow(message, mode, require_approval)
        else:
            yield from self._existing_project_flow(message, mode)

    def _scaffold_flow(self, message: str, mode: str, require_approval: bool) -> Iterator[dict]:
        draft = plan_for(message, name=self.root.name)
        plan_id = f"scaffold-{int(time.time()) % 100000:05d}"
        steps = draft["steps"]
        yield _event(
            "plan",
            f"Plan: {draft['template']['name']} ({len(draft['files'])} files)",
            {
                "id": plan_id,
                "template": draft["template"]["id"],
                "steps": steps,
                "files": draft["files"],
                "reasons": draft["reasons"],
                "questions": draft["questions"],
                "awaiting_approval": require_approval,
            },
        )
        if require_approval:
            self._pending = {"id": plan_id, "message": message, "mode": mode}
            yield _event("plan_pending", "Waiting for plan approval.", {"id": plan_id})
            return

        result = ScaffoldGenerator(self.root).create_from_request(message, name=self.root.name)
        if not result.ok:
            human = humanize(result.error)
            yield _event("error", human.title, human.to_dict())
            return
        if mode == "power":
            for path in result.files:
                yield _event("file", f"+ {path}", {"path": path})
        yield _event(
            "scaffold",
            f"Created {len(result.files)} files from the {draft['template']['name']} starter",
            {"files": result.files, "options": result.options, "warnings": result.warnings},
        )
        yield from self._verify_and_review(mode, generated_tests=True)
        yield from self._deploy_plan()

    def continue_plan(self, plan_id: str = "") -> Iterator[dict]:
        pending = getattr(self, "_pending", None)
        if not pending:
            yield _event("error", "No plan is waiting for approval.", {"plain": "nothing to resume"})
            return
        message = pending["message"]
        mode = pending.get("mode", "simple")
        result = ScaffoldGenerator(self.root).create_from_request(message, name=self.root.name)
        self._pending = None
        if not result.ok:
            human = humanize(result.error)
            yield _event("error", human.title, human.to_dict())
            return
        yield _event("scaffold", f"Created {len(result.files)} files", {"files": result.files, "options": result.options})
        yield from self._verify_and_review(mode, generated_tests=True)
        yield from self._deploy_plan()

    def _existing_project_flow(self, message: str, mode: str) -> Iterator[dict]:
        yield from self._verify_and_review(mode, generated_tests=False)
        yield from self._deploy_plan()
        self.remember(
            "assistant",
            "Reviewed the existing project. Connect a model (AGENT_MODEL_URL) for code edits; the review, tests and deploy plan above work without one.",
        )
        yield _event(
            "done",
            "Project reviewed. Connect a model to make code changes; everything above (tests, review, deploy plan) is real.",
            {"ok": True, "steps": 3},
        )

    # -------------------------------------------------------------- shared bits
    def _verify_and_review(self, mode: str, generated_tests: bool) -> Iterator[dict]:
        if generated_tests:
            gen = write_generated_tests(self.root)
            if gen["written"]:
                yield _event("tests", f"Generated {len(gen['written'])} test file(s)", gen)
        run = run_tests(self.root, timeout=300)
        yield _event(
            "tests",
            ("Tests pass" if run.ok else "Tests failed") + f" ({run.command})",
            run.to_dict(),
        )
        if not run.ok:
            human = humanize(run.failure_signal())
            yield _event("error", human.title, human.to_dict())

        review = CodeCritic().review(self.root)
        top = [finding.to_dict() for finding in review.top(4)]
        yield _event(
            "review",
            f"Review: {review.summary()}",
            {"score": review.score, "ok": review.ok, "findings": top, "counts": review.to_dict()["counts"]},
        )

    def _deploy_plan(self) -> Iterator[dict]:
        try:
            result = DeployEngine(self.root, dry_run=True).deploy()
            yield _event(
                "deploy_plan",
                f"Deploy plan ready: {result.plan.provider_label if result.plan else 'unknown'}",
                {
                    "provider": result.provider,
                    "steps": [step.to_dict() for step in (result.plan.steps if result.plan else [])],
                    "warnings": result.plan.warnings if result.plan else [],
                },
            )
        except Exception as exc:  # noqa: BLE001
            human = humanize(f"{type(exc).__name__}: {exc}")
            yield _event("error", human.title, human.to_dict())


def _event(event_type: str, message: str, data: dict | None = None) -> dict:
    return {"type": event_type, "message": message, "data": data or {}, "at": time.time()}


def _is_empty(root: Path) -> bool:
    if not root.exists():
        return True
    ignored = {".agent", ".git", ".venv", "venv", "node_modules", ".DS_Store"}
    return not any(path.name not in ignored for path in root.iterdir())
