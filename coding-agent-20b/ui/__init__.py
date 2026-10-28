"""Chat UI: a clean surface over the agent (chat, starters, preview, deploy).

Importing this package is cheap; FastAPI is only imported when you actually
build the app::

    from ui import create_app
    app = create_app("./my-app")
"""

from __future__ import annotations

from .preview import PreviewManager, list_workspace_files, read_workspace_file
from .runtime import AgentRuntime, http_generate_fn

__all__ = [
    "AgentRuntime",
    "PreviewManager",
    "create_app",
    "http_generate_fn",
    "list_workspace_files",
    "read_workspace_file",
]


def create_app(workspace: str = ".", model_url: str | None = None):
    """Lazy import so `import ui` works without FastAPI installed."""

    from .app import create_app as _create_app

    return _create_app(workspace, model_url=model_url)
