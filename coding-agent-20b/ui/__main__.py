"""``python -m ui [--workspace DIR] [--port 7860]``."""

from __future__ import annotations

import argparse
import os


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ui", description="Chat UI for the coding agent")
    parser.add_argument("--workspace", default=os.environ.get("AGENT_WORKSPACE", "."), help="project directory")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (keep 0.0.0.0 for remote previews)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGENT_UI_PORT", "7860")))
    parser.add_argument("--model-url", default=None, help="OpenAI-compatible endpoint (optional)")
    args = parser.parse_args(argv)

    from .app import serve

    serve(args.workspace, host=args.host, port=args.port, model_url=args.model_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
