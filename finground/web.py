"""User-facing web application backed by the standard ADK HTTP API."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi.staticfiles import StaticFiles
from google.adk.cli.fast_api import get_fast_api_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = PROJECT_ROOT / "web"


def create_web_app():
    """Create the ADK API server and mount the product UI on its root."""
    app = get_fast_api_app(
        agents_dir=str(PROJECT_ROOT),
        web=False,
        use_local_storage=True,
    )

    app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="finground-ui")
    return app


app = create_web_app()


def main() -> None:
    """Run the local product UI."""
    port = int(os.getenv("FINGROUND_WEB_PORT", "8000"))
    uvicorn.run("finground.web:app", host="127.0.0.1", port=port, reload=True)
