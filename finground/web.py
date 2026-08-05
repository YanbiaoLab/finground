"""User-facing web application backed by the standard ADK HTTP API."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import uvicorn
from fastapi.staticfiles import StaticFiles
from google.adk.cli.fast_api import get_fast_api_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = PROJECT_ROOT / "web"
RAW_PDF_ROOT = WEB_ROOT / "raw-pdf"


def create_web_app():
    """Create the ADK API server and mount the product UI on its root."""
    app = get_fast_api_app(
        agents_dir=str(PROJECT_ROOT),
        web=False,
        use_local_storage=True,
    )

    @app.get("/_finground/sample-reports")
    async def sample_reports() -> list[dict[str, str | int]]:
        return [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "url": f"/raw-pdf/{quote(path.name, safe='')}",
            }
            for path in sorted(RAW_PDF_ROOT.glob("*.pdf"), key=lambda item: item.name.casefold())
            if path.is_file()
        ]

    app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="finground-ui")
    return app


app = create_web_app()


def main() -> None:
    """Run the local product UI."""
    port = int(os.getenv("FINGROUND_WEB_PORT", "8000"))
    uvicorn.run("finground.web:app", host="127.0.0.1", port=port, reload=True)
