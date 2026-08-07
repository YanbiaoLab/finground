from fastapi.testclient import TestClient

from finground.web import app


def test_product_ui_assets_are_not_cached() -> None:
    with TestClient(app) as client:
        for resource in ("/", "/app.js", "/styles.css"):
            response = client.head(resource)

            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"
