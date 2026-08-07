from fastapi.testclient import TestClient

from finground.web import app


def test_product_ui_assets_are_not_cached() -> None:
    with TestClient(app) as client:
        for resource in ("/", "/app.js", "/styles.css"):
            response = client.head(resource)

            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"


def test_product_ui_uses_versioned_assets() -> None:
    with TestClient(app) as client:
        response = client.get("/")

        assert 'href="/styles.css?v=20260807-1"' in response.text
        assert 'src="/app.js?v=20260807-1"' in response.text
