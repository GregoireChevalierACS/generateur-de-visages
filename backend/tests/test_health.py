from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status():
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"


def test_health_returns_version():
    response = client.get("/health")
    data = response.json()
    assert "version" in data


def test_health_content_type_is_json():
    response = client.get("/health")
    assert "application/json" in response.headers["content-type"]
