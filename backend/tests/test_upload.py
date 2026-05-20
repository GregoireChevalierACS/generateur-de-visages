import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)

SAMPLE_IMAGES_DIR = Path(__file__).parent.parent.parent  # racine du projet


def make_png_bytes(width=100, height=100, color=(255, 0, 0)) -> bytes:
    """Crée un PNG valide en mémoire."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def make_jpeg_bytes(width=100, height=100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (0, 255, 0)).save(buf, format="JPEG")
    return buf.getvalue()


# --- Cas nominal ---

def test_upload_png_returns_200():
    response = client.post(
        "/api/upload",
        files={"file": ("test.png", make_png_bytes(), "image/png")},
    )
    assert response.status_code == 200


def test_upload_returns_id_and_url():
    response = client.post(
        "/api/upload",
        files={"file": ("test.png", make_png_bytes(), "image/png")},
    )
    data = response.json()
    assert "id" in data
    assert data["url"].startswith("/storage/uploads/")
    assert data["url"].endswith(".png")


def test_upload_returns_dimensions():
    response = client.post(
        "/api/upload",
        files={"file": ("test.png", make_png_bytes(200, 150), "image/png")},
    )
    data = response.json()
    assert data["width"] == 200
    assert data["height"] == 150


def test_upload_jpeg_returns_200():
    response = client.post(
        "/api/upload",
        files={"file": ("photo.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200


def test_upload_file_is_saved_on_disk():
    response = client.post(
        "/api/upload",
        files={"file": ("test.png", make_png_bytes(), "image/png")},
    )
    data = response.json()
    filename = Path(data["url"]).name
    from app.config import UPLOADS_DIR
    assert (UPLOADS_DIR / filename).exists()


def test_upload_with_real_sample_image():
    """Test avec une vraie image du projet (1.png)."""
    sample = SAMPLE_IMAGES_DIR / "1.png"
    if not sample.exists():
        pytest.skip("Image 1.png non trouvée")
    with open(sample, "rb") as f:
        response = client.post(
            "/api/upload",
            files={"file": ("1.png", f, "image/png")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["width"] > 0
    assert data["height"] > 0


# --- Cas d'erreur ---

def test_upload_rejects_unsupported_extension():
    response = client.post(
        "/api/upload",
        files={"file": ("doc.pdf", b"fake pdf content", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Extension" in response.json()["detail"]


def test_upload_rejects_invalid_image_content():
    response = client.post(
        "/api/upload",
        files={"file": ("fake.png", b"this is not an image", "image/png")},
    )
    assert response.status_code == 400


def test_upload_each_image_gets_unique_id():
    ids = set()
    for _ in range(3):
        response = client.post(
            "/api/upload",
            files={"file": ("test.png", make_png_bytes(), "image/png")},
        )
        ids.add(response.json()["id"])
    assert len(ids) == 3
