"""
Tests de l'endpoint POST /api/detect/{image_id}.
On upload d'abord une image, puis on appelle detect.
"""
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)
SAMPLES_DIR = Path(__file__).parent.parent.parent


def upload_image(png_bytes: bytes, filename: str = "test.png") -> str:
    """Helper : upload une image et retourne son image_id."""
    resp = client.post(
        "/api/upload",
        files={"file": (filename, png_bytes, "image/png")},
    )
    assert resp.status_code == 200
    return resp.json()["id"]


def make_png(width=200, height=200, color=(180, 180, 180)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


# --- Cas nominaux ---

def test_detect_returns_200_on_valid_image_id():
    image_id = upload_image(make_png())
    resp = client.post(f"/api/detect/{image_id}")
    assert resp.status_code == 200


def test_detect_response_has_required_fields():
    image_id = upload_image(make_png())
    data = client.post(f"/api/detect/{image_id}").json()
    assert "image_id" in data
    assert "face_count" in data
    assert "faces" in data


def test_detect_image_id_matches_request():
    image_id = upload_image(make_png())
    data = client.post(f"/api/detect/{image_id}").json()
    assert data["image_id"] == image_id


def test_detect_blank_image_returns_zero_faces():
    image_id = upload_image(make_png())
    data = client.post(f"/api/detect/{image_id}").json()
    assert data["face_count"] == 0
    assert data["faces"] == []


def test_detect_face_fields_structure():
    """Si des visages sont détectés, vérifie la structure de chaque objet."""
    sample = SAMPLES_DIR / "1.png"
    if not sample.exists():
        pytest.skip("1.png non trouvé")
    with open(sample, "rb") as f:
        image_id = upload_image(f.read(), "1.png")
    data = client.post(f"/api/detect/{image_id}").json()
    for face in data["faces"]:
        assert "face_id" in face
        assert "x" in face and "y" in face
        assert "width" in face and "height" in face
        assert "confidence" in face
        assert "detector" in face
        assert "crop_url" in face
        assert face["width"] > 0
        assert face["height"] > 0
        assert 0.0 <= face["confidence"] <= 1.0


def test_detect_crop_url_format():
    sample = SAMPLES_DIR / "1.png"
    if not sample.exists():
        pytest.skip("1.png non trouvé")
    with open(sample, "rb") as f:
        image_id = upload_image(f.read(), "1.png")
    data = client.post(f"/api/detect/{image_id}").json()
    for face in data["faces"]:
        assert face["crop_url"].startswith("/storage/faces/")
        assert face["crop_url"].endswith(".png")


def test_detect_face_count_matches_faces_list():
    sample = SAMPLES_DIR / "1.png"
    if not sample.exists():
        pytest.skip("1.png non trouvé")
    with open(sample, "rb") as f:
        image_id = upload_image(f.read(), "1.png")
    data = client.post(f"/api/detect/{image_id}").json()
    assert data["face_count"] == len(data["faces"])


def test_detect_is_idempotent():
    """Appeler detect deux fois sur la même image donne le même nombre de visages."""
    sample = SAMPLES_DIR / "1.png"
    if not sample.exists():
        pytest.skip("1.png non trouvé")
    with open(sample, "rb") as f:
        content = f.read()
    id1 = upload_image(content, "1.png")
    id2 = upload_image(content, "1.png")
    count1 = client.post(f"/api/detect/{id1}").json()["face_count"]
    count2 = client.post(f"/api/detect/{id2}").json()["face_count"]
    assert count1 == count2


# --- Cas d'erreur ---

def test_detect_unknown_image_id_returns_404():
    resp = client.post("/api/detect/nonexistent_image_id_xyz")
    assert resp.status_code == 404


def test_detect_404_has_detail_message():
    resp = client.post("/api/detect/nonexistent_image_id_xyz")
    assert "detail" in resp.json()
