"""
Tests des endpoints /api/generate/*, /api/faces/*, et la persistance dans detect.

Le patch cible toujours `app.routers.generate.generator_service` (référence locale
du routeur) et non le module source, pour éviter que le vrai modèle soit chargé.
"""
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.routers.generate as gen_router
import app.services.face_store as store_module
from app.main import app
from app.services.face_store import init_db, save_face
from app.services.generator import IMAGE_SIZE

client = TestClient(app)
SAMPLES_DIR = Path(__file__).parent.parent.parent


def make_png(w=100, h=100, color=(200, 180, 160)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "test.db")
    init_db()
    yield


@pytest.fixture
def mock_generator(tmp_path, monkeypatch):
    """
    Remplace generator_service dans le routeur par un mock.
    Redirige aussi GENERATED_DIR vers tmp_path.
    """
    import app.services.generator as gen_module
    monkeypatch.setattr(gen_module, "GENERATED_DIR", tmp_path)

    fake_image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (180, 140, 110))
    fake_url = "/storage/generated/fakeid.png"
    fake_image.save(tmp_path / "fakeid.png")

    svc_mock = MagicMock()
    svc_mock.is_ready = True
    svc_mock.generate.return_value = (fake_image, fake_url)

    # Patch la référence locale du routeur
    monkeypatch.setattr(gen_router, "generator_service", svc_mock)

    return svc_mock, fake_url


@pytest.fixture
def faces_in_db_with_crops(tmp_path, monkeypatch):
    """
    Crée un visage en DB et son fichier crop dans tmp_path.
    Patche FACES_DIR dans le routeur (référence locale) pour que
    _load_face_images() trouve le fichier au bon endroit.
    """
    # Patch la référence locale dans le routeur, pas app.config
    monkeypatch.setattr(gen_router, "FACES_DIR", tmp_path)

    face_img = Image.new("RGB", (256, 256), (210, 170, 130))
    face_img.save(tmp_path / "face1.png")
    save_face("face1", "img1", "/storage/faces/face1.png", "anime")
    return tmp_path


# ---- Status + Init ----

def test_generator_status_returns_ready_field():
    resp = client.get("/api/generate/status")
    assert resp.status_code == 200
    assert "ready" in resp.json()


def test_generator_status_has_loading_and_error_fields():
    resp = client.get("/api/generate/status")
    data = resp.json()
    assert "loading" in data
    assert "error" in data


def test_generator_status_initially_false():
    resp = client.get("/api/generate/status")
    data = resp.json()
    assert data["ready"] is False
    assert data["loading"] is False
    assert data["error"] is None


def test_generate_returns_503_when_model_not_ready():
    save_face("f1", "img1", "/storage/faces/f1.png", "anime")
    resp = client.post("/api/generate", json={})
    assert resp.status_code == 503


def test_init_endpoint_starts_background_load(monkeypatch):
    svc_mock = MagicMock()
    svc_mock.is_ready = False
    svc_mock.is_loading = False
    monkeypatch.setattr(gen_router, "generator_service", svc_mock)
    resp = client.post("/api/generate/init")
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"
    svc_mock.load_in_background.assert_called_once()


def test_init_endpoint_skips_if_already_loaded(monkeypatch):
    svc_mock = MagicMock()
    svc_mock.is_ready = True
    svc_mock.is_loading = False
    monkeypatch.setattr(gen_router, "generator_service", svc_mock)
    resp = client.post("/api/generate/init")
    assert resp.json()["status"] == "already_loaded"
    svc_mock.load_in_background.assert_not_called()


def test_init_endpoint_skips_if_already_loading(monkeypatch):
    svc_mock = MagicMock()
    svc_mock.is_ready = False
    svc_mock.is_loading = True
    monkeypatch.setattr(gen_router, "generator_service", svc_mock)
    resp = client.post("/api/generate/init")
    assert resp.json()["status"] == "loading"
    svc_mock.load_in_background.assert_not_called()


# ---- Génération mockée ----

def test_generate_returns_422_with_no_faces(mock_generator):
    resp = client.post("/api/generate", json={})
    assert resp.status_code == 422


def test_generate_returns_200_with_faces(mock_generator, faces_in_db_with_crops):
    resp = client.post("/api/generate", json={})
    assert resp.status_code == 200


def test_generate_response_has_url(mock_generator, faces_in_db_with_crops):
    data = client.post("/api/generate", json={}).json()
    assert "url" in data
    assert data["url"].startswith("/storage/generated/")


def test_generate_response_model_ready_true(mock_generator, faces_in_db_with_crops):
    data = client.post("/api/generate", json={}).json()
    assert data["model_ready"] is True


def test_generate_response_has_reference_faces_used(mock_generator, faces_in_db_with_crops):
    data = client.post("/api/generate", json={}).json()
    assert data["reference_faces_used"] >= 1


def test_generate_with_seed_passes_to_service(mock_generator, faces_in_db_with_crops):
    svc_mock, _ = mock_generator
    client.post("/api/generate", json={"seed": 42})
    call_kwargs = svc_mock.generate.call_args.kwargs
    assert call_kwargs["seed"] == 42


def test_generate_with_custom_ip_scale(mock_generator, faces_in_db_with_crops):
    svc_mock, _ = mock_generator
    client.post("/api/generate", json={"ip_scale": 0.3})
    call_kwargs = svc_mock.generate.call_args.kwargs
    assert call_kwargs["ip_scale"] == pytest.approx(0.3)


def test_generate_rejects_ip_scale_above_1():
    resp = client.post("/api/generate", json={"ip_scale": 1.5})
    assert resp.status_code == 422


def test_generate_rejects_ip_scale_below_0():
    resp = client.post("/api/generate", json={"ip_scale": -0.1})
    assert resp.status_code == 422


def test_generate_with_prompt_extra(mock_generator, faces_in_db_with_crops):
    svc_mock, _ = mock_generator
    client.post("/api/generate", json={"prompt_extra": "vibrant colors"})
    call_kwargs = svc_mock.generate.call_args.kwargs
    assert "vibrant colors" in call_kwargs["prompt"]


def test_generate_rejects_too_many_steps():
    resp = client.post("/api/generate", json={"steps": 100})
    assert resp.status_code == 422


# ---- Endpoint faces ----

def test_get_faces_empty():
    resp = client.get("/api/faces")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["faces"] == []


def test_get_faces_after_save():
    save_face("f1", "img1", "/storage/faces/f1.png", "anime")
    save_face("f2", "img1", "/storage/faces/f2.png", "frontal")
    data = client.get("/api/faces").json()
    assert data["total"] == 2


def test_get_face_count():
    save_face("f1", "img1", "/x.png", "anime")
    resp = client.get("/api/faces/count")
    assert resp.json()["count"] == 1


def test_face_entry_has_required_fields():
    save_face("f1", "img1", "/storage/faces/f1.png", "anime")
    faces = client.get("/api/faces").json()["faces"]
    face = faces[0]
    for field in ("face_id", "image_id", "crop_url", "detector", "created_at"):
        assert field in face


def test_delete_face_returns_200():
    save_face("f1", "img1", "/x.png", "anime")
    resp = client.delete("/api/faces/f1")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == "f1"


def test_delete_face_removes_from_db():
    save_face("f1", "img1", "/x.png", "anime")
    client.delete("/api/faces/f1")
    assert client.get("/api/faces").json()["total"] == 0


def test_delete_unknown_face_returns_404():
    resp = client.delete("/api/faces/doesnotexist")
    assert resp.status_code == 404


def test_clear_all_faces():
    for i in range(3):
        save_face(f"f{i}", "img1", f"/{i}.png", "anime")
    resp = client.delete("/api/faces")
    assert resp.json()["deleted_count"] == 3
    assert client.get("/api/faces").json()["total"] == 0
