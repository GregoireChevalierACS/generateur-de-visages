"""
Tests du stockage SQLite des visages et embeddings.
Utilise une DB temporaire isolée pour chaque test.
"""
import io
from pathlib import Path

import numpy as np
import pytest

import app.services.face_store as store_module
from app.services.face_store import (
    init_db,
    save_face,
    get_face,
    get_all_embeddings,
    list_faces,
    delete_face,
    count_faces,
    _serialize,
    _deserialize,
)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Redirige DB_PATH vers un fichier temporaire pour chaque test."""
    db = tmp_path / "test.db"
    monkeypatch.setattr(store_module, "DB_PATH", db)
    init_db()
    yield


def make_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.random(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


# --- Sérialisation ---

def test_serialize_deserialize_roundtrip():
    vec = make_embedding()
    assert np.allclose(vec, _deserialize(_serialize(vec)))


def test_serialize_produces_bytes():
    assert isinstance(_serialize(make_embedding()), bytes)


# --- CRUD de base ---

def test_save_and_get_face():
    save_face("face1", "img1", "/storage/faces/face1.png", "anime")
    result = get_face("face1")
    assert result is not None
    assert result["face_id"] == "face1"
    assert result["image_id"] == "img1"
    assert result["detector"] == "anime"


def test_save_face_with_embedding():
    emb = make_embedding()
    save_face("face2", "img1", "/storage/faces/face2.png", "frontal", emb)
    result = get_face("face2")
    assert result["embedding"] is not None
    assert np.allclose(result["embedding"], emb)


def test_get_unknown_face_returns_none():
    assert get_face("does_not_exist") is None


def test_save_face_without_embedding():
    save_face("face3", "img1", "/storage/faces/face3.png", "anime", None)
    result = get_face("face3")
    assert result["embedding"] is None


def test_delete_existing_face():
    save_face("face4", "img1", "/storage/faces/face4.png", "anime")
    assert delete_face("face4") is True
    assert get_face("face4") is None


def test_delete_nonexistent_face_returns_false():
    assert delete_face("ghost") is False


def test_count_faces():
    assert count_faces() == 0
    save_face("f1", "img1", "/x.png", "anime")
    save_face("f2", "img1", "/y.png", "frontal")
    assert count_faces() == 2


def test_list_faces_returns_all():
    save_face("f1", "img1", "/x.png", "anime")
    save_face("f2", "img2", "/y.png", "frontal")
    faces = list_faces()
    assert len(faces) == 2


def test_list_faces_does_not_include_embedding_blob():
    """list_faces ne doit pas exposer le BLOB binaire (trop lourd)."""
    save_face("f1", "img1", "/x.png", "anime", make_embedding())
    faces = list_faces()
    assert "embedding" not in faces[0]


# --- Embeddings ---

def test_get_all_embeddings_empty():
    assert get_all_embeddings() == []


def test_get_all_embeddings_returns_only_non_null():
    save_face("f1", "img1", "/x.png", "anime", make_embedding(1))
    save_face("f2", "img1", "/y.png", "anime", None)  # pas d'embedding
    save_face("f3", "img1", "/z.png", "anime", make_embedding(2))
    embs = get_all_embeddings()
    assert len(embs) == 2


def test_get_all_embeddings_correct_shape():
    for i in range(3):
        save_face(f"f{i}", "img1", f"/{i}.png", "anime", make_embedding(i))
    embs = get_all_embeddings()
    assert all(e.shape == (512,) for e in embs)


def test_upsert_replaces_existing_face():
    save_face("f1", "img1", "/x.png", "anime")
    save_face("f1", "img1", "/x.png", "anime", make_embedding())  # même face_id
    assert count_faces() == 1
    assert get_face("f1")["embedding"] is not None
