"""
Tests du service d'embedding CLIP.
Le modèle est chargé une seule fois par session de test (fixture scope=session).
"""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.services.embedder import Embedder, EMBEDDING_DIM

SAMPLES_DIR = Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def embedder():
    e = Embedder()
    e._ensure_loaded()
    return e


def make_pil(color=(200, 100, 50)) -> Image.Image:
    return Image.new("RGB", (64, 64), color)


def make_png_bytes(color=(200, 100, 50)) -> bytes:
    buf = io.BytesIO()
    make_pil(color).save(buf, format="PNG")
    return buf.getvalue()


# --- Shape et type ---

def test_embed_returns_numpy_array(embedder):
    result = embedder.embed(make_pil())
    assert isinstance(result, np.ndarray)


def test_embed_correct_shape(embedder):
    result = embedder.embed(make_pil())
    assert result.shape == (EMBEDDING_DIM,)


def test_embed_dtype_is_float32(embedder):
    result = embedder.embed(make_pil())
    assert result.dtype == np.float32


def test_embed_is_unit_normalized(embedder):
    result = embedder.embed(make_pil())
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)


# --- Sources d'entrée ---

def test_embed_from_pil(embedder):
    result = embedder.embed(make_pil())
    assert result.shape == (EMBEDDING_DIM,)


def test_embed_from_bytes(embedder):
    result = embedder.embed(make_png_bytes())
    assert result.shape == (EMBEDDING_DIM,)


def test_embed_from_path(embedder):
    sample = SAMPLES_DIR / "1.png"
    if not sample.exists():
        pytest.skip("1.png non trouvé")
    result = embedder.embed(sample)
    assert result.shape == (EMBEDDING_DIM,)


# --- Cohérence ---

def test_same_image_same_embedding(embedder):
    img = make_pil((123, 45, 67))
    e1 = embedder.embed(img)
    e2 = embedder.embed(img)
    assert np.allclose(e1, e2)


def test_different_images_different_embeddings(embedder):
    e1 = embedder.embed(make_pil((255, 0, 0)))
    e2 = embedder.embed(make_pil((0, 255, 0)))
    assert not np.allclose(e1, e2)


# --- Batch ---

def test_embed_batch_shape(embedder):
    images = [make_pil((i * 40, 0, 0)) for i in range(4)]
    result = embedder.embed_batch(images)
    assert result.shape == (4, EMBEDDING_DIM)


def test_embed_batch_matches_individual(embedder):
    images = [make_pil((100, 50, 200)), make_pil((20, 80, 150))]
    batch = embedder.embed_batch(images)
    for i, img in enumerate(images):
        assert np.allclose(batch[i], embedder.embed(img))


# --- Moyenne ---

def test_average_embedding_shape(embedder):
    embs = [embedder.embed(make_pil((i * 30, i * 20, 100))) for i in range(3)]
    avg = Embedder.average_embedding(embs)
    assert avg.shape == (EMBEDDING_DIM,)


def test_average_embedding_is_normalized(embedder):
    embs = [embedder.embed(make_pil()) for _ in range(3)]
    avg = Embedder.average_embedding(embs)
    assert np.linalg.norm(avg) == pytest.approx(1.0, abs=1e-5)


def test_average_of_one_equals_itself(embedder):
    emb = embedder.embed(make_pil())
    avg = Embedder.average_embedding([emb])
    assert np.allclose(avg, emb)


def test_average_embedding_raises_on_empty():
    with pytest.raises(ValueError):
        Embedder.average_embedding([])


# --- Images réelles ---

@pytest.mark.parametrize("img_file", ["1.png", "4.png"])
def test_embed_real_samples(embedder, img_file):
    sample = SAMPLES_DIR / img_file
    if not sample.exists():
        pytest.skip(f"{img_file} non trouvé")
    result = embedder.embed(sample)
    assert result.shape == (EMBEDDING_DIM,)
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)
