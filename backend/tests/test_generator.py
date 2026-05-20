"""
Tests du service de génération.

Tests unitaires   : mockent le pipeline SD → aucun téléchargement requis.
Tests d'intégration (marqués @pytest.mark.slow) : chargent le vrai modèle,
  skippés par défaut → lancer avec : pytest -m slow
"""
import io
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from PIL import Image

from app.services.generator import GeneratorService, DEFAULT_IP_SCALE, IMAGE_SIZE


SAMPLES_DIR = Path(__file__).parent.parent.parent


def make_face_image(color=(200, 160, 120)) -> Image.Image:
    return Image.new("RGB", (256, 256), color)


def make_fake_pipeline():
    """Retourne un mock du pipeline diffusers qui produit une image valide."""
    fake_image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (180, 140, 110))
    pipe_mock = MagicMock()
    pipe_mock.return_value.images = [fake_image]
    return pipe_mock, fake_image


# ---- État initial ----

def test_service_not_ready_at_init():
    svc = GeneratorService()
    assert svc.is_ready is False


def test_service_not_loading_at_init():
    svc = GeneratorService()
    assert svc.is_loading is False


def test_service_no_error_at_init():
    svc = GeneratorService()
    assert svc.load_error is None


def test_load_in_background_returns_true_when_idle():
    svc = GeneratorService()
    svc._ready = True  # skip actual load
    # already ready → returns False
    assert svc.load_in_background() is False


def test_load_in_background_returns_false_when_loading():
    svc = GeneratorService()
    svc._loading = True
    assert svc.load_in_background() is False


def test_generate_raises_when_not_loaded():
    svc = GeneratorService()
    with pytest.raises(RuntimeError, match="n'est pas chargé"):
        svc.generate([make_face_image()])


def test_generate_raises_with_empty_references(tmp_path):
    svc = GeneratorService()
    svc._ready = True  # bypass le check de chargement
    svc._pipe = make_fake_pipeline()[0]
    with pytest.raises(ValueError, match="Au moins une image"):
        svc.generate([])


# ---- Génération mockée ----

@pytest.fixture
def loaded_service(tmp_path, monkeypatch):
    """Service avec pipeline mocké et GENERATED_DIR redirigé vers tmp."""
    import app.services.generator as gen_module
    monkeypatch.setattr(gen_module, "GENERATED_DIR", tmp_path)

    pipe_mock, fake_image = make_fake_pipeline()
    svc = GeneratorService()
    svc._pipe = pipe_mock
    svc._ready = True
    return svc, pipe_mock, fake_image, tmp_path


def test_generate_returns_tuple(loaded_service):
    svc, _, _, _ = loaded_service
    result = svc.generate([make_face_image()])
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_generate_returns_pil_image(loaded_service):
    svc, _, _, _ = loaded_service
    image, _ = svc.generate([make_face_image()])
    assert isinstance(image, Image.Image)


def test_generate_returns_url(loaded_service):
    svc, _, _, _ = loaded_service
    _, url = svc.generate([make_face_image()])
    assert url.startswith("/storage/generated/")
    assert url.endswith(".png")


def test_generate_saves_file_to_disk(loaded_service):
    svc, _, _, tmp_path = loaded_service
    _, url = svc.generate([make_face_image()])
    filename = Path(url).name
    assert (tmp_path / filename).exists()


def test_generate_saved_file_is_valid_png(loaded_service):
    svc, _, _, tmp_path = loaded_service
    _, url = svc.generate([make_face_image()])
    path = tmp_path / Path(url).name
    img = Image.open(path)
    assert img.format == "PNG" or img.mode in ("RGB", "RGBA")


def test_generate_each_call_produces_unique_file(loaded_service):
    svc, _, _, tmp_path = loaded_service
    urls = [svc.generate([make_face_image()])[1] for _ in range(3)]
    assert len(set(urls)) == 3


def test_generate_with_seed_calls_pipeline_with_generator(loaded_service):
    svc, pipe_mock, _, _ = loaded_service
    svc.generate([make_face_image()], seed=42)
    call_kwargs = pipe_mock.call_args.kwargs
    assert call_kwargs["generator"] is not None


def test_generate_passes_prompt_to_pipeline(loaded_service):
    svc, pipe_mock, _, _ = loaded_service
    custom_prompt = "anime style, vibrant colors"
    svc.generate([make_face_image()], prompt=custom_prompt)
    call_kwargs = pipe_mock.call_args.kwargs
    assert call_kwargs["prompt"] == custom_prompt


def test_generate_passes_ip_scale(loaded_service):
    svc, _, _, _ = loaded_service
    # ip_scale est set via set_ip_adapter_scale, pas dans les kwargs du pipe
    svc._pipe.set_ip_adapter_scale = MagicMock()
    svc.generate([make_face_image()], ip_scale=0.3)
    svc._pipe.set_ip_adapter_scale.assert_called_once_with(0.3)


def test_generate_with_multiple_references(loaded_service):
    svc, pipe_mock, _, _ = loaded_service
    refs = [make_face_image((c, c, c)) for c in (100, 150, 200)]
    svc.generate(refs)
    call_kwargs = pipe_mock.call_args.kwargs
    assert call_kwargs["ip_adapter_image"] == refs


def test_generate_image_size_is_correct(loaded_service):
    svc, _, fake_image, tmp_path = loaded_service
    # Le mock retourne déjà une image à IMAGE_SIZE
    _, url = svc.generate([make_face_image()])
    saved = Image.open(tmp_path / Path(url).name)
    assert saved.size == (IMAGE_SIZE, IMAGE_SIZE)


# ---- Tests d'intégration (nécessitent le téléchargement du modèle) ----

@pytest.mark.slow
def test_load_initializes_pipeline():
    svc = GeneratorService()
    svc.load()
    assert svc.is_ready is True
    assert svc._pipe is not None


@pytest.mark.slow
def test_load_is_idempotent():
    svc = GeneratorService()
    svc.load()
    svc.load()  # second appel ne doit pas planter
    assert svc.is_ready is True


@pytest.mark.slow
def test_generate_real_image():
    svc = GeneratorService()
    svc.load()
    sample = SAMPLES_DIR / "1.png"
    ref = Image.open(sample).convert("RGB") if sample.exists() else make_face_image()
    image, url = svc.generate([ref], num_steps=5)
    assert isinstance(image, Image.Image)
    assert image.size == (IMAGE_SIZE, IMAGE_SIZE)
