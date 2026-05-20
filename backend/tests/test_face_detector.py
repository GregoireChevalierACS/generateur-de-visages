"""
Tests du service de détection de visages.
Teste la logique interne indépendamment de l'API HTTP.
"""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.services.face_detector import (
    FaceDetector,
    DetectedFace,
    _iou,
    _nms,
    _crop_and_save,
    CROP_SIZE,
)
from app.config import FACES_DIR

SAMPLES_DIR = Path(__file__).parent.parent.parent  # racine projet


# ---- Helpers ----------------------------------------------------------------

def make_blank_png(width=200, height=200) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 200, 200)).save(buf, format="PNG")
    return buf.getvalue()


def make_face_like_png() -> bytes:
    """Image avec un ovale chair-coloré au centre (ne trompe pas vraiment le
    détecteur, mais sert à tester le pipeline sans visage réel)."""
    img = Image.new("RGB", (300, 300), (180, 200, 220))
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 80, 200, 200], fill=(210, 170, 130))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---- Tests utilitaires ------------------------------------------------------

class TestIOU:
    def test_same_box_is_1(self):
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_no_overlap_is_0(self):
        assert _iou((0, 0, 5, 5), (10, 10, 5, 5)) == pytest.approx(0.0)

    def test_half_overlap(self):
        # Box B décalée de 5 en x → chevauchement de 5×10 sur union de 150
        val = _iou((0, 0, 10, 10), (5, 0, 10, 10))
        assert 0.2 < val < 0.5


class TestNMS:
    def test_keeps_unique_boxes(self):
        boxes = [(0, 0, 10, 10, 0.9, "a"), (50, 50, 10, 10, 0.8, "b")]
        assert len(_nms(boxes)) == 2

    def test_removes_duplicate(self):
        boxes = [
            (0, 0, 10, 10, 0.9, "a"),
            (1, 1, 10, 10, 0.8, "b"),  # très proche → supprimé
        ]
        assert len(_nms(boxes)) == 1

    def test_keeps_first_of_duplicates(self):
        boxes = [
            (0, 0, 10, 10, 0.9, "anime"),
            (1, 1, 10, 10, 0.8, "frontal"),
        ]
        kept = _nms(boxes)
        assert kept[0][4] == 0.9  # gardé celui avec la plus haute confiance


class TestCropAndSave:
    def test_saved_file_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.services.face_detector.FACES_DIR", tmp_path)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        url, path = _crop_and_save(img, 50, 50, 100, 100)
        assert path.exists()

    def test_crop_is_correct_size(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.services.face_detector.FACES_DIR", tmp_path)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        _, path = _crop_and_save(img, 50, 50, 100, 100)
        result = Image.open(path)
        assert result.size == CROP_SIZE

    def test_url_starts_with_storage(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.services.face_detector.FACES_DIR", tmp_path)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        url, _ = _crop_and_save(img, 10, 10, 50, 50)
        assert url.startswith("/storage/faces/")

    def test_crop_with_padding_clamps_to_image_bounds(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.services.face_detector.FACES_DIR", tmp_path)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Boîte au bord → le padding ne doit pas sortir de l'image
        url, path = _crop_and_save(img, 0, 0, 50, 50)
        assert path.exists()


# ---- Tests du détecteur (pipeline complet) ----------------------------------

class TestFaceDetector:
    def test_instantiation(self):
        detector = FaceDetector()
        assert detector is not None

    def test_detect_returns_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.services.face_detector.FACES_DIR", tmp_path)
        detector = FaceDetector()
        result = detector.detect(make_blank_png())
        assert isinstance(result, list)

    def test_detect_no_face_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.services.face_detector.FACES_DIR", tmp_path)
        detector = FaceDetector()
        result = detector.detect(make_blank_png())
        assert result == []

    def test_detect_result_is_detected_face(self, tmp_path, monkeypatch):
        """Si un visage est trouvé, chaque élément est un DetectedFace."""
        monkeypatch.setattr("app.services.face_detector.FACES_DIR", tmp_path)
        detector = FaceDetector()
        results = detector.detect(make_face_like_png())
        for face in results:
            assert isinstance(face, DetectedFace)

    def test_detect_face_fields_are_valid(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.services.face_detector.FACES_DIR", tmp_path)
        detector = FaceDetector()
        results = detector.detect(make_face_like_png())
        for face in results:
            assert face.width > 0
            assert face.height > 0
            assert 0.0 <= face.confidence <= 1.0
            assert face.detector in ("anime", "frontal", "profile")
            assert face.crop_url.startswith("/storage/faces/")

    def test_detect_from_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.services.face_detector.FACES_DIR", tmp_path)
        sample = SAMPLES_DIR / "1.png"
        if not sample.exists():
            pytest.skip("1.png non trouvé")
        detector = FaceDetector()
        results = detector.detect(sample)
        assert isinstance(results, list)

    @pytest.mark.parametrize("img_file", ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png"])
    def test_detect_on_all_samples(self, img_file, tmp_path, monkeypatch):
        """Teste que le pipeline tourne sans erreur sur toutes les images du projet."""
        monkeypatch.setattr("app.services.face_detector.FACES_DIR", tmp_path)
        sample = SAMPLES_DIR / img_file
        if not sample.exists():
            pytest.skip(f"{img_file} non trouvé")
        detector = FaceDetector()
        results = detector.detect(sample)
        # On n'exige pas de détection (le détecteur peut rater sur certains styles)
        # mais le pipeline ne doit pas crasher
        assert isinstance(results, list)
        print(f"\n  {img_file}: {len(results)} visage(s) détecté(s)")
