"""
Service de détection de visages pour images illustrées/anime.

Cascade de détecteurs (du plus spécialisé au plus générique) :
  1. LBP Anime cascade (nagadomi) — optimal pour anime / manga
  2. Haar frontal face (OpenCV) — fonctionne sur illustrations semi-réalistes
  3. Haar profile face (OpenCV) — visages de profil

Chaque détecteur tente de trouver des visages ; les résultats sont fusionnés
avec suppression des boîtes qui se chevauchent trop (NMS simple).
"""
import io
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from app.config import FACES_DIR

ANIME_CASCADE_URL = (
    "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface"
    "/master/lbpcascade_animeface.xml"
)
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
ANIME_CASCADE_PATH = MODELS_DIR / "lbpcascade_animeface.xml"

# Marge ajoutée autour du crop (proportion de la taille de la boîte)
CROP_PADDING = 0.15
# Taille normalisée des crops sauvegardés
CROP_SIZE = (256, 256)
# Seuil de chevauchement pour la suppression des doublons
IOU_THRESHOLD = 0.3


@dataclass
class DetectedFace:
    face_id: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    detector: str       # "anime", "frontal", "profile"
    crop_url: str       # URL relative du crop sauvegardé
    crop_path: Path


def _download_anime_cascade() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(ANIME_CASCADE_URL, ANIME_CASCADE_PATH)


def _iou(a: tuple, b: tuple) -> float:
    """Intersection over Union de deux boîtes (x, y, w, h)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    iw = max(0, min(ax + aw, bx + bw) - ix)
    ih = max(0, min(ay + ah, by + bh) - iy)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: list[tuple]) -> list[tuple]:
    """Supprime les boîtes redondantes (IoU > seuil)."""
    kept = []
    for box in boxes:
        dominated = False
        for ref in kept:
            if _iou(box[:4], ref[:4]) > IOU_THRESHOLD:
                dominated = True
                break
        if not dominated:
            kept.append(box)
    return kept


def _crop_and_save(img_rgb: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[str, Path]:
    """Crop avec padding, resize à CROP_SIZE, sauvegarde dans FACES_DIR."""
    ih, iw = img_rgb.shape[:2]
    pad_x = int(w * CROP_PADDING)
    pad_y = int(h * CROP_PADDING)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(iw, x + w + pad_x)
    y2 = min(ih, y + h + pad_y)

    crop = img_rgb[y1:y2, x1:x2]
    pil_crop = Image.fromarray(crop).resize(CROP_SIZE, Image.LANCZOS)

    face_id = uuid.uuid4().hex
    filename = f"{face_id}.png"
    dest = FACES_DIR / filename
    pil_crop.save(dest)

    return f"/storage/faces/{filename}", dest


class FaceDetector:
    def __init__(self):
        self._anime_cascade: Optional[cv2.CascadeClassifier] = None
        self._frontal_cascade: Optional[cv2.CascadeClassifier] = None
        self._profile_cascade: Optional[cv2.CascadeClassifier] = None

    def _load_cascades(self) -> None:
        if not ANIME_CASCADE_PATH.exists():
            _download_anime_cascade()
        self._anime_cascade = cv2.CascadeClassifier(str(ANIME_CASCADE_PATH))
        cv_data = cv2.data.haarcascades
        self._frontal_cascade = cv2.CascadeClassifier(cv_data + "haarcascade_frontalface_default.xml")
        self._profile_cascade = cv2.CascadeClassifier(cv_data + "haarcascade_profileface.xml")

    def _ensure_loaded(self) -> None:
        if self._anime_cascade is None:
            self._load_cascades()

    def detect(self, image_source: Path | bytes) -> list[DetectedFace]:
        """
        Détecte les visages dans une image.

        Args:
            image_source: chemin vers le fichier image, ou bytes bruts.

        Returns:
            Liste de DetectedFace, triée par surface décroissante.
        """
        self._ensure_loaded()

        if isinstance(image_source, (Path, str)):
            pil_img = Image.open(image_source).convert("RGB")
        else:
            pil_img = Image.open(io.BytesIO(image_source)).convert("RGB")

        img_rgb = np.array(pil_img)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        raw_boxes: list[tuple] = []  # (x, y, w, h, confidence, detector)

        # 1. Anime cascade
        anime_faces = self._anime_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )
        if len(anime_faces):
            for x, y, w, h in anime_faces:
                raw_boxes.append((int(x), int(y), int(w), int(h), 0.9, "anime"))

        # 2. Haar frontal
        frontal_faces = self._frontal_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(frontal_faces):
            for x, y, w, h in frontal_faces:
                raw_boxes.append((int(x), int(y), int(w), int(h), 0.8, "frontal"))

        # 3. Haar profile
        profile_faces = self._profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(profile_faces):
            for x, y, w, h in profile_faces:
                raw_boxes.append((int(x), int(y), int(w), int(h), 0.7, "profile"))

        # NMS + tri par surface
        deduped = _nms(raw_boxes)
        deduped.sort(key=lambda b: b[2] * b[3], reverse=True)

        results: list[DetectedFace] = []
        for x, y, w, h, conf, detector in deduped:
            crop_url, crop_path = _crop_and_save(img_rgb, x, y, w, h)
            face_id = crop_path.stem
            results.append(DetectedFace(
                face_id=face_id,
                x=x, y=y, width=w, height=h,
                confidence=conf,
                detector=detector,
                crop_url=crop_url,
                crop_path=crop_path,
            ))

        return results


# Singleton partagé entre les requêtes
face_detector = FaceDetector()
