"""
Extraction d'embeddings visuels via CLIP (ViT-B/32).

Chaque visage détecté est encodé en vecteur de 512 dimensions.
Ces vecteurs sont stockés en base pour conditionner la génération.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512


class Embedder:
    def __init__(self):
        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None

    def _load(self) -> None:
        self._processor = CLIPProcessor.from_pretrained(MODEL_ID)
        self._model = CLIPModel.from_pretrained(MODEL_ID)
        self._model.eval()

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._load()

    def embed(self, image_source: Path | bytes | Image.Image) -> np.ndarray:
        """
        Encode une image en vecteur CLIP normalisé (L2).

        Returns:
            np.ndarray de shape (512,), dtype float32.
        """
        self._ensure_loaded()

        if isinstance(image_source, Image.Image):
            img = image_source.convert("RGB")
        elif isinstance(image_source, (Path, str)):
            img = Image.open(image_source).convert("RGB")
        else:
            import io
            img = Image.open(io.BytesIO(image_source)).convert("RGB")

        inputs = self._processor(images=img, return_tensors="pt")
        with torch.no_grad():
            vision_out = self._model.vision_model(pixel_values=inputs["pixel_values"])
            # pooler_output : (1, hidden_size) — token CLS
            projected = self._model.visual_projection(vision_out.pooler_output)  # (1, 512)

        vec = projected[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_batch(self, images: list[Path | bytes | Image.Image]) -> np.ndarray:
        """
        Encode plusieurs images en une passe.

        Returns:
            np.ndarray de shape (N, 512), dtype float32.
        """
        return np.stack([self.embed(img) for img in images])

    @staticmethod
    def average_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
        """
        Calcule l'embedding moyen (re-normalisé) d'une collection de visages.
        C'est ce vecteur qui représente le "style collectif" des soumissions.
        """
        if not embeddings:
            raise ValueError("Aucun embedding à moyenner")
        mean = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else mean


# Singleton
embedder = Embedder()
