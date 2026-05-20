"""
Service de génération de visages via Stable Diffusion + IP-Adapter Face.

Modèle de base : Lykon/dreamshaper-8 (SD 1.5, style illustration)
IP-Adapter    : h94/IP-Adapter → ip-adapter-plus-face_sd15.bin

Optimisations 4 Go VRAM :
  - float16
  - enable_model_cpu_offload() : décharge les couches inutilisées sur CPU
  - Génération en 512×512
"""
import threading
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from app.config import FACES_DIR

GENERATED_DIR = Path(__file__).parent.parent.parent / "storage" / "generated"

BASE_MODEL_ID = "Lykon/dreamshaper-8"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "models"
IP_ADAPTER_WEIGHT = "ip-adapter-plus-face_sd15.bin"

DEFAULT_PROMPT = (
    "semi-realistic illustration, digital painting, character portrait, "
    "detailed face, graphic novel style, soft shading, high quality"
)
NEGATIVE_PROMPT = (
    "photorealistic, 3d render, hyperrealistic, blurry, deformed, "
    "extra limbs, bad anatomy, watermark, signature, text"
)

IMAGE_SIZE = 512
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 7.5
DEFAULT_IP_SCALE = 0.7   # 0 = ignore les références, 1 = très proche


class GeneratorService:
    def __init__(self):
        self._pipe = None
        self._ready = False
        self._loading = False
        self._load_error: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def load_in_background(self) -> bool:
        """Lance le chargement dans un thread séparé. Retourne False si déjà en cours."""
        if self._ready or self._loading:
            return False
        thread = threading.Thread(target=self._load_blocking, daemon=True)
        thread.start()
        return True

    def _load_blocking(self) -> None:
        """Exécuté dans un thread — bloque jusqu'à la fin du chargement."""
        self._loading = True
        self._load_error = None
        try:
            self.load()
        except Exception as e:
            self._load_error = str(e)
        finally:
            self._loading = False

    def load(self) -> None:
        """Télécharge et initialise le pipeline SD + IP-Adapter (appel unique)."""
        if self._ready:
            return

        from diffusers import StableDiffusionPipeline

        self._pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )

        self._pipe.load_ip_adapter(
            IP_ADAPTER_REPO,
            subfolder=IP_ADAPTER_SUBFOLDER,
            weight_name=IP_ADAPTER_WEIGHT,
        )
        self._pipe.set_ip_adapter_scale(DEFAULT_IP_SCALE)

        # Offload sur CPU pour tenir dans 4 Go VRAM
        self._pipe.enable_model_cpu_offload()

        self._ready = True

    def generate(
        self,
        reference_images: list[Image.Image],
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = NEGATIVE_PROMPT,
        num_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        ip_scale: float = DEFAULT_IP_SCALE,
        seed: Optional[int] = None,
    ) -> tuple[Image.Image, str]:
        """
        Génère un visage à partir d'images de référence.

        Args:
            reference_images : liste de crops de visages (PIL Images)
            ip_scale          : force du conditionnement (0-1)
            seed              : graine pour la reproductibilité

        Returns:
            (image PIL, url relative du fichier sauvegardé)
        """
        if not self._ready:
            raise RuntimeError("Le modèle n'est pas chargé. Appelez load() d'abord.")

        if not reference_images:
            raise ValueError("Au moins une image de référence est requise.")

        self._pipe.set_ip_adapter_scale(ip_scale)

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=reference_images,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            generator=generator,
        )

        image = result.images[0]
        file_id = uuid.uuid4().hex
        dest = GENERATED_DIR / f"{file_id}.png"
        image.save(dest)

        return image, f"/storage/generated/{file_id}.png"


# Singleton
generator_service = GeneratorService()
