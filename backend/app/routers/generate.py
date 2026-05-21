from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

import app.services.generator as _gen_module
from app.services.generator import DEFAULT_IP_SCALE, DEFAULT_STEPS, DEFAULT_GUIDANCE, random_diversity_tags
from app.services.face_store import list_faces
from app.config import FACES_DIR

router = APIRouter(tags=["generate"])


class GenerateRequest(BaseModel):
    ip_scale: float = Field(DEFAULT_IP_SCALE, ge=0.0, le=1.0,
                            description="Force du conditionnement sur les visages référence (0=libre, 1=proche)")
    steps: int = Field(DEFAULT_STEPS, ge=5, le=50)
    guidance_scale: float = Field(DEFAULT_GUIDANCE, ge=1.0, le=15.0)
    seed: int | None = Field(None, description="Graine pour résultats reproductibles")
    prompt_extra: str = Field("", description="Texte ajouté au prompt de base")


class GenerateResponse(BaseModel):
    url: str
    seed: int | None
    reference_faces_used: int
    model_ready: bool


def _load_face_images() -> list[Image.Image]:
    """Charge les crops des visages stockés en base."""
    faces = list_faces()
    images = []
    for f in faces:
        # crop_url = "/storage/faces/<filename>"
        filename = Path(f["crop_url"]).name
        path = FACES_DIR / filename
        if path.exists():
            images.append(Image.open(path).convert("RGB"))
    return images


@router.post("/generate/init")
def init_generator():
    """Lance le chargement SD + IP-Adapter en arrière-plan et retourne immédiatement."""
    svc = _gen_module.generator_service
    if svc.is_ready:
        return {"status": "already_loaded"}
    if svc.is_loading:
        return {"status": "loading"}
    svc.load_in_background()
    return {"status": "started"}


@router.get("/generate/status")
def generator_status():
    svc = _gen_module.generator_service
    return {
        "ready": svc.is_ready,
        "loading": svc.is_loading,
        "error": svc.load_error,
    }


@router.post("/generate", response_model=GenerateResponse)
def generate_face(req: GenerateRequest):
    svc = _gen_module.generator_service
    if not svc.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas initialisé. Appelez POST /api/generate/init d'abord.",
        )

    face_images = _load_face_images()
    if not face_images:
        raise HTTPException(
            status_code=422,
            detail="Aucun visage en base. Uploadez et détectez des images d'abord.",
        )

    from app.services.generator import DEFAULT_PROMPT
    prompt = f"{DEFAULT_PROMPT}, {random_diversity_tags()}"
    if req.prompt_extra:
        prompt = f"{prompt}, {req.prompt_extra}"

    try:
        _, url = svc.generate(
            reference_images=face_images,
            ip_scale=req.ip_scale,
            num_steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            prompt=prompt,
        )
    except Exception as exc:
        import traceback
        detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=detail)

    return GenerateResponse(
        url=url,
        seed=req.seed,
        reference_faces_used=len(face_images),
        model_ready=True,
    )
