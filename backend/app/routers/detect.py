from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.config import UPLOADS_DIR
from app.services.face_detector import face_detector
from app.services.embedder import embedder
from app.services.face_store import save_face

router = APIRouter(tags=["detect"])


class FaceResult(BaseModel):
    face_id: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    detector: str
    crop_url: str


class DetectResponse(BaseModel):
    image_id: str
    face_count: int
    faces: list[FaceResult]


def _embed_and_store(face_id: str, image_id: str, crop_url: str, detector: str, crop_path: Path):
    """Calcule l'embedding CLIP et persiste le visage en DB (tâche de fond)."""
    try:
        emb = embedder.embed(crop_path)
        save_face(face_id, image_id, crop_url, detector, emb)
    except Exception:
        # Ne pas faire crasher la requête si l'embedding échoue
        save_face(face_id, image_id, crop_url, detector, None)


@router.post("/detect/{image_id}", response_model=DetectResponse)
def detect_faces(image_id: str, background_tasks: BackgroundTasks):
    matches = list(UPLOADS_DIR.glob(f"{image_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' introuvable")

    image_path = matches[0]
    faces = face_detector.detect(image_path)

    for f in faces:
        background_tasks.add_task(
            _embed_and_store,
            f.face_id, image_id, f.crop_url, f.detector, f.crop_path,
        )

    return DetectResponse(
        image_id=image_id,
        face_count=len(faces),
        faces=[
            FaceResult(
                face_id=f.face_id,
                x=f.x, y=f.y,
                width=f.width, height=f.height,
                confidence=f.confidence,
                detector=f.detector,
                crop_url=f.crop_url,
            )
            for f in faces
        ],
    )
