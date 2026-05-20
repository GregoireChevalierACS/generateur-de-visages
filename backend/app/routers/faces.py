from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.face_store import list_faces, delete_face, count_faces, get_face

router = APIRouter(tags=["faces"])


class FaceEntry(BaseModel):
    face_id: str
    image_id: str
    crop_url: str
    detector: str
    created_at: str


class FacesListResponse(BaseModel):
    total: int
    faces: list[FaceEntry]


@router.get("/faces", response_model=FacesListResponse)
def get_faces():
    faces = list_faces()
    return FacesListResponse(total=len(faces), faces=faces)


@router.get("/faces/count")
def get_face_count():
    return {"count": count_faces()}


@router.delete("/faces/{face_id}")
def remove_face(face_id: str):
    if not delete_face(face_id):
        raise HTTPException(status_code=404, detail=f"Visage '{face_id}' introuvable")
    return {"deleted": face_id}


@router.delete("/faces")
def clear_all_faces():
    faces = list_faces()
    count = 0
    for f in faces:
        if delete_face(f["face_id"]):
            count += 1
    return {"deleted_count": count}
