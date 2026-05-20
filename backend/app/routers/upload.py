import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from app.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, UPLOADS_DIR

router = APIRouter(tags=["upload"])


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Extension non supportée : {ext}. Acceptées : {ALLOWED_EXTENSIONS}",
        )

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop lourd (max {MAX_FILE_SIZE_MB} Mo)",
        )

    try:
        from io import BytesIO
        img = Image.open(BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Fichier image invalide ou corrompu")

    file_id = uuid.uuid4().hex
    dest = UPLOADS_DIR / f"{file_id}{ext}"
    dest.write_bytes(contents)

    # Re-open after verify() to get dimensions (verify() consumes the stream)
    from io import BytesIO
    img = Image.open(BytesIO(contents))

    return {
        "id": file_id,
        "filename": file.filename,
        "url": f"/storage/uploads/{file_id}{ext}",
        "width": img.width,
        "height": img.height,
        "size_bytes": len(contents),
    }
