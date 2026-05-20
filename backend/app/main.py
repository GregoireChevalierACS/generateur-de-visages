from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import UPLOADS_DIR, FACES_DIR
from app.services.generator import GENERATED_DIR
from app.routers import upload, detect, faces, generate
from app.services.face_store import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Générateur de visages", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve stored images as static files
app.mount("/storage/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/storage/faces", StaticFiles(directory=str(FACES_DIR)), name="faces")
app.mount("/storage/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")

app.include_router(upload.router, prefix="/api")
app.include_router(detect.router, prefix="/api")
app.include_router(faces.router, prefix="/api")
app.include_router(generate.router, prefix="/api")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    import traceback
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()},
    )


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}
