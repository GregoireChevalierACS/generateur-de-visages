"""
Stockage persistant des visages détectés et de leurs embeddings (SQLite).

Table `faces` :
  face_id    TEXT PRIMARY KEY
  image_id   TEXT              -- image source (upload)
  crop_url   TEXT
  detector   TEXT
  embedding  BLOB              -- float32 numpy array sérialisé
  created_at TEXT
"""
import io
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).parent.parent.parent / "storage" / "faces.db"


def _serialize(vec: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, vec)
    return buf.getvalue()


def _deserialize(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob))


@contextmanager
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                face_id    TEXT PRIMARY KEY,
                image_id   TEXT NOT NULL,
                crop_url   TEXT NOT NULL,
                detector   TEXT NOT NULL,
                embedding  BLOB,
                created_at TEXT NOT NULL
            )
        """)


def save_face(
    face_id: str,
    image_id: str,
    crop_url: str,
    detector: str,
    embedding: np.ndarray | None = None,
) -> None:
    blob = _serialize(embedding) if embedding is not None else None
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO faces
              (face_id, image_id, crop_url, detector, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (face_id, image_id, crop_url, detector, blob, datetime.now(timezone.utc).isoformat()),
        )


def get_face(face_id: str) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM faces WHERE face_id = ?", (face_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    if result["embedding"]:
        result["embedding"] = _deserialize(result["embedding"])
    return result


def get_all_embeddings() -> list[np.ndarray]:
    """Retourne tous les embeddings disponibles (pour calcul de la moyenne)."""
    with _get_conn() as conn:
        rows = conn.execute("SELECT embedding FROM faces WHERE embedding IS NOT NULL").fetchall()
    return [_deserialize(r["embedding"]) for r in rows]


def list_faces() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT face_id, image_id, crop_url, detector, created_at FROM faces ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_face(face_id: str) -> bool:
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM faces WHERE face_id = ?", (face_id,))
    return cur.rowcount > 0


def count_faces() -> int:
    with _get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
