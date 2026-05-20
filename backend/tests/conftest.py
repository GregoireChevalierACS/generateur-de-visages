"""
Configuration pytest partagée entre tous les modules de test.

- Initialise la DB SQLite réelle une fois par session (pour les tests
  qui utilisent le TestClient sans context manager, donc sans lifespan).
- Expose un marker `slow` pour les tests d'intégration lourds.
"""
import pytest
from app.services.face_store import init_db as _init_db


@pytest.fixture(scope="session", autouse=True)
def init_real_db():
    """Crée la table `faces` dans la vraie DB avant la session de tests."""
    _init_db()
