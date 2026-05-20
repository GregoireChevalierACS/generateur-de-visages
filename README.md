# Générateur de visages

Outil de génération de visages illustrés à partir d'images soumises.  
Upload des dessins → détection automatique des visages → génération de nouveaux personnages via Stable Diffusion + IP-Adapter.

---

## Stack

| Couche | Technologie |
|---|---|
| Frontend | React 19 + TypeScript + Vite + Tailwind CSS |
| Backend | Python 3.12 + FastAPI + uvicorn |
| Détection | OpenCV (LBP anime cascade + Haar frontal) |
| Embeddings | CLIP ViT-B/32 (HuggingFace transformers) |
| Génération | Stable Diffusion 1.5 (DreamShaper-8) + IP-Adapter Face |
| Stockage | SQLite + fichiers locaux |

**Config matérielle recommandée :** GPU NVIDIA avec ≥ 4 Go VRAM (testé RTX 4060).  
La génération fonctionne aussi en CPU (~2-3 min/image).

---

## Installation

### Prérequis

- Python 3.12+
- Node.js 22+
- NVIDIA GPU (optionnel mais recommandé)

### Backend

```bash
cd backend
pip install -r requirements.txt
```

> PyTorch CUDA (RTX 4060 / CUDA 12.4) :
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```

### Frontend

```bash
cd frontend
npm install
```

---

## Lancement

Ouvrir deux terminaux :

```bash
# Terminal 1 — Backend
cd backend
python -m uvicorn app.main:app --reload
# → http://localhost:8000
```

```bash
# Terminal 2 — Frontend
cd frontend
npm run dev
# → http://localhost:5173
```

Documentation API interactive : **http://localhost:8000/docs**

---

## Utilisation

### 1. Uploader des images

Glisse-dépose tes illustrations dans la zone d'upload (PNG, JPG, WEBP, max 20 Mo).  
La détection de visages se lance automatiquement pour chaque image.

Les visages détectés apparaissent dans la galerie. Tu peux en supprimer individuellement ou tous effacer.

> **Note détection :** le détecteur fonctionne mieux sur les visages de face (style anime ou illustration semi-réaliste). Les visages de profil ou très stylisés peuvent nécessiter un recadrage manuel — cette fonctionnalité est prévue dans une prochaine itération.

### 2. Charger le modèle

Au premier lancement, clique sur **Charger le modèle**.  
Cela télécharge :
- DreamShaper-8 (~2 Go) — checkpoint Stable Diffusion optimisé pour l'illustration
- IP-Adapter Plus Face (~300 Mo) — conditionnement sur les visages de référence

**Durée : 3 à 5 minutes** selon la connexion. Une seule fois — le modèle est mis en cache par HuggingFace.

### 3. Générer

- **Fidélité aux références** : slider de 0 % (visage libre) à 100 % (très proche des références). Valeur recommandée : 60-75 %.
- **Graine fixe** : active-la pour obtenir des résultats reproductibles.
- Clique sur **Générer un visage** (~10-20 s avec GPU, ~2 min sans).

---

## API

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Santé du serveur |
| `POST` | `/api/upload` | Upload d'une image |
| `POST` | `/api/detect/{image_id}` | Détection des visages + embedding CLIP |
| `GET` | `/api/faces` | Liste des visages stockés |
| `GET` | `/api/faces/count` | Nombre de visages |
| `DELETE` | `/api/faces/{face_id}` | Supprime un visage |
| `DELETE` | `/api/faces` | Vide toute la bibliothèque |
| `GET` | `/api/generate/status` | Modèle chargé ? |
| `POST` | `/api/generate/init` | Charge le modèle SD + IP-Adapter |
| `POST` | `/api/generate` | Génère un visage |

### Paramètres de `/api/generate`

```json
{
  "ip_scale": 0.7,
  "steps": 30,
  "guidance_scale": 7.5,
  "seed": null,
  "prompt_extra": ""
}
```

---

## Tests

```bash
cd backend

# Suite rapide (115 tests, ~15 s)
python -m pytest

# Tests d'intégration réels — charge le vrai modèle SD
python -m pytest -m slow
```

---

## Structure du projet

```
generateur-de-visages/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI + lifespan
│   │   ├── config.py            # Chemins, constantes
│   │   ├── routers/
│   │   │   ├── upload.py        # POST /api/upload
│   │   │   ├── detect.py        # POST /api/detect/{id}
│   │   │   ├── faces.py         # GET/DELETE /api/faces
│   │   │   └── generate.py      # POST /api/generate
│   │   └── services/
│   │       ├── face_detector.py # Cascade LBP anime + Haar
│   │       ├── embedder.py      # CLIP ViT-B/32
│   │       ├── face_store.py    # SQLite CRUD
│   │       └── generator.py     # SD + IP-Adapter
│   ├── tests/                   # 115 tests pytest
│   ├── models/                  # Cascade anime (téléchargée auto)
│   ├── storage/
│   │   ├── uploads/             # Images uploadées
│   │   ├── faces/               # Crops des visages détectés
│   │   └── generated/           # Visages générés
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.tsx              # Logique principale
    │   ├── api.ts               # Client API typé
    │   └── components/
    │       ├── ImageUpload.tsx  # Zone drag & drop
    │       ├── FaceGallery.tsx  # Galerie des visages
    │       ├── GeneratePanel.tsx# Contrôles + résultat
    │       └── StatusBar.tsx    # Messages d'état
    └── package.json
```
