"""main.py — API FastAPI pour le modèle MultimodalFusion (21 pathologies thoraciques).

Endpoints :
  GET  /health   → statut du service + modèle chargé ou non
  POST /predict  → image (radio thoracique) + texte (findings) → probabilités par pathologie

Lancement local :
  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.inference import get_engine, ModelNotLoadedError
from app.config import LABEL_COLS, DEFAULT_THRESHOLD

app = FastAPI(
    title="CXR Multimodal Fusion API",
    description=(
        "Classification multi-label de 21 pathologies thoraciques à partir "
        "d'une radio (image) et d'un compte-rendu (texte), via fusion par "
        "cross-attention bidirectionnelle (BERT custom + ViT)."
    ),
    version="1.0.0",
)

# CORS ouvert par défaut — à restreindre en production si besoin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB


class PredictionItem(BaseModel):
    label: str
    probability: float
    positive: bool


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]
    positive_labels: list[str]
    threshold: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    labels: list[str]
    error: Optional[str] = None


@app.on_event("startup")
def _warm_up() -> None:
    """Force le chargement du modèle au démarrage du serveur plutôt qu'à la
    première requête, pour que la 1ère personne qui appelle /predict n'attende
    pas le temps de chargement (~quelques secondes selon le device)."""
    get_engine()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    engine = get_engine()
    return HealthResponse(
        status="ok" if engine.is_ready else "degraded",
        model_loaded=engine.is_ready,
        device=str(engine.device),
        labels=LABEL_COLS,
        error=engine.load_error,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="Radio thoracique (PNG/JPEG)"),
    findings: str = Form("", description="Compte-rendu radiologique (texte libre, peut être vide)"),
    threshold: float = Form(DEFAULT_THRESHOLD, description="Seuil de décision binaire (0-1)"),
) -> PredictResponse:
    engine = get_engine()

    if not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modèle non chargé sur le serveur. "
                f"Raison : {engine.load_error or 'inconnue'}."
            ),
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Type de fichier non supporté : {file.content_type}. "
                    f"Formats acceptés : {sorted(ALLOWED_CONTENT_TYPES)}.",
        )

    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(status_code=422, detail="threshold doit être entre 0 et 1.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=422, detail="Fichier image vide.")
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image trop volumineuse (max {MAX_IMAGE_SIZE_BYTES // (1024*1024)} MB).",
        )

    try:
        result = engine.predict(image_bytes, findings, threshold=threshold)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        # Erreur de décodage image (fichier corrompu), shape inattendue, etc.
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {exc}")

    return PredictResponse(**result)


@app.get("/")
def root():
    return {
        "service": "CXR Multimodal Fusion API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict (multipart/form-data: file, findings, threshold)",
    }
