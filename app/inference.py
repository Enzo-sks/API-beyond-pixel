"""
inference.py — Chargement du modèle MultimodalFusion + tokenizer depuis Hugging Face,
et fonction predict() utilisée par l'API FastAPI.

Le modèle est chargé UNE SEULE FOIS au démarrage (singleton via lru_cache),
pas à chaque requête — sinon chaque appel à /predict rechargerait 96M
paramètres depuis le disque (très lent, surtout sur Railway).
"""

from huggingface_hub import hf_hub_download

import io
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import ViTModel
from tokenizers import Tokenizer

from app.config import (
    TEXT_VOCAB_SIZE,
    TEXT_D,
    TEXT_H,
    TEXT_N,
    TEXT_D_FF,
    D_MODEL,
    N_HEADS,
    DROPOUT,
    N_LABELS,
    LABEL_COLS,
    IMAGE_SIZE,
    VIT_MEAN,
    VIT_STD,
    MAX_TEXT_LEN,
)
from app.models import BERTForMLM, MultimodalFusion


class ModelNotLoadedError(RuntimeError):
    """Levée quand le modèle/tokenizer n'a pas pu être chargé."""


# ── Transform image (identique à VAL_TF de train.py — PAS d'augmentation) ──
_IMAGE_TF = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
])


def _build_model(device: torch.device) -> MultimodalFusion:
    """Reconstruit l'architecture exacte utilisée à l'entraînement."""
    bert = BERTForMLM(TEXT_VOCAB_SIZE, TEXT_D, TEXT_H, TEXT_N, TEXT_D_FF)
    text_encoder = bert.encoder

    # ViT HuggingFace (même modèle qu'en entraînement)
    vit = ViTModel.from_pretrained("codewithdark/vit-chest-xray")

    model = MultimodalFusion(
        text_encoder=text_encoder,
        vit=vit,
        n_labels=N_LABELS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        dropout=DROPOUT,
    )
    return model.to(device)


class InferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: Optional[Tokenizer] = None
        self.model: Optional[MultimodalFusion] = None
        self._load_error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        try:
            print("[inference] Téléchargement du tokenizer depuis Hugging Face…")
            tokenizer_path = hf_hub_download(
                repo_id="Enzo-sks/cxr-multimodal-fusion2",
                filename="tokenizer.json"
            )
            self.tokenizer = Tokenizer.from_file(tokenizer_path)

            print("[inference] Téléchargement du checkpoint depuis Hugging Face…")
            ckpt_path = hf_hub_download(
                repo_id="Enzo-sks/cxr-multimodal-fusion2",
                filename="multimodal_fusion.pt"
            )

            print("[inference] Construction de l'architecture MultimodalFusion…")
            model = _build_model(self.device)

            print(f"[inference] Chargement des poids depuis {ckpt_path}")
            full_state = torch.load(
                ckpt_path, map_location=self.device, weights_only=True
            )

            # ───────────────────────────────────────────────────────────────
            # IMPORTANT :
            # On ignore tous les poids du ViT venant du checkpoint
            # car ils ne correspondent plus à la version actuelle du modèle HF.
            # On garde uniquement :
            #   - l'encodeur texte
            #   - la cross-attention
            #   - la tête de classification
            #   - le pooling image
            # ───────────────────────────────────────────────────────────────
            filtered_state = {
                k: v for k, v in full_state.items()
                if not k.startswith("vit.")
            }

            missing, unexpected = model.load_state_dict(filtered_state, strict=False)

            if missing:
                print(f"[inference] Clés manquantes (hors ViT) : {len(missing)} (ex: {missing[:5]})")
            if unexpected:
                print(f"[inference] Clés inattendues (hors ViT) : {len(unexpected)} (ex: {unexpected[:5]})")

            model.eval()
            self.model = model
            print(f"[inference] Modèle prêt sur device={self.device}.")

        except Exception as exc:
            self._load_error = str(exc)
            print(f"[inference] ÉCHEC chargement du modèle : {exc}")

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _encode_text(self, findings: str) -> torch.Tensor:
        text = (findings or "").strip()
        enc = self.tokenizer.encode(text)
        ids = enc.ids[:MAX_TEXT_LEN] if MAX_TEXT_LEN else enc.ids
        if len(ids) == 0:
            ids = [0]  # éviter une séquence vide
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def _encode_image(self, image_bytes: bytes) -> torch.Tensor:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pixel_values = _IMAGE_TF(img)
        return pixel_values.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image_bytes: bytes, findings: str, threshold: float = 0.5) -> dict:
        if not self.is_ready:
            raise ModelNotLoadedError(
                self._load_error or "Modèle non chargé."
            )

        input_ids = self._encode_text(findings)
        pixel_values = self._encode_image(image_bytes)

        logits = self.model(input_ids, pixel_values)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        results = [
            {
                "label": label,
                "probability": float(p),
                "positive": bool(p >= threshold),
            }
            for label, p in zip(LABEL_COLS, probs)
        ]
        results.sort(key=lambda r: r["probability"], reverse=True)

        positive_labels = [r["label"] for r in results if r["positive"]]

        return {
            "predictions": results,
            "positive_labels": positive_labels,
            "threshold": threshold,
        }


@lru_cache(maxsize=1)
def get_engine() -> InferenceEngine:
    return InferenceEngine()
