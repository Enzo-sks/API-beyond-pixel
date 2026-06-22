"""download_checkpoints.py — Télécharge tokenizer.json et multimodal_fusion.pt
au démarrage du conteneur, si les variables d'environnement TOKENIZER_URL /
FUSION_CKPT_URL sont définies et que les fichiers ne sont pas déjà présents.

Pourquoi : le ViT (codewithdark/vit-chest-xray, ~86M params) est déjà
téléchargé depuis le Hugging Face Hub au premier lancement (via transformers).
Le .pth de multimodal_fusion.pt + tokenizer.json sont SPÉCIFIQUES à ce repo et
doivent venir de votre propre stockage (S3, Hugging Face Hub privé/public,
Google Drive avec lien direct, GitHub Releases, etc.).

Deux stratégies possibles, au choix :
  1. COPIER les fichiers dans checkpoints/ AVANT le build Docker
     → rien à faire, ce script ne fait rien (fichiers déjà présents).
  2. Les héberger en ligne et définir TOKENIZER_URL / FUSION_CKPT_URL
     comme variables d'environnement sur Railway
     → ce script les télécharge au démarrage du conteneur.
"""

import os
import sys
import urllib.request

from app.config import (
    CKPT_DIR,
    TOKENIZER_PATH,
    FUSION_CKPT_PATH,
    TOKENIZER_URL,
    FUSION_CKPT_URL,
)


def _download(url: str, dest: str, label: str) -> None:
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"[download_checkpoints] {label} déjà présent ({size_mb:.1f} MB) — skip.")
        return

    if not url:
        print(
            f"[download_checkpoints] ATTENTION : {label} absent et aucune URL "
            f"fournie (dest={dest}). L'API ne pourra pas charger le modèle."
        )
        return

    print(f"[download_checkpoints] Téléchargement de {label} depuis {url} ...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp_dest = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp_dest)
        os.replace(tmp_dest, dest)
        size_mb = os.path.getsize(dest) / 1e6
        print(f"[download_checkpoints] {label} téléchargé ({size_mb:.1f} MB).")
    except Exception as exc:
        if os.path.exists(tmp_dest):
            os.remove(tmp_dest)
        print(f"[download_checkpoints] ÉCHEC téléchargement {label} : {exc}")
        raise


def main() -> None:
    os.makedirs(CKPT_DIR, exist_ok=True)
    _download(TOKENIZER_URL, TOKENIZER_PATH, "tokenizer.json")
    _download(FUSION_CKPT_URL, FUSION_CKPT_PATH, "multimodal_fusion.pt")


if __name__ == "__main__":
    main()
    # Si l'un des deux fichiers critiques manque encore après tentative de
    # téléchargement, on log un avertissement clair mais on NE bloque PAS le
    # démarrage : /health doit pouvoir répondre même modèle non chargé,
    # pour que Railway ne tue pas le conteneur en boucle.
    missing = [
        p for p in (TOKENIZER_PATH, FUSION_CKPT_PATH) if not os.path.exists(p)
    ]
    if missing:
        print(
            f"[download_checkpoints] Fichiers manquants après tentative : {missing}. "
            f"L'endpoint /predict renverra une erreur 503 jusqu'à ce qu'ils soient fournis."
        )
