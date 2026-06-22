# CXR Multimodal Fusion — API d'inférence

API FastAPI pour servir le modèle `MultimodalFusion` (BERT custom + ViT,
cross-attention bidirectionnelle) — classification multi-label de 21
pathologies thoraciques à partir d'une radio + d'un compte-rendu texte.

## ⚠️ Fichiers requis avant de lancer

Cette API a besoin de **2 fichiers** que le code seul ne contient pas :

| Fichier | Rôle | Où le mettre |
|---|---|---|
| `checkpoints/tokenizer.json` | Tokenizer BPE custom (vocab 4359 tokens) | `checkpoints/` |
| `checkpoints/multimodal_fusion.pt` | Poids entraînés du modèle (96M params) | `checkpoints/` |

Sans ces fichiers, l'API démarre quand même mais `/health` renverra
`"status": "degraded"` et `/predict` renverra une erreur `503`.

Le ViT pré-entraîné (`codewithdark/vit-chest-xray`) est téléchargé automatiquement
depuis Hugging Face Hub au premier démarrage — pas besoin de le fournir.

---

## Option A — Checkpoints inclus dans l'image Docker (le plus simple)

1. Placez vos fichiers :
   ```
   cxr-api/checkpoints/tokenizer.json
   cxr-api/checkpoints/multimodal_fusion.pt
   ```
2. Dans le `Dockerfile`, décommentez :
   ```dockerfile
   COPY checkpoints ./checkpoints
   ```
3. Build & run :
   ```bash
   docker build -t cxr-api .
   docker run -p 8000:8000 cxr-api
   ```

**Inconvénient** : si `multimodal_fusion.pt` fait plusieurs centaines de Mo,
chaque build Docker et chaque déploiement Railway re-uploadera l'image entière.

---

## Option B — Téléchargement au démarrage (recommandé pour Railway)

Hébergez vos 2 fichiers quelque part avec un lien de téléchargement direct
(Hugging Face Hub — même un repo privé avec token dans l'URL signée,
GitHub Releases, S3 avec URL pré-signée, etc.), puis définissez sur Railway :

```
TOKENIZER_URL=https://.../tokenizer.json
FUSION_CKPT_URL=https://.../multimodal_fusion.pt
```

Le script `app/download_checkpoints.py` les télécharge automatiquement au
démarrage du conteneur (et ne re-télécharge pas s'ils sont déjà présents).

**Important pour Railway** : sans volume persistant, le conteneur retélécharge
les poids à **chaque redémarrage/redéploiement**. Si c'est trop lent, attachez
un [Railway Volume](https://docs.railway.com/reference/volumes) monté sur
`/app/checkpoints` (réglez `CKPT_DIR=/app/checkpoints` en variable d'env).

---

## Lancer en local (sans Docker)

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
export CKPT_DIR=./checkpoints   # si différent du défaut
uvicorn app.main:app --reload --port 8000
```

## Endpoints

### `GET /health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "labels": ["Atelectasis", "Cardiomegaly", ...]
}
```

### `POST /predict`
Multipart form-data :
- `file` (required) — image PNG/JPEG de la radio
- `findings` (optional, default `""`) — texte du compte-rendu radiologique
- `threshold` (optional, default `0.5`) — seuil de décision binaire

**Exemple curl :**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@radio_patient.png" \
  -F "findings=Opacité basale droite, pas d'épanchement visible." \
  -F "threshold=0.5"
```

**Réponse :**
```json
{
  "predictions": [
    {"label": "Infiltration", "probability": 0.83, "positive": true},
    {"label": "Pneumonia", "probability": 0.61, "positive": true},
    {"label": "Normal", "probability": 0.04, "positive": false}
  ],
  "positive_labels": ["Infiltration", "Pneumonia"],
  "threshold": 0.5
}
```

`findings` peut être laissé vide (`""`) si vous voulez une prédiction
image-seule — le texte sera encodé comme une séquence quasi-vide.

Documentation interactive auto-générée : `http://localhost:8000/docs`

---

## Déploiement sur Railway

1. Poussez ce dossier sur un repo GitHub.
2. Sur [railway.app](https://railway.app) : **New Project → Deploy from GitHub repo**.
3. Railway détecte le `Dockerfile` automatiquement (confirmé par `railway.toml`).
4. Si vous utilisez l'Option B, ajoutez les variables d'environnement
   `TOKENIZER_URL` et `FUSION_CKPT_URL` dans l'onglet **Variables**.
5. Railway expose le service sur un domaine public — Railway injecte la
   variable `PORT` automatiquement, déjà gérée par le `CMD` du Dockerfile.
6. Vérifiez `https://<votre-app>.up.railway.app/health`.

### Limites de plan à surveiller
Le modèle complet (96M params, ViT + BERT custom) tourne sur CPU dans ce
setup. Sur le plan gratuit/Hobby de Railway (RAM limitée), une inférence prend
quelques secondes ; un trafic important pourrait nécessiter un plan avec plus
de RAM/CPU ou un passage en GPU (non géré par ce Dockerfile en l'état).

---

## Structure du projet

```
cxr-api/
├── app/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── layers.py        # RoPE, MHA, Block
│   │   ├── encoder.py        # Encoder texte (stack de Blocks)
│   │   ├── bert_mlm.py       # BERTForMLM (pré-entraînement, on récupère .encoder)
│   │   └── fusion.py         # BidirectionalCrossAttention + MultimodalFusion
│   ├── config.py             # Constantes / hyperparamètres / chemins
│   ├── inference.py          # Chargement modèle (singleton) + predict()
│   ├── download_checkpoints.py  # Téléchargement optionnel au démarrage
│   └── main.py                # FastAPI : /health, /predict
├── checkpoints/               # tokenizer.json + multimodal_fusion.pt (à fournir)
├── Dockerfile
├── requirements.txt
├── railway.toml
└── .dockerignore
```
