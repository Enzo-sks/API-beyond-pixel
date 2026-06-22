"""config.py — Constantes du modèle et chemins de checkpoints."""

import os

# ── Chemins ──────────────────────────────────────────────────────────────
# En local : ./checkpoints/...
# Sur Railway : monté via volume ou téléchargé au démarrage (cf. download_checkpoints.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(BASE_DIR, "checkpoints"))

TOKENIZER_PATH = os.path.join(CKPT_DIR, "tokenizer.json")
FUSION_CKPT_PATH = os.path.join(CKPT_DIR, "multimodal_fusion.pt")

# URLs optionnelles pour téléchargement au démarrage (si les fichiers ne sont
# pas inclus dans l'image Docker — cf. download_checkpoints.py)
TOKENIZER_URL = os.environ.get("TOKENIZER_URL", "")
FUSION_CKPT_URL = os.environ.get("FUSION_CKPT_URL", "")

# ── Hyperparamètres du modèle (identiques à train.py) ───────────────────
VIT_NAME = "codewithdark/vit-chest-xray"

TEXT_VOCAB_SIZE = 4359   # taille du vocab du tokenizer custom (cf. README)
TEXT_D = 256
TEXT_H = 8
TEXT_N = 6
TEXT_D_FF = 512

D_MODEL = 512
N_HEADS = 8
DROPOUT = 0.1

N_LABELS = 21
LABEL_COLS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Pneumonia", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Infiltration", "Mass", "Nodule",
    "Hernia", "Fracture", "Pleural_Thickening", "Opacity", "Consolidation",
    "Granuloma", "Calcinosis", "Scoliosis", "Atherosclerosis", "Normal",
]

# ── Image preprocessing (identique à VAL_TF de train.py) ────────────────
IMAGE_SIZE = (224, 224)
VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD = [0.5, 0.5, 0.5]

# ── Inference ─────────────────────────────────────────────────────────────
MAX_TEXT_LEN = 256          # tronque les findings trop longs (sécurité mémoire)
DEFAULT_THRESHOLD = 0.5     # seuil binaire par défaut pour "maladie détectée"
