"""fusion.py — Multimodal fusion avec cross-attention bidirectionnelle.

Copie fidèle du model.py original du repo multimodal_fusion/.

Architecture:
  text_tokens  (B, L, 256)  ─┐
                               ├─► BidirectionalCrossAttention ─► pool ─► concat ─► head
  image_tokens (B, 197, 768) ─┘

Les deux modalités sont projetées vers un espace commun d_model=512 avant
le cross-attention. La direction Text→Image et Image→Text s'effectuent en
parallèle (même features projetées en entrée) pour un vrai signal bidirectionnel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalCrossAttention(nn.Module):
    """
    Cross-attention bidirectionnel entre séquences texte et image.

    Les deux directions sont calculées en parallèle sur les features projetées,
    puis chaque branche reçoit son propre bloc FFN + LayerNorm résiduel.

    Args:
        text_dim  : dimension de l'encoder texte (256)
        image_dim : dimension du backbone ViT   (768)
        d_model   : dimension de projection commune (512)
        n_heads   : têtes d'attention (doit diviser d_model)
        dropout   : taux de dropout
    """

    def __init__(
        self,
        text_dim: int  = 256,
        image_dim: int = 768,
        d_model: int   = 512,
        n_heads: int   = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"

        self.text_proj  = nn.Linear(text_dim,  d_model, bias=False)
        self.image_proj = nn.Linear(image_dim, d_model, bias=False)

        # Text queries × Image keys/values
        self.t2i = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # Image queries × Text keys/values
        self.i2t = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Norms post-attention (Pre-LN sur résidu)
        self.norm_t1 = nn.LayerNorm(d_model)
        self.norm_i1 = nn.LayerNorm(d_model)

        # FFN pour chaque branche : d_model → 2*d_model → d_model
        self.ffn_t = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ffn_i = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        self.norm_t2 = nn.LayerNorm(d_model)
        self.norm_i2 = nn.LayerNorm(d_model)

    def forward(
        self,
        text_tokens:  torch.Tensor,   # (B, L, text_dim)
        image_tokens: torch.Tensor,   # (B, N, image_dim)  N=197 pour ViT-B/16
    ):
        T = self.text_proj(text_tokens)    # (B, L, d_model)
        I = self.image_proj(image_tokens)  # (B, N, d_model)

        # Les deux cross-attentions sont calculées sur les mêmes projections T, I
        # pour un flux d'information simultané dans les deux directions.
        T_ca, _ = self.t2i(T, I, I)   # text enrichi par l'image
        I_ca, _ = self.i2t(I, T, T)   # image enrichie par le texte

        T = self.norm_t1(T + T_ca)
        I = self.norm_i1(I + I_ca)

        T = self.norm_t2(T + self.ffn_t(T))
        I = self.norm_i2(I + self.ffn_i(I))

        return T, I   # (B, L, d_model), (B, N, d_model)


class MultimodalFusion(nn.Module):
    """
    Classifieur multi-label combinant un encoder texte custom (d=256)
    et un ViT pré-entraîné (d=768) via cross-attention bidirectionnel.

    Pooling:
      - Texte  : token [CLS] (index 0) du tenseur fusionné
      - Image  : attention pooling souple sur tous les tokens fusionnés

    La concaténation des deux représentations (B, 2*d_model) alimente la tête.

    Args:
        text_encoder : Encoder de Text classification/models/encoder.py
        vit          : ViTModel HuggingFace (expose .last_hidden_state et .config.hidden_size)
        n_labels     : nombre de classes de sortie (défaut 21)
        d_model      : dimension de projection commune (défaut 512)
        n_heads      : têtes de cross-attention (défaut 8)
        dropout      : taux de dropout (défaut 0.1)
    """

    def __init__(
        self,
        text_encoder,
        vit,
        n_labels: int  = 21,
        d_model:  int  = 512,
        n_heads:  int  = 8,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.vit          = vit

        text_dim  = text_encoder.d            # 256
        image_dim = vit.config.hidden_size    # 768

        self.cross_attn = BidirectionalCrossAttention(
            text_dim=text_dim, image_dim=image_dim,
            d_model=d_model, n_heads=n_heads, dropout=dropout,
        )

        # Poids d'attention pour le pooling image (appris)
        self.image_pool_w = nn.Linear(d_model, 1, bias=False)

        # Tête de classification : [text_CLS ‖ image_pool] → n_labels
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_labels),
        )

    def forward(
        self,
        input_ids:    torch.Tensor,             # (B, L)
        pixel_values: torch.Tensor,             # (B, 3, H, W)
    ) -> torch.Tensor:                          # (B, n_labels)
        # ── Encodeurs ────────────────────────────────────────────────────
        text_tokens  = self.text_encoder(input_ids)                          # (B, L, 256)
        image_tokens = self.vit(pixel_values=pixel_values).last_hidden_state # (B, 197, 768)

        # ── Cross-attention bidirectionnel ───────────────────────────────
        fused_text, fused_image = self.cross_attn(text_tokens, image_tokens)
        # fused_text : (B, L, d_model)
        # fused_image: (B, 197, d_model)

        # ── Pooling texte : token [CLS] ──────────────────────────────────
        text_repr = fused_text[:, 0]                                   # (B, d_model)

        # ── Pooling image : attention souple ────────────────────────────
        attn_w     = F.softmax(self.image_pool_w(fused_image), dim=1)  # (B, 197, 1)
        image_repr = (attn_w * fused_image).sum(dim=1)                 # (B, d_model)

        # ── Classification ───────────────────────────────────────────────
        fused = torch.cat([text_repr, image_repr], dim=1)              # (B, 2*d_model)
        return self.head(fused)
