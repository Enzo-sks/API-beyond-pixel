"""bert_mlm.py — BERT encoder + MLM prediction head."""

import torch.nn as nn
from .encoder import Encoder


class BERTForMLM(nn.Module):
    def __init__(self, V, d=256, h=8, N=6, d_ff=512):
        super().__init__()
        self.encoder = Encoder(V, d, h, N, d_ff)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.RMSNorm(d), nn.Linear(d, V))
    def forward(self, ids): return self.head(self.encoder(ids))
