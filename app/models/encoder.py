"""encoder.py — Transformer Encoder (stack of Blocks)."""

import torch.nn as nn
from .layers import RoPE, Block


class Encoder(nn.Module):
    def __init__(self, V, d=256, h=8, N=6, d_ff=512):
        super().__init__()
        self.d, self.emb = d, nn.Embedding(V, d)
        rope = RoPE(d // h)
        self.layers = nn.ModuleList([Block(d, h, d_ff, rope) for _ in range(N)])
        self.norm = nn.RMSNorm(d)
    def forward(self, src):
        x = self.emb(src)
        for l in self.layers: x = l(x)
        return self.norm(x)
