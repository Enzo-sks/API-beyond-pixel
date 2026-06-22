"""layers.py — RoPE, Multi-Head Attention, Transformer Block."""

import torch, torch.nn as nn, torch.nn.functional as F


class RoPE(nn.Module):
    def __init__(self, d, max_len=4096):
        super().__init__()
        freqs = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
        angles = torch.outer(torch.arange(max_len).float(), freqs)
        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())
    def forward(self, x):
        T = x.size(2); c, s = self.cos[:T], self.sin[:T]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1*c - x2*s, x1*s + x2*c), -1).flatten(-2)


class MHA(nn.Module):
    def __init__(self, d, h, rope=None):
        super().__init__()
        self.h, self.dk, self.rope = h, d//h, rope
        self.W_q, self.W_k, self.W_v, self.W_o = (nn.Linear(d, d) for _ in range(4))
    def forward(self, q, k, v):
        B = q.size(0)
        proj = lambda W, x: W(x).view(B, -1, self.h, self.dk).transpose(1, 2)
        Q, K, V = proj(self.W_q, q), proj(self.W_k, k), proj(self.W_v, v)
        if self.rope: Q, K = self.rope(Q), self.rope(K)
        out = F.scaled_dot_product_attention(Q, K, V)
        return self.W_o(out.transpose(1, 2).contiguous().view(B, -1, self.h*self.dk))


class Block(nn.Module):
    def __init__(self, d, h, d_ff, rope):
        super().__init__()
        self.attn, self.ffn = MHA(d, h, rope), nn.Sequential(nn.Linear(d, d_ff), nn.SiLU(), nn.Linear(d_ff, d))
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
    def forward(self, x):
        xn = self.n1(x); x = x + self.attn(xn, xn, xn)
        return x + self.ffn(self.n2(x))
