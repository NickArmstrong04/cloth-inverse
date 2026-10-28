"""Inverse models: ResNet encoder on the material grid with either a direct
regression head ("cnn_reg") or a heatmap head for (u, v) ("cnn_heat")."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.GroupNorm(8, cout)
        self.skip = (nn.Identity() if cin == cout and stride == 1
                     else nn.Conv2d(cin, cout, 1, stride, bias=False))

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.gelu(h + self.skip(x))


class Encoder(nn.Module):
    """128x128x6 -> feature maps at 1/4 (for heatmaps) and 1/16 (pooled)."""

    def __init__(self, cin=6, w=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, w, 3, 1, 1), nn.GroupNorm(8, w), nn.GELU())
        self.l1 = Block(w, w, 2)          # 64
        self.l2 = Block(w, 2 * w, 2)      # 32
        self.l3 = Block(2 * w, 4 * w, 2)  # 16
        self.l4 = Block(4 * w, 8 * w, 2)  # 8
        self.c_mid, self.c_out = 2 * w, 8 * w

    def forward(self, x):
        h = self.stem(x)
        h = self.l1(h)
        mid = self.l2(h)
        out = self.l4(self.l3(mid))
        return mid, out


class CNNReg(nn.Module):
    """Global-pool + MLP, predicts (2, 7) directly (normalized targets)."""

    def __init__(self, w=32, hidden=256, p_drop=0.2):
        super().__init__()
        self.enc = Encoder(6, w)
        self.head = nn.Sequential(
            nn.Linear(self.enc.c_out + 6, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, 14))

    def forward(self, x, g):
        _, out = self.enc(x)
        f = torch.cat([out.mean((-1, -2)), g], 1)
        return self.head(f).view(-1, 2, 7), None


class CNNHeat(nn.Module):
    """Heatmap head for (u, v) per force slot at 1/4 resolution with
    soft-argmax; the other 5 params regressed from pooled features plus
    heatmap-attended local features."""

    def __init__(self, w=32, hidden=256, p_drop=0.2, uv_scale=None):
        super().__init__()
        self.enc = Encoder(6, w)
        cm = self.enc.c_mid
        self.heat = nn.Sequential(nn.Conv2d(cm, cm, 3, 1, 1), nn.GroupNorm(8, cm), nn.GELU(),
                                  nn.Conv2d(cm, 2, 1))
        self.reg = nn.Sequential(
            nn.Linear(self.enc.c_out + 6 + cm, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, 5))
        self.uv_scale = uv_scale  # (mean, std) of u,v to emit normalized coords

    def forward(self, x, g):
        mid, out = self.enc(x)
        B, _, H, W = mid.shape
        logits = self.heat(mid).view(B, 2, H * W)
        prob = logits.softmax(-1).view(B, 2, H, W)
        jj = (torch.arange(H, device=x.device).float() + 0.5) / H  # v
        ii = (torch.arange(W, device=x.device).float() + 0.5) / W  # u
        u = (prob.sum(2) * ii).sum(-1)  # (B,2)
        v = (prob.sum(3) * jj).sum(-1)
        att = torch.einsum("bkhw,bchw->bkc", prob, mid)  # (B,2,cm)
        glob = torch.cat([out.mean((-1, -2)), g], 1).unsqueeze(1).expand(-1, 2, -1)
        rest = self.reg(torch.cat([glob, att], -1))  # (B,2,5)
        uv = torch.stack([u, v], -1)
        if self.uv_scale is not None:
            m, s = self.uv_scale
            uv = (uv - m) / s
        return torch.cat([uv, rest], -1), prob


def build(name: str, uv_scale=None, **kw):
    if name == "cnn_reg":
        return CNNReg(**kw)
    if name == "cnn_heat":
        return CNNHeat(uv_scale=uv_scale, **kw)
    raise ValueError(name)


def pair_loss(pred_n, y_n, w_dir=1.0):
    """Permutation-invariant loss on normalized targets. pred_n, y_n: (B,2,7).
    MSE on (u,v,logmag,dur); (1 - cos) on direction."""
    def one(p, t):
        mse = ((p[..., [0, 1, 5, 6]] - t[..., [0, 1, 5, 6]]) ** 2).mean(-1)
        d = F.normalize(p[..., 2:5], dim=-1)
        cos = (d * t[..., 2:5]).sum(-1)
        return mse + w_dir * (1.0 - cos)  # (B,2)
    l_id = one(pred_n, y_n).sum(-1)
    l_sw = one(pred_n, y_n.flip(1)).sum(-1)
    return torch.minimum(l_id, l_sw).mean()
