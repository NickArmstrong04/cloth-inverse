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
    """Material-grid encoder. Downsamples by 2 per block until the map is ~8 px
    (4 blocks for 128 input, 7 for 768). Returns the map two blocks before the
    end ("mid", used for heatmaps: 32 px at 128, 24 px at 768) and the final
    map ("out", pooled for regression). Channel width doubles per block up to 8w.
    A stride-2 stem is used for inputs above 256 to keep activations affordable."""

    def __init__(self, cin=6, w=32, in_size=128):
        super().__init__()
        stem_stride = 2 if in_size > 256 else 1
        size = in_size // stem_stride
        self.stem = nn.Sequential(nn.Conv2d(cin, w, 3, stem_stride, 1), nn.GroupNorm(8, w), nn.GELU())
        blocks, c = [], w
        while size > 8:
            cout = min(8 * w, c * 2) if blocks else w
            blocks.append(Block(c, cout, 2))
            c, size = cout, size // 2
        assert len(blocks) >= 3, "input too small"
        self.blocks = nn.ModuleList(blocks)
        self.c_mid = blocks[-3].c2.out_channels
        self.c_out = c
        self.mid_index = len(blocks) - 3

    def forward(self, x):
        h = self.stem(x)
        mid = None
        for i, b in enumerate(self.blocks):
            h = b(h)
            if i == self.mid_index:
                mid = h
        return mid, h


class CNNReg(nn.Module):
    """Global-pool + MLP, predicts (2, 7) directly (normalized targets)."""

    def __init__(self, w=32, hidden=256, p_drop=0.2, in_size=128, cin=6, gdim=6, **_):
        super().__init__()
        self.enc = Encoder(cin, w, in_size)
        self.head = nn.Sequential(
            nn.Linear(self.enc.c_out + gdim, hidden), nn.GELU(), nn.Dropout(p_drop),
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

    def __init__(self, w=32, hidden=256, p_drop=0.2, uv_scale=None, in_size=128, cin=6, gdim=6, **_):
        super().__init__()
        self.enc = Encoder(cin, w, in_size)
        cm = self.enc.c_mid
        self.heat = nn.Sequential(nn.Conv2d(cm, cm, 3, 1, 1), nn.GroupNorm(8, cm), nn.GELU(),
                                  nn.Conv2d(cm, 2, 1))
        self.reg = nn.Sequential(
            nn.Linear(self.enc.c_out + gdim + cm, hidden), nn.GELU(), nn.Dropout(p_drop),
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


class PointTF(nn.Module):
    """Observation-ladder rung 3: an UNORDERED point cloud. Each step samples
    ``n_points`` random grid points (random subset + random order), so the model
    never sees the material indexing. Per-point MLP -> transformer encoder ->
    attention pooling -> (2, 7). Input per point: centred position (3) and,
    if present, velocity (3). Global features g are appended after pooling."""

    def __init__(self, cin=6, gdim=6, d=128, layers=4, heads=4, n_points=2048, hidden=256, p_drop=0.1, **_):
        super().__init__()
        self.n_points = n_points
        self.embed = nn.Sequential(nn.Linear(cin, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=p_drop, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.tf = nn.TransformerEncoder(layer, layers)
        self.pool_q = nn.Parameter(torch.randn(1, 4, d) * 0.02)  # 4 learned pooling queries
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(4 * d + gdim, hidden), nn.GELU(), nn.Dropout(p_drop),
                                  nn.Linear(hidden, 14))

    def forward(self, x, g):
        B, C, H, W = x.shape
        pts = x.reshape(B, C, H * W).transpose(1, 2)  # (B, N, C)
        n = min(self.n_points, H * W)
        idx = torch.stack([torch.randperm(H * W, device=x.device)[:n] for _ in range(B)])  # fresh subset + order
        pts = torch.gather(pts, 1, idx.unsqueeze(-1).expand(-1, -1, C))
        h = self.tf(self.embed(pts))
        q = self.pool_q.expand(B, -1, -1)
        pooled, _ = self.pool(q, h, h)
        f = torch.cat([pooled.reshape(B, -1), g], 1)
        return self.head(f).view(-1, 2, 7), None


def build(name: str, uv_scale=None, **kw):
    if name == "cnn_reg":
        return CNNReg(**kw)
    if name == "cnn_heat":
        return CNNHeat(uv_scale=uv_scale, **kw)
    if name == "pointtf":
        kw.pop("in_size", None); kw.pop("w", None)
        return PointTF(**kw)
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
