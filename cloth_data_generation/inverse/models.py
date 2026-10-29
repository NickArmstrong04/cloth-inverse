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


def aux_field_loss(pred, target, m0=1.0):
    """MSE between a predicted force-field map and the dense target, both
    (B, 3, h, w) (the target is average-pooled to the prediction's size),
    divided by ``m0`` = the dataset's mean squared target per cell, so that
    ~1.0 = predicting zero everywhere. A constant normaliser (not per sample):
    a view may hide the whole patch, leaving an all-zero target."""
    t = F.adaptive_avg_pool2d(target, pred.shape[-2:]) if target.shape[-2:] != pred.shape[-2:] else target
    return ((pred.float() - t) ** 2).mean() / m0


class CNNReg(nn.Module):
    """Global-pool + MLP, predicts (2, 7) directly (normalized targets).
    With ``aux=True`` a light decoder on the mid-resolution map also predicts the
    dense force field (3, h, w); forward returns its loss when a target is given."""

    def __init__(self, w=32, hidden=256, p_drop=0.2, in_size=128, cin=6, gdim=6, aux=False, **_):
        super().__init__()
        self.enc = Encoder(cin, w, in_size)
        self.head = nn.Sequential(
            nn.Linear(self.enc.c_out + gdim, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, 14))
        cm = self.enc.c_mid
        self.aux = (nn.Sequential(nn.Conv2d(cm, cm, 3, 1, 1), nn.GroupNorm(8, cm), nn.GELU(), nn.Conv2d(cm, 3, 1))
                    if aux else None)
        self.aux_m0 = 1.0  # set by train.py: mean squared target per cell

    def forward(self, x, g, a=None):
        mid, out = self.enc(x)
        f = torch.cat([out.mean((-1, -2)), g], 1)
        aux = aux_field_loss(self.aux(mid), a, self.aux_m0) if (self.aux is not None and a is not None) else None
        return self.head(f).view(-1, 2, 7), aux


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

    def forward(self, x, g, a=None):
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

    def __init__(self, cin=6, gdim=6, d=128, layers=4, heads=None, n_points=2048, hidden=256, p_drop=0.1,
                 aux=False, **_):
        super().__init__()
        heads = heads or max(4, d // 32)
        self.n_points = n_points
        self.aux = nn.Linear(d, 3) if aux else None  # per-point force-field head
        self.aux_m0 = 1.0
        self.embed = nn.Sequential(nn.Linear(cin, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=p_drop, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.tf = nn.TransformerEncoder(layer, layers)
        self.pool_q = nn.Parameter(torch.randn(1, 4, d) * 0.02)  # 4 learned pooling queries
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(4 * d + gdim, hidden), nn.GELU(), nn.Dropout(p_drop),
                                  nn.Linear(hidden, 14))

    def forward(self, x, g, a=None, vis=None):
        """x: (B, C, H, W) grid (points are H*W). vis: optional (B, H*W) bool, sample only
        visible points (rung 4a). a: optional (B, 3, H, W) dense target for the aux head."""
        B, C, H, W = x.shape
        N = H * W
        pts = x.reshape(B, C, N).transpose(1, 2)  # (B, N, C)
        n = min(self.n_points, N)
        if vis is None:
            idx = torch.stack([torch.randperm(N, device=x.device)[:n] for _ in range(B)])  # fresh subset + order
        else:
            w = vis.float() + 1e-6  # visible points (with replacement only if fewer than n are visible)
            idx = torch.stack([torch.multinomial(w[b], n, replacement=bool(vis[b].sum() < n)) for b in range(B)])
        pts = torch.gather(pts, 1, idx.unsqueeze(-1).expand(-1, -1, C))
        h = self.tf(self.embed(pts))
        q = self.pool_q.expand(B, -1, -1)
        pooled, _ = self.pool(q, h, h)
        f = torch.cat([pooled.reshape(B, -1), g], 1)
        aux = None
        if self.aux is not None and a is not None:
            t = torch.gather(a.reshape(B, 3, N).transpose(1, 2), 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            p = self.aux(h).float()
            aux = ((p - t) ** 2).mean() / self.aux_m0
        return self.head(f).view(-1, 2, 7), aux


class DinoReg(nn.Module):
    """Pretrained-feature model: a frozen DINOv2 ViT (facebookresearch/dinov2 via
    torch.hub) on an image of the observation, with a small head. Input channels
    (positions of the material grid, or depth+mask of a rendered view) go through
    a learned 1x1 adapter to 3 channels at ``size`` px (multiple of 14). Head:
    attention pooling over patch tokens + CLS + g -> (2, 7). Optional aux head
    predicts the force field per patch (3, size/14, size/14).
    Pre-download on a node with internet (weights cache in ~/.cache/torch/hub):
      python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"
    """

    def __init__(self, cin=6, gdim=6, arch="dinov2_vits14", size=224, hidden=256, p_drop=0.1, finetune=False,
                 aux=False, **_):
        super().__init__()
        self.size = size
        self.backbone = torch.hub.load("facebookresearch/dinov2", arch, trust_repo=True, skip_validation=True)
        d = self.backbone.embed_dim
        self.finetune = finetune
        for p in self.backbone.parameters():
            p.requires_grad = finetune
        self.adapter = nn.Conv2d(cin, 3, 1)
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.pool_q = nn.Parameter(torch.randn(1, 4, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 8, batch_first=True)
        self.head = nn.Sequential(nn.Linear(5 * d + gdim, hidden), nn.GELU(), nn.Dropout(p_drop),
                                  nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p_drop),
                                  nn.Linear(hidden, 14))
        self.aux = nn.Linear(d, 3) if aux else None
        self.aux_m0 = 1.0

    def forward(self, x, g, a=None):
        B = x.shape[0]
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        img = (self.adapter(x) - self.im_mean) / self.im_std
        with torch.set_grad_enabled(self.finetune):
            feats = self.backbone.forward_features(img)
        cls, tok = feats["x_norm_clstoken"], feats["x_norm_patchtokens"]  # (B, d), (B, P, d)
        q = self.pool_q.expand(B, -1, -1)
        pooled, _ = self.pool(q, tok, tok)
        f = torch.cat([pooled.reshape(B, -1), cls, g], 1)
        aux = None
        if self.aux is not None and a is not None:
            hw = self.size // 14
            pmap = self.aux(tok).transpose(1, 2).reshape(B, 3, hw, hw)
            aux = aux_field_loss(pmap, a, self.aux_m0)
        return self.head(f).view(-1, 2, 7), aux


def build(name: str, uv_scale=None, **kw):
    if name == "cnn_reg":
        return CNNReg(**kw)
    if name == "cnn_heat":
        return CNNHeat(uv_scale=uv_scale, **kw)
    if name == "pointtf":
        kw.pop("in_size", None); kw.pop("w", None)
        return PointTF(**kw)
    if name == "dino_reg":
        kw.pop("in_size", None); kw.pop("w", None)
        return DinoReg(**kw)
    raise ValueError(name)


def _d4_targets(y_n, uv_mean, uv_std):
    """The 8 D4 relabelings of the (normalized) target (u, v); other fields unchanged.
    Returns (8, B, 2, 7)."""
    u = y_n[..., 0] * uv_std[0] + uv_mean[0]
    v = y_n[..., 1] * uv_std[1] + uv_mean[1]
    outs = []
    for (uu, vv) in ((u, v), (1 - u, v), (u, 1 - v), (1 - u, 1 - v), (v, u), (1 - v, u), (v, 1 - u), (1 - v, 1 - u)):
        t = y_n.clone()
        t[..., 0] = (uu - uv_mean[0]) / uv_std[0]
        t[..., 1] = (vv - uv_mean[1]) / uv_std[1]
        outs.append(t)
    return torch.stack(outs)


def pair_loss(pred_n, y_n, w_dir=1.0, sym_uv=None):
    """Permutation-invariant loss on normalized targets. pred_n, y_n: (B,2,7).
    MSE on (u,v,logmag,dur); (1 - cos) on direction. With ``sym_uv=(mean, std)``
    of the (u, v) normalization, the target may also be relabeled by any D4
    symmetry of the sheet (unknown-pose training)."""
    def one(p, t):
        mse = ((p[..., [0, 1, 5, 6]] - t[..., [0, 1, 5, 6]]) ** 2).mean(-1)
        d = F.normalize(p[..., 2:5], dim=-1)
        cos = (d * t[..., 2:5]).sum(-1)
        return mse + w_dir * (1.0 - cos)  # (B,2)
    targets = _d4_targets(y_n, *sym_uv) if sym_uv is not None else y_n.unsqueeze(0)
    losses = []
    for t in targets:
        losses.append(one(pred_n, t).sum(-1))
        losses.append(one(pred_n, t.flip(1)).sum(-1))
    return torch.stack(losses).min(0).values.mean()
