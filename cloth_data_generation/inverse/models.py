"""Inverse models: ResNet encoder on the material grid with either a direct
regression head ("cnn_reg"), a heatmap head for (u, v) ("cnn_heat"), or a dense
force-field head decoded by peak finding ("cnn_field"); a point transformer
("pointtf"); pretrained vision ("dino_reg") and point-cloud ("sonata_reg") backbones.

Options shared by the grid models:
  intrinsic=True  rotation-invariant inputs (strain + curvature in material coordinates,
                  geometry.py) and directions predicted in the local tangent frame.
  flow=True       an extra conditional flow-matching head that samples K candidate
                  force pairs from the pooled features (multimodal posterior), for
                  re-simulation selection (select_resim.py)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry import intrinsic, local_to_world, to_local_field, world_to_local


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


# ----------------------------------------------------------------------------- intrinsic inputs

class IntrinsicInput(nn.Module):
    """Turns raw centred positions (B, 3[+3], H, W) into invariant features and keeps the
    local frames for the direction heads. A learnable per-channel affine replaces the
    dataset-level input standardisation that train.py applies to raw positions."""

    def __init__(self, with_vel: bool):
        super().__init__()
        self.cin = 9 if with_vel else 6
        self.scale = nn.Parameter(torch.ones(1, self.cin, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, self.cin, 1, 1))
        self.frames = None

    def forward(self, x):
        vel = x[:, 3:6] if (self.cin == 9 and x.shape[1] >= 6) else None
        f, self.frames = intrinsic(x[:, :3].float(), vel.float() if vel is not None else None)
        return (f * self.scale + self.shift).to(x.dtype)


def _uv_from_norm(pred, uv_scale):
    m, s = uv_scale
    return pred[..., :2] * s + m


# ----------------------------------------------------------------------------- flow-matching head

class FlowHead(nn.Module):
    """Conditional flow matching from N(0, I) to the normalised (2, 7) target, conditioned on
    the encoder's pooled features. The target set is symmetric (two unordered forces, and the
    D4 relabelings when the pose is unknown): each training step pairs the noise sample with
    the nearest equivalent target, which keeps the learned flow single-valued.
    Sampling: Euler, ``steps`` steps, K candidates per item."""

    def __init__(self, cdim, hidden=256, layers=3, steps=32):
        super().__init__()
        self.steps = steps
        self.cond = nn.Sequential(nn.Linear(cdim, hidden), nn.GELU())
        self.tproj = nn.Linear(1, hidden)
        mods = [nn.Linear(14 + hidden, hidden), nn.GELU()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.GELU()]
        mods += [nn.Linear(hidden, 14)]
        self.net = nn.Sequential(*mods)

    def v(self, z, t, c):
        h = self.cond(c) + self.tproj(t.view(-1, 1))
        return self.net(torch.cat([z, h], -1))

    def loss(self, c, targets):
        """c: (B, cdim) features; targets: (S, B, 2, 7) the S equivalent normalised targets."""
        S, B = targets.shape[:2]
        z1 = targets.reshape(S, B, 14).float()
        e = torch.randn(B, 14, device=c.device)
        k = ((z1 - e.unsqueeze(0)) ** 2).sum(-1).argmin(0)  # nearest equivalent target per item
        z1 = z1[k, torch.arange(B, device=c.device)]
        t = torch.rand(B, device=c.device)
        zt = (1 - t[:, None]) * e + t[:, None] * z1
        return ((self.v(zt, t, c.float()) - (z1 - e)) ** 2).mean()

    @torch.no_grad()
    def sample(self, c, K):
        B = c.shape[0]
        cK = c.float().repeat_interleave(K, 0)
        z = torch.randn(B * K, 14, device=c.device)
        dt = 1.0 / self.steps
        for i in range(self.steps):
            t = torch.full((B * K,), i * dt, device=c.device)
            z = z + dt * self.v(z, t, cK)
        return z.view(B, K, 2, 7)


class GridModel(nn.Module):
    """Shared machinery for the grid models: optional intrinsic input, optional flow head,
    direction frame handling."""

    def _setup(self, cin, intrinsic_in, flow, cdim, uv_scale, flow_steps):
        self.intrinsic = intrinsic_in
        self.pre = IntrinsicInput(with_vel=cin >= 6) if intrinsic_in else None
        self.uv_scale = uv_scale
        self.flow = None  # built by the subclass once the feature width is known
        self.flow_targets = None  # set per step by train.py: (S, B, 2, 7) equivalent targets
        self._feat = None
        return self.pre.cin if intrinsic_in else cin

    def _inputs(self, x):
        return self.pre(x) if self.pre is not None else x

    def _dirs_to_world(self, pred):
        """pred (B, 2, 7) with local-frame directions -> world directions (intrinsic only)."""
        if self.pre is None:
            return pred
        uv = _uv_from_norm(pred.float(), self.uv_scale)
        d = local_to_world(pred[..., 2:5].float(), uv, self.pre.frames)
        return torch.cat([pred[..., :2], d.to(pred.dtype), pred[..., 5:]], -1)

    def _aux_target(self, a):
        return to_local_field(a, self.pre.frames) if (self.pre is not None and a is not None) else a

    def _flow_loss(self, feat):
        """Flow-matching loss on the targets train.py handed over for this batch (consumed once);
        stored in ``last_flow_loss`` and added by train.py with weight 1 (independent of --aux-w)."""
        self.last_flow_loss = None
        if self.flow is None or self.flow_targets is None:
            return None
        t, self.flow_targets = self.flow_targets, None
        if self.pre is not None:  # flow generates local-frame directions
            S, B = t.shape[:2]
            uv = _uv_from_norm(t.reshape(S * B, 2, 7), self.uv_scale)
            fr = self.pre.frames.repeat(S, 1, 1, 1, 1)
            d = world_to_local(t.reshape(S * B, 2, 7)[..., 2:5], uv, fr).view(S, B, 2, 3)
            t = torch.cat([t[..., :2], d, t[..., 5:]], -1)
        self.last_flow_loss = self.flow.loss(feat, t)
        return self.last_flow_loss

    @torch.no_grad()
    def sample(self, x, g, K):
        """K candidate (2, 7) normalised predictions per item, world directions: (B, K, 2, 7)."""
        self.forward(x, g)
        z = self.flow.sample(self._feat, K)
        if self.pre is not None:
            B = z.shape[0]
            out = []
            for k in range(K):
                out.append(self._dirs_to_world(z[:, k]))
            z = torch.stack(out, 1)
        z[..., 2:5] = F.normalize(z[..., 2:5], dim=-1)
        return z


class CNNReg(GridModel):
    """Global-pool + MLP, predicts (2, 7) directly (normalized targets).
    With ``aux=True`` a light decoder on the mid-resolution map also predicts the
    dense force field (3, h, w); forward returns its loss when a target is given."""

    def __init__(self, w=32, hidden=256, p_drop=0.2, in_size=128, cin=6, gdim=6, aux=False, intrinsic=False,
                 flow=False, uv_scale=None, flow_steps=32, **_):
        super().__init__()
        enc_in = self._setup(cin, intrinsic, flow, 0, uv_scale, flow_steps)
        self.enc = Encoder(enc_in, w, in_size)
        self.gdim = gdim
        cdim = self.enc.c_out + gdim
        if flow:
            self.flow = FlowHead(cdim, steps=flow_steps)
        self.head = nn.Sequential(
            nn.Linear(cdim, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, 14))
        cm = self.enc.c_mid
        self.aux = (nn.Sequential(nn.Conv2d(cm, cm, 3, 1, 1), nn.GroupNorm(8, cm), nn.GELU(), nn.Conv2d(cm, 3, 1))
                    if aux else None)
        self.aux_m0 = 1.0  # set by train.py: mean squared target per cell

    def forward(self, x, g, a=None):
        if self.pre is not None:
            g = torch.zeros_like(g)  # world-frame globals carry pose; invariant model ignores them
        mid, out = self.enc(self._inputs(x))
        f = torch.cat([out.mean((-1, -2)), g], 1)
        self._feat = f
        a = self._aux_target(a)
        aux = aux_field_loss(self.aux(mid), a, self.aux_m0) if (self.aux is not None and a is not None) else None
        self._flow_loss(f)
        pred = self._dirs_to_world(self.head(f).view(-1, 2, 7))
        return pred, aux


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


class CNNField(GridModel):
    """Recommendation 2: the dense force field is the PRIMARY output. A light decoder
    upsamples the mid map (x4) and predicts (i) a presence map trained against the Gaussian
    patch weights of both forces (BCE) and (ii) the force-field map trained against the dense
    target (MSE, as the aux loss). The (2, 7) answer is decoded from the maps: the two
    strongest non-maximum-suppressed peaks, a differentiable soft-argmax in a window around
    each for (u, v), the field averaged over the window for the direction, and log-magnitude
    / duration from an MLP on pooled + window-attended features. The permutation and D4
    ambiguities never enter the dense losses; pair_loss on the decoded answer is kept as a
    light second term. Forward returns (pred, dense_loss) so train.py treats the dense loss
    as the aux loss (weight --aux-w, forced to >= 1 for this model)."""

    def __init__(self, w=32, hidden=256, p_drop=0.2, in_size=128, cin=6, gdim=6, uv_scale=None, intrinsic=False,
                 flow=False, flow_steps=32, win=3, heat_w=1.0, **_):
        super().__init__()
        enc_in = self._setup(cin, intrinsic, flow, 0, uv_scale, flow_steps)
        self.enc = Encoder(enc_in, w, in_size)
        cm, cd = self.enc.c_mid, max(32, self.enc.c_mid // 2)
        self.dec = nn.Sequential(
            nn.Conv2d(cm, cd, 3, 1, 1), nn.GroupNorm(8, cd), nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(cd, cd, 3, 1, 1), nn.GroupNorm(8, cd), nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(cd, cd, 3, 1, 1), nn.GroupNorm(8, cd), nn.GELU())
        self.heat = nn.Conv2d(cd, 1, 1)
        self.field = nn.Conv2d(cd, 3, 1)
        cdim = self.enc.c_out + gdim
        if flow:
            self.flow = FlowHead(cdim, steps=flow_steps)
        self.reg = nn.Sequential(nn.Linear(cdim + cd, hidden), nn.GELU(), nn.Dropout(p_drop), nn.Linear(hidden, 2))
        self.win, self.heat_w = win, heat_w
        self.aux_m0 = 1.0

    def _peaks(self, p):
        """Two NMS peaks of p (B, 1, h, w) -> indices (B, 2) flat."""
        B, _, h, w = p.shape
        k = 2 * self.win + 1
        nms = torch.where(F.max_pool2d(p, k, 1, k // 2) == p, p, torch.zeros_like(p)).view(B, -1)
        return nms.topk(2, -1).indices

    def _window(self, idx, h, w):
        """Window cell indices (B, 2, k*k) and a validity mask around each peak."""
        r = self.win
        py, px = idx // w, idx % w
        dy, dx = torch.meshgrid(torch.arange(-r, r + 1, device=idx.device), torch.arange(-r, r + 1, device=idx.device),
                                indexing="ij")
        yy = py[..., None] + dy.reshape(-1)
        xx = px[..., None] + dx.reshape(-1)
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        return (yy.clamp(0, h - 1) * w + xx.clamp(0, w - 1)), ok, yy, xx

    def forward(self, x, g, a=None):
        if self.pre is not None:
            g = torch.zeros_like(g)
        mid, out = self.enc(self._inputs(x))
        d = self.dec(mid)
        B, cd, h, w = d.shape
        logits, field = self.heat(d), self.field(d)
        p = torch.sigmoid(logits.float())
        idx = self._peaks(p.detach())
        cells, ok, yy, xx = self._window(idx, h, w)
        wts = torch.gather(p.view(B, 1, -1).expand(-1, 2, -1), 2, cells) * ok  # (B,2,k2)
        wts = wts / wts.sum(-1, keepdim=True).clamp_min(1e-6)
        u = (wts * xx.float()).sum(-1) / (w - 1)
        v = (wts * yy.float()).sum(-1) / (h - 1)
        fl = field.float().view(B, 3, -1)
        dirs = torch.stack([torch.gather(fl[:, c].unsqueeze(1).expand(-1, 2, -1), 2, cells) for c in range(3)], -1)
        dirs = (wts.unsqueeze(-1) * dirs).sum(2)  # (B,2,3), local frame if intrinsic
        dv = d.float().view(B, cd, -1)
        att = torch.stack([torch.gather(dv[:, c].unsqueeze(1).expand(-1, 2, -1), 2, cells) for c in range(cd)], -1)
        att = (wts.unsqueeze(-1) * att).sum(2)  # (B,2,cd)
        gf = torch.cat([out.mean((-1, -2)), g], 1)
        self._feat = gf
        rest = self.reg(torch.cat([gf.unsqueeze(1).expand(-1, 2, -1).float(), att], -1))  # (B,2,2) logmag, dur
        m, s = self.uv_scale
        uvn = (torch.stack([u, v], -1) - m) / s
        pred = self._dirs_to_world(torch.cat([uvn, dirs, rest], -1))
        dense = None
        if a is not None:
            al = self._aux_target(a)
            at = F.adaptive_avg_pool2d(al, (h, w)) if al.shape[-2:] != (h, w) else al
            wt = F.adaptive_max_pool2d(a.norm(dim=1, keepdim=True), (h, w)).clamp(0, 1)
            dense = (self.heat_w * F.binary_cross_entropy_with_logits(logits.float(), wt)
                     + ((field.float() - at) ** 2).mean() / self.aux_m0)
        self._flow_loss(gf)
        return pred, dense


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
    """Pretrained-feature model: a DINOv2 / DINOv3 ViT (torch.hub) on an image of the
    observation, with a small head. Input channels (positions of the material grid, or
    depth+mask(+shade) of a rendered view) go through a learned 1x1 adapter to 3 channels at
    ``size`` px (multiple of the patch size). Head: attention pooling over patch tokens + CLS
    + g -> (2, 7). Optional aux head predicts the force field per patch.
    ``finetune=True`` unfreezes the backbone (recommendation 5: frozen features lost by 0.7-2.4 cm).
    repo: "facebookresearch/dinov2" (arch dinov2_vits14, patch 14) or "facebookresearch/dinov3"
    (arch dinov3_vits16, patch 16; weights are gated: pass ``weights=<local .pth>``).
    Pre-download on a node with internet (cache in ~/.cache/torch/hub):
      python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"
    """

    def __init__(self, cin=6, gdim=6, arch="dinov2_vits14", size=224, hidden=256, p_drop=0.1, finetune=False,
                 aux=False, repo="facebookresearch/dinov2", weights=None, **_):
        super().__init__()
        self.size = size
        kw = {"weights": weights} if weights else {}
        self.backbone = torch.hub.load(repo, arch, trust_repo=True, skip_validation=True, **kw)
        d = self.backbone.embed_dim
        self.patch = getattr(self.backbone, "patch_size", 16 if "dinov3" in arch else 14)
        assert size % self.patch == 0, f"--view-size must be a multiple of the patch size {self.patch}"
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
        with torch.set_grad_enabled(self.finetune and torch.is_grad_enabled()):
            feats = self.backbone.forward_features(img)
        cls, tok = feats["x_norm_clstoken"], feats["x_norm_patchtokens"]  # (B, d), (B, P, d)
        q = self.pool_q.expand(B, -1, -1)
        pooled, _ = self.pool(q, tok, tok)
        f = torch.cat([pooled.reshape(B, -1), cls, g], 1)
        aux = None
        if self.aux is not None and a is not None:
            hw = self.size // self.patch
            pmap = self.aux(tok).transpose(1, 2).reshape(B, 3, hw, hw)
            aux = aux_field_loss(pmap, a, self.aux_m0)
        return self.head(f).view(-1, 2, 7), aux


class SonataReg(nn.Module):
    """Recommendation 4: Sonata (CVPR 2025), a self-supervised Point Transformer V3 encoder,
    fine-tuned on the point cloud (rungs 3 / 4a). UNTESTED here: needs the sonata package
    (cluster/setup_sonata.sh) and the checkpoint cached on the login node.
    Input per point: coord (m) + the 6 features the checkpoint expects (colour = 0, normal);
    normals come from the material grid (cross product of the grid tangents), which a real
    point cloud would estimate locally. Points are voxelised at ``grid_size`` (duplicates
    dropped), encoded, hierarchy unrolled back to the kept points, attention-pooled -> (2, 7).
    ``vis`` restricts to the visible points (rung 4a)."""

    def __init__(self, cin=6, gdim=6, hidden=256, p_drop=0.1, n_points=2048, grid_size=0.004, ckpt="sonata",
                 finetune=True, aux=False, **_):
        super().__init__()
        import sonata  # noqa: F401  (facebookresearch/sonata, package mode)
        self.backbone = sonata.model.load(ckpt, repo_id="facebook/sonata") if not ckpt.endswith(".pth") \
            else sonata.model.load(ckpt)
        for p in self.backbone.parameters():
            p.requires_grad = finetune
        self.n_points, self.grid_size = n_points, grid_size
        d = sum(self.backbone.enc_channels) if hasattr(self.backbone, "enc_channels") else 1232
        self.proj = nn.LazyLinear(hidden)
        self.pool_q = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.pool = nn.MultiheadAttention(hidden, 8, batch_first=True)
        self.head = nn.Sequential(nn.Linear(4 * hidden + gdim, hidden), nn.GELU(), nn.Dropout(p_drop),
                                  nn.Linear(hidden, 14))
        self.aux = nn.LazyLinear(3) if aux else None
        self.aux_m0 = 1.0

    @staticmethod
    def _unroll(point):
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point

    def forward(self, x, g, a=None, vis=None):
        B, C, H, W = x.shape
        N = H * W
        pos = x[:, :3].float()
        _, frames = intrinsic(pos)
        normal = frames[..., 2].reshape(B, N, 3)
        pts = pos.reshape(B, 3, N).transpose(1, 2)
        n = min(self.n_points, N)
        if vis is None:
            idx = torch.stack([torch.randperm(N, device=x.device)[:n] for _ in range(B)])
        else:
            w = vis.float() + 1e-6
            idx = torch.stack([torch.multinomial(w[b], n, replacement=bool(vis[b].sum() < n)) for b in range(B)])
        coords, feats, batch, keep_idx = [], [], [], []
        for b in range(B):
            c = pts[b, idx[b]]
            gc = torch.floor(c / self.grid_size).long()
            key = gc[:, 0] * 73856093 ^ gc[:, 1] * 19349663 ^ gc[:, 2] * 83492791
            vox, inv = torch.unique(key, return_inverse=True)
            rep = torch.zeros(len(vox), dtype=torch.long, device=x.device)
            rep.scatter_(0, inv, torch.arange(len(key), device=x.device))  # one representative point per voxel
            sel = idx[b][rep]
            coords.append(pts[b, sel]); feats.append(torch.cat([torch.zeros(len(sel), 3, device=x.device), normal[b, sel]], 1))
            batch.append(torch.full((len(sel),), b, device=x.device, dtype=torch.long)); keep_idx.append(sel)
        point = {"coord": torch.cat(coords), "feat": torch.cat(feats), "batch": torch.cat(batch),
                 "grid_size": self.grid_size}
        point = self._unroll(self.backbone(point))
        feat = self.proj(point.feat.float())  # (sum n_b, hidden)
        bidx = point.batch if hasattr(point, "batch") else point["batch"]
        lens = torch.bincount(bidx, minlength=B)
        M = int(lens.max())
        dense = torch.zeros(B, M, feat.shape[-1], device=x.device)
        mask = torch.ones(B, M, dtype=torch.bool, device=x.device)
        pos_in = torch.cat([torch.arange(int(l), device=x.device) for l in lens])
        dense[bidx, pos_in] = feat
        mask[bidx, pos_in] = False
        q = self.pool_q.expand(B, -1, -1)
        pooled, _ = self.pool(q, dense, dense, key_padding_mask=mask)
        f = torch.cat([pooled.reshape(B, -1), g], 1)
        aux = None
        if self.aux is not None and a is not None:
            tgt = torch.cat([a.reshape(B, 3, N).transpose(1, 2)[b, k] for b, k in enumerate(keep_idx)])
            aux = ((self.aux(feat) - tgt) ** 2).mean() / self.aux_m0
        return self.head(f).view(-1, 2, 7), aux


def build(name: str, uv_scale=None, **kw):
    if name == "cnn_reg":
        return CNNReg(uv_scale=uv_scale, **kw)
    if name == "cnn_heat":
        return CNNHeat(uv_scale=uv_scale, **kw)
    if name == "cnn_field":
        return CNNField(uv_scale=uv_scale, **kw)
    if name == "pointtf":
        kw.pop("in_size", None); kw.pop("w", None)
        return PointTF(**kw)
    if name == "dino_reg":
        kw.pop("in_size", None); kw.pop("w", None)
        return DinoReg(**kw)
    if name == "sonata_reg":
        kw.pop("in_size", None); kw.pop("w", None)
        return SonataReg(**kw)
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


def equivalent_targets(y_n, sym_uv=None):
    """All equivalent normalised targets of a batch: (S, B, 2, 7) with S = 2 (slot order) or
    16 (slot order x D4). Used by the flow head."""
    base = _d4_targets(y_n, *sym_uv) if sym_uv is not None else y_n.unsqueeze(0)
    return torch.cat([base, base.flip(2)], 0)


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
