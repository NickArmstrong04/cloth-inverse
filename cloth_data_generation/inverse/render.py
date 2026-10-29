"""Observation-ladder rung 4a: a single orthographic view of the sheet.

Given centred vertex positions (B, N, 3) already expressed in the camera frame
(camera on +z looking down -z), rasterise a depth image with a z-buffer and
report which vertices are visible from that view. Any per-vertex feature (e.g.
the dense force-field target) can be splatted into the same image so that the
aux supervision lives in image space too. Pure torch, batched, runs on the GPU.

Orthographic projection: image x <- world x, image y <- world y, both over a
fixed metric window of ``extent`` metres (the sheet is 0.5 m), so the image
carries absolute scale; a monocular photo would not, and ``--view-norm`` in
train.py rescales by the visible extent to mimic that.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def render(pts: torch.Tensor, feats: torch.Tensor | None = None, size: int = 224, extent: float = 0.6,
           splat: int = 1):
    """pts: (B, N, 3) camera-frame positions. feats: (B, N, C) or None.
    Returns img (B, 2 + C, size, size): [depth (m, 0 where empty), mask, feats...],
    and vis (B, N) bool: vertex is the closest to the camera at its pixel."""
    B, N, _ = pts.shape
    dev = pts.device
    s = size / extent
    px = ((pts[..., 0] + extent / 2) * s).long().clamp(0, size - 1)
    py = ((pts[..., 1] + extent / 2) * s).long().clamp(0, size - 1)
    pix = py * size + px  # (B, N)
    z = pts[..., 2]
    zbuf = torch.full((B, size * size), -1e9, device=dev, dtype=z.dtype)
    zbuf.scatter_reduce_(1, pix, z, reduce="amax", include_self=True)
    zb = torch.gather(zbuf, 1, pix)
    vis = z >= zb - 1e-6  # (B, N)
    hit = zbuf > -1e8
    depth = torch.where(hit, zbuf, torch.zeros_like(zbuf))
    chans = [depth, hit.to(z.dtype)]
    if feats is not None:
        C = feats.shape[-1]
        fimg = torch.zeros((B, size * size, C), device=dev, dtype=feats.dtype)
        # write the visible vertices' features (last writer wins among equals, all visible so fine)
        idx = pix.unsqueeze(-1).expand(-1, -1, C)
        fimg.scatter_(1, idx, torch.where(vis.unsqueeze(-1), feats, torch.zeros_like(feats)))
        # keep only pixels whose writer was visible: re-scatter a visibility flag
        chans.extend(fimg.unbind(-1))
    img = torch.stack(chans, 1).view(B, -1, size, size)
    if splat > 1:  # dilate to close pin holes between neighbouring vertices
        d = F.max_pool2d(img[:, :1], splat, 1, splat // 2)
        m = F.max_pool2d(img[:, 1:2], splat, 1, splat // 2)
        img = torch.cat([d, m, img[:, 2:]], 1)
    return img, vis


def view_batch(x: torch.Tensor, g: torch.Tensor, y: torch.Tensor, a: torch.Tensor | None, R: torch.Tensor,
               size: int = 224, extent: float = 0.6, splat: int = 1, norm: bool = False):
    """Rotate a grid batch into camera frames R (B,3,3) and render it.
    x: (B, C, H, W) with positions in channels 0:3; a: (B, 3, H, W) force field or None.
    Returns img (B, 2[+3], S, S), g_cam (B, gdim), y_cam (B, 2, 7), a_img or None,
    vis (B, H*W) and pts_cam (B, H*W, 3) for point-cloud models."""
    B, C, H, W = x.shape
    nvec = C // 3
    xr = x.clone()
    for k in range(nvec):
        xr[:, 3 * k:3 * k + 3] = torch.einsum("bij,bjhw->bihw", R, x[:, 3 * k:3 * k + 3])
    gr = g.clone()
    for k in range(g.shape[1] // 3):
        gr[:, 3 * k:3 * k + 3] = torch.einsum("bij,bj->bi", R, g[:, 3 * k:3 * k + 3])
    yr = y.clone()
    yr[:, :, 2:5] = torch.einsum("bij,bkj->bki", R, y[:, :, 2:5])
    pts = xr[:, :3].reshape(B, 3, H * W).transpose(1, 2)
    feats = None
    ar = None
    if a is not None:
        ar = torch.einsum("bij,bjhw->bihw", R, a)  # the force field is a vector field: rotate it too
        feats = ar.reshape(B, 3, H * W).transpose(1, 2)
    if norm:  # monocular-style: no absolute scale; fit the visible cloud to the window
        ext = (pts[..., :2].amax(1) - pts[..., :2].amin(1)).amax(-1).clamp_min(1e-3)  # (B,)
        pts = pts * (0.8 * extent / ext)[:, None, None]
    img, vis = render(pts, feats, size, extent, splat)
    a_img = img[:, 2:5] if a is not None else None
    return img[:, :2], gr, yr, a_img, vis, pts
