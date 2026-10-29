"""Rotation-invariant surface features for the material grid (recommendation 1).

By the fundamental theorem of surfaces a deformed sheet is determined up to a rigid
motion by its first fundamental form (in-plane strain) and second fundamental form
(curvature), both expressed in material coordinates. Feeding those instead of world
positions makes the model invariant to the unknown pose by construction, so rung 2
(unknown pose) costs the same as rung 1 for position. Directions are not invariant:
the model predicts them in the local tangent frame (e_u, e_v, n) of the sheet at the
predicted patch and they are mapped back to the world with that frame (equivariant).

Grid convention (data.py): x is (B, 3, H, W) centred positions, axis -2 = v, -1 = u.
Rest spacing h = 0.5 m / (N - 1) per axis (block-pooled grids keep the same extent).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SHEET_SIZE = 0.5


def _grad(x: torch.Tensor, hv: float, hu: float):
    """Central differences (one-sided at the borders) of a (B, C, H, W) field.
    Returns (d/dv, d/du)."""
    dv, du = torch.gradient(x, spacing=(hv, hu), dim=(-2, -1), edge_order=1)
    return dv, du


def intrinsic(x: torch.Tensor, vel: torch.Tensor | None = None, curv_scale: float = 50.0, strain_scale: float = 10.0):
    """x: (B, 3, H, W) positions (any rigid pose). Returns
    feats (B, 6[+3], H, W): [g_uu-1, g_vv-1, g_uv, b_uu, b_uv, b_vv (, v_local)] scaled to O(1),
    frames (B, H, W, 3, 3): columns (e_u, e_v, n) so that world = F @ local.
    Strain: metric tensor of the deformed sheet relative to the flat rest state (0 = unstretched).
    Curvature: normal components of the second derivatives, in 1/m, squashed with asinh."""
    B, _, H, W = x.shape
    hu, hv = SHEET_SIZE / (W - 1), SHEET_SIZE / (H - 1)
    tv, tu = _grad(x, hv, hu)  # (B,3,H,W) each
    g_uu = (tu * tu).sum(1) - 1.0
    g_vv = (tv * tv).sum(1) - 1.0
    g_uv = (tu * tv).sum(1)
    n = F.normalize(torch.cross(tu, tv, dim=1), dim=1)
    x_uv, x_uu = _grad(tu, hv, hu)
    x_vv, _ = _grad(tv, hv, hu)
    b_uu = (x_uu * n).sum(1)
    b_uv = (x_uv * n).sum(1)
    b_vv = (x_vv * n).sum(1)
    feats = [g_uu * strain_scale, g_vv * strain_scale, g_uv * strain_scale,
             torch.asinh(b_uu / curv_scale), torch.asinh(b_uv / curv_scale), torch.asinh(b_vv / curv_scale)]
    e1 = F.normalize(tu, dim=1)
    e2 = torch.cross(n, e1, dim=1)
    frames = torch.stack([e1, e2, n], dim=-1).permute(0, 2, 3, 1, 4)  # (B,H,W,3,3) columns e1,e2,n
    if vel is not None:  # express the velocity field in the local frame too
        vl = torch.einsum("bhwij,bihw->bjhw", frames, vel)
        feats.extend(vl.unbind(1))
    return torch.stack(feats, 1), frames


def to_local_field(a: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """World vector field (B, 3, H, W) -> components in each cell's local frame (F^T a)."""
    return torch.einsum("bhwij,bihw->bjhw", frames, a)


def sample_frames(frames: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample the frame field at material coordinates uv (B, K, 2) in [0, 1]
    (u along W, v along H; cell i sits at u = i / (W - 1), as in data.force_field_map).
    Returns (B, K, 3, 3). Differentiable in uv."""
    B, H, W, _, _ = frames.shape
    f = frames.reshape(B, H, W, 9).permute(0, 3, 1, 2)  # (B,9,H,W)
    grid = (uv.clamp(0, 1) * 2 - 1).unsqueeze(1)  # (B,1,K,2) as (x=u, y=v)
    s = F.grid_sample(f, grid, mode="bilinear", align_corners=True)  # (B,9,1,K)
    return s[:, :, 0].permute(0, 2, 1).reshape(B, -1, 3, 3)


def local_to_world(d_local: torch.Tensor, uv: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """d_local (B, K, 3) in the tangent frame at uv -> world direction (B, K, 3), unit length."""
    Fk = sample_frames(frames, uv)
    return F.normalize(torch.einsum("bkij,bkj->bki", Fk, d_local), dim=-1)


def world_to_local(d_world: torch.Tensor, uv: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    Fk = sample_frames(frames, uv)
    return F.normalize(torch.einsum("bkji,bkj->bki", Fk, d_world), dim=-1)


def shade(img: torch.Tensor, extent: float, light=(0.3, 0.3, 0.9)) -> torch.Tensor:
    """Lambertian shading of a rendered (depth, mask) image (B, 2, S, S): returns (B, 3, S, S)
    [depth, mask, shade]. Normals come from the depth gradient over the metric pixel size, so a
    pretrained vision backbone sees something closer to a photograph than a raw depth map."""
    depth, mask = img[:, :1], img[:, 1:2]
    S = img.shape[-1]
    px = extent / S
    dzdy, dzdx = torch.gradient(depth, spacing=(px, px), dim=(-2, -1), edge_order=1)
    n = F.normalize(torch.cat([-dzdx, -dzdy, torch.ones_like(depth)], 1), dim=1)
    l = torch.tensor(light, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    l = l / l.norm()
    sh = (n * l).sum(1, keepdim=True).clamp_min(0) * mask
    return torch.cat([depth, mask, sh], 1)
