"""Observation-ladder rung 4b, step 1: synthetic "photos" of a deformed sheet.

A photo carries less than the simulator state in a specific way: one viewpoint, perspective
projection, self-occlusion (back faces and the insides of folds are hidden), no absolute
scale, and only pixels. This module renders exactly that from ``final_grid`` and nothing
more realistic: no cast shadows, no sensor noise, flat background. Photographic polish is
a later, separate step (domain randomisation / Blender) if SAM turns out to need it.

Pipeline per sample, batched over views on the GPU:
  centre grid -> random SO(3) camera (data.random_rotation) at distance d on +z, looking
  down -z -> pinhole projection to the image plane -> z-buffer splat (render.render) of
  per-vertex (u, v, normal) -> nearest-to-camera hole filling -> printed texture looked up
  through (u, v) -> two-sided Lambertian shading with one light -> RGB PNG.

Normals are the true mesh normals (cross product of the grid's u and v tangents), rotated
into the camera frame, rather than depth-gradient normals (geometry.shade): under
perspective the depth image's pixel spacing is not metric, so that shortcut is wrong here.

Usage (inside the WSL venv, from this directory):
  python photo.py --h5 ~/cloth/smoke_const/b256 --out ../../cloth_results/photos/b256 --views 4
  python photo.py --check            # flat / edge-on / folded sanity renders + numbers
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import random_rotation  # noqa: E402
from render import render  # noqa: E402

SHEET_SIZE = 0.5  # m, the simulated sheet (see data.SHEET_SIZE)


# ----------------------------------------------------------------------------- geometry

def grid_normals(x: torch.Tensor) -> torch.Tensor:
    """Per-vertex unit normals of a grid (B, ny, nx, 3): cross(d/du, d/dv), so a flat sheet
    in the xy plane has normal +z. Finite differences along axis1 (u = i) and axis0 (v = j)."""
    dv, du = torch.gradient(x, dim=(1, 2))
    n = torch.cross(du, dv, dim=-1)
    return F.normalize(n, dim=-1, eps=1e-9)


def grid_uv(ny: int, nx: int, dev) -> torch.Tensor:
    """Material coordinates (ny, nx, 2) = (u, v) in [0, 1], matching p = j*nx + i."""
    v, u = torch.meshgrid(torch.linspace(0, 1, ny, device=dev), torch.linspace(0, 1, nx, device=dev),
                          indexing="ij")
    return torch.stack([u, v], -1)


def sample_cameras(x: torch.Tensor, n: int, dist=(0.9, 1.4), min_extent=0.1):
    """Draw n random cameras for a centred cloud x (N, 3). Returns R (n,3,3), d (n,).
    A camera is a rotation R (world -> camera, camera on +z looking down -z) and a distance d
    along the camera's z axis. Views whose projected footprint is too small (edge-on) are
    redrawn so no picture is a degenerate line. Uses torch's global RNG (seed it once)."""
    dev = x.device
    Rs, ds = [], []
    tries = 0
    while len(Rs) < n and tries < 20:
        tries += 1
        m = 4 * n
        R = random_rotation(m, dev)
        d = dist[0] + (dist[1] - dist[0]) * torch.rand(m, device=dev)
        p = torch.einsum("bij,nj->bni", R, x)
        p[..., 2] -= d[:, None]
        proj = p[..., :2] / (-p[..., 2:3])
        ext = (proj.amax(1) - proj.amin(1)).amin(-1)  # the smaller side of the footprint
        ok = ext > min_extent  # image-plane units at focal length 1: ~10 cm at 1 m
        for k in torch.nonzero(ok).flatten().tolist():
            if len(Rs) < n:
                Rs.append(R[k])
                ds.append(d[k])
    if len(Rs) < n:
        raise RuntimeError("could not find non-degenerate views")
    return torch.stack(Rs), torch.stack(ds)


# ----------------------------------------------------------------------------- rendering

def fill_nearest(img: torch.Tensor, splat: int) -> torch.Tensor:
    """Close pin holes between neighbouring vertices: every pixel takes all channels from the
    nearest-to-camera pixel in its splat x splat window (channel 0 = z key, 1 = mask)."""
    if splat <= 1:
        return img
    B, C, S, _ = img.shape
    key = torch.where(img[:, 1:2] > 0, img[:, :1], torch.full_like(img[:, :1], -1e9))
    _, idx = F.max_pool2d(key, splat, 1, splat // 2, return_indices=True)
    flat = img.view(B, C, S * S)
    out = torch.gather(flat, 2, idx.view(B, 1, S * S).expand(-1, C, -1)).view(B, C, S, S)
    out[:, :1] = torch.where(out[:, 1:2] > 0, out[:, :1], torch.zeros_like(out[:, :1]))
    return out


def checker_texture(T: int = 512, cells: int = 8, dev="cpu") -> torch.Tensor:
    """A printed sheet: two-tone checker with a red corner marker at (u, v) = (0, 0) and a dark
    stripe along u = 0, the asymmetric cue ROADMAP.md asks for. Returns (3, T, T) in [0, 1]."""
    yy, xx = torch.meshgrid(torch.arange(T, device=dev), torch.arange(T, device=dev), indexing="ij")
    chk = ((xx * cells // T + yy * cells // T) % 2).float()
    c0 = torch.tensor([0.92, 0.92, 0.88], device=dev).view(3, 1, 1)
    c1 = torch.tensor([0.35, 0.55, 0.80], device=dev).view(3, 1, 1)
    tex = c0 * (1 - chk) + c1 * chk
    corner = (xx < T // cells) & (yy < T // cells)
    tex = torch.where(corner, torch.tensor([0.85, 0.15, 0.15], device=dev).view(3, 1, 1).expand_as(tex), tex)
    stripe = xx < T // (4 * cells)
    tex = torch.where(stripe, torch.tensor([0.1, 0.1, 0.1], device=dev).view(3, 1, 1).expand_as(tex), tex)
    return tex


def load_texture(path: str | None, dev) -> torch.Tensor:
    if path is None:
        return checker_texture(dev=dev)
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(im).permute(2, 0, 1).to(dev)


def photograph(grid: torch.Tensor, R: torch.Tensor, d: torch.Tensor, texture: torch.Tensor | None,
               size: int = 512, fov_deg: float = 40.0, splat: int = 3, bg: float = 0.5,
               light=(0.3, 0.4, 1.0), ambient: float = 0.35, plain=(0.75, 0.75, 0.72), vis_tol: float = 0.004):
    """Render B views of one grid (ny, nx, 3) with cameras R (B,3,3), d (B,).
    Returns rgb (B, 3, S, S) in [0, 1] and gt dict {mask (B,S,S), depth (B,S,S) metres,
    uv (B,2,S,S), vis (B, ny*nx)} for a later --gt dump; only rgb is written now."""
    dev = grid.device
    ny, nx, _ = grid.shape
    B = R.shape[0]
    x = grid - grid.reshape(-1, 3).mean(0)
    n = grid_normals(x[None])[0].reshape(-1, 3)  # (N, 3) world frame
    uv = grid_uv(ny, nx, dev).reshape(-1, 2)
    xf = x.reshape(-1, 3)
    p = torch.einsum("bij,nj->bni", R, xf)  # camera frame
    p[..., 2] -= d[:, None]
    nc = torch.einsum("bij,nj->bni", R, n)
    depth = -p[..., 2]
    proj = p[..., :2] / depth[..., None]  # pinhole: image-plane coords at focal length 1
    extent = 2 * math.tan(math.radians(fov_deg) / 2)  # image-plane width for the FOV
    pts = torch.cat([proj, p[..., 2:3]], -1)  # z key: larger (less negative) = nearer
    feats = torch.cat([uv[None].expand(B, -1, -1), nc], -1)  # (B, N, 5)
    img, vis0 = render(pts, None, size, extent, splat=1)  # depth + mask + exact per-pixel winners
    # features of the winning vertex per pixel. render()'s own feature path scatters all
    # vertices (zeros for hidden ones) in arbitrary order, which leaves black speckle where
    # several vertices share a pixel; here the z-buffer winner is picked by index instead.
    sc = size / extent
    px = ((proj[..., 0] + extent / 2) * sc).long().clamp(0, size - 1)
    py = ((proj[..., 1] + extent / 2) * sc).long().clamp(0, size - 1)
    pix = py * size + px
    N = pix.shape[1]
    vid = torch.arange(N, device=dev)[None].expand(B, -1)
    win = torch.full((B, size * size), -1, device=dev, dtype=torch.long)
    win.scatter_reduce_(1, pix, torch.where(vis0, vid, torch.full_like(vid, -1)), reduce="amax", include_self=True)
    fimg = torch.gather(feats, 1, win.clamp_min(0).unsqueeze(-1).expand(-1, -1, feats.shape[-1]))
    fimg = fimg * (win >= 0).unsqueeze(-1)
    img = torch.cat([img, fimg.transpose(1, 2).reshape(B, -1, size, size)], 1)
    img = fill_nearest(img, splat)
    mask = img[:, 1] > 0
    z = img[:, 0]
    # visibility against the filled z-buffer: a vertex is seen if nothing in its pixel's
    # window is nearer than it by more than vis_tol (metres); render()'s own flag is
    # pixel-exact and leaks through folds when the two layers land on different pixels
    zb = torch.gather(z.reshape(B, -1), 1, pix)
    vis = pts[..., 2] >= zb - vis_tol
    uv_img = img[:, 2:4]
    nrm = F.normalize(img[:, 4:7], dim=1, eps=1e-9)
    l = torch.tensor(light, device=dev, dtype=torch.float32)
    l = (l / l.norm()).view(1, 3, 1, 1)
    lam = (nrm * l).sum(1, keepdim=True).abs()  # two-sided: a thin sheet shows its back
    shade = ambient + (1 - ambient) * lam
    if texture is not None:
        g = (uv_img * 2 - 1).permute(0, 2, 3, 1)  # grid_sample wants (B, S, S, 2) with x=u, y=v
        albedo = F.grid_sample(texture[None].expand(B, -1, -1, -1), g, mode="bilinear",
                               padding_mode="border", align_corners=False)
    else:
        albedo = torch.tensor(plain, device=dev).view(1, 3, 1, 1).expand(B, 3, size, size)
    rgb = albedo * shade
    m = mask[:, None].float()
    rgb = rgb * m + bg * (1 - m)
    rgb = rgb.flip(-2)  # image y up -> rows down, so the picture is not mirrored vertically
    gt = {"mask": mask.flip(-2), "depth": torch.where(mask, -z, torch.zeros_like(z)).flip(-2),
          "uv": uv_img.flip(-2), "vis": vis}
    return rgb.clamp(0, 1), gt


def to_png(rgb: torch.Tensor, path: str) -> None:
    from PIL import Image
    arr = (rgb.permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ----------------------------------------------------------------------------- data

def list_h5(spec: list[str]) -> list[str]:
    out = []
    for s in spec:
        s = os.path.expanduser(s)
        if os.path.isdir(s):
            out += sorted(glob.glob(os.path.join(s, "*.h5")))
        else:
            out += sorted(glob.glob(s))
    return out


def load_grid(path: str):
    with h5py.File(path, "r") as h:
        grid = h["final_grid"][()].astype(np.float32)
        fp = h["force_params"][()]
        fp = json.loads(fp.decode() if isinstance(fp, bytes) else fp)
        cfg = h["sim_config"][()]
        cfg = json.loads(cfg.decode() if isinstance(cfg, bytes) else cfg)
    return grid, fp, cfg.get("sheet", {}).get("material", "cloth")


# ----------------------------------------------------------------------------- sanity

def check(args, dev):
    """Flat sheet frontal -> full square, all vertices visible; edge-on -> near-empty;
    folded in half -> about half the vertices visible."""
    n = 256  # the dataset resolution: vertex spacing ~2 mm, ~1.4 px at 512 px / 40 deg / 1 m
    v, u = torch.meshgrid(torch.linspace(-0.25, 0.25, n), torch.linspace(-0.25, 0.25, n), indexing="ij")
    flat = torch.stack([u, v, torch.zeros_like(u)], -1).to(dev)
    folded = flat.clone()
    right = folded[..., 0] > 0
    folded[..., 0] = torch.where(right, -folded[..., 0], folded[..., 0])  # fold the right half over the left
    folded[..., 2] = torch.where(right, torch.full_like(folded[..., 2], 0.01), folded[..., 2])
    I = torch.eye(3, device=dev)
    c, s = math.cos(math.radians(89)), math.sin(math.radians(89))
    Rx = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], device=dev)
    d = torch.tensor([1.0], device=dev)
    tex = checker_texture(dev=dev)
    os.makedirs(args.out, exist_ok=True)
    expect_frac = (SHEET_SIZE / (2 * math.tan(math.radians(args.fov) / 2) * 1.0)) ** 2
    for name, g, R in [("flat_frontal", flat, I), ("flat_edge_on", flat, Rx), ("folded_frontal", folded, I),
                       ("folded_back", folded, torch.diag(torch.tensor([1.0, -1.0, -1.0], device=dev)))]:
        rgb, gt = photograph(g, R[None], d, tex, args.size, args.fov, splat=args.splat, bg=args.bg)
        frac = gt["mask"].float().mean().item()
        visf = gt["vis"].float().mean().item()
        print(f"{name:16s} mask frac {frac:.3f} (flat frontal expects ~{expect_frac:.3f})  "
              f"visible vertices {visf:.3f}")
        to_png(rgb[0], os.path.join(args.out, f"check_{name}.png"))
    print("wrote check_*.png to", args.out)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", nargs="*", default=[], help="h5 files, globs or directories")
    ap.add_argument("--out", default="photos")
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fov", type=float, default=40.0, help="horizontal field of view, degrees")
    ap.add_argument("--splat", type=int, default=3, help="hole-filling window in pixels (odd)")
    ap.add_argument("--bg", type=float, default=0.5, help="background grey level")
    ap.add_argument("--plain", action="store_true", help="untextured sheet")
    ap.add_argument("--texture", default=None, help="PNG to print on the sheet (default: checker + corner mark)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="render only the first N files")
    ap.add_argument("--check", action="store_true", help="sanity renders of synthetic sheets")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.check:
        check(args, dev)
        return
    files = list_h5(args.h5)
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit("no h5 files found")
    os.makedirs(args.out, exist_ok=True)
    tex = None if args.plain else load_texture(args.texture, dev)
    torch.manual_seed(args.seed)
    manifest = {"args": vars(args), "fov_deg": args.fov, "camera": "R maps world (centred grid) to camera; "
                "camera at origin looking down -z; sheet centre at (0, 0, -d)", "views": []}
    t0 = time.time()
    for f in files:
        grid_np, fp, material = load_grid(f)
        grid = torch.from_numpy(grid_np).to(dev)
        R, d = sample_cameras(grid.reshape(-1, 3) - grid.reshape(-1, 3).mean(0), args.views)
        rgb, gt = photograph(grid, R, d, tex, args.size, args.fov, args.splat, args.bg)
        stem = os.path.splitext(os.path.basename(f))[0]
        for k in range(args.views):
            name = f"{stem}_v{k}.png"
            to_png(rgb[k], os.path.join(args.out, name))
            manifest["views"].append({"png": name, "h5": f, "material": material, "view": k,
                                      "R": R[k].tolist(), "d": float(d[k]),
                                      "visible_frac": float(gt["vis"][k].float().mean()),
                                      "mask_frac": float(gt["mask"][k].float().mean()),
                                      "force_params": fp})
        print(f"{stem}: {args.views} views, visible {gt['vis'].float().mean(1).tolist()}")
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"{len(files)} samples x {args.views} views in {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
