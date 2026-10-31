"""Make a packaged split look like it came through the photo chain (Phase C of the rung-4b -> rung-1
plan): for every sample, K random fixed-calibrated cameras; the vertices the renderer sees keep
their position up to a stereo-like depth error applied in the camera frame; the hidden ones are
filled harmonically from the visible boundary (grid_from_views.harmonic_fill); velocity is zeroed.
Writes a new split root with K files per sample (`<stem>_v<k>.h5`, same keys as the source) and a
matching dataset_manifest.json, so train.py / eval_grid.py read it unchanged:

  python degrade_split.py --root $S3 --out $S3_photo --views 2            # cluster, GPU node
  python degrade_split.py --root ~/cloth/ds256_const_test16 --out ~/cloth/test16_photo --views 2   # flat dir

Depth-error model (per view, camera frame, from lift_stereo on 32 test pairs: scale 0.5% median /
1.9% p90, affine residual 0.8 cm, depth+tilt residual 0.5 cm, visible-vertex error 2.3 cm):
  d' = d (1 + s) + b + cx x + cy y + smooth(sigma_px) * a_lf + white * a_hf
with s ~ N(0, scale), b ~ N(0, offset), cx, cy ~ N(0, tilt) per unit image-plane coordinate,
a smooth zero-mean field of amplitude a_lf (Gaussian-filtered white noise on the image) and white
per-vertex noise a_hf. `--noise none` keeps the visible vertices exact (prices visibility alone).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import sys
import time

import h5py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_template import pixel_rays  # noqa: E402
from grid_from_views import cam_to_world, harmonic_fill  # noqa: E402
from photo import photograph, sample_cameras  # noqa: E402

PRESETS = {  # scale, offset m, tilt m per image-plane unit, a_lf m, sigma_px, a_hf m
    "none": dict(scale=0, offset=0, tilt=0, a_lf=0, sigma_px=8, a_hf=0),
    "measured": dict(scale=0.01, offset=0.005, tilt=0.01, a_lf=0.012, sigma_px=12, a_hf=0.004),
    "mild": dict(scale=0.005, offset=0.003, tilt=0.005, a_lf=0.006, sigma_px=12, a_hf=0.002),
}


def perturb_depth(depth_v: np.ndarray, vpix: np.ndarray, size: int, fov: float, p: dict, rng: np.random.Generator):
    """Stereo-like error on the per-vertex depth (N,) of one view; vpix (N, 2) col,row."""
    ext = math.tan(math.radians(fov) / 2)
    x = ((vpix[:, 0] + 0.5) / size - 0.5) * 2 * ext
    y = (0.5 - (vpix[:, 1] + 0.5) / size) * 2 * ext
    s, b = rng.normal(0, p["scale"]), rng.normal(0, p["offset"])
    cx, cy = rng.normal(0, p["tilt"], 2)
    d = depth_v * (1 + s) + b + cx * x + cy * y
    if p["a_lf"] > 0:
        field = gaussian_filter(rng.normal(size=(size, size)), p["sigma_px"], mode="reflect")
        field *= p["a_lf"] / max(field.std(), 1e-9)
        d = d + field[np.clip(vpix[:, 1], 0, size - 1), np.clip(vpix[:, 0], 0, size - 1)]
    if p["a_hf"] > 0:
        d = d + rng.normal(0, p["a_hf"], len(d))
    return d.astype(np.float32)


def degrade_file(src: str, dst_stem: str, out_dir: str, K: int, size: int, fov: float, p: dict, rng, dev):
    with h5py.File(src, "r") as h:
        pos = h["final_grid"][()].astype(np.float32)
        extra = {k: h[k][()] for k in h.keys() if k not in ("final_grid", "final_vel")}
        attrs = dict(h.attrs)
    ny, nx = pos.shape[:2]
    c = pos.reshape(-1, 3).mean(0).astype(np.float32)
    grid = torch.from_numpy(pos).to(dev)
    R, d = sample_cameras(grid.reshape(-1, 3) - grid.reshape(-1, 3).mean(0), K)
    _, gt = photograph(grid, R, d, None, size, fov)
    vis = gt["vis"].cpu().numpy().reshape(K, ny, nx)
    vpix = gt["vpix"].cpu().numpy().reshape(K, ny, nx, 2)
    depth = gt["depth"].cpu().numpy()  # (K, S, S) metres, rows down
    names, fracs = [], []
    for k in range(K):
        Rk, dk = R[k].cpu().numpy().astype(np.float32), float(d[k])
        vp = vpix[k].reshape(-1, 2)
        dv = depth[k][np.clip(vp[:, 1], 0, size - 1), np.clip(vp[:, 0], 0, size - 1)]
        pv = vis[k].reshape(-1) & (dv > 0)
        dv = perturb_depth(dv, vp, size, fov, p, rng)
        rays = pixel_rays(vp, size, fov)
        lifted = cam_to_world(rays * dv[:, None], Rk, dk, c).reshape(ny, nx, 3).astype(np.float32)
        X = harmonic_fill(np.where(pv.reshape(ny, nx)[..., None], lifted, 0).astype(np.float32), pv.reshape(ny, nx))
        name = f"{dst_stem}_v{k}.h5"
        with h5py.File(os.path.join(out_dir, name), "w") as h:
            h.create_dataset("final_grid", data=X)
            h.create_dataset("final_vel", data=np.zeros_like(X))
            for kk, v in extra.items():
                h.create_dataset(kk, data=v)
            for kk, v in attrs.items():
                h.attrs[kk] = v
            h.attrs["degraded_from"] = os.path.basename(src)
            h.attrs["visible_frac"] = float(pv.mean())
            h.attrs["view_R"] = Rk.ravel()
            h.attrs["view_d"] = dk
            h.attrs["err_all_cm"] = float(np.linalg.norm(X - pos, axis=-1).mean() * 100)
        names.append(name)
        fracs.append(float(pv.mean()))
    return names, fracs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="split root (train/val/test) or a flat dir of h5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--noise", choices=list(PRESETS), default="measured")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    p = PRESETS[args.noise]
    splits = [s for s in ("train", "val", "test") if os.path.isdir(os.path.join(args.root, s))]
    flat = not splits
    man_src = os.path.join(args.root, "dataset_manifest.json")
    manifest = json.load(open(man_src)) if os.path.exists(man_src) else None
    t0 = time.time()
    for split in (splits or [""]):
        src_dir = os.path.join(args.root, split)
        out_dir = os.path.join(args.out, split)
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(src_dir, "*.h5")))[: args.limit or None]
        entries = {e["file"]: e for e in manifest["splits"][split]} if manifest and split else {}
        new_entries, fr, n = [], [], 0
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            names, fracs = degrade_file(f, stem, out_dir, args.views, args.size, args.fov, p, rng, dev)
            fr += fracs
            for name in names:
                e = dict(entries.get(os.path.basename(f), {"file": os.path.basename(f)}))
                e["file"] = name
                e["degraded_from"] = os.path.basename(f)
                new_entries.append(e)
            n += 1
            if n % 50 == 0 or n == len(files):
                print(f"  {split or 'flat'}: {n}/{len(files)} files, visible {np.mean(fr):.2f}, {time.time() - t0:.0f}s", flush=True)
        if manifest and split:
            manifest["splits"][split] = new_entries
    if manifest:
        manifest["degrade"] = vars(args) | {"preset": p}
        with open(os.path.join(args.out, "dataset_manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
    with open(os.path.join(args.out, "degrade.json"), "w") as fh:
        json.dump(vars(args) | {"preset": p, "flat": flat}, fh, indent=1)
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
