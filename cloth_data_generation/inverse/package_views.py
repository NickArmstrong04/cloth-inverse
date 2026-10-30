"""Rung 4b, step 3: SAM mask x oracle depth -> rung-4a point-model input.

For each view with a `_gt.npz` (photo.py --gt) and a SAM mask (segment.py), unproject the
pixels inside the mask with the oracle depth through the same pinhole model photo.py used,
centre the cloud, and write it with the force labels in the camera frame. The same package
is written from the oracle mask, so feeding both to one checkpoint measures the cost of
segmentation alone (monocular depth comes later).

Package <stem>_pts_<source>.npz: pts (N, 3) float32 camera-frame positions, centred;
g (6,) = camera-frame centroid of the cloud before centring + zeros (no velocity, matches
gdim=6); y (2, 7) labels as data.encode_forces, directions rotated into the camera frame as
render.view_batch does; R (3, 3), d, size, fov_deg, n_oracle_visible, source.
PointTF takes x (B, 3, H, W) and samples H*W points, so a loader reshapes pts to (3, N, 1).

  python package_views.py --photos ../../cloth_results/photos/cloth256 --prompt "a piece of cloth"
  python package_views.py --photos ../../cloth_results/photos/check --oracle-only --roundtrip
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]


def unproject(mask: np.ndarray, depth: np.ndarray, fov_deg: float) -> np.ndarray:
    """Pixels (rows down) inside mask -> camera-frame points (N, 3), camera at the origin
    looking down -z. Inverse of photo.photograph: px = floor((proj + extent/2) * size/extent)
    on the un-flipped image, so pixel centres map back to proj = (px + 0.5) * extent/size - extent/2."""
    S = mask.shape[0]
    extent = 2 * math.tan(math.radians(fov_deg) / 2)
    rows, cols = np.nonzero(mask & (depth > 0))
    py = S - 1 - rows  # undo the vertical flip of the written image
    z = depth[rows, cols]
    x = ((cols + 0.5) * extent / S - extent / 2) * z
    y = ((py + 0.5) * extent / S - extent / 2) * z
    return np.stack([x, y, -z], -1).astype(np.float32)


def reproject(pts: np.ndarray, size: int, fov_deg: float) -> np.ndarray:
    """Camera-frame points -> bool mask (rows down), for the round-trip check."""
    extent = 2 * math.tan(math.radians(fov_deg) / 2)
    sc = size / extent
    proj = pts[:, :2] / (-pts[:, 2:3])
    px = np.clip(((proj[:, 0] + extent / 2) * sc).astype(int), 0, size - 1)
    py = np.clip(((proj[:, 1] + extent / 2) * sc).astype(int), 0, size - 1)
    m = np.zeros((size, size), bool)
    m[size - 1 - py, px] = True
    return m


def package(gt_path: str, mask: np.ndarray, source: str, out_dir: str) -> dict:
    g = np.load(gt_path)
    depth, R, d = g["depth"], g["R"], float(g["d"])
    size, fov = int(g["size"]), float(g["fov_deg"])
    pts = unproject(mask, depth, fov)
    centroid = pts.mean(0) if len(pts) else np.zeros(3, np.float32)
    from data import encode_forces  # lazy: data.py needs h5py, which the SAM/VGGT venv (lift.py) lacks
    fp = json.loads(str(g["force_params"]))
    y = encode_forces(fp) if fp else np.zeros((2, 7), np.float32)
    y[:, 2:5] = y[:, 2:5] @ R.T  # directions into the camera frame (view_batch: R @ dir)
    stem = os.path.basename(gt_path)[: -len("_gt.npz")]
    path = os.path.join(out_dir, f"{stem}_pts_{source}.npz")
    np.savez_compressed(path, pts=pts - centroid, g=np.concatenate([centroid, np.zeros(3, np.float32)]),
                        y=y, R=R, d=d, size=size, fov_deg=fov, n_oracle_visible=int(g["vis"].sum()),
                        source=source, material=str(g["material"]))
    return {"stem": stem, "source": source, "N": int(len(pts)), "path": path, "size": size, "fov": fov}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", nargs="+", required=True)
    ap.add_argument("--prompt", default=None, help="SAM prompt whose masks (sam3/<stem>__<slug>.png) to package")
    ap.add_argument("--oracle-only", action="store_true")
    ap.add_argument("--out", default=None, help="default <photos>/views")
    ap.add_argument("--roundtrip", action="store_true", help="re-project the oracle package and compare masks")
    args = ap.parse_args()
    for pdir in args.photos:
        out = args.out or os.path.join(pdir, "views")
        os.makedirs(out, exist_ok=True)
        recs = []
        for gp in sorted(glob.glob(os.path.join(pdir, "*_gt.npz"))):
            g = np.load(gp)
            recs.append(package(gp, g["mask"], "oracle", out))
            if args.roundtrip:
                z = np.load(recs[-1]["path"])
                m = reproject(z["pts"] + z["g"][:3], int(z["size"]), float(z["fov_deg"]))
                gm = g["mask"] & (g["depth"] > 0)
                recs[-1]["roundtrip_mismatch_px"] = int((m != gm).sum())
            if not args.oracle_only and args.prompt:
                stem = os.path.basename(gp)[: -len("_gt.npz")]
                mp = os.path.join(pdir, "sam3", f"{stem}__{slug(args.prompt)}.png")
                if os.path.exists(mp):
                    sm = np.asarray(Image.open(mp).convert("L")) > 127
                    recs.append(package(gp, sm, "sam3", out))
                else:
                    print(f"  no SAM mask for {stem} ({mp})")
        with open(os.path.join(out, "index.json"), "w") as fh:
            json.dump({"prompt": args.prompt, "records": recs}, fh, indent=1)
        for r in recs:
            extra = f"  roundtrip mismatch {r['roundtrip_mismatch_px']} px" if "roundtrip_mismatch_px" in r else ""
            print(f"{r['stem']:40s} {r['source']:6s} N={r['N']:7d}{extra}")
        print(f"{len(recs)} packages -> {out}")


if __name__ == "__main__":
    main()
