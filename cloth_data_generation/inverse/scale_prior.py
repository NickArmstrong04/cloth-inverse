"""Known-size scale check for the photo chain: the sheet is a 0.5 m square of near-inextensible paper,
so the 3D distance between visible vertices L cells apart, divided by their material distance
L * h, is a scale-free statistic of the sheet (chord / arc, ~1 for a gently creased sheet). On a
lifted grid it is that statistic times the depth-scale error, so prior / measured estimates the
scale correction without any oracle.

  python scale_prior.py --photos <photo.py set with views/ packages> [--h5 <split dir for the prior>]

Prints, per base view, the true correction (oracle scale / photometric scale, from the lift_stereo
package), the measured span ratio with oracle and decoded correspondences, and the true sheet's
span ratio; with --h5, the prior (median true span ratio over visible vertices of the split's
grids from random cameras is not needed: all vertices are used) for each L.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_template import pixel_rays  # noqa: E402
from grid_from_views import span_ratio, stereo_depth_map  # noqa: E402
from eval_grid import load_h5  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", nargs="*", default=[])
    ap.add_argument("--h5", default=None, help="split dir: median span ratio of the true grids (the prior)")
    ap.add_argument("--L", nargs="*", type=int, default=[8, 16, 32])
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--remap", nargs=2, default=None, metavar=("OLD", "NEW"), help="rewrite source h5 paths (renamed splits)")
    args = ap.parse_args()
    if args.h5:
        files = sorted(glob.glob(os.path.join(args.h5, "*.h5")))[: args.limit]
        for L in args.L:
            r = [span_ratio(load_h5(f)[0], None, L)[0] for f in files]
            print(f"prior L={L}: median {np.median(r):.4f}  p10 {np.percentile(r, 10):.4f}  p90 {np.percentile(r, 90):.4f}  (n={len(r)})")
    rows = []
    for pdir in args.photos:
        man = json.load(open(os.path.join(pdir, "manifest.json")))
        for v in man["views"]:
            stem = os.path.splitext(v["png"])[0]
            if v["partner"] < v["view"]:
                continue
            gt = np.load(os.path.join(pdir, f"{stem}_gt.npz"))
            depth, s_geo = stereo_depth_map(pdir, stem, gt)
            if depth is None:
                continue
            z = np.load(os.path.join(pdir, "views", f"{stem}_pts_stereo.npz"))
            need = float(z["scale_to_oracle"]) / s_geo
            pos = load_h5(v["h5"].replace(*args.remap) if args.remap else v["h5"])[0]
            ny, nx = pos.shape[:2]
            vpix = gt["vpix"].reshape(ny, nx, 2)
            size, fov = int(gt["size"]), float(gt["fov_deg"])
            dv = depth[vpix[..., 1], vpix[..., 0]]
            pvis = gt["vis"].reshape(ny, nx) & (dv > 0)
            P = pixel_rays(vpix.reshape(-1, 2), size, fov).reshape(ny, nx, 3) * dv[..., None]
            row = dict(stem=stem, need=need)
            cp = os.path.join(pdir, "decode", f"{stem}_corr.npz")
            Pd = has = None
            if os.path.exists(cp):
                cz = np.load(cp)
                est, has = cz["vpix_est"].reshape(ny, nx, 2), cz["has_est"].reshape(ny, nx)
                ei = np.clip(np.round(est).astype(int), 0, size - 1)
                de = depth[ei[..., 1], ei[..., 0]]
                has = has & (de > 0)
                Pd = pixel_rays(est.reshape(-1, 2), size, fov).reshape(ny, nx, 3) * de[..., None]
            for L in args.L:
                row[f"true{L}"] = span_ratio(pos, pvis, L)[0]
                row[f"orc{L}"] = span_ratio(P, pvis, L)[0]
                row[f"dec{L}"] = span_ratio(Pd, has, L)[0] if Pd is not None else float("nan")
            rows.append(row)
            print(stem, " ".join(f"{k} {v:.3f}" for k, v in row.items() if k != "stem"))
    if rows:
        need = np.array([r["need"] for r in rows])
        bad = np.abs(need - 1) > 0.1
        print(f"\n{len(rows)} views, {bad.sum()} with |scale error| > 10%")
        for L in args.L:
            for src in ("orc", "dec"):
                m = np.array([r[f"{src}{L}"] for r in rows])
                t = np.array([r[f"true{L}"] for r in rows])
                prior = np.nanmedian(t)
                k = prior / m  # predicted correction with the prior
                print(f"L={L:2d} {src}: true-ratio median {prior:.3f} (spread p10-p90 {np.nanpercentile(t,10):.3f}-{np.nanpercentile(t,90):.3f}) | "
                      f"measured/(true*need^-1) median {np.nanmedian(m * need / t):.3f} | "
                      f"corr(log k, log need) {np.corrcoef(np.log(k[np.isfinite(k)]), np.log(need[np.isfinite(k)]))[0,1]:.2f} | "
                      f"|k-need| median {np.nanmedian(np.abs(k - need)):.3f}, on bad views {np.nanmedian(np.abs(k[bad] - need[bad])) if bad.any() else float('nan'):.3f}")


if __name__ == "__main__":
    main()
