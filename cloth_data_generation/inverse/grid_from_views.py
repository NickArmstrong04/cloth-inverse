"""Turn a photo pair into the rung-1 model's input: an indexed 256x256 grid in the world frame.

For every base view of a `photo.py --stereo --gt` set (with SAM masks from segment.py and stereo
packages from `lift_stereo.py --sam`), builds a ladder of inputs, each one degradation further
from the oracle, so eval_grid.py can price every barrier separately:

  oracle   true grid, true velocity, true g                      (= the model's own number)
  novel    true grid, velocity channels and mean velocity zeroed (a photo has no velocity)
  hidden   visible vertices at their TRUE position, hidden ones filled (harmonic interpolation
           from the visible boundary), g true                     (cost of the hidden side)
  stereo   visible vertices at the STEREO depth of their pixel (oracle correspondences; the
           visible set is gated by the SAM mask and by stereo coverage), hidden filled, g true
  photo    as stereo, with g = centroid of the lifted grid         (everything photo-derived)

Camera model: the fixed calibrated camera. photo.py places the camera relative to the deformed
sheet (p = R (x - c) + (0, 0, -d)); the calibration a fixed camera would know is that pose in the
world frame, (R, t = -R c - (0, 0, d)), so a lifted camera-frame point maps back to the world as
x = R^T (p - t), drift included. The hidden fill is `--fill harmonic` (sparse Laplace solve,
exact) or `--fill sft` (fit_template.fit_view started from the stereo surface, slower).

  python grid_from_views.py --photos ../../cloth_results/photos/stereo20 [--out <photos>/grids]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import encode_forces  # noqa: E402
from eval_grid import grid_input, load_h5  # noqa: E402
from fit_template import pixel_rays  # noqa: E402


def harmonic_fill(X: np.ndarray, known: np.ndarray) -> np.ndarray:
    """(ny, nx, 3) grid, known (ny, nx) bool: unknown vertices solve the discrete Laplace equation
    with the known ones as boundary (exact sparse solve, 4-neighbour, Neumann at the sheet edge)."""
    ny, nx, _ = X.shape
    n = ny * nx
    idx = np.arange(n).reshape(ny, nx)
    rows, cols = [], []
    for a, b in [(idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])]:
        rows += [a.ravel(), b.ravel()]
        cols += [b.ravel(), a.ravel()]
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(A.sum(1)).ravel()) - A  # graph Laplacian
    k = known.ravel()
    u = ~k
    if not u.any():
        return X.copy()
    Luu = L[u][:, u].tocsc()
    Luk = L[u][:, k]
    out = X.reshape(n, 3).copy()
    rhs = -Luk @ out[k]
    sol = spla.splu(Luu).solve(rhs)
    out[u] = sol
    return out.reshape(ny, nx, 3)


def stereo_depth_map(pdir: str, stem: str, gt) -> tuple[np.ndarray | None, float]:
    """Per-pixel metric stereo depth of the base view, rebuilt from the lift_stereo package (pixel
    order = np.nonzero of the mask it was made with), scaled by the photometric scale (no oracle)."""
    vp = os.path.join(pdir, "views", f"{stem}_pts_stereo.npz")
    if not os.path.exists(vp):
        return None, 0.0
    z = np.load(vp)
    mask = gt["mask"] & (gt["depth"] > 0)
    if str(z["mask_source"]) == "sam":
        mp = os.path.join(pdir, "sam3", f"{stem}__fabric.png")
        mask &= np.asarray(Image.open(mp).convert("L")) > 127
    rr, cc = np.nonzero(mask)
    pts = z["pts"] + z["g"][:3]
    assert len(pts) == len(rr), f"{stem}: package / mask pixel count mismatch"
    depth = np.zeros(gt["depth"].shape, np.float32)
    depth[rr, cc] = -pts[:, 2] * float(z["scale_geo"])
    return depth, float(z["scale_geo"])


def cam_to_world(p_cam: np.ndarray, R: np.ndarray, d: float, c: np.ndarray) -> np.ndarray:
    """Inverse of photo.photograph with the fixed-camera calibration (R, t = -R c - (0,0,d))."""
    t = -R @ c - np.array([0, 0, d], np.float32)
    return (p_cam - t) @ R  # R^T (p - t) for row vectors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", nargs="+", required=True)
    ap.add_argument("--fill", choices=["harmonic", "sft"], default="harmonic")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="default <photos>/grids")
    args = ap.parse_args()
    for pdir in args.photos:
        out = args.out or os.path.join(pdir, "grids")
        os.makedirs(out, exist_ok=True)
        man = json.load(open(os.path.join(pdir, "manifest.json")))
        views = {os.path.splitext(v["png"])[0]: v for v in man["views"]}
        recs, cache = [], {}
        n = 0
        for stem, v in sorted(views.items()):
            if v.get("partner", -1) >= 0 and v["partner"] < v["view"]:
                continue  # partner views: the base view carries the pair
            if args.limit and n >= args.limit:
                break
            gt = np.load(os.path.join(pdir, f"{stem}_gt.npz"))
            if v["h5"] not in cache:
                cache[v["h5"]] = load_h5(v["h5"])
            pos, vel, fp = cache[v["h5"]]
            y = encode_forces(fp)
            ny, nx = pos.shape[:2]
            c = pos.reshape(-1, 3).mean(0).astype(np.float32)
            R, d = gt["R"].astype(np.float32), float(gt["d"])
            vis = gt["vis"].reshape(ny, nx)
            vpix = gt["vpix"].reshape(ny, nx, 2)
            depth, s_geo = stereo_depth_map(pdir, stem, gt)
            if depth is None:
                print(f"{stem}: no stereo package, skipped")
                continue
            n += 1
            t0 = time.time()
            # photo-visible: seen by the renderer AND lifted (inside the SAM mask with stereo depth)
            dvert = depth[vpix[..., 1], vpix[..., 0]]
            pvis = vis & (dvert > 0)
            rays = pixel_rays(vpix.reshape(-1, 2), int(gt["size"]), float(gt["fov_deg"])).reshape(ny, nx, 3)
            lifted_cam = rays * dvert[..., None]
            lifted = cam_to_world(lifted_cam.reshape(-1, 3), R, d, c).reshape(ny, nx, 3).astype(np.float32)
            # ladder
            X_hidden = harmonic_fill(pos, vis)  # true visible positions, hidden filled
            X_stereo = harmonic_fill(np.where(pvis[..., None], lifted, 0).astype(np.float32), pvis)
            if args.fill == "sft":
                raise SystemExit("--fill sft: not wired yet (fit_template.fit_view on the stereo surface)")
            variants = {
                "oracle": grid_input(pos, vel, 1, "oracle"),
                "novel": grid_input(pos, None, 1, "oracle"),
                "hidden": grid_input(X_hidden, None, 1, "override", g_override=c),
                "stereo": grid_input(X_stereo, None, 1, "override", g_override=c),
                "photo": grid_input(X_stereo, None, 1, "oracle"),  # g = centroid of the lifted grid
            }
            err = lambda X, m: float(np.linalg.norm((X - pos)[m], axis=-1).mean() * 100) if m.any() else float("nan")
            rec = dict(stem=stem, set=os.path.basename(os.path.normpath(pdir)), h5=v["h5"], scale_geo=s_geo,
                       visible_frac=float(vis.mean()), photo_visible_frac=float(pvis.mean()),
                       err_vis_stereo_cm=err(lifted, pvis), err_hidden_fill_true_cm=err(X_hidden, ~vis),
                       err_hidden_fill_stereo_cm=err(X_stereo, ~pvis), err_all_stereo_cm=err(X_stereo, np.ones_like(vis)),
                       drift_err_cm=float(np.linalg.norm(X_stereo.reshape(-1, 3).mean(0) - c) * 100),
                       drift_cm=float(np.linalg.norm(c) * 100), seconds=time.time() - t0)
            recs.append(rec)
            save = {"y": y, "vis": vis, "pvis": pvis, "stem": stem}
            for k, (x, g) in variants.items():
                save[f"X_{k}"], save[f"G_{k}"] = x, g
            np.savez_compressed(os.path.join(out, f"{stem}_grids.npz"), **save)
            print(f"{stem:34s} vis {rec['visible_frac']:.2f} (photo {rec['photo_visible_frac']:.2f}) | stereo visible err "
                  f"{rec['err_vis_stereo_cm']:5.2f} cm | hidden fill from truth {rec['err_hidden_fill_true_cm']:5.1f}, "
                  f"from stereo {rec['err_hidden_fill_stereo_cm']:5.1f} | all {rec['err_all_stereo_cm']:5.2f} | "
                  f"drift err {rec['drift_err_cm']:4.1f} of {rec['drift_cm']:4.0f} cm | {rec['seconds']:.1f}s")
        med = lambda k: float(np.nanmedian([r[k] for r in recs]))
        summary = {k: med(k) for k in ["visible_frac", "photo_visible_frac", "err_vis_stereo_cm", "err_hidden_fill_true_cm",
                                       "err_hidden_fill_stereo_cm", "err_all_stereo_cm", "drift_err_cm"]}
        summary["n"] = len(recs)
        print(os.path.basename(os.path.normpath(pdir)), json.dumps(summary))
        with open(os.path.join(out, "grids.json"), "w") as fh:
            json.dump({"args": vars(args), "records": recs, "summary": summary}, fh, indent=1)


if __name__ == "__main__":
    main()
