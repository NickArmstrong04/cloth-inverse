"""Shape-from-template, step 1 (oracle correspondences): recover the sheet mesh from a photo's
mask + per-pixel material correspondences + known intrinsics + the known rest grid, with NO depth
input. Tests whether known knowledge (node count, rest edge lengths, inextensibility) can replace
monocular depth prediction (VGGT: 4 cm median depth error as a smooth inclination ramp).

Unknowns: every vertex of the (pooled) rest grid in the camera frame, free 3D points. A vertex the
oracle marks visible is held to its pixel (photo.py --gt `vpix`) by a soft reprojection term with a
half-pixel tolerance (the correspondence is quantised to the pixel). Energy: (0) reprojection; (1) isometry, every grid edge (4-neighbour + both diagonals) keeps its rest length;
(2) bending, a weak Laplacian; (3) silhouette, no vertex may project outside the mask (distance
transform of the mask, bilinearly sampled). Scale and inclination are fixed by (1) under the
perspective camera (isometric shape-from-template is well posed), so the score is an ABSOLUTE
camera-frame error against the true grid (R @ (grid - centroid) + (0, 0, -d)), no alignment.

Oracle inputs used: mask, vis, vpix, R / d (scoring only), the h5 grid (scoring only), fov. Not
used: depth. Step 2 replaces vis / vpix by estimated correspondences.

  python fit_template.py --photos ../../cloth_results/photos/cloth256 --limit 4 --pool 4
Outputs <photos>/fit/<stem>_fit.npz (X (ny,nx,3) fitted, X_true, vis) and fit/scores.json.
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
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from photo import load_grid  # noqa: E402


def pixel_rays(vpix: np.ndarray, size: int, fov_deg: float) -> np.ndarray:
    """(N, 2) col,row (image rows down) -> (N, 3) unit-depth rays (x, y, -1) in the camera frame,
    the inverse of photo.photograph's projection at the pixel centre."""
    extent = 2 * math.tan(math.radians(fov_deg) / 2)
    col, row = vpix[:, 0].astype(np.float64), vpix[:, 1].astype(np.float64)
    py = size - 1 - row
    x = (col + 0.5) * extent / size - extent / 2
    y = (py + 0.5) * extent / size - extent / 2
    return np.stack([x, y, -np.ones_like(x)], -1).astype(np.float32)


def project(X: torch.Tensor, size: int, extent: float) -> torch.Tensor:
    """Camera-frame points (..., 3) -> continuous pixel coords (col, row), rows down."""
    depth = (-X[..., 2]).clamp_min(1e-3)
    proj = X[..., :2] / depth[..., None]
    sc = size / extent
    col = (proj[..., 0] + extent / 2) * sc - 0.5
    py = (proj[..., 1] + extent / 2) * sc - 0.5
    return torch.stack([col, size - 1 - py], -1)


def grid_edges(ny: int, nx: int):
    """Edge index pairs (E, 2) and their rest-length multipliers (1 for axis edges, sqrt2 diagonals)."""
    idx = np.arange(ny * nx).reshape(ny, nx)
    e, m = [], []
    e.append(np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1)); m.append(np.ones(len(e[-1])))
    e.append(np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], 1)); m.append(np.ones(len(e[-1])))
    e.append(np.stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel()], 1)); m.append(np.full(len(e[-1]), math.sqrt(2)))
    e.append(np.stack([idx[1:, :-1].ravel(), idx[:-1, 1:].ravel()], 1)); m.append(np.full(len(e[-1]), math.sqrt(2)))
    return np.concatenate(e), np.concatenate(m).astype(np.float32)


def laplacian(X: torch.Tensor) -> torch.Tensor:
    """4-neighbour umbrella operator on an (ny, nx, 3) grid (interior only)."""
    return X[1:-1, 1:-1] - 0.25 * (X[:-2, 1:-1] + X[2:, 1:-1] + X[1:-1, :-2] + X[1:-1, 2:])


def init_hidden(X: torch.Tensor, vis: torch.Tensor, iters: int = 300) -> torch.Tensor:
    """Fill hidden vertices by Jacobi relaxation from the visible ones (harmonic interpolation)."""
    X = X.clone()
    pad = lambda t: F.pad(t.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="replicate")[0].permute(1, 2, 0)
    for _ in range(iters):
        P = pad(X)
        avg = 0.25 * (P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:])
        X = torch.where(vis[..., None], X, avg)
    return X


def fit_view(gt, grid_true: np.ndarray, pool: int, dev, iters: int, lr: float, w_iso: float, w_bend: float,
             w_sil: float, init_depth: float, log_every: int = 0, oracle_lengths: bool = False,
             init_true_depth: bool = False, w_rep: float = 1.0, sigma_px: float = 0.5,
             init_truth_noise: float | None = None, vggt_depth: np.ndarray | None = None):
    size, fov = int(gt["size"]), float(gt["fov_deg"])
    extent = 2 * math.tan(math.radians(fov) / 2)
    ny, nx = grid_true.shape[:2]
    sel_y, sel_x = np.arange(0, ny, pool), np.arange(0, nx, pool)
    my, mx = len(sel_y), len(sel_x)
    sub = np.ix_(sel_y, sel_x)
    vis = gt["vis"].reshape(ny, nx)[sub]
    vpix = gt["vpix"].reshape(ny, nx, 2)[sub]
    rays = torch.from_numpy(pixel_rays(vpix.reshape(-1, 2), size, fov)).to(dev).view(my, mx, 3)
    visT = torch.from_numpy(vis).to(dev)
    # true camera-frame grid for scoring (R maps the centred grid to the camera; sheet centre at -d)
    R, d = gt["R"].astype(np.float32), float(gt["d"])
    Xt = (grid_true.reshape(-1, 3) - grid_true.reshape(-1, 3).mean(0)) @ R.T
    Xt[:, 2] -= d
    Xt = torch.from_numpy(Xt.astype(np.float32)).to(dev).view(ny, nx, 3)[sub]
    # rest lengths: flat sheet, pooled cell
    cell = float(gt["sheet_size"]) / (nx - 1) * pool if "sheet_size" in gt else 0.5 / (nx - 1) * pool
    E, mult = grid_edges(my, mx)
    E = torch.from_numpy(E).to(dev)
    rest = torch.from_numpy(mult).to(dev) * cell
    if oracle_lengths:  # diagnostic: the TRUE deformed edge lengths (the simulated cloth stretches)
        rest = (Xt.view(-1, 3)[E[:, 0]] - Xt.view(-1, 3)[E[:, 1]]).norm(dim=-1)
    if init_true_depth:  # diagnostic: start at the true median depth (ambiguity vs local minimum)
        init_depth = float((-Xt[..., 2])[visT].median())
    # silhouette: distance (px) to the mask, 0 inside
    mask = gt["mask"]
    dist = distance_transform_edt(~mask).astype(np.float32)
    distT = torch.from_numpy(dist).to(dev)[None, None]
    # unknowns
    depth = torch.full((my, mx), init_depth, device=dev)
    if vggt_depth is not None:  # start the visible vertices at VGGT's depth of their pixel (already scaled)
        vd = torch.from_numpy(vggt_depth[vpix[..., 1], vpix[..., 0]].astype(np.float32)).to(dev)
        depth = torch.where(visT & (vd > 0), vd, depth)
    X0 = rays * depth[..., None]
    X0 = init_hidden(X0, visT)
    if init_truth_noise is not None:  # diagnostic: start at the true surface (+ noise, m)
        X0 = Xt + init_truth_noise * torch.randn_like(Xt)
    # all vertices free; visible ones are held to their pixel by a soft reprojection term (sigma px):
    # pinning them to the pixel-centre ray quantises each vertex by up to half a pixel (~0.8 mm at 1 m),
    # ~10% of a pooled edge, which the isometry term cannot absorb.
    Xv = X0.clone().requires_grad_(True)
    opt = torch.optim.Adam([Xv], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    hid_mask = ~visT
    pixT = torch.from_numpy(vpix.reshape(my, mx, 2).astype(np.float32)).to(dev)

    def assemble():
        return Xv

    with torch.no_grad():  # the initialiser's own error (VGGT alone when --init-vggt), same vertices, same metric
        e0 = (X0 - Xt).norm(dim=-1) * 100
        dz0 = ((-X0[..., 2]) - (-Xt[..., 2])) * 100
        init_rec = {"init_vis_err_mean_cm": float(e0[visT].mean()), "init_vis_err_rmse_cm": float(e0[visT].pow(2).mean().sqrt()),
                    "init_vis_depth_abs_err_med_cm": float(dz0[visT].abs().median()), "init_vis_depth_bias_cm": float(dz0[visT].median())}

    for it in range(iters):
        opt.zero_grad()
        X = assemble()
        flat = X.view(-1, 3)
        el = (flat[E[:, 0]] - flat[E[:, 1]]).norm(dim=-1)
        e_iso = ((el - rest) / cell).pow(2).mean()
        e_bend = (laplacian(X) / cell).pow(2).sum(-1).mean()
        pix = project(X, size, extent)
        gridc = torch.stack([pix[..., 0] / (size - 1) * 2 - 1, pix[..., 1] / (size - 1) * 2 - 1], -1)
        dpx = F.grid_sample(distT, gridc[None], align_corners=True, padding_mode="border")[0, 0]
        e_sil = (dpx[hid_mask] / 10.0).pow(2).mean() if hid_mask.any() else dpx.sum() * 0
        e_rep = ((pix[visT] - pixT[visT]) / sigma_px).pow(2).sum(-1).mean()
        loss = w_iso * e_iso + w_bend * e_bend + w_sil * e_sil + w_rep * e_rep
        loss.backward()
        opt.step()
        sched.step()
        if log_every and (it % log_every == 0 or it == iters - 1):
            with torch.no_grad():
                err = (assemble() - Xt).norm(dim=-1)
                print(f"  it {it:5d} loss {loss.item():.4e} iso {e_iso.item():.3e} bend {e_bend.item():.3e} "
                      f"sil {e_sil.item():.3e} rep {e_rep.item():.3e} | vis err {err[visT].mean().item() * 100:.2f} cm "
                      f"hid err {err[~visT].mean().item() * 100 if (~visT).any() else 0:.2f} cm")
    with torch.no_grad():
        X = assemble()
        err = (X - Xt).norm(dim=-1) * 100
        dz = ((-X[..., 2]) - (-Xt[..., 2])) * 100
        el = (X.view(-1, 3)[E[:, 0]] - X.view(-1, 3)[E[:, 1]]).norm(dim=-1)
        rec = {**init_rec, "n_vis": int(visT.sum()), "n_hidden": int((~visT).sum()),
               "vis_err_mean_cm": float(err[visT].mean()), "vis_err_rmse_cm": float(err[visT].pow(2).mean().sqrt()),
               "vis_err_p90_cm": float(err[visT].quantile(0.9)),
               "vis_depth_abs_err_med_cm": float(dz[visT].abs().median()), "vis_depth_bias_cm": float(dz[visT].median()),
               "hidden_err_mean_cm": float(err[~visT].mean()) if (~visT).any() else None,
               "edge_stretch_rms": float(((el - rest) / rest).pow(2).mean().sqrt()),
               "true_edge_stretch_rms": float((((Xt.view(-1, 3)[E[:, 0]] - Xt.view(-1, 3)[E[:, 1]]).norm(dim=-1) - rest) / rest).pow(2).mean().sqrt()),
               "depth_range_true_cm": float(((-Xt[..., 2])[visT].max() - (-Xt[..., 2])[visT].min()) * 100)}
    return rec, X.detach().cpu().numpy(), Xt.cpu().numpy(), vis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pool", type=int, default=4, help="fit every k-th vertex (256 / k grid)")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--w-iso", type=float, default=1.0)
    ap.add_argument("--w-bend", type=float, default=1e-3)
    ap.add_argument("--w-sil", type=float, default=1.0)
    ap.add_argument("--w-rep", type=float, default=1.0, help="reprojection weight for the visible vertices")
    ap.add_argument("--sigma-px", type=float, default=0.5, help="reprojection tolerance (pixel quantisation)")
    ap.add_argument("--init-depth", type=float, default=1.15, help="m, the middle of the camera-distance prior")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--oracle-lengths", action="store_true", help="diagnostic: use the true deformed edge lengths")
    ap.add_argument("--init-true-depth", action="store_true", help="diagnostic: initialise at the true median depth")
    ap.add_argument("--init-truth-noise", type=float, default=None,
                    help="diagnostic: initialise at the true surface plus Gaussian noise of this sigma (m)")
    ap.add_argument("--vggt-raw-scale", action="store_true",
                    help="with --init-vggt: use VGGT's own metric depth (no oracle scale factor)")
    ap.add_argument("--init-vggt", action="store_true",
                    help="initialise the visible vertices at VGGT's depth (views/<stem>_pts_vggt.npz, oracle-scaled)")
    ap.add_argument("--out", default=None, help="default <photos>/fit")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for pdir in args.photos:
        out = args.out or os.path.join(pdir, "fit")
        os.makedirs(out, exist_ok=True)
        man = json.load(open(os.path.join(pdir, "manifest.json")))
        h5_of = {os.path.splitext(v["png"])[0]: v["h5"] for v in man["views"]}
        recs = []
        grids = {}
        for gp in sorted(glob.glob(os.path.join(pdir, "*_gt.npz")))[: args.limit or None]:
            stem = os.path.basename(gp)[: -len("_gt.npz")]
            gt = np.load(gp)
            h5 = h5_of[stem]
            if h5 not in grids:
                grids[h5] = load_grid(h5)[0]
            t0 = time.time()
            vggt_depth = None
            if args.init_vggt:
                vp = os.path.join(pdir, "views", f"{stem}_pts_vggt.npz")
                if not os.path.exists(vp):
                    print(f"{stem}: no VGGT package, skipped")
                    continue
                vz = np.load(vp)
                from PIL import Image
                sm = np.asarray(Image.open(os.path.join(pdir, "sam3", f"{stem}__fabric.png")).convert("L")) > 127
                m = sm & (gt["depth"] > 0)
                rr, cc = np.nonzero(m)
                pts = vz["pts"] + vz["g"][:3]
                assert len(pts) == len(rr), "VGGT cloud / mask pixel count mismatch"
                vggt_depth = np.zeros(gt["depth"].shape, np.float32)
                vggt_depth[rr, cc] = -pts[:, 2] * (1.0 if args.vggt_raw_scale else float(vz["scale_to_oracle"]))
            rec, X, Xt, vis = fit_view(gt, grids[h5], args.pool, dev, args.iters, args.lr, args.w_iso, args.w_bend,
                                       args.w_sil, args.init_depth, args.log_every, args.oracle_lengths,
                                       args.init_true_depth, args.w_rep, args.sigma_px, args.init_truth_noise,
                                       vggt_depth)
            rec.update(stem=stem, set=os.path.basename(os.path.normpath(pdir)), pool=args.pool,
                       visible_frac=float(gt["vis"].mean()), material=str(gt["material"]), seconds=time.time() - t0)
            recs.append(rec)
            np.savez_compressed(os.path.join(out, f"{stem}_fit.npz"), X=X, X_true=Xt, vis=vis, pool=args.pool)
            print(f"{stem:36s} vis {rec['visible_frac']:.2f}  init err {rec['init_vis_err_mean_cm']:.2f} -> "
                  f"vis err mean {rec['vis_err_mean_cm']:.2f} cm "
                  f"rmse {rec['vis_err_rmse_cm']:.2f} p90 {rec['vis_err_p90_cm']:.2f} | depth |dz| med "
                  f"{rec['vis_depth_abs_err_med_cm']:.2f} bias {rec['vis_depth_bias_cm']:+.2f} | hidden "
                  f"{(rec['hidden_err_mean_cm'] or 0):.1f} cm | stretch {rec['edge_stretch_rms']:.3f} "
                  f"(true {rec['true_edge_stretch_rms']:.3f}) | {rec['seconds']:.0f}s")
        med = lambda k: float(np.median([r[k] for r in recs if r.get(k) is not None]))
        summary = {"n": len(recs), "pool": args.pool, "init_vis_err_mean_cm": med("init_vis_err_mean_cm"),
                   "init_vis_depth_abs_err_med_cm": med("init_vis_depth_abs_err_med_cm"),
                   "vis_err_mean_cm": med("vis_err_mean_cm"),
                   "vis_err_rmse_cm": med("vis_err_rmse_cm"), "vis_depth_abs_err_med_cm": med("vis_depth_abs_err_med_cm"),
                   "hidden_err_mean_cm": med("hidden_err_mean_cm"), "edge_stretch_rms": med("edge_stretch_rms")}
        print(os.path.basename(pdir), json.dumps(summary))
        with open(os.path.join(out, "scores.json"), "w") as fh:
            json.dump({"args": vars(args), "records": recs, "summary": summary}, fh, indent=1)


if __name__ == "__main__":
    main()
