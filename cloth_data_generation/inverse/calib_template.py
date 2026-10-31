"""Low-DOF template calibration of VGGT depth: d' = a*dv + b + c*x + d*y (depth scale, offset,
image-plane tilt), the four numbers that explain the VGGT error (depth_decomp). Fitted WITHOUT the
oracle depth: visible pooled vertices at their (oracle, step-2: estimated) pixels, energy =
isometry on the edges whose ends are both visible (flat rest lengths, 0.5 m sheet) + weak bending.
Start: raw VGGT depth (no oracle scale). Variants: --dof 1 (scale), 2 (scale+offset), 4 (all).

Cloth venv. --tag vggt (single-view packages) or stereo (lift_stereo packages); falls back to the
oracle mask when no SAM mask exists. Result (2026-10-31): the isometry prior cannot set these
numbers at the cm level (1% of scale = 1 cm of depth at 1 m; the simulated cloth stretches 8%).
  ~/cloth/.venv/bin/python calib_template.py --photos ../../cloth_results/photos/cloth256 --dof 4
"""
import argparse, glob, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_template import pixel_rays, grid_edges, laplacian
from photo import load_grid
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--photos", nargs="+", required=True)
ap.add_argument("--pool", type=int, default=4)
ap.add_argument("--dof", type=int, default=4)
ap.add_argument("--w-bend", type=float, default=1e-3)
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--oracle-lengths", action="store_true")
ap.add_argument("--l1", action="store_true", help="L1 stretch loss (robust to the stretched edges)")
ap.add_argument("--tag", default="vggt", help="package tag: vggt | stereo")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=None)
args = ap.parse_args()
dev = torch.device("cuda")
recs = []
for pdir in args.photos:
    man = json.load(open(os.path.join(pdir, "manifest.json")))
    h5_of = {os.path.splitext(v["png"])[0]: v["h5"] for v in man["views"]}
    grids = {}
    for gp in sorted(glob.glob(os.path.join(pdir, "*_gt.npz")))[: args.limit or None]:
        stem = os.path.basename(gp)[:-7]
        vp = os.path.join(pdir, "views", f"{stem}_pts_{args.tag}.npz"); mp = os.path.join(pdir, "sam3", f"{stem}__fabric.png")
        if not os.path.exists(vp): continue
        gt = np.load(gp); vz = np.load(vp)
        h5 = h5_of[stem]
        if h5 not in grids: grids[h5] = load_grid(h5)[0]
        grid = grids[h5]
        sm = (np.asarray(Image.open(mp).convert("L")) > 127) if os.path.exists(mp) else gt["mask"]
        m = sm & (gt["depth"] > 0); rr, cc = np.nonzero(m)
        pts = vz["pts"] + vz["g"][:3]
        if len(pts) != len(rr): continue
        dv_map = np.zeros(gt["depth"].shape, np.float32); dv_map[rr, cc] = -pts[:, 2]  # RAW vggt depth
        size, fov = int(gt["size"]), float(gt["fov_deg"])
        ny, nx = grid.shape[:2]; P = args.pool
        sub = np.ix_(np.arange(0, ny, P), np.arange(0, nx, P)); my, mx = sub[0].shape[0], sub[1].shape[1]
        vis = gt["vis"].reshape(ny, nx)[sub]; vpix = gt["vpix"].reshape(ny, nx, 2)[sub]
        dv = dv_map[vpix[..., 1], vpix[..., 0]]
        vis = vis & (dv > 0)
        rays = torch.from_numpy(pixel_rays(vpix.reshape(-1, 2), size, fov)).to(dev).view(my, mx, 3)
        R, d = gt["R"].astype(np.float32), float(gt["d"])
        Xt = (grid.reshape(-1, 3) - grid.reshape(-1, 3).mean(0)) @ R.T; Xt[:, 2] -= d
        Xt = torch.from_numpy(Xt.astype(np.float32)).to(dev).view(ny, nx, 3)[sub]
        visT = torch.from_numpy(vis).to(dev); dvT = torch.from_numpy(dv.astype(np.float32)).to(dev)
        cell = 0.5 / (nx - 1) * P
        E, mult = grid_edges(my, mx); E = torch.from_numpy(E).to(dev)
        rest = torch.from_numpy(mult).to(dev) * cell
        if args.oracle_lengths:
            rest = (Xt.view(-1, 3)[E[:, 0]] - Xt.view(-1, 3)[E[:, 1]]).norm(dim=-1)
        vflat = visT.view(-1); ev = vflat[E[:, 0]] & vflat[E[:, 1]]
        # bending only where the 4-neighbourhood is visible
        bm = visT.clone(); bm[1:-1, 1:-1] = visT[1:-1, 1:-1] & visT[:-2, 1:-1] & visT[2:, 1:-1] & visT[1:-1, :-2] & visT[1:-1, 2:]
        bm[0, :] = bm[-1, :] = bm[:, 0] = bm[:, -1] = False
        # scale init: the known sheet size vs the raw cloud is unknown; start a so that the median
        # visible edge length equals the rest length (a closed-form isometric scale guess)
        def depth_of(p):
            la, b, c, dd = p
            return torch.exp(la) * dvT + b + c * rays[..., 0] + dd * rays[..., 1]
        with torch.no_grad():
            X0 = rays * dvT[..., None]
            el0 = (X0.view(-1, 3)[E[:, 0]] - X0.view(-1, 3)[E[:, 1]]).norm(dim=-1)[ev]
            a0 = float((rest[ev] / el0).median())
        p = torch.tensor([math.log(a0), 0.0, 0.0, 0.0], device=dev, requires_grad=True)
        free = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}[args.dof]
        mask_p = torch.zeros(4, device=dev); mask_p[free] = 1
        opt = torch.optim.Adam([p], lr=2e-2)
        def energy():
            X = rays * depth_of(p)[..., None]
            el = (X.view(-1, 3)[E[:, 0]] - X.view(-1, 3)[E[:, 1]]).norm(dim=-1)
            e_iso = (((el - rest) / cell).abs() if args.l1 else ((el - rest) / cell) ** 2)[ev].mean()
            e_bend = ((laplacian(X) / cell) ** 2).sum(-1)[bm[1:-1, 1:-1]].mean() if bm.any() else 0 * e_iso
            return e_iso + args.w_bend * e_bend, X
        t0 = time.time()
        for it in range(args.iters):
            opt.zero_grad(); e, X = energy(); e.backward()
            p.grad *= mask_p; opt.step()
        with torch.no_grad():
            e, X = energy()
            X0 = rays * (a0 * dvT)[..., None]
            err0 = (X0 - Xt).norm(dim=-1)[visT] * 100; err = (X - Xt).norm(dim=-1)[visT] * 100
            dz = (-X[..., 2] + Xt[..., 2])[visT] * 100
            s_or = float(vz["scale_to_oracle"])
            Xs = rays * (s_or * dvT)[..., None]; errs = (Xs - Xt).norm(dim=-1)[visT] * 100
        rec = dict(stem=stem, set=os.path.basename(os.path.normpath(pdir)), n_vis=int(vis.sum()),
                   scale_oracle=s_or, scale_iso_init=a0, scale_fit=float(torch.exp(p[0])),
                   err_oracle_scale_cm=float(errs.mean()), err_iso_scale_cm=float(err0.mean()),
                   err_fit_cm=float(err.mean()), err_fit_p90_cm=float(err.quantile(0.9)),
                   depth_abs_err_med_cm=float(dz.abs().median()), depth_bias_cm=float(dz.median()),
                   params=p.detach().cpu().tolist(), energy=float(e), seconds=time.time() - t0)
        recs.append(rec)
        print(f"{stem:36s} vis {rec['n_vis']:4d} scale or {s_or:.3f} iso {a0:.3f} fit {rec['scale_fit']:.3f} | "
              f"err oracle-scale {rec['err_oracle_scale_cm']:5.2f} iso-scale {rec['err_iso_scale_cm']:5.2f} "
              f"fit {rec['err_fit_cm']:5.2f} cm | dz med {rec['depth_abs_err_med_cm']:5.2f} bias {rec['depth_bias_cm']:+5.2f} | {rec['seconds']:.1f}s")
med = lambda k: float(np.median([r[k] for r in recs]))
summary = {k: med(k) for k in ["err_oracle_scale_cm", "err_iso_scale_cm", "err_fit_cm", "err_fit_p90_cm", "depth_abs_err_med_cm", "depth_bias_cm"]}
summary["n"] = len(recs); summary["dof"] = args.dof
summary["scale_ratio_fit_over_oracle_med"] = med("scale_fit") and float(np.median([r["scale_fit"] / r["scale_oracle"] for r in recs]))
summary["scale_ratio_log_std"] = float(np.std(np.log([r["scale_fit"] / r["scale_oracle"] for r in recs])))
print(json.dumps(summary, indent=1))
if args.out: json.dump({"args": vars(args), "records": recs, "summary": summary}, open(args.out, "w"), indent=1)
