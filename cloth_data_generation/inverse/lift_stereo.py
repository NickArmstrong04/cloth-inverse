"""Stereo pair through VGGT: the two photos of a --stereo render set go in together; VGGT's
predicted camera translation vs the KNOWN baseline (a phone moved a measured distance) gives
the metric scale with no oracle. Scores the base view's depth at the oracle-mask pixels:
  single view, oracle scale (today's reference)      | stereo, baseline scale (no oracle)
  + per-view affine z / depth+tilt residuals (how much linear error remains in each).

Runs in the SAM 3 venv (VGGT). Photos from `photo.py --stereo DEG --gt`; --sam scores inside the SAM
mask (segment.py first) instead of the oracle mask. Writes views/<stem>_pts_stereo.npz packages
and <photos-parent>/vggt_stereo.json.
  ~/sam3env/bin/python lift_stereo.py ../../cloth_results/photos/stereo20
"""
import glob, json, math, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lift import load_vggt, unproject_vggt
from package_views import unproject
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

dev = torch.device("cuda"); dtype = torch.bfloat16
model = load_vggt(dev)
use_sam = "--sam" in sys.argv
WINDOW = float(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else 1.4
_skip = {str(WINDOW)}
pdirs = [a for a in sys.argv[1:] if not a.startswith("--") and a not in _skip]

def run(pngs, known_K=None):
    images = load_and_preprocess_images(pngs).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        pred = model(images)
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    W0, H0 = Image.open(pngs[0]).size
    out = []
    for i in range(len(pngs)):
        depth = pred["depth"][0, i, ..., 0].float(); h, w = depth.shape
        depth = F.interpolate(depth[None, None], (H0, W0), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        K = np.diag([W0 / w, H0 / h, 1.0]) @ intr[0, i].float().cpu().numpy()
        E = extr[0, i].float().cpu().numpy()  # (3,4) world->cam OpenCV
        C = -E[:, :3].T @ E[:, 3]
        out.append((depth, K, C))
    return out

def fits(dv, do, u, v, ref):
    def st(dfit):
        P = np.stack([u * dfit, v * dfit, -dfit], 1)
        return [float(np.median(np.abs(dfit - do)) * 100), float(np.sqrt((((P - P.mean(0)) - (ref - ref.mean(0))) ** 2).sum(1).mean()) * 100)]
    A = np.stack([dv, np.ones_like(dv)], 1); c1, *_ = np.linalg.lstsq(A, do, rcond=None)
    A2 = np.stack([dv, np.ones_like(dv), u, v], 1); c2, *_ = np.linalg.lstsq(A2, do, rcond=None)
    return st, st(A @ c1), st(A2 @ c2)

def true_K(size, fov):
    f = size / (2 * math.tan(math.radians(fov) / 2))
    return np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])


def cam_from_world_cv(R, d):
    """photo.py camera (p = R x + (0,0,-d), y up, z back) as OpenCV cam-from-world (R_cv, t_cv)."""
    Fl = np.diag([1.0, -1.0, -1.0])
    return Fl @ R, Fl @ np.array([0, 0, -d])


def scale_from_baseline(d0, d1, m0, m1, K, R0, t0, R1, t1, I0=None, I1=None, s_prior=None, window=1.4):
    """Metric scale of VGGT's (shape-correct, scale-free) depth pair from the KNOWN relative pose,
    without VGGT's pose head: the scale s for which view 0's points, scaled by s and moved into
    camera 1 by the known transform, land where they should in view 1. Two consistency tests:
    photometric (I0, I1 given: the pixel's grey value must match in view 1; the printed texture
    makes this sharp everywhere) and, without images, depth (view 1's depth at the landing pixel
    must agree; degenerate on flat parts of the sheet, so it fails on near-planar pairs).
    Minimises the median mismatch over s (log grid + refinement). Returns (s, cost); cost 1e9
    means no overlap found (fall back to VGGT's own baseline)."""
    rr, cc = np.nonzero(m0)
    z = d0[rr, cc]
    x = (cc + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (rr + 0.5 - K[1, 2]) / K[1, 1] * z
    P0 = np.stack([x, y, z], 1)  # OpenCV cam 0, VGGT scale
    g0 = I0[rr, cc] if I0 is not None else None
    Rrel = R1 @ R0.T
    trel = t1 - Rrel @ t0  # cam1 = Rrel cam0 + trel (metric)
    H, W = d1.shape

    def cost(s):
        P1 = (s * P0) @ Rrel.T + trel
        zz = P1[:, 2]
        ok = zz > 1e-3
        uu = K[0, 0] * P1[ok, 0] / zz[ok] + K[0, 2] - 0.5
        vv = K[1, 1] * P1[ok, 1] / zz[ok] + K[1, 2] - 0.5
        inside = (uu >= 0) & (uu < W - 1) & (vv >= 0) & (vv < H - 1)
        if inside.sum() < 50:
            return 1e9
        uu, vv = uu[inside], vv[inside]
        ui, vi = np.floor(uu).astype(int), np.floor(vv).astype(int)
        hit = m1[vi, ui] & m1[vi + 1, ui + 1]
        if hit.sum() < 50:
            return 1e9
        if g0 is not None:  # bilinear grey value of view 1 at the landing point
            fu, fv = (uu - ui)[hit], (vv - vi)[hit]
            ui, vi = ui[hit], vi[hit]
            g1 = ((1 - fu) * (1 - fv) * I1[vi, ui] + fu * (1 - fv) * I1[vi, ui + 1]
                  + (1 - fu) * fv * I1[vi + 1, ui] + fu * fv * I1[vi + 1, ui + 1])
            return float(np.mean(np.abs(g1 - g0[ok][inside][hit])))
        zt = zz[ok][inside][hit]
        return float(np.median(np.abs(s * d1[vi[hit], ui[hit]] - zt) / zt))

    # search window: the repeating checker makes the photometric cost periodic in s (aliases at
    # +-50-65%), so the search stays within a factor `window` of the prior (VGGT's own baseline
    # scale, 10% median / 25% p90 off) when one is given
    lo, hi = (s_prior / window, s_prior * window) if s_prior else (0.2, 5.0)
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 160))
    s = grid[int(np.argmin([cost(c) for c in grid]))]
    for step in [0.05, 0.02, 0.007, 0.002]:
        cand = s * np.exp(np.linspace(-step, step, 11))
        s = cand[int(np.argmin([cost(c) for c in cand]))]
    return float(s), cost(s)


rows = []
for pdir in pdirs:
    man = json.load(open(os.path.join(pdir, "manifest.json")))
    views = {os.path.splitext(v["png"])[0]: v for v in man["views"]}
    for stem, v in sorted(views.items()):
        if v["partner"] < 0 or v["partner"] < v["view"]:
            continue  # base views only (partner index is larger)
        pstem = stem[: stem.rfind("_v")] + f"_v{v['partner']}"
        g = np.load(os.path.join(pdir, f"{stem}_gt.npz"))
        mask = g["mask"] & (g["depth"] > 0)
        if use_sam:
            mp = os.path.join(pdir, "sam3", f"{stem}__fabric.png")
            if not os.path.exists(mp): continue
            mask &= np.asarray(Image.open(mp).convert("L")) > 127
        if mask.sum() < 50: continue
        rr, cc = np.nonzero(mask)
        ref = unproject(mask, g["depth"], float(g["fov_deg"])); do = -ref[:, 2]
        size = int(g["size"]); ext = math.tan(math.radians(float(g["fov_deg"])) / 2)
        u = ((cc + 0.5) / size - 0.5) * 2 * ext; vv = (0.5 - (rr + 0.5) / size) * 2 * ext
        # single
        d1, K1, _ = run([os.path.join(pdir, stem + ".png")])[0]
        dv1 = -unproject_vggt(mask, d1, K1)[:, 2]
        st1, aff1, quad1 = fits(dv1, do, u, vv, ref)
        s_or = float(np.median(do) / np.median(dv1))
        # stereo
        (d2, K2, C0), (d2p, _, C1) = run([os.path.join(pdir, stem + ".png"), os.path.join(pdir, pstem + ".png")])
        dv2 = -unproject_vggt(mask, d2, K2)[:, 2]
        gp = np.load(os.path.join(pdir, f"{pstem}_gt.npz"))
        m1 = gp["mask"] & (gp["depth"] > 0)
        if use_sam:
            mp1 = os.path.join(pdir, "sam3", f"{pstem}__fabric.png")
            if os.path.exists(mp1):
                m1 &= np.asarray(Image.open(mp1).convert("L")) > 127
        Kt = true_K(int(g["size"]), float(g["fov_deg"]))
        R0cv, t0cv = cam_from_world_cv(g["R"], float(g["d"]))
        R1cv, t1cv = cam_from_world_cv(gp["R"], float(gp["d"]))
        I0 = np.asarray(Image.open(os.path.join(pdir, stem + ".png")).convert("L"), np.float32) / 255
        I1 = np.asarray(Image.open(os.path.join(pdir, pstem + ".png")).convert("L"), np.float32) / 255
        s_vggt = float(v["baseline"]) / max(float(np.linalg.norm(C0 - C1)), 1e-6)  # VGGT's own baseline: the prior
        s_geo, geo_cost = scale_from_baseline(d2, d2p, mask, m1, Kt, R0cv, t0cv, R1cv, t1cv, I0, I1,
                                              s_prior=s_vggt, window=WINDOW)
        if geo_cost >= 1e9:
            s_geo = s_vggt
        dv2k = -unproject_vggt(mask, d2, Kt)[:, 2]  # VGGT depth with the TRUE intrinsics
        st2, aff2, quad2 = fits(dv2, do, u, vv, ref)
        b_pred = float(np.linalg.norm(C0 - C1)); b_true = float(v["baseline"])
        s_base = b_true / max(b_pred, 1e-6)
        s_or2 = float(np.median(do) / np.median(dv2))
        row = dict(stem=stem, set=os.path.basename(os.path.normpath(pdir)), baseline_cm=b_true * 100,
                   single_oracle_scale=st1(dv1 * s_or), single_affz=aff1, single_quad=quad1,
                   stereo_oracle_scale=st2(dv2 * s_or2), stereo_baseline_scale=st2(dv2 * s_base),
                   stereo_affz=aff2, stereo_quad=quad2, scale_err=s_base / s_or2 - 1,
                   stereo_geo_scale=st2(dv2 * s_geo), stereo_geo_scale_trueK=st2(dv2k * s_geo),
                   scale_err_geo=s_geo / s_or2 - 1, geo_cost=geo_cost,
                   f_ratio=float(K2[0, 0] / (size / (2 * ext))))
        rows.append(row)
        os.makedirs(os.path.join(pdir, "views"), exist_ok=True)
        pts = unproject_vggt(mask, d2, K2); cen = pts.mean(0)
        np.savez_compressed(os.path.join(pdir, "views", f"{stem}_pts_stereo.npz"), pts=pts - cen,
                            g=np.concatenate([cen, np.zeros(3, np.float32)]), scale_to_oracle=s_or2,
                            scale_baseline=s_base, scale_geo=s_geo, K=K2, source="stereo", material=str(g["material"]),
                            R=g["R"], d=float(g["d"]), size=size, fov_deg=float(g["fov_deg"]),
                            mask_source="sam" if use_sam else "oracle")
        print(f"{stem:34s} b {b_true*100:4.1f}cm | single or-scale {row['single_oracle_scale'][0]:5.2f} quad {quad1[0]:5.2f} | "
              f"stereo or-scale {row['stereo_oracle_scale'][0]:5.2f} BASELINE-scale {row['stereo_baseline_scale'][0]:5.2f} "
              f"({row['scale_err']*100:+5.1f}%) GEO-scale {row['stereo_geo_scale'][0]:5.2f} ({row['scale_err_geo']*100:+5.1f}%, "
              f"cost {geo_cost:.3f}) affz {aff2[0]:5.2f} quad {quad2[0]:5.2f} | f {row['f_ratio']:.2f}")
    med = lambda k, i: float(np.median([r[k][i] for r in rows if r["set"] == os.path.basename(os.path.normpath(pdir))]))
    sub = [r for r in rows if r["set"] == os.path.basename(os.path.normpath(pdir))]
    print(f"\n{os.path.basename(os.path.normpath(pdir))}: median over {len(sub)} pairs, med|err| cm / point RMSE cm")
    for k in ["single_oracle_scale", "single_affz", "single_quad", "stereo_oracle_scale", "stereo_baseline_scale",
              "stereo_geo_scale", "stereo_geo_scale_trueK", "stereo_affz", "stereo_quad"]:
        print(f"  {k:22s} {med(k,0):5.2f} / {med(k,1):5.2f}")
    se = np.array([r["scale_err"] for r in sub])
    print(f"  baseline scale error: median {np.median(np.abs(se))*100:.1f}%, p90 {np.percentile(np.abs(se), 90)*100:.1f}%, bias {np.median(se)*100:+.1f}%")
    sg = np.array([r["scale_err_geo"] for r in sub])
    print(f"  GEO scale error (known pose vs the depth pair): median {np.median(np.abs(sg))*100:.1f}%, "
          f"p90 {np.percentile(np.abs(sg), 90)*100:.1f}%, bias {np.median(sg)*100:+.1f}%\n")
json.dump(rows, open(os.path.join(os.path.dirname(os.path.normpath(pdirs[0])), "vggt_stereo.json"), "w"), indent=1)
