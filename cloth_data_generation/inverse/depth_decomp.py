"""How much of the VGGT depth error is a per-view linear calibration?
For every lifted view: oracle depth do and VGGT depth dv (same masked pixels).
Fits, each per view, and the residual median |err| (cm) and point RMSE (cm):
  scale     dv*s
  affine_z  a*dv + b
  quad_z    a*dv + b + c*u + d*v   (depth + image-plane tilt)
  affine3d  12-parameter affine on the 3D points (ceiling of any linear fix)

Cloth venv; needs the single-view VGGT packages (lift.py) and SAM masks.
  ~/cloth/.venv/bin/python depth_decomp.py ../../cloth_results/photos/cloth256 ../../cloth_results/photos/paper256
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from package_views import unproject

def pix_dirs(rows, cols, size, fov):
    import math
    ext = math.tan(math.radians(fov) / 2)
    x = ((cols + 0.5) / size - 0.5) * 2 * ext
    y = (0.5 - (rows + 0.5) / size) * 2 * ext
    return x, y

rows_out = []
for pdir in sys.argv[1:]:
    for gp in sorted(glob.glob(os.path.join(pdir, "*_gt.npz"))):
        stem = os.path.basename(gp)[:-7]
        vp = os.path.join(pdir, "views", f"{stem}_pts_vggt.npz")
        mp = os.path.join(pdir, "sam3", f"{stem}__fabric.png")
        if not (os.path.exists(vp) and os.path.exists(mp)):
            continue
        from PIL import Image
        g = np.load(gp); vg = np.load(vp)
        sam = np.asarray(Image.open(mp).convert("L")) > 127
        m = sam & (g["depth"] > 0)
        rr, cc = np.nonzero(m)
        pts = vg["pts"] + vg["g"][:3]
        if len(pts) != len(rr):
            continue
        do = g["depth"][rr, cc]
        dv = -pts[:, 2]
        ref = unproject(m, g["depth"], float(g["fov_deg"]))
        u, v = pix_dirs(rr, cc, int(g["size"]), float(g["fov_deg"]))
        def stats(dfit):
            err = dfit - do
            P = np.stack([u * dfit, v * dfit, -dfit], 1)
            rmse = np.sqrt((((P - P.mean(0)) - (ref - ref.mean(0))) ** 2).sum(1).mean())
            return float(np.median(np.abs(err)) * 100), float(np.percentile(np.abs(err), 90) * 100), float(rmse * 100)
        r = {"stem": stem, "set": os.path.basename(pdir), "range_cm": float((do.max() - do.min()) * 100)}
        s = np.median(do) / np.median(dv)
        r["scale"] = stats(dv * s)
        A = np.stack([dv, np.ones_like(dv)], 1)
        coef, *_ = np.linalg.lstsq(A, do, rcond=None); r["affine_z"] = stats(A @ coef); r["affine_z_coef"] = coef.tolist()
        A2 = np.stack([dv, np.ones_like(dv), u, v], 1)
        coef2, *_ = np.linalg.lstsq(A2, do, rcond=None); r["quad_z"] = stats(A2 @ coef2)
        # full 3D affine on the VGGT points (scaled first so the fit is well conditioned)
        Ps = pts * s
        A3 = np.concatenate([Ps, np.ones((len(Ps), 1))], 1)
        M, *_ = np.linalg.lstsq(A3, ref, rcond=None)
        fit3 = A3 @ M
        rm = np.sqrt(((fit3 - ref) ** 2).sum(1).mean()) * 100
        r["affine3d"] = (float(np.median(np.abs(fit3[:, 2] - ref[:, 2])) * 100), float(np.percentile(np.abs(fit3[:, 2] - ref[:, 2]), 90) * 100), float(rm))
        # plane-only truth: how non-planar is the sheet? residual of the oracle itself to its best plane
        Ao = np.stack([ref[:, 0], ref[:, 1], np.ones(len(ref))], 1)
        cp, *_ = np.linalg.lstsq(Ao, ref[:, 2], rcond=None)
        r["oracle_plane_resid_cm"] = float(np.median(np.abs(Ao @ cp - ref[:, 2])) * 100)
        rows_out.append(r)
        print(f"{stem:38s} range {r['range_cm']:5.1f}  scale {r['scale'][0]:5.2f}/{r['scale'][2]:5.2f}  affz {r['affine_z'][0]:5.2f}/{r['affine_z'][2]:5.2f} (a={coef[0]:.2f})  quad {r['quad_z'][0]:5.2f}/{r['quad_z'][2]:5.2f}  aff3d {r['affine3d'][0]:5.2f}/{r['affine3d'][2]:5.2f}  plane {r['oracle_plane_resid_cm']:4.1f}")

def med(k, i): return float(np.median([r[k][i] for r in rows_out]))
print("\nmedian over %d views: med|err| cm / p90 / point RMSE cm" % len(rows_out))
for k in ["scale", "affine_z", "quad_z", "affine3d"]:
    print(f"  {k:9s} {med(k,0):5.2f} / {med(k,1):5.2f} / {med(k,2):5.2f}")
print("  affine_z slope a: median %.3f, IQR %.3f-%.3f" % tuple(np.percentile([r['affine_z_coef'][0] for r in rows_out], [50, 25, 75])))
print("  oracle plane residual median %.2f cm" % np.median([r['oracle_plane_resid_cm'] for r in rows_out]))
json.dump(rows_out, open(os.path.join(os.path.dirname(sys.argv[1]), "depth_decomp.json"), "w"), indent=1)
