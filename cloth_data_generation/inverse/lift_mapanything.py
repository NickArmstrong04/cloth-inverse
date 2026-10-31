"""MapAnything (Meta, facebook/map-anything) on the stereo pairs: a metric-capable model that takes
the KNOWN intrinsics and the KNOWN camera motion as inputs, which VGGT cannot (lift_stereo.py:
VGGT's own baseline is 10% off). Three input modes per pair, base view scored at the oracle-mask
(or --sam) pixels against the oracle depth:
  img          images only                         (like VGGT: shape up to scale)
  K            + true intrinsics                   (scale from the model's metric head)
  K+pose       + true relative pose, metric flag   (scale from the measured baseline)
Reported per mode: depth error at ORACLE scale (shape), RAW metric depth error (no oracle), the
scale error, and the depth+tilt residual. Runs in the SAM 3 venv (install: map-anything clone,
pip install -e .; numpy 2.x is fine for SAM 3 and VGGT despite their pins).

  ~/sam3env/bin/python lift_mapanything.py ../../cloth_results/photos/stereo20 [--sam] [--single]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from package_views import unproject  # noqa: E402

FLIP = np.diag([1.0, -1.0, -1.0])  # our camera frame (y up, z backward) -> OpenCV (y down, z forward)


def true_K(size: int, fov_deg: float) -> np.ndarray:
    f = size / (2 * math.tan(math.radians(fov_deg) / 2))
    return np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])


def cam2world_cv(R: np.ndarray, d: float) -> np.ndarray:
    """photo.py camera (p = R x + (0,0,-d)) as an OpenCV cam2world 4x4 in the centred-grid world."""
    Rcv = FLIP @ R
    tcv = FLIP @ np.array([0, 0, -d])
    T = np.eye(4)
    T[:3, :3] = Rcv.T
    T[:3, 3] = -Rcv.T @ tcv
    return T


def fits(dv, do, u, v, ref):
    def st(dfit):
        P = np.stack([u * dfit, v * dfit, -dfit], 1)
        return [float(np.median(np.abs(dfit - do)) * 100),
                float(np.sqrt((((P - P.mean(0)) - (ref - ref.mean(0))) ** 2).sum(1).mean()) * 100)]
    A2 = np.stack([dv, np.ones_like(dv), u, v], 1)
    c2, *_ = np.linalg.lstsq(A2, do, rcond=None)
    return st, st(A2 @ c2)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--sam", action="store_true", help="score inside the SAM 'fabric' mask (segment.py first)")
    ap.add_argument("--single", action="store_true", help="also run the base view alone (img, K)")
    ap.add_argument("--model", default="facebook/map-anything")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs
    dev = torch.device("cuda")
    t0 = time.time()
    model = MapAnything.from_pretrained(args.model).to(dev).eval()
    print(f"{args.model} loaded in {time.time() - t0:.0f}s")

    def run(items, **kw):
        """items: list of (png, K or None, T or None). Returns z-depth (H0, W0) per view."""
        views = []
        for png, K, T in items:
            v = {"img": np.asarray(Image.open(png).convert("RGB"))}
            if K is not None:
                v["intrinsics"] = torch.from_numpy(K.astype(np.float32))
            if T is not None:
                v["camera_poses"] = torch.from_numpy(T.astype(np.float32))
                v["is_metric_scale"] = torch.tensor([True])
            views.append(v)
        proc = preprocess_inputs(views)
        with torch.no_grad():
            preds = model.infer(proc, memory_efficient_inference=False, use_amp=True, amp_dtype="bf16",
                                apply_mask=False, mask_edges=False, **kw)
        W0, H0 = Image.open(items[0][0]).size
        out = []
        for p in preds:
            z = p["depth_z"][0, ..., 0].float()[None, None]
            out.append(F.interpolate(z, (H0, W0), mode="bilinear", align_corners=False)[0, 0].cpu().numpy())
        return out

    rows = []
    for pdir in args.photos:
        man = json.load(open(os.path.join(pdir, "manifest.json")))
        views = {os.path.splitext(v["png"])[0]: v for v in man["views"]}
        n = 0
        for stem, v in sorted(views.items()):
            if v.get("partner", -1) < 0 or v["partner"] < v["view"]:
                continue
            if args.limit and n >= args.limit:
                break
            pstem = stem[: stem.rfind("_v")] + f"_v{v['partner']}"
            g = np.load(os.path.join(pdir, f"{stem}_gt.npz"))
            gp = np.load(os.path.join(pdir, f"{pstem}_gt.npz"))
            mask = g["mask"] & (g["depth"] > 0)
            if args.sam:
                mp = os.path.join(pdir, "sam3", f"{stem}__fabric.png")
                if not os.path.exists(mp):
                    continue
                mask &= np.asarray(Image.open(mp).convert("L")) > 127
            if mask.sum() < 50:
                continue
            n += 1
            rr, cc = np.nonzero(mask)
            size, fov = int(g["size"]), float(g["fov_deg"])
            ref = unproject(mask, g["depth"], fov)
            do = -ref[:, 2]
            ext = math.tan(math.radians(fov) / 2)
            u = ((cc + 0.5) / size - 0.5) * 2 * ext
            vv = (0.5 - (rr + 0.5) / size) * 2 * ext
            K = true_K(size, fov)
            T0, T1 = cam2world_cv(g["R"], float(g["d"])), cam2world_cv(gp["R"], float(gp["d"]))
            b_true = float(np.linalg.norm(T0[:3, 3] - T1[:3, 3]))
            png0, png1 = os.path.join(pdir, stem + ".png"), os.path.join(pdir, pstem + ".png")
            modes = {"img": [(png0, None, None), (png1, None, None)],
                     "K": [(png0, K, None), (png1, K, None)],
                     "K+pose": [(png0, K, T0), (png1, K, T1)]}
            if args.single:
                modes["single_img"] = [(png0, None, None)]
                modes["single_K"] = [(png0, K, None)]
            row = {"stem": stem, "set": os.path.basename(os.path.normpath(pdir)), "baseline_cm": b_true * 100}
            line = f"{stem:34s} b {b_true * 100:4.1f}cm"
            for name, items in modes.items():
                t1 = time.time()
                z = run(items)[0]
                dv = z[rr, cc]
                ok = np.isfinite(dv) & (dv > 0)
                if ok.sum() < 50:
                    row[name] = None
                    line += f" | {name}: no depth"
                    continue
                st, quad = fits(dv[ok], do[ok], u[ok], vv[ok], ref[ok])
                s_or = float(np.median(do[ok]) / np.median(dv[ok]))
                row[name] = {"oracle_scale": st(dv[ok] * s_or), "raw": st(dv[ok]), "quad": quad,
                             "scale_err": 1 / s_or - 1, "valid_frac": float(ok.mean()), "seconds": time.time() - t1}
                line += (f" | {name}: shape {row[name]['oracle_scale'][0]:5.2f} RAW {row[name]['raw'][0]:5.2f} "
                         f"({row[name]['scale_err'] * 100:+5.1f}%) quad {quad[0]:4.2f}")
            rows.append(row)
            print(line)
        sub = [r for r in rows if r["set"] == os.path.basename(os.path.normpath(pdir))]
        print(f"\n{os.path.basename(os.path.normpath(pdir))}: median over {len(sub)} pairs, "
              f"depth med|err| cm / point RMSE cm; raw = metric, no oracle")
        for name in modes:
            rs = [r[name] for r in sub if r.get(name)]
            if not rs:
                continue
            med = lambda k, i: float(np.median([r[k][i] for r in rs]))
            se = np.abs([r["scale_err"] for r in rs])
            print(f"  {name:10s} shape(oracle scale) {med('oracle_scale', 0):5.2f} / {med('oracle_scale', 1):5.2f}   "
                  f"RAW {med('raw', 0):5.2f} / {med('raw', 1):5.2f}   quad {med('quad', 0):4.2f}   "
                  f"|scale err| med {np.median(se) * 100:.1f}% p90 {np.percentile(se, 90) * 100:.1f}%  "
                  f"bias {np.median([r['scale_err'] for r in rs]) * 100:+.1f}%  n={len(rs)}")
        print()
    out = os.path.join(os.path.dirname(os.path.normpath(args.photos[0])), "stereo", "mapanything.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"args": vars(args), "records": rows}, open(out, "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
