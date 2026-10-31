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
pdirs = [a for a in sys.argv[1:] if not a.startswith("--")]

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
        (d2, K2, C0), (_, _, C1) = run([os.path.join(pdir, stem + ".png"), os.path.join(pdir, pstem + ".png")])
        dv2 = -unproject_vggt(mask, d2, K2)[:, 2]
        st2, aff2, quad2 = fits(dv2, do, u, vv, ref)
        b_pred = float(np.linalg.norm(C0 - C1)); b_true = float(v["baseline"])
        s_base = b_true / max(b_pred, 1e-6)
        s_or2 = float(np.median(do) / np.median(dv2))
        row = dict(stem=stem, set=os.path.basename(os.path.normpath(pdir)), baseline_cm=b_true * 100,
                   single_oracle_scale=st1(dv1 * s_or), single_affz=aff1, single_quad=quad1,
                   stereo_oracle_scale=st2(dv2 * s_or2), stereo_baseline_scale=st2(dv2 * s_base),
                   stereo_affz=aff2, stereo_quad=quad2, scale_err=s_base / s_or2 - 1,
                   f_ratio=float(K2[0, 0] / (size / (2 * ext))))
        rows.append(row)
        os.makedirs(os.path.join(pdir, "views"), exist_ok=True)
        pts = unproject_vggt(mask, d2, K2); cen = pts.mean(0)
        np.savez_compressed(os.path.join(pdir, "views", f"{stem}_pts_stereo.npz"), pts=pts - cen,
                            g=np.concatenate([cen, np.zeros(3, np.float32)]), scale_to_oracle=s_or2,
                            scale_baseline=s_base, K=K2, source="stereo", material=str(g["material"]),
                            R=g["R"], d=float(g["d"]), size=size, fov_deg=float(g["fov_deg"]), mask_source="oracle")
        print(f"{stem:34s} b {b_true*100:4.1f}cm | single or-scale {row['single_oracle_scale'][0]:5.2f} quad {quad1[0]:5.2f} | "
              f"stereo or-scale {row['stereo_oracle_scale'][0]:5.2f} BASELINE-scale {row['stereo_baseline_scale'][0]:5.2f} "
              f"(scale err {row['scale_err']*100:+5.1f}%) affz {aff2[0]:5.2f} quad {quad2[0]:5.2f} | f {row['f_ratio']:.2f}")
    med = lambda k, i: float(np.median([r[k][i] for r in rows if r["set"] == os.path.basename(os.path.normpath(pdir))]))
    sub = [r for r in rows if r["set"] == os.path.basename(os.path.normpath(pdir))]
    print(f"\n{os.path.basename(os.path.normpath(pdir))}: median over {len(sub)} pairs, med|err| cm / point RMSE cm")
    for k in ["single_oracle_scale", "single_affz", "single_quad", "stereo_oracle_scale", "stereo_baseline_scale", "stereo_affz", "stereo_quad"]:
        print(f"  {k:22s} {med(k,0):5.2f} / {med(k,1):5.2f}")
    se = np.array([r["scale_err"] for r in sub])
    print(f"  baseline scale error: median {np.median(np.abs(se))*100:.1f}%, p90 {np.percentile(np.abs(se), 90)*100:.1f}%, bias {np.median(se)*100:+.1f}%\n")
json.dump(rows, open(os.path.join(os.path.dirname(os.path.normpath(pdirs[0])), "vggt_stereo.json"), "w"), indent=1)
