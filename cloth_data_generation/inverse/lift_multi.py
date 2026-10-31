"""VGGT with all views of a sample fed together (multi-view) vs one at a time. Per view, at the
SAM-mask pixels: depth error after oracle scale (median abs cm), after a per-view affine z
(a*dv+b) and after depth+tilt (a*dv+b+c*x+d*y) least squares, point RMSE. Scale = one factor per
SAMPLE for the multi-view run (VGGT's frames share a scale), one per view for the single run.

Runs in the SAM 3 venv. Uses the single-view sets (all views of a sample, random SO(3) cameras).
  ~/sam3env/bin/python lift_multi.py ../../cloth_results/photos/cloth256 ../../cloth_results/photos/paper256
"""
import glob, json, math, os, re, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lift import load_vggt, unproject_vggt
from package_views import unproject
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

dev = torch.device("cuda"); dtype = torch.bfloat16
model = load_vggt(dev)
pdirs = sys.argv[1:]
out_rows = []

def run(pngs):
    images = load_and_preprocess_images(pngs).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        pred = model(images)
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    W0, H0 = Image.open(pngs[0]).size
    res = []
    for i in range(len(pngs)):
        depth = pred["depth"][0, i, ..., 0].float()
        h, w = depth.shape
        depth = F.interpolate(depth[None, None], (H0, W0), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        K = np.diag([W0 / w, H0 / h, 1.0]) @ intr[0, i].float().cpu().numpy()
        res.append((depth, K))
    return res

def score(depth, K, g, sam):
    m = sam & (g["depth"] > 0); rr, cc = np.nonzero(m)
    pts = unproject_vggt(m, depth, K); ref = unproject(m, g["depth"], float(g["fov_deg"]))
    dv, do = -pts[:, 2], -ref[:, 2]
    fov = float(g["fov_deg"]); size = int(g["size"]); ext = math.tan(math.radians(fov) / 2)
    u = ((cc + 0.5) / size - 0.5) * 2 * ext; v = (0.5 - (rr + 0.5) / size) * 2 * ext
    def st(dfit):
        P = np.stack([u * dfit, v * dfit, -dfit], 1)
        return float(np.median(np.abs(dfit - do)) * 100), float(np.sqrt((((P - P.mean(0)) - (ref - ref.mean(0))) ** 2).sum(1).mean()) * 100)
    A = np.stack([dv, np.ones_like(dv)], 1); c1, *_ = np.linalg.lstsq(A, do, rcond=None)
    A2 = np.stack([dv, np.ones_like(dv), u, v], 1); c2, *_ = np.linalg.lstsq(A2, do, rcond=None)
    return dict(scale_view=float(np.median(do) / np.median(dv)), affz=st(A @ c1), quad=st(A2 @ c2), dv=dv, do=do, st=st)

for pdir in pdirs:
    stems = sorted(os.path.basename(p)[:-7] for p in glob.glob(os.path.join(pdir, "*_gt.npz")))
    samples = {}
    for s in stems: samples.setdefault(re.sub(r"_v\d+$", "", s), []).append(s)
    for samp, vs in samples.items():
        gts, sams, pngs, keep = [], [], [], []
        for s in vs:
            mp = os.path.join(pdir, "sam3", f"{s}__fabric.png")
            if not os.path.exists(mp): continue
            sam = np.asarray(Image.open(mp).convert("L")) > 127
            g = np.load(os.path.join(pdir, f"{s}_gt.npz"))
            if (sam & (g["depth"] > 0)).sum() < 50: continue
            gts.append(g); sams.append(sam); pngs.append(os.path.join(pdir, s + ".png")); keep.append(s)
        if len(pngs) < 2: continue
        multi = run(pngs)
        single = [run([p])[0] for p in pngs]
        sc_m = [score(d, K, g, sm) for (d, K), g, sm in zip(multi, gts, sams)]
        sc_s = [score(d, K, g, sm) for (d, K), g, sm in zip(single, gts, sams)]
        # one scale per sample for the multi-view run: median over all views' depth ratios
        s_samp = float(np.median(np.concatenate([r["do"] for r in sc_m])) / np.median(np.concatenate([r["dv"] for r in sc_m])))
        for s, rm, rs in zip(keep, sc_m, sc_s):
            row = dict(stem=s, set=os.path.basename(os.path.normpath(pdir)), n_views=len(pngs),
                       single_scale=rs["st"](rs["dv"] * rs["scale_view"]), single_affz=rs["affz"], single_quad=rs["quad"],
                       multi_scale_view=rm["st"](rm["dv"] * rm["scale_view"]), multi_scale_sample=rm["st"](rm["dv"] * s_samp),
                       multi_affz=rm["affz"], multi_quad=rm["quad"],
                       multi_scale_ratio_view_over_sample=rm["scale_view"] / s_samp)
            out_rows.append(row)
            print(f"{s:36s} single: scale {row['single_scale'][0]:5.2f} affz {row['single_affz'][0]:5.2f} quad {row['single_quad'][0]:5.2f} | "
                  f"multi({len(pngs)}): scale/view {row['multi_scale_view'][0]:5.2f} scale/sample {row['multi_scale_sample'][0]:5.2f} "
                  f"affz {row['multi_affz'][0]:5.2f} quad {row['multi_quad'][0]:5.2f}  (view/sample scale {row['multi_scale_ratio_view_over_sample']:.3f})")
med = lambda k, i: float(np.median([r[k][i] for r in out_rows]))
print(f"\nmedian over {len(out_rows)} views, med|err| cm / point RMSE cm")
for k in ["single_scale", "single_affz", "single_quad", "multi_scale_view", "multi_scale_sample", "multi_affz", "multi_quad"]:
    print(f"  {k:20s} {med(k,0):5.2f} / {med(k,1):5.2f}")
print("  multi view/sample scale ratio log-std %.3f" % np.std(np.log([r["multi_scale_ratio_view_over_sample"] for r in out_rows])))
json.dump(out_rows, open(os.path.join(os.path.dirname(os.path.normpath(pdirs[0])), "vggt_multi.json"), "w"), indent=1)
