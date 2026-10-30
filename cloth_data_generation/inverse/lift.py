"""Rung 4b, step 3: lift the segmented photo to 3D with VGGT (Meta) in place of the oracle depth.

Runs in the SAM 3 venv (cluster/setup_sam3.sh installs VGGT; weights facebook/VGGT-1B).
For every view with a `_gt.npz` (photo.py --gt) and a SAM mask (segment.py, default prompt
"fabric"): VGGT predicts a depth map and camera intrinsics from the single RGB image; the
pixels inside the SAM mask are unprojected into VGGT's camera frame and converted to ours
(camera at the origin looking down -z, y up: x -> x, y -> -y, z -> -z).

A monocular lift has no absolute scale, so the cloud is scored after a single scale factor
fitted to the oracle (median ratio of depths inside the mask) and that factor is reported;
the package stores the unscaled cloud plus `scale_to_oracle`, and the loader side decides
(ROADMAP: normalise by the known 0.5 m sheet, or --view-norm). Metrics per view, oracle
pixels inside the SAM mask only: depth relative error after scale (median |pred*s/gt - 1|),
point RMSE after scale (cm), and the scale factor itself (a real photo cannot have it).

  python lift.py --photos ../../cloth_results/photos/cloth256 ../../cloth_results/photos/paper256 --prompt fabric
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from package_views import unproject  # noqa: E402  (oracle pinhole, for the reference cloud)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]


def load_vggt(dev):
    for cand in [os.path.expanduser("~/vggt")]:
        if os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
    return model


def predict_depth(model, png: str, dev, dtype):
    """VGGT on one image. Returns depth (H, W) in VGGT's camera frame at the original image
    size, confidence (H, W), and intrinsics K (3, 3) rescaled to the original size."""
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    images = load_and_preprocess_images([png]).to(dev)  # (1, 3, h, w), h/w multiples of 14
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        pred = model(images)
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    depth = pred["depth"][0, 0, ..., 0].float()  # (h, w)
    conf = pred["depth_conf"][0, 0].float()
    K = intr[0, 0].float().cpu().numpy()
    W0, H0 = Image.open(png).size
    h, w = depth.shape
    depth = F.interpolate(depth[None, None], (H0, W0), mode="bilinear", align_corners=False)[0, 0]
    conf = F.interpolate(conf[None, None], (H0, W0), mode="bilinear", align_corners=False)[0, 0]
    S = np.diag([W0 / w, H0 / h, 1.0])
    return depth.cpu().numpy(), conf.cpu().numpy(), S @ K


def similarity_rmse_cm(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE (cm) of a vs b after the best similarity transform a -> b (Umeyama: rotation,
    translation, isotropic scale) on corresponding points: what a pose- and scale-invariant
    model sees of the lifting error."""
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(A.T @ B / len(a))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    sc = (S * np.diag(D)).sum() / (A ** 2).sum(1).mean()
    res = sc * A @ R - B
    return float(np.sqrt((res ** 2).sum(1).mean())) * 100


def unproject_vggt(mask: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """OpenCV pinhole (x right, y down, z forward) -> our camera frame (y up, z backward)."""
    rows, cols = np.nonzero(mask)
    z = depth[rows, cols]
    x = (cols + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (rows + 0.5 - K[1, 2]) / K[1, 1] * z
    return np.stack([x, -y, -z], -1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", nargs="+", required=True)
    ap.add_argument("--prompt", default="fabric")
    ap.add_argument("--oracle-mask", action="store_true", help="use the oracle mask instead of the SAM mask")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--known-fov", action="store_true",
                    help="unproject VGGT's depth with the render's true intrinsics (a phone reports its focal "
                         "length too) instead of VGGT's predicted ones: separates depth error from intrinsics error")
    args = ap.parse_args()
    dev = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    t0 = time.time()
    model = load_vggt(dev)
    print(f"VGGT loaded in {time.time() - t0:.0f}s")
    for pdir in args.photos:
        out = os.path.join(pdir, "views")
        os.makedirs(out, exist_ok=True)
        recs = []
        gts = sorted(glob.glob(os.path.join(pdir, "*_gt.npz")))[: args.limit or None]
        for gp in gts:
            stem = os.path.basename(gp)[: -len("_gt.npz")]
            g = np.load(gp)
            if args.oracle_mask:
                mask = g["mask"]
            else:
                mp = os.path.join(pdir, "sam3", f"{stem}__{slug(args.prompt)}.png")
                if not os.path.exists(mp):
                    continue
                mask = np.asarray(Image.open(mp).convert("L")) > 127
            mask = mask & (g["depth"] > 0)  # score only where the oracle has depth
            if mask.sum() < 50:
                recs.append({"stem": stem, "N": int(mask.sum()), "skipped": True})
                continue
            depth, conf, K = predict_depth(model, os.path.join(pdir, stem + ".png"), dev, dtype)
            if args.known_fov:
                Wd = mask.shape[1]
                f = Wd / (2 * math.tan(math.radians(float(g["fov_deg"])) / 2))
                K = np.array([[f, 0, Wd / 2], [0, f, mask.shape[0] / 2], [0, 0, 1.0]])
            pts = unproject_vggt(mask, depth, K)
            ref = unproject(mask, g["depth"], float(g["fov_deg"]))  # same pixels, same order
            s = float(np.median(-ref[:, 2]) / max(np.median(-pts[:, 2]), 1e-6))
            rel = np.abs(pts[:, 2] * s / ref[:, 2] - 1)
            ps = pts * s
            rmse = float(np.sqrt((((ps - ps.mean(0)) - (ref - ref.mean(0))) ** 2).sum(1).mean())) * 100
            # focal length check: our render used a 40 deg horizontal FOV -> f = W / (2 tan 20)
            f_true = mask.shape[1] / (2 * math.tan(math.radians(float(g["fov_deg"])) / 2))
            sim = similarity_rmse_cm(pts, ref)
            rec = {"stem": stem, "N": int(len(pts)), "scale_to_oracle": s, "point_rmse_cm_similarity": sim,
                   "depth_rel_err_median": float(np.median(rel)), "depth_rel_err_p90": float(np.percentile(rel, 90)),
                   "point_rmse_cm_scaled_centred": rmse, "f_pred_over_true": float(K[0, 0] / f_true),
                   "conf_median": float(np.median(conf[mask])), "material": str(g["material"])}
            recs.append(rec)
            centroid = pts.mean(0)
            tag = "vggt_knownfov" if args.known_fov else "vggt"
            np.savez_compressed(os.path.join(out, f"{stem}_pts_{tag}.npz"), pts=pts - centroid,
                                g=np.concatenate([centroid, np.zeros(3, np.float32)]),
                                scale_to_oracle=s, K=K, source="vggt", material=str(g["material"]),
                                R=g["R"], d=float(g["d"]), size=int(g["size"]), fov_deg=float(g["fov_deg"]))
            print(f"{stem:40s} N={len(pts):6d} scale {s:.2f}  rel err med {rec['depth_rel_err_median']:.3f} "
                  f"p90 {rec['depth_rel_err_p90']:.3f}  rmse {rmse:.2f} cm  sim {sim:.2f} cm  f ratio {rec['f_pred_over_true']:.2f}")
        ok = [r for r in recs if not r.get("skipped")]
        if ok:
            med = lambda k: float(np.median([r[k] for r in ok]))
            summary = {"n": len(ok), "depth_rel_err_median": med("depth_rel_err_median"),
                       "depth_rel_err_p90": med("depth_rel_err_p90"), "point_rmse_cm": med("point_rmse_cm_scaled_centred"),
                       "point_rmse_cm_similarity": med("point_rmse_cm_similarity"),
                       "scale_median": med("scale_to_oracle"),
                       "scale_spread": float(np.std(np.log([r["scale_to_oracle"] for r in ok]))),
                       "f_ratio_median": med("f_pred_over_true")}
            print(os.path.basename(pdir), json.dumps(summary))
        with open(os.path.join(out, f"{'vggt_knownfov' if args.known_fov else 'vggt'}_scores.json"), "w") as fh:
            json.dump({"prompt": args.prompt, "records": recs, "summary": summary if ok else None}, fh, indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
