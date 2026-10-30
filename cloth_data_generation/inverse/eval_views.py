"""Rung 4b, step 4: run a rung-4a point model on the photo packages (oracle / SAM / VGGT).

Loads a `--view` checkpoint (train.py: model.pt with args, state, scaler, xstd, gstd): the
point model (pointtf, visible camera-frame points) or an image model (cnn_reg / dino_reg on
the orthographic depth+mask view, re-rendered here from the package points with render.render
at the run's size / extent / splat), and evaluates it on
`<photos>/views/<stem>_pts_<source>.npz` from package_views.py (oracle, sam3) and lift.py
(vggt, vggt_knownfov). The same views appear under every source, so the differences are the
perception costs: oracle -> sam3 = segmentation, sam3 -> vggt = 3D lifting.

Input conventions follow train.forward_batch for the point model: camera-frame positions
restricted to the visible points (here: all package points), divided by the saved xstd;
the global feature is the sheet's WORLD centroid in training (its drift from the rest position,
rotated into the camera frame): in a free-flight dataset the sheet flies ~1 m along the applied
force, so it is a strong cue that a photo aimed at the sheet does not show. `--g zero` (default)
feeds zeros, `--g oracle` the true drift carried by the oracle package (photo.py --gt stores the
centroid; package_views writes g_world). Scale: `--scale metric` feeds metres as stored (oracle/SAM are metric;
VGGT is multiplied by its `scale_to_oracle`, i.e. an oracle scale, flagged in the output);
`--scale norm` rescales every cloud so its largest x/y extent is 0.8 * extent, as
`--view-norm` does in training (no oracle; use with checkpoints trained with --view-norm).
Targets are the package `y` (camera frame); metrics are D4-symmetric as in --view training.

  python eval_views.py --run runs/M9i --photos ../../cloth_results/photos/cloth256 [--sources oracle sam3 vggt]
  (M9i: cnn_reg on the depth view; M9f/M9g: pointtf on the visible points. Both are --view runs.)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import shade  # noqa: E402
from metrics import NAMES, pair_metrics, summarize  # noqa: E402
from models import PointTF, build  # noqa: E402
from render import render  # noqa: E402


class Run:
    """A --view checkpoint: the point model (pointtf) or an image model (cnn_reg / dino_reg on the
    rendered depth+mask view). Rebuilds the network as train.py did and applies its input scaling."""

    def __init__(self, run: str, dev):
        ck = torch.load(os.path.join(run, "model.pt"), map_location=dev, weights_only=False)
        self.args = a = ck["args"]
        assert a.get("view"), "eval_views needs a --view run"
        state = {k[2:] if k.startswith("m.") else k: v for k, v in ck["state"].items()}
        self.gdim = ck["gstd"].shape[1]
        self.xstd, self.gstd = ck["xstd"].to(dev), ck["gstd"].to(dev)
        mean, std = ck["scaler"]
        self.mean, self.std = mean.to(dev), std.to(dev)
        self.dev = dev
        self.point = a["model"] == "pointtf"
        self.size, self.extent, self.splat = a.get("view_size", 224), a.get("view_extent", 0.6), a.get("view_splat", 3)
        self.shade = bool(a.get("view_shade"))
        if self.point:
            d, cin = state["embed.0.weight"].shape
            self.model = PointTF(cin=cin, gdim=self.gdim, d=d, layers=a.get("pt_layers", 4),
                                 n_points=a.get("n_points", 2048), aux="aux.weight" in state)
        else:
            self.model = build(a["model"], uv_scale=(self.mean[:2], self.std[:2]), w=a.get("width", 32),
                               in_size=self.size, cin=3 if self.shade else 2, gdim=self.gdim, aux=a.get("aux_w", 0) > 0,
                               pool="max" if a.get("pool_max") else "mean", strain_scale=a.get("strain_scale") or 10.0,
                               arch=a.get("dino_arch"), size=self.size, finetune=a.get("dino_finetune", False),
                               repo=a.get("dino_repo"), weights=a.get("dino_weights"))
        self.model = self.model.to(dev)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        # x_scale / g_scale (GridModel buffers for --hybrid, M16) postdate the M9 checkpoints; they
        # default to ones and are unused outside hybrid mode, so an old checkpoint may lack them.
        missing = [k for k in missing if k not in ("x_scale", "g_scale")]
        assert not missing, f"missing tensors: {missing[:5]}"
        assert not unexpected, f"unexpected tensors: {unexpected[:5]}"
        self.model.eval()

    def describe(self):
        return (f"pointtf n_points={self.model.n_points}" if self.point else
                f"{self.args['model']} on a {self.size}px orthographic view, extent {self.extent} m, shade {self.shade}")

    @torch.no_grad()
    def predict(self, p: torch.Tensor, g_world: np.ndarray | None = None) -> np.ndarray:
        """p: (N, 3) camera-frame points, already scaled; g_world: camera-frame sheet centroid (3,) or None
        (zeros). Returns (2, 7) raw-scale prediction."""
        g = torch.zeros(1, self.gdim, device=self.dev)
        if g_world is not None:
            g[0, :3] = torch.from_numpy(np.asarray(g_world, np.float32)).to(self.dev) / self.gstd[0, :3]
        if self.point:
            x = p.T.reshape(1, 3, -1, 1) / self.xstd[:, :3]
            vis = torch.ones(1, x.shape[2], dtype=torch.bool, device=self.dev)
            pred, _ = self.model(x, g, None, vis=vis)
        else:
            img, _ = render(p[None], None, self.size, self.extent, self.splat)  # (1, 2, S, S) depth + mask
            if self.shade:
                img = shade(img, self.extent)
            scale = torch.tensor([self.extent, 1.0, 1.0], device=self.dev).view(1, 3, 1, 1)[:, : img.shape[1]]
            pred, _ = self.model(img / scale, g, None)
        pred = pred.float() * self.std + self.mean
        pred[:, :, 2:5] = torch.nn.functional.normalize(pred[:, :, 2:5], dim=-1)
        return pred[0].cpu().numpy()


def load_packages(photos: list[str], source: str) -> list[dict]:
    out = []
    for pdir in photos:
        for f in sorted(glob.glob(os.path.join(pdir, "views", f"*_pts_{source}.npz"))):
            z = np.load(f)
            if len(z["pts"]) < 16:
                continue
            stem = os.path.basename(f).split("_pts_")[0]
            out.append({"stem": stem, "set": os.path.basename(pdir), "pts": z["pts"].astype(np.float32),
                        "y": z["y"] if "y" in z else None,
                        "g_world": z["g_world"] if ("g_world" in z and bool(z["has_centroid"])) else None,
                        "scale_to_oracle": float(z["scale_to_oracle"]) if "scale_to_oracle" in z else 1.0,
                        "material": str(z["material"]) if "material" in z else ""})
    return out


def scaled_points(pts: np.ndarray, dev, scale: str, extent: float, scale_to_oracle: float) -> torch.Tensor:
    p = torch.from_numpy(pts).to(dev)
    if scale == "metric":
        return p * scale_to_oracle
    ext = (p[:, :2].amax(0) - p[:, :2].amin(0)).amax().clamp_min(1e-3)  # --view-norm: fit to 0.8 * extent
    return p * (0.8 * extent / ext)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--photos", nargs="+", required=True)
    ap.add_argument("--sources", nargs="*", default=["oracle", "sam3", "vggt"])
    ap.add_argument("--scale", choices=["metric", "norm"], default=None,
                    help="default: norm if the run used --view-norm, else metric")
    ap.add_argument("--g", choices=["zero", "oracle"], default="zero",
                    help="global feature: zeros, or the true world centroid (drift) from the oracle package")
    ap.add_argument("--passes", type=int, default=4, help="average the metrics over random point subsets")
    ap.add_argument("--out", default=None, help="json path (default <run>/eval_views.json)")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run(args.run, dev)
    scale = args.scale or ("norm" if run.args.get("view_norm") else "metric")
    extent = run.extent
    print(f"run {args.run}: {run.describe()}; trained view_norm={run.args.get('view_norm')} -> scale {scale}")
    orc = load_packages(args.photos, "oracle")  # labels (and the drift oracle) live in the oracle package
    labels = {p["stem"]: p["y"] for p in orc}
    gws = {p["stem"]: p["g_world"] for p in orc}
    if args.g == "oracle":
        missing = [s for s, v in gws.items() if v is None]
        assert not missing, f"--g oracle needs packages from photo.py --gt with the centroid (re-render); missing e.g. {missing[:3]}"
    else:
        gws = {s: None for s in gws}
    print(f"global feature: {args.g}")
    res = {"run": args.run, "scale": scale, "g": args.g, "sources": {}}
    rows = []
    for src in args.sources:
        pk = [p for p in load_packages(args.photos, src) if p["stem"] in labels and labels[p["stem"]] is not None]
        if not pk:
            print(f"{src}: no packages")
            continue
        ms = []
        for _ in range(args.passes):
            preds = np.stack([run.predict(scaled_points(p["pts"], dev, scale, extent, p["scale_to_oracle"]),
                                          gws[p["stem"]]) for p in pk])
            ms.append(pair_metrics(preds, np.stack([labels[p["stem"]] for p in pk]), sym=True))
        m = np.mean(ms, 0)
        s = summarize(m)
        s["n"] = len(pk)
        s["oracle_scale_used"] = scale == "metric" and src.startswith("vggt")
        per = {p["stem"]: dict(zip(NAMES, m[i].tolist())) for i, p in enumerate(pk)}
        res["sources"][src] = {"summary": s, "per_view": per}
        rows.append((src, s))
        print(f"{src:14s} n={len(pk):3d}  pos {s['pos_cm_mean']:5.1f} cm  dir {s['dir_deg_mean']:5.1f} deg  "
              f"mag {s['mag_rel_mean']:.2f}  impulse {s['impulse_rel_mean']:.2f}"
              + ("  (VGGT scaled with the oracle factor)" if s["oracle_scale_used"] else ""))
    out = args.out or os.path.join(args.run, "eval_views.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
