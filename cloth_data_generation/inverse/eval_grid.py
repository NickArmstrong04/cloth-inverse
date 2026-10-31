"""Score a rung-1 grid checkpoint (cnn_reg, e.g. M16-a `--hybrid`) on indexed-grid inputs that did
NOT come through train.py's loader: oracle grids from h5 files with controlled degradations
(velocity zeroed, global feature replaced), or the photo-derived grids of grid_from_views.py
(hidden vertices filled, visible ones at the stereo depth, g from the lifted cloud). Rebuilds the
network as train.py did (models.build + the saved args) and applies the same input convention:
a hybrid / intrinsic model takes raw metres and raw g (it standardises with its x_scale / g_scale
buffers), a plain model takes x / xstd and g / gstd. Metrics = train.py's (pair_metrics, sym off).

  python eval_grid.py --run runs/M16a --h5 ~/cloth/ds256_const_test16          # sanity: = metrics.json test
  python eval_grid.py --run runs/M16a --h5 ... --vel zero --g cloud              # cost of no velocity, photo g
  python eval_grid.py --run runs/M16a --npz ../../cloth_results/photos/stereo20_test/grids --variants all
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
from data import _pool, encode_forces  # noqa: E402
from geometry import kabsch  # noqa: E402
from metrics import NAMES, pair_metrics, summarize  # noqa: E402
from models import build  # noqa: E402
from photo import list_h5  # noqa: E402


class GridRun:
    def __init__(self, run: str, dev):
        ck = torch.load(os.path.join(run, "model.pt"), map_location=dev, weights_only=False)
        self.args = a = ck["args"]
        assert not a.get("view"), "eval_grid is for grid (rung 1/2) checkpoints; use eval_views for --view runs"
        state = {k[2:] if k.startswith("m.") else k: v for k, v in ck["state"].items()}
        self.xstd, self.gstd = ck["xstd"].to(dev), ck["gstd"].to(dev)
        self.cin, self.gdim = self.xstd.shape[1], self.gstd.shape[1]
        mean, std = ck["scaler"]
        self.mean, self.std = mean.to(dev), std.to(dev)
        self.dev = dev
        self.pool = int(a.get("pool") or 1)
        self.hybrid = bool(a.get("hybrid"))
        self.intrinsic = bool(a.get("intrinsic")) or self.hybrid
        self.kabsch = bool(a.get("kabsch"))
        # input resolution the encoder was built for: 8 px after n stride-2 blocks (stem stride 2 above 256)
        n_blocks = 1 + max(int(k.split(".")[2]) for k in state if k.startswith("enc.blocks."))
        self.in_size = 8 * 2 ** n_blocks if n_blocks <= 5 else 16 * 2 ** n_blocks
        self.model = build(a["model"], uv_scale=(self.mean[:2], self.std[:2]), w=a.get("width", 32),
                           in_size=self.in_size, cin=self.cin, gdim=self.gdim, aux=(a.get("aux_w") or 0) > 0,
                           intrinsic=bool(a.get("intrinsic")), pool="max" if a.get("pool_max") else "mean",
                           strain_scale=a.get("strain_scale") or 10.0, hybrid=self.hybrid,
                           intrinsic_rel=bool(a.get("intrinsic_rel")), patch_head=a.get("patch_head") or 0).to(dev)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if not self.hybrid:
            missing = [k for k in missing if k not in ("x_scale", "g_scale")]
        assert not missing, f"missing tensors: {missing[:5]}"
        assert not unexpected, f"unexpected tensors: {unexpected[:5]}"
        self.model.eval()

    def describe(self):
        a = self.args
        return (f"{a['model']} cin={self.cin} gdim={self.gdim} in_size={self.in_size} hybrid={self.hybrid} "
                f"intrinsic={self.intrinsic} kabsch={self.kabsch} no_vel={bool(a.get('no_vel'))}")

    @torch.no_grad()
    def predict(self, X: np.ndarray, G: np.ndarray, bs: int = 8) -> np.ndarray:
        """X (N, C, H, W) raw metres (C >= cin; extra channels dropped), G (N, gdim) raw. Returns (N, 2, 7)."""
        out = []
        for i in range(0, len(X), bs):
            x = torch.from_numpy(X[i:i + bs, : self.cin]).to(self.dev)
            g = torch.from_numpy(G[i:i + bs, : self.gdim]).to(self.dev)
            R = None
            if self.kabsch:
                R = kabsch(x[:, :3])
                rot = lambda t: torch.einsum("bij,bjhw->bihw", R, t)
                x = torch.cat([rot(x[:, 3 * k:3 * k + 3]) for k in range(x.shape[1] // 3)], 1)
                g = torch.cat([torch.einsum("bij,bj->bi", R, g[:, 3 * k:3 * k + 3]) for k in range(g.shape[1] // 3)], 1)
            raw = self.intrinsic
            pred, _ = self.model(x if raw else x / self.xstd, g if raw else g / self.gstd, None)
            pred = pred.float() * self.std + self.mean
            if R is not None:
                pred[..., 2:5] = torch.einsum("bji,bkj->bki", R, pred[..., 2:5])
            pred[..., 2:5] = torch.nn.functional.normalize(pred[..., 2:5], dim=-1)
            out.append(pred.cpu().numpy())
        return np.concatenate(out)


def grid_input(pos: np.ndarray, vel: np.ndarray | None, pool: int, g_mode: str = "oracle",
               g_override: np.ndarray | None = None):
    """train.py's X / G from a (ny, nx, 3) world-frame grid: X = (pos - centroid, vel), G = (centroid, mean vel),
    centroid taken before pooling (data.load_split). vel None -> zero velocity channels and zero mean velocity."""
    c = pos.reshape(-1, 3).mean(0)
    v = vel if vel is not None else np.zeros_like(pos)
    mv = v.reshape(-1, 3).mean(0) if vel is not None else np.zeros(3, np.float32)
    x = _pool(np.concatenate([pos - c, v], -1).astype(np.float32), pool).transpose(2, 0, 1)
    if g_mode == "zero":
        g = np.zeros(6, np.float32)
    elif g_mode == "override":
        g = np.concatenate([g_override.astype(np.float32), mv])
    else:
        g = np.concatenate([c, mv]).astype(np.float32)
    return x, g


def load_h5(path: str):
    import h5py
    with h5py.File(path, "r") as h:
        pos = h["final_grid"][()].astype(np.float32)
        vel = h["final_vel"][()].astype(np.float32) if "final_vel" in h else None
        fp = json.loads(h["force_params"][()].decode() if isinstance(h["force_params"][()], bytes)
                        else str(h["force_params"][()]))
    return pos, vel, fp


def score(run: GridRun, X, G, Y, sym: bool = False):
    pred = run.predict(np.stack(X), np.stack(G))
    m = pair_metrics(pred, np.stack(Y), sym=sym)
    return summarize(m), m, pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--h5", nargs="*", default=[], help="h5 files / dirs / globs: oracle grids")
    ap.add_argument("--vel", choices=["oracle", "zero"], default="oracle", help="velocity channels for --h5")
    ap.add_argument("--g", choices=["oracle", "zero"], default="oracle", help="global feature for --h5")
    ap.add_argument("--npz", default=None, help="dir of grid_from_views.py outputs")
    ap.add_argument("--variants", nargs="*", default=["all"], help="npz variant names (default: all found)")
    ap.add_argument("--sym", action="store_true", help="D4-symmetric metrics (train.py: off for rung 1)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = GridRun(args.run, dev)
    print(f"run {args.run}: {run.describe()}")
    res = {"run": args.run, "args": vars(args), "rows": {}}

    def report(name, s, n):
        res["rows"][name] = s | {"n": n}
        print(f"{name:34s} n={n:3d}  pos {s['pos_cm_mean']:5.2f} cm (med {s['pos_cm_median']:5.2f})  "
              f"dir {s['dir_deg_mean']:5.1f} deg (med {s['dir_deg_median']:5.1f})  mag {s['mag_rel_mean']:.3f}  "
              f"impulse {s['impulse_rel_mean']:.3f}  F {s['mag_N_mean']:.3f} N")

    if args.h5:
        files = list_h5(args.h5)[: args.limit or None]
        X, G, Y = [], [], []
        for f in files:
            pos, vel, fp = load_h5(f)
            pool = pos.shape[0] // run.in_size  # grid resolution -> the encoder's input size (train.py --pool)
            assert pool * run.in_size == pos.shape[0], f"{f}: grid {pos.shape[0]} vs model input {run.in_size}"
            x, g = grid_input(pos, None if args.vel == "zero" else vel, pool, args.g)
            X.append(x); G.append(g); Y.append(encode_forces(fp))
        s, m, pred = score(run, X, G, Y, args.sym)
        report(f"h5 vel={args.vel} g={args.g}", s, len(X))
        res["per_sample"] = {os.path.basename(f): dict(zip(NAMES, m[i].tolist())) for i, f in enumerate(files)}
    if args.npz:
        files = sorted(glob.glob(os.path.join(args.npz, "*_grids.npz")))[: args.limit or None]
        assert files, f"no *_grids.npz in {args.npz}"
        first = np.load(files[0])
        names = [k[2:] for k in first.files if k.startswith("X_")]
        if args.variants != ["all"]:
            names = [n for n in args.variants if n in names]
        for name in names:
            X, G, Y = [], [], []
            for f in files:
                z = np.load(f)
                x = z[f"X_{name}"]
                pool = x.shape[-1] // run.in_size
                if pool > 1:  # grid resolution -> the encoder's input size (train.py --pool)
                    x = _pool(x.transpose(1, 2, 0), pool).transpose(2, 0, 1)
                X.append(x); G.append(z[f"G_{name}"]); Y.append(z["y"])
            s, m, pred = score(run, X, G, Y, args.sym)
            report(name, s, len(X))
            res.setdefault("per_view", {})[name] = {os.path.basename(f)[: -len("_grids.npz")]: dict(zip(NAMES, m[i].tolist()))
                                                   for i, f in enumerate(files)}
    out = args.out or os.path.join(args.run, "eval_grid.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
