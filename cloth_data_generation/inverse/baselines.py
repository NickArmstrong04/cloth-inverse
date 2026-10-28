"""Non-learned / linear baselines: training-mean and ridge regression on
hand features. Forces are canonically ordered by u for the linear model."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from data import load_split
from metrics import pair_metrics, summarize


def features(d: dict) -> np.ndarray:
    x = d["x"].numpy()  # (N,6,ny,nx)
    N = x.shape[0]
    small = torch.nn.functional.avg_pool2d(d["x"], 8).numpy().reshape(N, -1)  # 6x16x16
    g = d["g"].numpy()
    pos = x[:, 0:3].reshape(N, 3, -1)
    cov = np.stack([np.cov(p) for p in pos]).reshape(N, -1)  # shape tensor (9)
    ext = np.stack([pos.max(-1) - pos.min(-1)])[0]  # extents (3)
    return np.concatenate([g, cov, ext, small], 1)


def canon(y: np.ndarray) -> np.ndarray:
    out = y.copy()
    for k in range(len(y)):
        if y[k, 0, 0] > y[k, 1, 0]:
            out[k] = y[k, ::-1]
    return out


def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xn = (X - mu) / sd
    Xa = np.concatenate([Xn, np.ones((len(X), 1))], 1)
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1])
    A[-1, -1] -= lam
    W = np.linalg.solve(A, Xa.T @ Y)
    return lambda Xq: np.concatenate([(Xq - mu) / sd, np.ones((len(Xq), 1))], 1) @ W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--out", default="runs/baselines")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tr, va, te = (load_split(args.root, s) for s in ("train", "val", "test"))
    Ytr = canon(tr["y"].numpy())
    res = {}

    mean = Ytr.mean(0, keepdims=True)
    for nm, d in (("val", va), ("test", te)):
        m = pair_metrics(np.repeat(mean, len(d["y"]), 0), d["y"].numpy())
        res[f"mean/{nm}"] = summarize(m)

    Xtr, Xva, Xte = features(tr), features(va), features(te)
    best = None
    for lam in (1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0):
        f = ridge_fit(Xtr, Ytr.reshape(len(Ytr), -1), lam)
        mv = pair_metrics(f(Xva).reshape(-1, 2, 7), va["y"].numpy())
        score = mv[:, 0].mean() + mv[:, 1].mean() / 10
        if best is None or score < best[0]:
            best = (score, lam, f)
    _, lam, f = best
    for nm, d, X in (("val", va, Xva), ("test", te, Xte)):
        m = pair_metrics(f(X).reshape(-1, 2, 7), d["y"].numpy())
        res[f"ridge/{nm}"] = summarize(m) | {"lambda": lam}
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    for k, v in res.items():
        print(f"{k:12s} pos {v['pos_cm_mean']:5.1f} cm  dir {v['dir_deg_mean']:5.1f} deg  "
              f"mag {v['mag_rel_mean']:.2f}  dur {v['dur_s_mean']:.3f} s  impulse {v['impulse_rel_mean']:.2f}")


if __name__ == "__main__":
    main()
