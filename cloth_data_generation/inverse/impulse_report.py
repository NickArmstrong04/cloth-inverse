"""Impulse (F*t) as a single target: read-only analysis of test_predictions.json files.

usage: python impulse_report.py runs/M13a runs/M13b ...   (dirs or json paths)

Per paired force: eF = log(F_pred/F_true), et = log(t_pred/t_true), eJ = eF + et.
Reports median relative errors for F, t, J, a constant-predictor floor, the
correlation of eF with et (negative = model trades force against duration),
and the variance split var(eJ) = var(eF) + var(et) + 2 cov.
"""
import json, os, sys
import numpy as np

D4 = [lambda u, v: (u, v), lambda u, v: (1 - u, v), lambda u, v: (u, 1 - v), lambda u, v: (1 - u, 1 - v),
      lambda u, v: (v, u), lambda u, v: (1 - v, u), lambda u, v: (v, 1 - u), lambda u, v: (1 - v, 1 - u)]


def cost(p, t, f):
    u, v = f(p["u"], p["v"])
    pos = np.hypot(u - t["u"], v - t["v"]) * 100.0  # sheet units -> ~cm scale, matches metrics weighting
    ang = np.degrees(np.arccos(np.clip(np.dot(p["direction"], t["direction"]), -1, 1)))
    return pos + ang / 10


def pair(pred, true):
    best, bc = None, None
    for f in D4:
        for a, b in ((0, 1), (1, 0)):
            c = cost(pred[0], true[a], f) + cost(pred[1], true[b], f)
            if bc is None or c < bc:
                best, bc = ((pred[0], true[a]), (pred[1], true[b])), c
    return best


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "test_predictions.json")
    return json.load(open(path))


def analyse(recs):
    seen = set()  # runs trained before 2026-10-30 with --no-rot hold every test sample twice
    recs = [r for r in recs if not (r["file"] in seen or seen.add(r["file"]))]
    eF, et, F, T, Fp = [], [], [], [], []
    for r in recs:
        for p, t in pair(r["pred"], r["true"]):
            eF.append(np.log(p["magnitude"] / t["magnitude"]))
            et.append(np.log(p["duration"] / t["duration"]))
            F.append(t["magnitude"]); T.append(t["duration"]); Fp.append(p["magnitude"])
    eF, et, F, T, Fp = map(np.array, (eF, et, F, T, Fp))
    eJ = eF + et
    rel = lambda e: np.median(np.abs(np.exp(e) - 1))
    # constant-predictor floor: predict the test-set geometric mean / median
    cF = np.log(np.exp(np.log(F).mean()) / F)
    cT = np.log(np.median(T) / T)
    cJ = np.log(np.exp(np.log(F * T).mean()) / (F * T))
    vF, vT, cov = eF.var(), et.var(), np.cov(eF, et)[0, 1]
    return dict(n=len(eF), F=rel(eF), t=rel(et), J=rel(eJ), F0=rel(cF), t0=rel(cT), J0=rel(cJ),
                r=np.corrcoef(eF, et)[0, 1], vF=vF, vT=vT, cov=cov, vJ=eJ.var(),
                # R^2 of log-prediction vs log-truth
                R2F=1 - vF / np.log(F).var(), R2J=1 - eJ.var() / np.log(F * T).var(),
                # absolute force error in newtons (mean / median) and linear R^2: the units the
                # tactile-sensing literature reports (force range here 0.15-1.25 N)
                maeN=np.abs(Fp - F).mean(), medN=np.median(np.abs(Fp - F)), maeN0=np.abs(F.mean() - F).mean(),
                R2lin=1 - ((Fp - F) ** 2).mean() / F.var())


if __name__ == "__main__":
    print(f"{'run':14s} {'n':>5s} | median |rel err|  F      t      J   | const floor  F      t      J   | "
          f"corr(eF,et)  var eF   var et   2cov    var eJ |  R2 logF  R2 logJ |  MAE N  med N  const N  R2 lin")
    for a in sys.argv[1:]:
        s = analyse(load(a))
        name = os.path.basename(a.rstrip("/\\").replace("test_predictions.json", "").rstrip("/\\"))
        print(f"{name:14s} {s['n']:5d} |                 {s['F']:.3f}  {s['t']:.3f}  {s['J']:.3f} |"
              f"              {s['F0']:.3f}  {s['t0']:.3f}  {s['J0']:.3f} |"
              f"   {s['r']:+.2f}      {s['vF']:.3f}    {s['vT']:.3f}   {2*s['cov']:+.3f}   {s['vJ']:.3f} |"
              f"  {s['R2F']:+.2f}    {s['R2J']:+.2f} |  {s['maeN']:.3f}  {s['medN']:.3f}  {s['maeN0']:.3f}   {s['R2lin']:+.2f}")
