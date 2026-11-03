"""Extra diagnostics for refine.py outputs (M22): oracle-best guess (lower bound for any selector over the
simulated guesses), how often best-by-fit picks the truly best guess, and fit vs error correlation.
  python refine_stats.py runs/M21c/refine_clean [more dirs]
"""
import json
import sys

import numpy as np

for d in sys.argv[1:]:
    rows = [r for r in json.load(open(f"{d}/refine.json"))["rows"] if sum("fit" in g for g in r["guesses"]) >= 3]
    pos = np.array([[g["m"]["pos_cm"] for g in r["guesses"][:3]] for r in rows if len(r["guesses"]) >= 3])
    dr = np.array([[g["m"]["dir_deg"] for g in r["guesses"][:3]] for r in rows if len(r["guesses"]) >= 3])
    fit = np.array([[g["fit"]["rmse_centered_m"] * 100 for g in r["guesses"][:3]] for r in rows if len(r["guesses"]) >= 3])
    bf = fit.argmin(1)
    ob = pos.argmin(1)
    i = np.arange(len(pos))
    r_all = np.corrcoef(fit.ravel(), pos.ravel())[0, 1]
    r_within = np.mean([np.corrcoef(f, p)[0, 1] for f, p in zip(fit, pos) if np.std(f) > 0 and np.std(p) > 0])
    print(f"== {d} (n={len(pos)} with 3 simulated guesses)")
    print(f"  guess1 {pos[:, 0].mean():.2f} cm / {dr[:, 0].mean():.1f} deg | best-by-fit {pos[i, bf].mean():.2f} / {dr[i, bf].mean():.1f}"
          f" | oracle-best-of-3 {pos[i, ob].mean():.2f} / {dr[i, ob].mean():.1f}")
    print(f"  best-by-fit = oracle-best in {np.mean(bf == ob) * 100:.0f}% | best-by-fit picks guess 1/2/3: {np.bincount(bf, minlength=3).tolist()}")
    print(f"  guess k better than guess 1 (pos): k2 {np.mean(pos[:, 1] < pos[:, 0]) * 100:.0f}%  k3 {np.mean(pos[:, 2] < pos[:, 0]) * 100:.0f}%")
    print(f"  corr(fit, pos err): pooled {r_all:.2f}, within-sample mean {r_within:.2f}")
    print(f"  by guess-1 error: <3 cm n={np.sum(pos[:, 0] < 3)} best-fit delta {np.mean(pos[pos[:, 0] < 3][np.arange(np.sum(pos[:, 0] < 3)), bf[pos[:, 0] < 3]] - pos[pos[:, 0] < 3, 0]):+.2f} cm;"
          f" >=3 cm n={np.sum(pos[:, 0] >= 3)} delta {np.mean(pos[pos[:, 0] >= 3][np.arange(np.sum(pos[:, 0] >= 3)), bf[pos[:, 0] >= 3]] - pos[pos[:, 0] >= 3, 0]) if np.any(pos[:, 0] >= 3) else float('nan'):+.2f} cm")
