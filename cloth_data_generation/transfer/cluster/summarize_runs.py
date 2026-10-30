#!/usr/bin/env python3
"""One line per training run: python3 cluster/summarize_runs.py [runs_glob ...]
Reads ../inverse/runs/<name>/metrics.json (stdlib only, fine on the login node).
Default: every run under inverse/runs. Example: python3 cluster/summarize_runs.py 'M1*' 'M2*'"""
import glob, json, os, sys
inv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "inverse", "runs")
pats = sys.argv[1:] or ["*"]
dirs = sorted({d for p in pats for d in glob.glob(os.path.join(inv, p))})
print(f"{'run':28} {'split':10} {'pos cm':>7} {'dir deg':>8} {'mag rel':>8} {'mag N':>6} {'dur s':>6} {'imp rel':>8}")
for d in dirs:
    p = os.path.join(d, "metrics.json")
    name = os.path.basename(d)
    if not os.path.exists(p):
        print(f"{name:28} (no metrics.json yet)"); continue
    m = json.load(open(p))
    for split in ("train", "val", "test"):
        keys = [k for k in m if k.startswith(split + "/")]
        for k in keys:
            r = m[k]
            if k.endswith("/resim"):  # eval_resim.py: centred RMSE in m, score 0 = floor, 1 = chance
                g = lambda key: r.get(key, {}).get("rmse_centered_m", float("nan"))
                sc, fb = r.get("score"), r.get("frac_beats_chance")
                print(f"{name:28} {k:10} resim n={r['n']} pred {g('pred'):.4f} floor {g('floor'):.4f} chance {g('chance'):.4f} m"
                      + (f"  score {sc:.2f}  beats chance {fb:.0%}" if sc is not None else ""))
                continue
            mn = f"{r['mag_N_mean']:6.3f}" if "mag_N_mean" in r else "     -"
            print(f"{name:28} {k:10} {r['pos_cm_mean']:7.2f} {r['dir_deg_mean']:8.1f} {r['mag_rel_mean']:8.2f} {mn} {r['dur_s_mean']:6.3f} {r['impulse_rel_mean']:8.2f}")
