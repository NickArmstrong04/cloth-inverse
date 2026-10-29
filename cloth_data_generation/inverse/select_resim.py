"""Recommendation 3, second half: pick among the flow head's K candidate force pairs by
re-simulation (analysis by synthesis). For each test sample, every candidate from
test_candidates.json (train.py --flow) is simulated with generate_sample.py and compared with
the true final grid; the candidate with the smallest centred RMSE is selected. Reports the
parameter metrics of (i) the selected candidate, (ii) the best-of-K oracle, (iii) the mean
candidate, so the gain of re-sim selection over the regression head is explicit.

Cost: K simulations per test sample (~45 s each at 256 on one GPU). Use --limit / --k and run
as a job array (--shard i/n) for the full split; merge the shard JSONs with --merge."""

from __future__ import annotations

import argparse
import glob
import json
import os

import h5py
import numpy as np

from eval_resim import grid_error, sim

NAMES = ["pos_cm", "dir_deg", "mag_rel", "dur_s", "impulse_rel"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True, help="test_candidates.json from train.py --flow")
    ap.add_argument("--root", required=True, help="splits dir (reads <root>/test/<file>)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N test samples")
    ap.add_argument("--k", type=int, default=None, help="use only the first k candidates")
    ap.add_argument("--shard", default=None, help="i/n: process every n-th sample starting at i (job arrays)")
    ap.add_argument("--merge", action="store_true", help="merge <out>/shard_*.json into resim_select.json and exit")
    args = ap.parse_args()
    out = os.path.abspath(args.out or os.path.join(os.path.dirname(args.cands), "resim_select"))
    os.makedirs(out, exist_ok=True)
    if args.merge:
        rows = sum((json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "shard_*.json")))), [])
        _report(rows, out)
        return
    items = json.load(open(args.cands))[: args.limit]
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        items = items[i::n]
    rows = []
    for j, it in enumerate(items):
        name = it["file"]
        with h5py.File(os.path.join(args.root, "test", name), "r") as h:
            true_grid = h["final_grid"][()]
        base = os.path.join(out, name.replace(".h5", ""))
        cands = it["candidates"][: args.k]
        errs = []
        for k, forces in enumerate(cands):
            p = f"{base}_c{k}.h5"
            try:
                sim(forces, p, seed=900 + j)
                with h5py.File(p, "r") as h:
                    errs.append(grid_error(h["final_grid"][()], true_grid)["rmse_centered_m"])
            except Exception as e:  # a candidate that blows the simulator is simply not selected
                print(f"  {name} candidate {k} failed: {str(e)[-200:]}", flush=True)
                errs.append(float("inf"))
            finally:
                if os.path.exists(p):
                    os.remove(p)
        m = np.asarray(it["cand_metrics"])[: len(cands)]  # (k, 5) parameter metrics per candidate
        sel = int(np.argmin(errs))
        rows.append({"file": name, "selected": sel, "resim_rmse": errs, "selected_metrics": m[sel].tolist(),
                     "oracle_metrics": m[m[:, 0].argmin()].tolist(), "mean_metrics": m.mean(0).tolist()})
        print(f"{name}: selected {sel} (rmse {errs[sel]:.3f} m) pos {m[sel, 0]:.1f} cm, oracle {m[:, 0].min():.1f} cm, "
              f"mean {m[:, 0].mean():.1f} cm", flush=True)
    tag = args.shard.replace("/", "of") if args.shard else "all"
    json.dump(rows, open(os.path.join(out, f"shard_{tag}.json"), "w"))
    _report(rows, out)


def _report(rows, out):
    summ = {"n": len(rows)}
    for key in ("selected_metrics", "oracle_metrics", "mean_metrics"):
        v = np.asarray([r[key] for r in rows])
        summ[key] = {f"{n}_mean": float(v[:, i].mean()) for i, n in enumerate(NAMES)}
    json.dump({"summary": summ, "rows": rows}, open(os.path.join(out, "resim_select.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
