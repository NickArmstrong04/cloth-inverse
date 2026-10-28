"""Re-simulate predicted forces for the test split and compare the resulting
final state with the true one. Also re-simulates the TRUE forces in the same
(non-deterministic) mode to measure the irreducible chaos floor.

Runs generate_sample.py from the parent cloth directory as a subprocess."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLOTH = os.path.dirname(HERE)


def sim(forces: list[dict], out_h5: str, seed: int) -> None:
    fj = out_h5.replace(".h5", "_forces.json")
    with open(fj, "w") as f:
        json.dump(forces, f)
    p = subprocess.run([sys.executable, os.path.join(CLOTH, "generate_sample.py"), "--seed", str(seed),
                        "--force-json", fj, "--out", out_h5], capture_output=True, text=True, cwd=CLOTH)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])


def grid_error(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.linalg.norm(a - b, axis=-1)
    ca, cb = a.reshape(-1, 3).mean(0), b.reshape(-1, 3).mean(0)
    dc = np.linalg.norm((a - ca) - (b - cb), axis=-1)
    return {"rmse_m": float(np.sqrt((d ** 2).mean())), "max_m": float(d.max()),
            "centroid_err_m": float(np.linalg.norm(ca - cb)),
            "rmse_centered_m": float(np.sqrt((dc ** 2).mean()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="test_predictions.json from train.py")
    ap.add_argument("--root", default=os.path.join(CLOTH, "dataset"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-floor", action="store_true")
    args = ap.parse_args()
    out = os.path.abspath(args.out or os.path.join(os.path.dirname(args.preds), "resim"))
    os.makedirs(out, exist_ok=True)
    preds = json.load(open(args.preds))[: args.limit]
    rows = []
    for k, item in enumerate(preds):
        name = item["file"]
        with h5py.File(os.path.join(args.root, "test", name), "r") as h:
            true_grid = h["final_grid"][()]
        base = os.path.join(out, name.replace(".h5", ""))
        sim(item["pred"], base + "_pred.h5", seed=900 + k)
        with h5py.File(base + "_pred.h5", "r") as h:
            pg = h["final_grid"][()]
        row = {"file": name, "pred_vs_true": grid_error(pg, true_grid), "param_metrics": item["metrics"]}
        if not args.skip_floor:
            sim(item["true"], base + "_floor.h5", seed=900 + k)
            with h5py.File(base + "_floor.h5", "r") as h:
                fg = h["final_grid"][()]
            row["floor_resim_true_vs_true"] = grid_error(fg, true_grid)
        rows.append(row)
        print(name, "pred rmse %.3f m (centered %.3f)" % (row["pred_vs_true"]["rmse_m"], row["pred_vs_true"]["rmse_centered_m"]),
              ("floor rmse %.3f m (centered %.3f)" % (row["floor_resim_true_vs_true"]["rmse_m"],
                                                        row["floor_resim_true_vs_true"]["rmse_centered_m"]) if not args.skip_floor else ""),
              flush=True)
    summ = {"n": len(rows)}
    for key in ("pred_vs_true", "floor_resim_true_vs_true"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            summ[key] = {m: float(np.mean([v[m] for v in vals])) for m in vals[0]}
    # scale reference: extent of the true deformed sheets
    json.dump({"summary": summ, "rows": rows}, open(os.path.join(out, "resim_metrics.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
