"""Simulation-in-the-loop refinement (analysis by synthesis, 2026-11-02, JOBS.md M22).

The network f proposes, the simulator checks, the network corrects:

  y_0     = f(obs)                                  guess 1: the network on the observed sheet
  s_k     = sim(y_k)                                the guess simulated with the dataset's own physics
  y_{k+1} = y_k + step * (y_0 - f(s_k))             f(s_k) is how the network reads the guess's own sheet;
                                                    if y_k were right it would read y_0 again
  fit_k   = centred RMSE(s_k, obs)                  likelihood of guess k in observation space

i.e. a fixed-point / Newton iteration on f(sim(y)) = f(obs) with the network as the approximate
inverse Jacobian (no extra training). Force slots are an unordered pair, so f(s_k) is matched to
y_k's slots before differencing; directions are renormalised, log-magnitude and duration add.
With --iters K, guesses 1..K are simulated and guess K+1 is the last unchecked update; "best fit"
is the simulated guess with the lowest fit_k. --floor also simulates the true forces from the same
(different from the sample's) initial jitter: the chaos floor of fit and of f's self-consistency.

Observations: --obs <dir of h5> (a clean split dir = perfect information; a chain split's
test dir = the photo path, grids from cluster/chain_split.sbatch). The simulator config and the
true forces are read from each file (chain files carry the source's sim_config / force_params).

  python refine.py --run runs/M21c --obs $S5/test --out runs/M21c/refine_clean --limit 4 --iters 3 --floor
  python refine.py ... --shard i/n          then   python refine.py --out <same> --merge
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import decode_forces, encode_forces  # noqa: E402
from eval_grid import GridRun, grid_input  # noqa: E402
from eval_resim import Simulator, grid_error, load_sample, with_footprint  # noqa: E402
from metrics import NAMES, pair_metrics  # noqa: E402

SEED = 4242  # initial jitter of every simulation: not the sample's own (the microstate is unknown)


def predict(run: GridRun, grid: np.ndarray) -> np.ndarray:
    pool = grid.shape[0] // run.in_size
    x, g = grid_input(grid.astype(np.float32), None, pool, "oracle")
    return run.predict(x[None], g[None])[0]  # (2, 7)


def match(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """a's two slots reordered to ref's (smaller summed u, v, direction distance)."""
    cost = lambda b: sum(np.linalg.norm(b[i, :5] - ref[i, :5]) for i in range(2))
    return min((a[list(p)] for p in itertools.permutations(range(2))), key=cost)


def update(y: np.ndarray, y0: np.ndarray, ys: np.ndarray, step: float) -> np.ndarray:
    out = y + step * (y0 - match(ys, y))
    for i in range(2):
        out[i, 2:5] /= max(np.linalg.norm(out[i, 2:5]), 1e-8)
        out[i, 0:2] = np.clip(out[i, 0:2], 0.02, 0.98)
        out[i, 6] = np.clip(out[i, 6], 0.1, 1.0)
    return out


def metrics(y: np.ndarray, ytrue: np.ndarray) -> dict:
    return dict(zip(NAMES, pair_metrics(y[None], ytrue[None], sym=False)[0].tolist()))


def run_one(path: str, run: GridRun, simulator_box: list, args) -> dict:
    obs, _, cfg, true_fp = load_sample(path)
    ytrue = encode_forces(true_fp)
    if simulator_box[0] is None:
        simulator_box[0] = Simulator(cfg, workdir=args.out, device=args.device)
    sim = simulator_box[0]
    settle = float(cfg["t_capture"]) - max(float(f["duration"]) for f in true_fp[:2])  # dataset constant (cap)

    def simulate(y):
        fp = with_footprint(decode_forces(y), true_fp)
        t_cap = max(f["duration"] for f in fp) + settle
        g, _ = sim.run(fp, t_cap, seed=SEED)
        if not np.isfinite(g).all():
            raise FloatingPointError("non-finite simulated grid")
        return g

    row = {"file": os.path.basename(path), "guesses": []}
    t0 = time.time()
    y0 = predict(run, obs)
    y = y0.copy()
    for k in range(args.iters + 1):
        gk = {"k": k + 1, "y": y.tolist(), "m": metrics(y, ytrue)}
        if k < args.iters:
            try:
                s = simulate(y)
            except Exception as e:  # a guess the simulator cannot run ends the loop (kept, unchecked)
                gk["sim_failed"] = str(e)[-200:]
                row["guesses"].append(gk)
                break
            gk["fit"] = grid_error(s, obs)
            ys = predict(run, s)
            gk["self"] = metrics(match(ys, y), y)  # how far f reads its own guess back
            y = update(y, y0, ys, args.step)
        row["guesses"].append(gk)
    if args.floor:
        try:
            s = simulate(ytrue)
            row["floor"] = {"fit": grid_error(s, obs), "self": metrics(match(predict(run, s), ytrue), ytrue)}
        except Exception as e:
            row["floor"] = {"failed": str(e)[-200:]}
    row["seconds"] = time.time() - t0
    fits = [g["fit"]["rmse_centered_m"] for g in row["guesses"] if "fit" in g]
    row["best_fit_k"] = int(np.argmin(fits)) + 1 if fits else 1
    print(f"{row['file']}: pos " + " -> ".join(f"{g['m']['pos_cm']:.1f}" for g in row["guesses"])
          + " cm | dir " + " -> ".join(f"{g['m']['dir_deg']:.0f}" for g in row["guesses"])
          + " | fit " + " ".join(f"{f * 100:.1f}" for f in fits)
          + (f" | floor fit {row['floor']['fit']['rmse_centered_m'] * 100:.1f} cm" if args.floor and "fit" in row.get("floor", {}) else "")
          + f" cm | best k={row['best_fit_k']} | {row['seconds']:.0f}s", flush=True)
    return row


def report(rows: list[dict], out: str) -> None:
    K = max(len(r["guesses"]) for r in rows)
    summ = {"n": len(rows), "per_guess": []}
    for k in range(K):
        ms = [r["guesses"][min(k, len(r["guesses"]) - 1)]["m"] for r in rows]  # a stopped loop keeps its last guess
        fits = [r["guesses"][k]["fit"]["rmse_centered_m"] for r in rows if k < len(r["guesses"]) and "fit" in r["guesses"][k]]
        summ["per_guess"].append({"k": k + 1, "pos_cm": float(np.mean([m["pos_cm"] for m in ms])),
                                  "pos_cm_median": float(np.median([m["pos_cm"] for m in ms])),
                                  "dir_deg": float(np.mean([m["dir_deg"] for m in ms])),
                                  "dir_deg_median": float(np.median([m["dir_deg"] for m in ms])),
                                  "mag_rel": float(np.mean([m["mag_rel"] for m in ms])),
                                  "dur_s": float(np.mean([m["dur_s"] for m in ms])),
                                  "fit_cm": float(np.mean(fits) * 100) if fits else None})
    best = [r["guesses"][r["best_fit_k"] - 1]["m"] for r in rows]
    summ["best_fit"] = {"pos_cm": float(np.mean([m["pos_cm"] for m in best])),
                        "pos_cm_median": float(np.median([m["pos_cm"] for m in best])),
                        "dir_deg": float(np.mean([m["dir_deg"] for m in best])),
                        "dir_deg_median": float(np.median([m["dir_deg"] for m in best]))}
    fl = [r["floor"] for r in rows if "fit" in r.get("floor", {})]
    if fl:
        summ["floor"] = {"fit_cm": float(np.mean([f["fit"]["rmse_centered_m"] for f in fl]) * 100),
                         "self_pos_cm": float(np.mean([f["self"]["pos_cm"] for f in fl])),
                         "self_dir_deg": float(np.mean([f["self"]["dir_deg"] for f in fl])), "n": len(fl)}
    summ["n_sim_failed"] = sum(any("sim_failed" in g for g in r["guesses"]) for r in rows)
    json.dump({"summary": summ, "rows": rows}, open(os.path.join(out, "refine.json"), "w"), indent=1)
    for p in summ["per_guess"]:
        print(f"guess {p['k']}: pos {p['pos_cm']:.2f} (med {p['pos_cm_median']:.2f}) cm  dir {p['dir_deg']:.1f} "
              f"(med {p['dir_deg_median']:.1f}) deg  mag {p['mag_rel']:.2f}  dur {p['dur_s']:.3f} s  fit {p['fit_cm']}")
    print("best fit:", json.dumps(summ["best_fit"]), "| floor:", json.dumps(summ.get("floor")),
          "| sim failed:", summ["n_sim_failed"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None)
    ap.add_argument("--obs", default=None, help="dir of observation h5 (clean split dir or chain test dir)")
    ap.add_argument("--glob", default="*.h5", help="file pattern in --obs (e.g. '*_v0.h5': one view per sheet)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=3, help="simulations per sample")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.merge:
        rows = sum((json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.out, "shard_*.json")))), [])
        report(rows, args.out)
        return
    files = sorted(glob.glob(os.path.join(args.obs, args.glob)))[: args.limit or None]
    tag = "all"
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        files, tag = files[i::n], f"{i}of{n}"
    run = GridRun(args.run, torch.device(args.device))
    box, rows = [None], []
    for f in files:
        rows.append(run_one(f, run, box, args))
        json.dump(rows, open(os.path.join(args.out, f"shard_{tag}.json"), "w"))
    report(rows, args.out)


if __name__ == "__main__":
    main()
