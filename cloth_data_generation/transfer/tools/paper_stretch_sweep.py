"""Paper membrane-stretch sweep: is the 10-35% stretch a material or a solver effect?

Runs the paper preset on a few dataset force configurations, varying one setting at
a time (VBD iterations, substeps, membrane stiffness tri_ke, crease softening), and
reports how far the membrane stretches.

Reading the result:
- stretch falls as ITERATIONS / SUBSTEPS rise  -> the solver is under-converged
  (tension does not reach the rest of the sheet within a substep)
- stretch does not fall (or rises) with TRI_KE at fixed iterations -> stiffness is not
  the lever until the solver converges
- the max sits on creases and disappears with soften 1.0 -> crease softening, not membrane

Forces are drawn exactly as generate_batch.py draws dataset slot k (seed SEED0 + k,
first attempt), and the resolution-scaled radius/margin/gap match the dataset runs.
Each output HDF5 is kept, so a rerun skips finished runs and only re-scores them.

Run from cloth_data_generation/transfer:   python tools/paper_stretch_sweep.py
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import csv
import json
import os
import time

import h5py
import numpy as np
import warp as wp

from cloth_sim.validate import edge_stretch_map
from generate_batch import sample_forces
from generate_sample import SessionCache, build_parser, configure, generate

# ----------------------------------------------------------------------------- CONFIG
OUTDIR = "paper_stretch_sweep"   # relative to transfer/
DEVICE = "cuda:0"                # "cpu" works but a 256 sample takes hours on CPU
NX = 256                         # dataset resolution (not swept)
SEED0 = 100000                   # dataset seed0; slot k uses SEED0 + k
SLOTS = [0, 1, 2]                # which dataset slots (force configurations) to run
LOADING = "free"                 # force distribution of the paper datasets

# Paper physics exactly as cluster/paper.sh pins it (the baseline of every run).
BASE = {
    "yield_curvature": 50.0,
    "edge_ke": 200.0,
    "crease_soften": 0.1,
    "settle": 0.0,
    "tri_ke": 1.0e4,     # also sets tri_ka (generate_sample.py)
    "iterations": 20,    # SimConfig default
    "substeps": 30,      # SimConfig default
}

# Each run changes only the listed keys relative to BASE.
RUNS = {
    "base":            {},
    "it60":            {"iterations": 60},
    "it200":           {"iterations": 200},
    "sub90":           {"substeps": 90},
    "ke3e4":           {"tri_ke": 3.0e4},
    "ke1e5":           {"tri_ke": 1.0e5},
    "ke1e5_it200":     {"tri_ke": 1.0e5, "iterations": 200},
    "no_soften":       {"crease_soften": 1.0},
}

PATCH_RADIUS_CELLS = 9.0  # a stretch maximum within this many cells of a force centre is "patch"
# ------------------------------------------------------------------------------------


def res_scale() -> float:
    return NX / 128.0


def sample_argv(params: dict, seed: int, force_json: str, out: str) -> list[str]:
    s = res_scale()
    return [
        "--seed", str(seed), "--force-json", force_json, "--out", out,
        "--nx", str(NX), "--ny", str(NX), "--material", "paper", "--device", DEVICE,
        # resolution scaling exactly as generate_batch.py applies it
        "--radius", f"{2.0e-3 / s:.6g}", "--margin", f"{1.0e-3 / s:.6g}",
        "--gap", f"{0.5e-3 / s:.6g}",
        "--yield-curvature", str(params["yield_curvature"]),
        "--edge-ke", str(params["edge_ke"]),
        "--crease-soften", str(params["crease_soften"]),
        "--settle", str(params["settle"]),
        "--tri-ke", str(params["tri_ke"]),
        "--iterations", str(params["iterations"]),
        "--substeps", str(params["substeps"]),
    ]


def write_forces(slot: int) -> tuple[int, str, list[dict]]:
    seed = SEED0 + slot
    rng = np.random.default_rng(10_000 + seed)  # same draw as generate_batch.run_one
    forces = sample_forces(rng, res_scale(), LOADING)
    path = os.path.join(OUTDIR, f"forces_slot{slot:04d}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(forces, f, indent=2)
    return seed, path, forces


def score(h5_path: str, forces: list[dict]) -> dict:
    """Stretch statistics at capture and over the saved snapshots, plus where the max sits."""
    with h5py.File(h5_path, "r") as h:
        cfg = json.loads(h["sim_config"][()])
        final = h["final_grid"][()].astype(np.float64)
        snaps = h["intermediate_grids"][()].astype(np.float64)
        vmax = float(h["sim_log/max_speed"][()].max())
        crease = h["plastic/crease_map"][()] if "plastic/crease_map" in h else None
    rest_h = cfg["sheet"]["size_x"] / (cfg["nx"] - 1)
    if not np.all(np.isfinite(final)):
        return {"nan": True}

    s_final = edge_stretch_map(final, rest_h)
    s_peak = max(float(edge_stretch_map(g, rest_h).max()) for g in snaps) if len(snaps) else np.nan

    j, i = np.unravel_index(int(np.argmax(s_final)), s_final.shape)
    centres = [(f["u"] * (NX - 1), f["v"] * (NX - 1)) for f in forces]
    d_patch = min(np.hypot(i - cu, j - cv) for cu, cv in centres)
    return {
        "nan": False,
        "wall_s": cfg["wall_time_s"],
        "stretch_p50": float(np.percentile(s_final, 50)),
        "stretch_p99": float(np.percentile(s_final, 99)),
        "stretch_max": float(s_final.max()),
        "stretch_peak_snapshots": s_peak,
        "max_at": "patch" if d_patch <= PATCH_RADIUS_CELLS else "elsewhere",
        "max_dist_cells": float(d_patch),
        "crease_at_max_rad": float(crease[j, i]) if crease is not None else np.nan,
        "vmax": vmax,
    }


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    wp.init()
    parser = build_parser()
    cache = SessionCache()
    slots = [write_forces(k) for k in SLOTS]
    rows = []
    for name, change in RUNS.items():
        params = BASE | change
        for (seed, fpath, forces), slot in zip(slots, SLOTS):
            out = os.path.join(OUTDIR, f"{name}_slot{slot:04d}.h5")
            if not os.path.exists(out):
                print(f"[{name} slot {slot}] {change or 'baseline'}", flush=True)
                t0 = time.time()
                spec = configure(parser.parse_args(sample_argv(params, seed, fpath, out)))
                generate(spec, cache=cache, verbose=False)
                print(f"    done in {time.time() - t0:.0f} s", flush=True)
            rows.append({"run": name, "slot": slot, **params, **score(out, forces)})

    keys = ["run", "slot", "iterations", "substeps", "tri_ke", "crease_soften", "nan", "wall_s",
            "stretch_p50", "stretch_p99", "stretch_max", "stretch_peak_snapshots",
            "max_at", "max_dist_cells", "crease_at_max_rad", "vmax"]
    csv_path = os.path.join(OUTDIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'run':<14}{'p99':>8}{'max':>8}{'peak':>8}{'max at':>11}{'wall s':>9}  (mean over slots)")
    for name in RUNS:
        r = [x for x in rows if x["run"] == name]
        if any(x["nan"] for x in r):
            print(f"{name:<14}  NaN in {sum(x['nan'] for x in r)}/{len(r)} slots")
            continue
        at = "/".join(x["max_at"] for x in r)
        print(f"{name:<14}{np.mean([x['stretch_p99'] for x in r]):8.3f}"
              f"{np.mean([x['stretch_max'] for x in r]):8.3f}"
              f"{np.mean([x['stretch_peak_snapshots'] for x in r]):8.3f}"
              f"{at:>11}{np.mean([x['wall_s'] for x in r]):9.0f}")
    print(f"\nper-sample table: {csv_path}")


if __name__ == "__main__":
    main()
