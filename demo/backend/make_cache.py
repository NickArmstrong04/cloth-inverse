"""Build the precomputed-result cache (demo step 6, kept as record).

Draws spread-out force configurations, runs the warm sim + M16-a on each,
and stores one ``*.json`` per entry in ``demo/backend/cache/`` holding the
exact job result payload plus its true forces.

Usage (WSL, warm GPU, ~15 x 13 s):
    /home/narmstrong/cloth/.venv/bin/python demo/backend/make_cache.py \
        --ckpt demo/backend/checkpoints/M16a --n 15 --seed0 200000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "cloth_data_generation", "transfer"))

from generate_batch import sample_forces  # noqa: E402
from simulate import SimService, downsample_frames, encode_frames, validate_forces  # noqa: E402
import inference as inf  # noqa: E402
from cache import CACHE_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed0", type=int, default=200000)
    args = ap.parse_args()
    os.makedirs(CACHE_DIR, exist_ok=True)
    ckpt, _ = inf.load_checkpoint(args.ckpt)
    svc = SimService()
    made, skipped = 0, 0
    seed = args.seed0
    while made < args.n:
        forces = sample_forces(np.random.default_rng(seed), 2.0, "free")
        seed += 1
        try:
            forces = validate_forces(forces)
        except ValueError:
            skipped += 1
            continue
        t0 = time.time()
        try:
            sim = svc.run(forces, nx=256, snapshots=30, seed=seed)
        except Exception as e:
            print(f"seed {seed}: sim FAILED ({e}), skipped", flush=True)
            skipped += 1
            continue
        pred, predicted = inf.predict_forces(ckpt, sim["final_pos"], sim["final_vel"])
        per_force, mean = inf.score_forces(ckpt, pred, forces)
        entry = {
            "frames": encode_frames(downsample_frames(sim["frames"], 64)),
            "true_forces": [dict(f, impulse=f["magnitude"] * f["duration"]) for f in forces],
            "predicted_forces": predicted,
            "reversal_forces": [dict(f, direction=[-x for x in f["direction"]]) for f in predicted],
            "metrics_mean": {k: mean[k] for k in ("pos_cm", "dir_deg", "impulse_rel")},
            "timings": {"sim_s": sim["wall_s"]},
            "grid": 256,
            "seed": seed,
        }
        with open(os.path.join(CACHE_DIR, f"entry_{made:02d}.json"), "w") as fh:
            json.dump(entry, fh)
        made += 1
        print(f"entry {made}/{args.n}: pos {mean['pos_cm']:.2f}cm dir {mean['dir_deg']:.1f}deg "
              f"({time.time() - t0:.1f}s, skipped {skipped})", flush=True)
    print(f"-> {CACHE_DIR} ({made} entries, {skipped} skipped)")


if __name__ == "__main__":
    main()
