"""128-vs-256 domain-shift measurement (demo step 2, kept as experiment record).

Draws 20 force configurations with ``sample_forces`` (``free`` loading, same
as ``ds256_const``), simulates each at 128 and at 256 with the same seed
(same u/v/direction/magnitude/duration; only the footprint in cells
differs), upsamples the 128 final state to 256, scores both with the M16-a
wrapper, and writes the error tables + grid decision to ``demo/README.md``.

Usage (WSL, warm GPU):
    /home/narmstrong/cloth/.venv/bin/python demo/backend/measure_grid.py \
        --ckpt demo/backend/checkpoints/M16a --n 20 --seed0 100000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "cloth_data_generation", "transfer"))

from generate_batch import sample_forces  # noqa: E402
from simulate import SimService  # noqa: E402
import inference as inf  # noqa: E402


def draw(seed: int, res_scale: float) -> list[dict]:
    return sample_forces(np.random.default_rng(seed), res_scale, "free")


def run_pair(svc: SimService, ckpt, seed: int, sims: dict) -> dict | None:
    f128, f256 = draw(seed, 1.0), draw(seed, 2.0)
    try:
        r128, r256 = sims[128][seed], sims[256][seed]
    except Exception as e:  # NaN / QC failure: skip this seed, note it
        return {"seed": seed, "failed": str(e)}
    up, uvel = inf.upsample_indexed(r128["final_pos"], r128["final_vel"], 256)
    p128, _ = inf.predict_forces(ckpt, up, uvel)
    p256, _ = inf.predict_forces(ckpt, r256["final_pos"], r256["final_vel"])
    _, m128 = inf.score_forces(ckpt, p128, f128)
    _, m256 = inf.score_forces(ckpt, p256, f256)
    return {"seed": seed, "m128": m128, "m256": m256,
            "wall128": r128["wall_s"], "wall256": r256["wall_s"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=100000)
    args = ap.parse_args()
    ckpt, _ = inf.load_checkpoint(args.ckpt)
    svc = SimService()
    seeds = [args.seed0 + i for i in range(args.n)]
    # one grid at a time: the warm session is per-grid, no rebuild per seed
    sims = {}
    for nx, rs in ((128, 1.0), (256, 2.0)):
        sims[nx] = {}
        for s in seeds:
            try:
                sims[nx][s] = svc.run(draw(s, rs), nx=nx, snapshots=5, seed=s)
                print(f"nx={nx} seed {s}: ok ({sims[nx][s]['wall_s']:.1f}s)", flush=True)
            except Exception as e:
                sims[nx][s] = e
                print(f"nx={nx} seed {s}: FAILED ({e})", flush=True)
    rows, fails = [], []
    for s in seeds:
        if isinstance(sims[128][s], Exception) or isinstance(sims[256][s], Exception):
            fails.append({"seed": s, "failed": f"128: {sims[128][s]} / 256: {sims[256][s]}"})
            continue
        r = run_pair(svc, ckpt, s, sims)
        rows.append(r)
        print(f"seed {r['seed']}: 128 pos {r['m128']['pos_cm']:.2f}cm dir {r['m128']['dir_deg']:.1f}deg "
              f"imp {r['m128']['impulse_rel']:.3f} ({r['wall128']:.1f}s) | 256 pos {r['m256']['pos_cm']:.2f}cm "
              f"dir {r['m256']['dir_deg']:.1f}deg imp {r['m256']['impulse_rel']:.3f} ({r['wall256']:.1f}s)",
              flush=True)
    assert rows, "all simulations failed"
    s128 = {k: float(np.mean([r["m128"][k] for r in rows])) for k in
            ("pos_cm", "dir_deg", "mag_rel", "dur_s", "impulse_rel", "mag_N")}
    s256 = {k: float(np.mean([r["m256"][k] for r in rows])) for k in
            ("pos_cm", "dir_deg", "mag_rel", "dur_s", "impulse_rel", "mag_N")}
    med128 = float(np.median([r["m128"]["impulse_rel"] for r in rows]))
    med256 = float(np.median([r["m256"]["impulse_rel"] for r in rows]))
    w128 = float(np.mean([r["wall128"] for r in rows]))
    w256 = float(np.mean([r["wall256"] for r in rows]))
    print(f"\n128 path (upsampled): n={len(rows)} pos {s128['pos_cm']:.2f}cm dir {s128['dir_deg']:.1f}deg "
          f"imp-rel mean {s128['impulse_rel']:.3f} median {med128:.3f} wall {w128:.1f}s")
    print(f"256 path:             n={len(rows)} pos {s256['pos_cm']:.2f}cm dir {s256['dir_deg']:.1f}deg "
          f"imp-rel mean {s256['impulse_rel']:.3f} median {med256:.3f} wall {w256:.1f}s")
    ratio_pos = s128["pos_cm"] / max(s256["pos_cm"], 1e-9)
    ratio_dir = s128["dir_deg"] / max(s256["dir_deg"], 1e-9)
    grid = 256 if (ratio_pos > 2.0 or ratio_dir > 2.0) else 128
    print(f"ratios 128/256: pos {ratio_pos:.2f}x dir {ratio_dir:.2f}x -> grid {grid}")
    if fails:
        print(f"failed seeds ({len(fails)}): {[f['seed'] for f in fails]}")

    out = {"rows": rows, "failed": fails,
           "summary128": s128 | {"impulse_rel_median": med128, "wall_s": w128},
           "summary256": s256 | {"impulse_rel_median": med256, "wall_s": w256},
           "decision": grid}
    with open(os.path.join(_HERE, "grid_measure.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    md = [f"# Demo log (measurements; {__import__('datetime').date.today().isoformat()})",
          "",
          "## 128-vs-256 grid decision (step 2)",
          "",
          "20 random `free` force configurations (`sample_forces`, seeds "
          f"{args.seed0}-{args.seed0 + args.n - 1}), simulated at 128 and 256 "
          "with the same seeds, M16-a scored on both (128 upsampled bilinear "
          "to 256). Means; impulse-rel median in brackets.",
          "",
          "| path | pos cm | dir deg | impulse rel (median) | sim wall s |",
          "|---|---|---|---|---|",
          f"| 128 -> 256 | {s128['pos_cm']:.2f} | {s128['dir_deg']:.1f} | "
          f"{s128['impulse_rel']:.3f} ({med128:.3f}) | {w128:.1f} |",
          f"| 256 | {s256['pos_cm']:.2f} | {s256['dir_deg']:.1f} | "
          f"{s256['impulse_rel']:.3f} ({med256:.3f}) | {w256:.1f} |",
          "",
          f"Ratios 128/256: pos {ratio_pos:.2f}x, dir {ratio_dir:.2f}x. "
          f"**Decision: grid {grid}.**",
          ""]
    if fails:
        md += [f"Failed seeds ({len(fails)}, excluded): " +
               ", ".join(str(f["seed"]) for f in fails) + ".", ""]
    with open(os.path.join(_REPO, "demo", "README.md"), "w") as fh:
        fh.write("\n".join(md))
    print(f"-> demo/README.md (grid {grid}), demo/backend/grid_measure.json")


if __name__ == "__main__":
    main()
