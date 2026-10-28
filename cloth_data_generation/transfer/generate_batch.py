"""Run N independent cloth samples with random force parameters.

Batch strategy: one OS process per worker (default 8), each running
generate_sample.py as a subprocess. Instances trivially cannot interact
(no shared model/state). One HDF5 file per sample.

Newton does provide ModelBuilder.replicate() for multi-world single-model
batching; per-world force routing and per-world contact logging were not
verified within this prototype's budget, so process parallelism is used.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def sample_forces(rng: np.random.Generator, res_scale: float = 1.0) -> list[dict]:
    """Broad dataset distribution: full (u,v) coverage, uniform-sphere
    directions, log-uniform magnitudes (incl. weak folds), same-side pushes
    allowed, plus a 15% near-symmetric branch. Min material separation 0.2
    keeps the two patches distinct.
    """
    def one():
        d = rng.normal(size=3)
        d = d / np.linalg.norm(d)
        return {
            "u": float(rng.uniform(0.05, 0.95)),
            "v": float(rng.uniform(0.05, 0.95)),
            "direction": [float(x) for x in d],
            "magnitude": float(np.exp(rng.uniform(np.log(0.3), np.log(2.5)))),
            "duration": float(rng.uniform(0.3, 0.7)),
            "sigma_cells": 3.0 * res_scale, "radius_cells": 9.0 * res_scale,
        }
    if rng.uniform() < 0.15:
        f1 = one()
        f2 = dict(f1)
        f2["u"] = 1.0 - f1["u"]
        f2["v"] = f1["v"]
        d = np.array(f1["direction"])
        f2["direction"] = [float(-d[0]), float(d[1]), float(d[2])]
        return [f1, f2]
    for _ in range(100):
        f1, f2 = one(), one()
        dist = ((f1["u"] - f2["u"]) ** 2 + (f1["v"] - f2["v"]) ** 2) ** 0.5
        if dist >= 0.2:
            return [f1, f2]
    return [f1, f2]


def run_one(slot: int, seed: int, outdir: str, extra: list[str],
            max_attempts: int = 3, resume: bool = False,
            res_scale: float = 1.0, check_pierce: bool = True) -> dict:
    """Run one dataset slot (with QC-gated retries).

    With ``resume=True`` a slot whose passing HDF5 already exists in
    ``outdir`` is accepted as-is (QC re-run for the manifest), and attempts
    whose output already sits in ``outdir/rejected`` are not re-simulated.
    Seeds per attempt are deterministic, so a resumed run reproduces the
    exact sequence an uninterrupted run would have taken.
    """
    import glob

    from cloth_sim.validate import qc_sample_h5

    attempts = []
    rej = os.path.join(outdir, "rejected")
    if resume:
        done = sorted(glob.glob(os.path.join(outdir, f"sample_slot{slot:04d}_seed*.h5")))
        if done:
            opath = done[-1]
            aseed = int(os.path.basename(opath).split("_seed")[1].split(".")[0])
            qc = qc_sample_h5(opath, check_pierce=check_pierce)
            attempts.append({"seed": aseed, "ok": qc["ok"], "wall": None,
                             "qc": qc["metrics"], "reasons": qc["reasons"],
                             "resumed": True})
            if qc["ok"]:
                return {"slot": slot, "seed": aseed, "ok": True,
                        "attempts": attempts,
                        "file": os.path.basename(opath)}
            os.makedirs(rej, exist_ok=True)
            os.rename(opath, os.path.join(rej, os.path.basename(opath)))
            attempts = []
    for attempt in range(max_attempts):
        aseed = seed if attempt == 0 else seed + 50000 + attempt
        rng = np.random.default_rng(10_000 + aseed)
        forces = sample_forces(rng, res_scale)
        fpath = os.path.join(outdir, f"forces_slot{slot:04d}_seed{aseed}.json")
        with open(fpath, "w") as f:
            json.dump(forces, f, indent=2)
        opath = os.path.join(outdir, f"sample_slot{slot:04d}_seed{aseed}.h5")
        rpath = os.path.join(rej, os.path.basename(opath))
        if resume and os.path.exists(rpath):
            qc = qc_sample_h5(rpath, check_pierce=check_pierce)
            attempts.append({"seed": aseed, "ok": False, "wall": None,
                             "qc": qc["metrics"], "reasons": qc["reasons"],
                             "resumed": True})
            continue
        t0 = time.time()
        p = subprocess.run(
            [sys.executable, os.path.join(HERE, "generate_sample.py"),
             "--seed", str(aseed), "--force-json", fpath, "--out", opath,
             *extra],
            capture_output=True, text=True,
        )
        wall = time.time() - t0
        if p.returncode != 0:
            attempts.append({"seed": aseed, "ok": False, "wall": wall,
                             "error": p.stderr[-1000:]})
            continue
        qc = qc_sample_h5(opath, check_pierce=check_pierce)
        attempts.append({"seed": aseed, "ok": qc["ok"], "wall": wall,
                         "qc": qc["metrics"], "reasons": qc["reasons"]})
        if qc["ok"]:
            return {"slot": slot, "seed": aseed, "ok": True,
                    "attempts": attempts,
                    "file": os.path.basename(opath)}
        os.makedirs(rej, exist_ok=True)
        os.rename(opath, rpath)
    return {"slot": slot, "seed": seed, "ok": False, "attempts": attempts,
            "file": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--outdir", default="batch")
    ap.add_argument("--seed0", type=int, default=1000,
                    help="base seed; sample seed = seed0 + slot")
    ap.add_argument("--slot0", type=int, default=0,
                    help="first slot index handled by this process (job arrays: "
                         "each task takes slots slot0..slot0+n-1)")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=None, help="default: nx")
    ap.add_argument("--pierce-qc", choices=["auto", "on", "off"], default="auto",
                    help="self-piercing QC check (CPU, slow at high res); auto = on for nx<=256")
    ap.add_argument("--no-res-scaling", action="store_true",
                    help="keep 128-grid patch (3/9 cells) and self-contact "
                         "margin/gap (1.0/0.5 mm) regardless of --nx")
    ap.add_argument("--sample-args", default="",
                    help="extra args forwarded to generate_sample.py")
    ap.add_argument("--resume", action="store_true",
                    help="skip slots already passing in --outdir and attempts "
                         "already in --outdir/rejected")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    extra = args.sample_args.split() if args.sample_args else []
    ny = args.ny or args.nx
    extra += ["--nx", str(args.nx), "--ny", str(ny)]
    # Physical consistency across resolutions: the 128-grid reference has a
    # 3.937 mm cell, Gaussian patch sigma 3 cells, self-contact margin 1.0 mm
    # (~25% of spacing) and gap 0.5 mm. Scale patch size in cells and shrink
    # margin/gap with the cell so the physics stays comparable.
    res_scale = 1.0 if args.no_res_scaling else args.nx / 128.0
    if not args.no_res_scaling and args.nx != 128:
        extra += ["--margin", f"{1.0e-3 / res_scale:.6g}",
                  "--gap", f"{0.5e-3 / res_scale:.6g}"]
    tag = f"{args.slot0:05d}"
    check_pierce = (args.nx <= 256) if args.pierce_qc == "auto" else (args.pierce_qc == "on")

    t0 = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, args.slot0 + k, args.seed0 + args.slot0 + k,
                          args.outdir, extra, 3, args.resume, res_scale, check_pierce): k
                for k in range(args.n)}
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            results.append(r)
            att = len(r["attempts"])
            print(f"slot {r['slot']}: ok={r['ok']} attempts={att} "
                  f"seed={r['seed']} file={r['file']}", flush=True)
            with open(os.path.join(args.outdir, f"progress_{tag}.jsonl"), "a") as f:
                f.write(json.dumps(r) + "\n")
    total = time.time() - t0
    ok = sum(1 for r in results if r["ok"])
    results.sort(key=lambda r: r["slot"])
    # Job arrays: one summary per task (merge with cluster/merge_summaries.py);
    # a single whole-dataset run (slot0 == 0) also writes batch_summary.json.
    names = [f"batch_summary_{tag}.json"] + (["batch_summary.json"] if args.slot0 == 0 else [])
    summary = {"n": args.n, "ok": ok, "total_wall_s": total,
               "jobs": args.jobs, "slot0": args.slot0, "seed0": args.seed0,
               "nx": args.nx, "ny": ny, "res_scale": res_scale, "pierce_qc": check_pierce,
               "sample_args": extra,
               "distribution": sample_forces.__doc__,
               "results": results}
    for name in names:
        with open(os.path.join(args.outdir, name), "w") as f:
            json.dump(summary, f, indent=2)
    print(f"batch: {ok}/{args.n} ok, total {total:.1f}s "
          f"({total/max(ok,1):.1f}s per sample at {args.jobs} jobs)")


if __name__ == "__main__":
    main()
