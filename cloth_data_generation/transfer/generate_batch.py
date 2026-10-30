"""Run N independent cloth samples with random force parameters.

Batch strategy: one OS process per worker (default 8). Each worker imports
the simulator once and keeps one SimSession (model + VBD solver + CUDA graph)
for the batch's configuration, so per sample only the state is reset and the
frames replayed (the solver setup alone was ~4 s per sample at 128x128, more
than the simulation itself once the graph is used). ``--subprocess`` restores
the original one-process-per-sample isolation. One HDF5 file per sample.

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

# QC stretch limit per material: paper is near-inextensible (healthy runs
# ~1.05), so a stretch cloth tolerates means a torn or exploded paper sheet.
STRETCH_MAX = {"cloth": 2.0, "paper": 1.3}


LOADINGS = ("free", "pusher")
# log-uniform, N. free: 0.15-1.25 N since 2026-10-29 (constant-force model delivers the full
# impulse; the 0.3-2.5 N range of ds768_night/ds256_free tore 1/3 of attempts at full impulse,
# and at the legacy ~53% impulse it was equivalent to 0.16-1.3 N).
MAG_RANGE = {"free": (np.log(0.15), np.log(1.25)), "pusher": (np.log(0.1), np.log(1.0))}


def sample_direction(rng: np.random.Generator, loading: str, u: float = 0.5, v: float = 0.5) -> np.ndarray:
    """free: uniform on the sphere (original datasets). For pusher the direction is
    assigned pairwise in sample_forces (the two pushes oppose each other)."""
    d = rng.normal(size=3)
    return d / np.linalg.norm(d)


def _pusher_pair(rng: np.random.Generator, f1: dict, f2: dict) -> None:
    """Two-hands scrunch on a table: each patch pushes toward the other (+-30 deg
    heading jitter), tilted 0-20 deg into the table, never a pull. A single push
    stronger than table friction (0.1 N paper, 0.25 N cloth) just slides the sheet
    (smoke test 2026-10-28); opposing pushes cancel the net force and compress the
    sheet between the patches so it buckles against the table."""
    for a, b in ((f1, f2), (f2, f1)):
        th = np.arctan2(b["v"] - a["v"], b["u"] - a["u"]) + np.deg2rad(rng.uniform(-30.0, 30.0))
        ph = np.deg2rad(rng.uniform(0.0, 20.0))
        a["direction"] = [float(np.cos(ph) * np.cos(th)), float(np.cos(ph) * np.sin(th)), float(-np.sin(ph))]


def sample_forces(rng: np.random.Generator, res_scale: float = 1.0,
                  loading: str = "free") -> list[dict]:
    """Broad dataset distribution: full (u,v) coverage, log-uniform magnitudes
    (range per loading, MAG_RANGE), plus a 15% near-symmetric branch. Min material
    separation 0.2 keeps the two patches distinct. free: uniform-sphere directions,
    same-side pushes allowed (original datasets). pusher: the two pushes oppose
    each other (_pusher_pair). Forces depend only on the seed, so cloth and paper
    datasets with the same seed0 are paired.
    """
    def one():
        d = sample_direction(rng, loading)  # drawn first: keeps legacy seeds reproducible
        u, v = float(rng.uniform(0.05, 0.95)), float(rng.uniform(0.05, 0.95))
        return {
            "u": u,
            "v": v,
            "direction": [float(x) for x in d],
            # free flight: the whole sheet accelerates, 0.15-1.25 N folds it. On a table
            # friction anchors the sheet and the same force tears the patch out (smoke
            # test 2026-10-28: 1.2-2 N tore cloth and NaN'd paper, 0.5-0.85 N were fine).
            "magnitude": float(np.exp(rng.uniform(*MAG_RANGE[loading]))),
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
        pair = [f1, f2]
    else:
        for _ in range(100):
            f1, f2 = one(), one()
            dist = ((f1["u"] - f2["u"]) ** 2 + (f1["v"] - f2["v"]) ** 2) ** 0.5
            if dist >= 0.2:
                break
        pair = [f1, f2]
    if loading == "pusher":
        _pusher_pair(rng, pair[0], pair[1])
    return pair


_CACHE = None  # per-worker SessionCache (created lazily after fork)


def _simulate_in_process(argv: list[str]) -> tuple[int, str]:
    """Run generate_sample in this worker, reusing its session. Returns (rc, stderr)."""
    global _CACHE
    import io
    import traceback
    import contextlib

    import warp as wp

    import generate_sample as gs  # imported in the worker, after the fork

    if _CACHE is None:
        wp.init()
        _CACHE = gs.SessionCache()
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            spec = gs.configure(gs.build_parser().parse_args(argv))
            gs.generate(spec, cache=_CACHE, verbose=False)
        return 0, err.getvalue()
    except Exception:
        _CACHE = None  # rebuild the session after any failure
        return 1, err.getvalue() + traceback.format_exc()


def run_one(slot: int, seed: int, outdir: str, extra: list[str],
            max_attempts: int = 3, resume: bool = False,
            res_scale: float = 1.0, check_pierce: bool = True,
            stretch_max: float = 2.0, loading: str = "free",
            in_process: bool = True) -> dict:
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
            qc = qc_sample_h5(opath, check_pierce=check_pierce, stretch_max=stretch_max)
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
        forces = sample_forces(rng, res_scale, loading)
        fpath = os.path.join(outdir, f"forces_slot{slot:04d}_seed{aseed}.json")
        with open(fpath, "w") as f:
            json.dump(forces, f, indent=2)
        opath = os.path.join(outdir, f"sample_slot{slot:04d}_seed{aseed}.h5")
        rpath = os.path.join(rej, os.path.basename(opath))
        if resume and os.path.exists(rpath):
            qc = qc_sample_h5(rpath, check_pierce=check_pierce, stretch_max=stretch_max)
            attempts.append({"seed": aseed, "ok": False, "wall": None,
                             "qc": qc["metrics"], "reasons": qc["reasons"],
                             "resumed": True})
            continue
        t0 = time.time()
        argv = ["--seed", str(aseed), "--force-json", fpath, "--out", opath, *extra]
        if in_process:
            rc, stderr = _simulate_in_process(argv)
        else:
            p = subprocess.run(
                [sys.executable, os.path.join(HERE, "generate_sample.py"), *argv],
                capture_output=True, text=True,
            )
            rc, stderr = p.returncode, p.stderr
        wall = time.time() - t0
        if rc != 0:
            attempts.append({"seed": aseed, "ok": False, "wall": wall,
                             "error": stderr[-1000:]})
            fatal = ("primary context" in stderr or "CUDA driver" in stderr
                     or "no CUDA-capable device" in stderr or "ModuleNotFoundError" in stderr)
            if fatal:
                # the GPU/environment is broken for this process; retrying other
                # seeds cannot help, and a resubmission (resume) will redo the slot
                return {"slot": slot, "seed": seed, "ok": False, "attempts": attempts,
                        "file": None, "fatal": True}
            continue
        qc = qc_sample_h5(opath, check_pierce=check_pierce, stretch_max=stretch_max)
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
    ap.add_argument("--material", choices=sorted(STRETCH_MAX), default="cloth",
                    help="sheet preset forwarded to generate_sample.py (paper = plastic creasing)")
    ap.add_argument("--loading", choices=LOADINGS, default="free",
                    help="force-direction model: free (uniform sphere, original) or pusher "
                         "(tangential push tilted into the sheet, what a bench rig applies)")
    ap.add_argument("--table", action="store_true",
                    help="bench scenario: gravity + ground plane (forwarded to generate_sample.py)")
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
    ap.add_argument("--subprocess", action="store_true",
                    help="one generate_sample.py process per sample (original isolation) "
                         "instead of reusing one solver session per worker")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    extra = args.sample_args.split() if args.sample_args else []
    ny = args.ny or args.nx
    extra += ["--nx", str(args.nx), "--ny", str(ny)]
    if args.material != "cloth":
        extra += ["--material", args.material]
    if args.table:
        extra += ["--table"]
    # Physical consistency across resolutions: the 128-grid reference has a
    # 3.937 mm cell, Gaussian patch sigma 3 cells, particle radius 2.0 mm (~50%
    # of spacing), self-contact margin 1.0 mm (~25%) and gap 0.5 mm. Scale patch
    # size in cells and shrink radius/margin/gap with the cell so the physics
    # stays comparable. (Until 2026-10-28 the radius was NOT scaled: at 256 it
    # equalled the spacing and at 768 it was 3 cells, so compressed sheets had
    # every neighbour in permanent contact; bench smoke tests exploded at 256
    # and 44% of ds768_night attempts tore.)
    res_scale = 1.0 if args.no_res_scaling else args.nx / 128.0
    if not args.no_res_scaling and args.nx != 128:
        extra += ["--radius", f"{2.0e-3 / res_scale:.6g}",
                  "--margin", f"{1.0e-3 / res_scale:.6g}",
                  "--gap", f"{0.5e-3 / res_scale:.6g}"]
    tag = f"{args.slot0:05d}"
    check_pierce = (args.nx <= 256) if args.pierce_qc == "auto" else (args.pierce_qc == "on")

    t0 = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, args.slot0 + k, args.seed0 + args.slot0 + k,
                          args.outdir, extra, 3, args.resume, res_scale, check_pierce,
                          STRETCH_MAX[args.material], args.loading, not args.subprocess): k
                for k in range(args.n)}
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            results.append(r)
            if r.get("fatal"):
                for f in futs:
                    f.cancel()
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
               "material": args.material, "stretch_max": STRETCH_MAX[args.material],
               "loading": args.loading, "table": args.table,
               "in_process": not args.subprocess,
               "sample_args": extra,
               "distribution": sample_forces.__doc__,
               "results": results}
    for name in names:
        with open(os.path.join(args.outdir, name), "w") as f:
            json.dump(summary, f, indent=2)
    print(f"batch: {ok}/{args.n} ok, total {total:.1f}s "
          f"({total/max(ok,1):.1f}s per sample at {args.jobs} jobs)")
    if any(r.get("fatal") for r in results):
        print("FATAL: GPU or environment unusable on this node (see attempts[].error); "
              "resubmit with --resume, excluding this node if it repeats", file=sys.stderr)
        sys.exit(3)
    if ok == 0 and args.n > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
