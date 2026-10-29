"""Merge per-task batch_summary_*.json files from a job array into one
batch_summary.json (consumed by package_splits.py) plus a human-readable
dataset_summary.json with QC statistics and the list of missing/failed slots."""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--expect-n", type=int, default=None,
                    help="total slots expected (default: highest slot seen + 1)")
    args = ap.parse_args()

    parts = sorted(glob.glob(os.path.join(args.indir, "batch_summary_*.json")))
    results: dict[int, dict] = {}
    wall, meta = 0.0, {}
    for p in parts:
        d = json.load(open(p))
        meta = meta or d
        wall += d.get("total_wall_s", 0.0)
        for r in d["results"]:
            prev = results.get(r["slot"])
            if prev is None or (r["ok"] and not prev["ok"]):
                results[r["slot"]] = r
    # Files on disk that no summary mentions (task killed before it wrote one).
    from_disk = 0
    for h5 in glob.glob(os.path.join(args.indir, "sample_slot*_seed*.h5")):
        b = os.path.basename(h5)
        slot = int(b.split("slot")[1].split("_")[0])
        if slot not in results or not results[slot]["ok"]:
            seed = int(b.split("_seed")[1].split(".")[0])
            results[slot] = {"slot": slot, "seed": seed, "ok": True, "file": b,
                             "attempts": [], "note": "from disk, no summary"}
            from_disk += 1
    assert results, f"nothing found in {args.indir}"
    # Finiteness QC: a sample whose final state holds NaN/inf passed the generator's checks
    # once (ds256_free slot 1519, 2026-10-29) and made every val loss NaN. Reject it here so it
    # is never packaged (train.py also drops such samples defensively).
    import h5py
    non_finite = 0
    for r in results.values():
        if r["ok"] and r.get("file"):
            try:
                with h5py.File(os.path.join(args.indir, r["file"]), "r") as h:
                    fin = bool(np.isfinite(h["final_grid"][()]).all() and np.isfinite(h["final_vel"][()]).all())
            except Exception as e:  # unreadable file counts as failed too
                fin, r["note"] = False, f"unreadable: {e}"
            if not fin:
                r["ok"], r["note"] = False, (r.get("note", "") + " non-finite final state").strip()
                non_finite += 1
    if non_finite:
        print(f"rejected {non_finite} sample(s) with a non-finite final state")
    res = [results[k] for k in sorted(results)]
    ok = [r for r in res if r["ok"]]
    n_expect = args.expect_n or (max(results) + 1)
    missing = sorted(set(range(n_expect)) - set(results))
    failed = [r["slot"] for r in res if not r["ok"]]

    merged = {"n": n_expect, "ok": len(ok), "total_wall_s": wall, "jobs": "slurm-array",
              "nx": meta.get("nx"), "ny": meta.get("ny"), "seed0": meta.get("seed0"),
              "res_scale": meta.get("res_scale"), "sample_args": meta.get("sample_args"),
              "distribution": meta.get("distribution"), "results": res}
    with open(os.path.join(args.indir, "batch_summary.json"), "w") as f:
        json.dump(merged, f, indent=2)

    qc = {k: [] for k in ("ke_final", "vmax_max", "stretch_max", "piercing_pairs")}
    walls, attempts = [], []
    for r in ok:
        if r["attempts"]:
            a = r["attempts"][-1]
            attempts.append(len(r["attempts"]))
            if a.get("wall"):
                walls.append(a["wall"])
            for k in qc:
                if a.get("qc") and a["qc"].get(k) is not None:  # piercing may be skipped (None)
                    qc[k].append(a["qc"][k])
    stats = {k: {"min": float(np.min(v)), "median": float(np.median(v)), "max": float(np.max(v))}
             for k, v in qc.items() if v}
    summary = {"tasks_with_summary": len(parts), "slots_expected": n_expect, "ok": len(ok),
               "failed_slots": failed, "missing_slots": missing, "from_disk_only": from_disk,
               "attempts_needed": {"1": attempts.count(1), "2": attempts.count(2), "3": attempts.count(3)},
               "sim_wall_s": ({"mean": float(np.mean(walls)), "min": float(np.min(walls)),
                               "max": float(np.max(walls))} if walls else None),
               "gpu_hours_total": wall / 3600.0, "qc": stats,
               "nx": meta.get("nx"), "sample_args": meta.get("sample_args")}
    with open(os.path.join(args.indir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if missing or failed:
        print(f"\n{len(missing)} missing + {len(failed)} failed slots. Re-run submit.py with "
              f"the same --name to resume and fill them.")


if __name__ == "__main__":
    main()
