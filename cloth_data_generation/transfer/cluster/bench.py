#!/usr/bin/env python3
"""Submit resolution-scaling benchmarks, one job per GPU type.

Each job runs the SAME two-force scenario at increasing grid resolutions
(default 64..384) and records per-sample wall time, GPU utilization and peak
memory. Submission time is recorded so the report can compute the real queue
wait of every job (Submit -> Start via sacct).

  python3 cluster/bench.py --dry-run
  python3 cluster/bench.py                                   # default types & sizes
  python3 cluster/bench.py --types a40,l40s,h100 --nx 128,256,512 --repeats 2
  python3 cluster/bench_report.py                            # afterwards
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_types import DEFAULT_TYPES, GPU_TYPES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
BENCH_DIR = os.path.join(CODE_DIR, "bench")


def load_env(path):
    cfg = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--types", default=",".join(DEFAULT_TYPES))
    ap.add_argument("--nx", default="64,128,192,256,384", help="resolutions, ascending")
    ap.add_argument("--repeats", type=int, default=2,
                    help="samples per resolution (first one includes kernel compile); 0 = as many as fit in --budget-min")
    ap.add_argument("--one-job-per-nx", action="store_true",
                    help="submit a separate job per resolution (each gets the full --time/--budget-min)")
    ap.add_argument("--time", default="03:00:00")
    ap.add_argument("--budget-min", type=int, default=150, help="stop sweeping after this many minutes of work")
    ap.add_argument("--mem", default="32G")
    ap.add_argument("--partition", default=None, help="override partition for all types (e.g. scavenge_gpu)")
    ap.add_argument("--sample-args", default=None)
    ap.add_argument("--run", default=None, help="run name (default bench_<timestamp>); an existing run is extended")
    ap.add_argument("--per-gpu", type=int, default=1,
                    help="run this many samples concurrently on the one GPU (tests whether the GPU is underused)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_env(os.path.join(HERE, "cluster.env"))
    account = cfg.get("ACCOUNT", "")
    sample_args = args.sample_args if args.sample_args is not None else cfg.get("SAMPLE_ARGS", "")
    run = args.run or time.strftime("bench_%Y%m%d_%H%M")
    run_dir = os.path.join(BENCH_DIR, run)
    template = open(os.path.join(HERE, "bench.sbatch")).read()
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    nx_list = " ".join(args.nx.split(","))
    mpath = os.path.join(run_dir, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {"run": run, "jobs": {}}
    manifest.update({"nx": args.nx, "repeats": args.repeats, "sample_args": sample_args,
                     "one_job_per_nx": args.one_job_per_nx})
    print(f"run {run}: types {types}, nx {nx_list}, repeats {args.repeats or 'as many as fit'}, "
          f"time {args.time}, budget {args.budget_min} min" + (", one job per nx" if args.one_job_per_nx else ""))
    units = []  # (key, gtype, nx list for the job, outdir)
    for t in types:
        if t not in GPU_TYPES:
            print(f"skip unknown type {t}")
            continue
        suffix = f"_x{args.per_gpu}" if args.per_gpu > 1 else ""
        if args.one_job_per_nx:
            for nx in nx_list.split():
                units.append((f"{t}/nx{nx}{suffix}", t, nx, os.path.join(run_dir, t, f"nx{nx}{suffix}")))
        else:
            units.append((f"{t}{suffix}", t, nx_list, os.path.join(run_dir, t + suffix)))
    script = ""
    for key, t, nxs, outdir in units:
        part = args.partition or GPU_TYPES[t][0]
        script = template
        for k, v in {"PARTITION": part, "GTYPE": t, "MEM": args.mem, "TIME": args.time, "OUTDIR": outdir,
                     "ACCOUNT_LINE": f"#SBATCH --account={account}" if account else "", "CODE_DIR": CODE_DIR,
                     "NX_LIST": nxs, "REPEATS": args.repeats, "BUDGET_S": args.budget_min * 60,
                     "SAMPLE_ARGS": sample_args, "CONC": args.per_gpu}.items():
            script = script.replace(f"__{k}__", str(v))
        path = os.path.join(outdir, "bench.sbatch")
        print(f"  {key:30s} partition {part:12s} nx {nxs}")
        if args.dry_run:
            continue
        os.makedirs(outdir, exist_ok=True)
        open(path, "w").write(script)
        p = subprocess.run(["sbatch", "--parsable", path], capture_output=True, text=True)
        if p.returncode != 0:
            print("    sbatch failed:", p.stderr.strip())
            manifest["jobs"][key] = {"error": p.stderr.strip(), "gpu": t}
            continue
        job = p.stdout.strip().split(";")[0]
        manifest["jobs"][key] = {"job_id": job, "gpu": t, "partition": part, "nx": nxs, "per_gpu": args.per_gpu,
                                 "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")}
        print(f"    submitted job {job}")
    if args.dry_run:
        print("\n(dry run) first job script would be:\n" + script[:1200] + "\n...")
        return
    os.makedirs(run_dir, exist_ok=True)
    json.dump(manifest, open(mpath, "w"), indent=2)
    print(f"\nmanifest: {run_dir}/manifest.json")
    print(f"watch   : squeue -u $USER -n cloth_bench")
    print(f"report  : python3 cluster/bench_report.py --run {run}")


if __name__ == "__main__":
    main()
