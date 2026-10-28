#!/usr/bin/env python3
"""Stability sweep: re-simulate force configurations that FAILED QC in a
production dataset under alternative solver settings, one job per setting,
and report pass rate, max speed (is the velocity cap lifted?), impulse
transfer and cost per sample.

  python3 cluster/stability.py --dataset ds768_night --n-configs 12          # submit
  python3 cluster/stability.py --dataset ds768_night --report                # afterwards

Default settings compared (generate_sample.py flags):
  base    : production settings (30 substeps, 20 iterations)
  sub60   : --substeps 60
  sub90   : --substeps 90
  sub150  : --substeps 150
  it40    : --iterations 40
Add your own with --settings 'label=flags;label2=flags'.

Bench profile (table + gravity, 2026-10-28): every bench smoke sample on the
workstation went NaN once the sheet crumpled against the table, so sweep the
solver on the bench scenario before any production run. --forces-dir takes any
directory of forces_*.json (e.g. a smoke dataset, no failures needed) and
--base-flags are added to every setting (use the = form, argparse otherwise eats the leading --):
  python3 cluster/stability.py --dataset smoke256_bench --forces-dir \
      $DATA/smoke256_bench --all-configs --base-flags="--table" \
      --settings "base=;sub60=--substeps 60;sub120=--substeps 120;ke1e4=--contact-ke 1e4;\
sub60ke1e4=--substeps 60 --contact-ke 1e4;it40=--iterations 40;stiff=--tri-ke 1e4 --substeps 60"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_types import GPU_TYPES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
BENCH_DIR = os.path.join(CODE_DIR, "bench")
DEFAULT_SETTINGS = "base=;sub60=--substeps 60;sub90=--substeps 90;sub150=--substeps 150;it40=--iterations 40"


def load_env(path):
    cfg = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


def pick_failed_configs(dataset_dir: str, n: int, seed: int) -> list[str]:
    """Force JSONs of attempts that were rejected (file in rejected/) or that
    crashed with NaN, read from the per-task progress logs."""
    bad = {}
    for p in glob.glob(os.path.join(dataset_dir, "progress_*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            for a in r["attempts"]:
                if not a["ok"] and not a.get("error"):
                    fj = os.path.join(dataset_dir, f"forces_slot{r['slot']:04d}_seed{a['seed']}.json")
                    if os.path.exists(fj):
                        bad[fj] = a.get("reasons", [])
    files = sorted(bad)
    random.Random(seed).shuffle(files)
    return files[:n]


def report(run_dir: str) -> None:
    rows = []
    for p in glob.glob(os.path.join(run_dir, "*", "results.jsonl")):
        rows += [json.loads(l) for l in open(p)]
    if not rows:
        sys.exit("no results yet")
    by = {}
    for r in rows:
        by.setdefault(r["setting"], []).append(r)
    import numpy as np
    print(f"{'setting':10s} {'flags':22s} {'n':>3s} {'pass':>5s} {'NaN':>4s} {'stretch med/max':>16s} {'vmax med/max':>13s} {'impulse%':>9s} {'sim s med':>9s}")
    for s, rs in sorted(by.items(), key=lambda kv: kv[0]):
        n = len(rs); ok = sum(r["ok"] for r in rs)
        nan = sum(1 for r in rs if r.get("nan_time") is not None or (r.get("error") and "nan" in r["error"].lower()))
        st = [r["stretch_max"] for r in rs if r.get("stretch_max") is not None]
        vm = [r["vmax_max"] for r in rs if r.get("vmax_max") is not None]
        imp = [r["impulse_ratio_at_force_end"] for r in rs if r.get("impulse_ratio_at_force_end") is not None]
        sw = [r["sim_wall_s"] for r in rs if r.get("sim_wall_s")]
        f = lambda v, fmt: fmt % v if v else "-"
        print(f"{s:10s} {rs[0]['extra'][:22]:22s} {n:>3d} {ok/n*100:>4.0f}% {nan:>4d} "
              f"{(f(np.median(st), '%.2f') + '/' + f(max(st), '%.1f')) if st else '-':>16s} "
              f"{(f(np.median(vm), '%.2f') + '/' + f(max(vm), '%.2f')) if vm else '-':>13s} "
              f"{f(np.median(imp) * 100, '%.0f') if imp else '-':>9s} {f(np.median(sw), '%.0f') if sw else '-':>9s}")
    print("\npass = QC (stretch<=2, no NaN, KE/speed sane) on configs that FAILED under production settings."
          "\nvmax identical across configs = velocity clamp still active; impulse% = momentum delivered / F*t.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset name under DATA_DIR (source of failed configs)")
    ap.add_argument("--n-configs", type=int, default=12)
    ap.add_argument("--forces-dir", default=None, help="use the forces_*.json in this directory instead of failed configs")
    ap.add_argument("--all-configs", action="store_true", help="with --forces-dir: every config, not n-configs")
    ap.add_argument("--base-flags", default="", help="generate_sample.py flags added to every setting, e.g. '--table --material paper'")
    ap.add_argument("--material", default="cloth", help="forwarded to every setting")
    ap.add_argument("--nx", type=int, default=None, help="default: from cluster.env NX")
    ap.add_argument("--settings", default=DEFAULT_SETTINGS)
    ap.add_argument("--time", default="04:00:00")
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--partition", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_env(os.path.join(HERE, "cluster.env"))
    data_dir = os.path.expandvars(cfg.get("DATA_DIR", ""))
    dataset_dir = os.path.join(data_dir, args.dataset)
    run = args.run or f"stability_{args.dataset}"
    run_dir = os.path.join(BENCH_DIR, run)
    if args.report:
        report(run_dir)
        return

    nx = args.nx or int(cfg.get("NX", 768))
    if args.forces_dir:
        configs = sorted(glob.glob(os.path.join(args.forces_dir, "forces_*.json")))
        if not args.all_configs:
            random.Random(args.seed).shuffle(configs); configs = configs[:args.n_configs]
    else:
        configs = pick_failed_configs(dataset_dir, args.n_configs, args.seed)
    if not configs:
        sys.exit(f"no force configs found ({args.forces_dir or dataset_dir})")
    base = (args.base_flags + (f" --material {args.material}" if args.material != "cloth" else "")).strip()
    forces_dir = os.path.join(run_dir, "forces")
    print(f"run {run}: {len(configs)} failed configs from {args.dataset}, nx={nx}")
    if not args.dry_run:
        os.makedirs(forces_dir, exist_ok=True)
        for c in configs:
            shutil.copy(c, forces_dir)
    account = cfg.get("ACCOUNT", "")
    exclude = cfg.get("EXCLUDE_NODES", "")
    template = open(os.path.join(HERE, "stability.sbatch")).read()
    manifest = {"run": run, "dataset": args.dataset, "nx": nx, "base_flags": base,
                "configs": [os.path.basename(c) for c in configs], "jobs": {}}
    for item in args.settings.split(";"):
        label, _, flags = item.partition("=")
        label, flags = label.strip(), flags.strip()
        outdir = os.path.join(run_dir, label)
        script = template
        for k, v in {"PARTITION": args.partition or cfg.get("PARTITION", "gpu"), "GPUS": args.gpus or cfg.get("GPUS", "1"),
                     "TIME": args.time, "OUTDIR": outdir, "CODE_DIR": CODE_DIR, "NX": nx, "SETTING": label,
                     "EXTRA": (base + " " + flags).strip(), "FORCES_DIR": forces_dir,
                     "ACCOUNT_LINE": f"#SBATCH --account={account}" if account else "",
                     "EXCLUDE_LINE": f"#SBATCH --exclude={exclude}" if exclude else ""}.items():
            script = script.replace(f"__{k}__", str(v))
        print(f"  {label:8s} flags '{flags}'")
        if args.dry_run:
            continue
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "stability.sbatch")
        open(path, "w").write(script)
        p = subprocess.run(["sbatch", "--parsable", path], capture_output=True, text=True)
        if p.returncode != 0:
            print("    sbatch failed:", p.stderr.strip()); continue
        manifest["jobs"][label] = {"job_id": p.stdout.strip().split(";")[0], "flags": flags}
        print(f"    submitted job {manifest['jobs'][label]['job_id']}")
    if not args.dry_run:
        json.dump(manifest, open(os.path.join(run_dir, "manifest.json"), "w"), indent=2)
        print(f"\nreport when done: python3 cluster/stability.py --dataset {args.dataset} --report")


if __name__ == "__main__":
    main()
