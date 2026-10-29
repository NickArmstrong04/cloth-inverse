#!/usr/bin/env python3
"""Submit cloth data generation as a Slurm job array (run ON the cluster).

Examples (from the copied transfer/ directory):
  python3 cluster/submit.py --n 1000 --dry-run          # show what would run
  python3 cluster/submit.py --n 1000                    # 256x256, 40 tasks x 25
  python3 cluster/submit.py --n 1000 --nx 128 --chunk 50 --name ds128_1k
  python3 cluster/submit.py --n 1000 --name ds256_1000  # again later = resume
  python3 cluster/submit.py --status ds256_1000         # progress + queue
  python3 cluster/submit.py --n 1000 --tasks 3,7-9      # re-run specific tasks
  python3 cluster/submit.py --n 1000 --material paper   # creasing paper sheet -> ds768_1000_paper

Only the Python standard library is used, so it runs with the login node's
python3. Settings come from cluster/cluster.env; flags override them.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CODE_DIR = os.path.dirname(HERE)


def load_env(path: str) -> dict:
    cfg = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = os.path.expandvars(v.strip().strip('"').strip("'"))
    return cfg


def parse_tasks(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def compress(tasks: list[int]) -> str:
    """[0,1,2,5,7,8] -> '0-2,5,7-8' for --array."""
    if not tasks:
        return ""
    runs, start, prev = [], tasks[0], tasks[0]
    for t in tasks[1:]:
        if t == prev + 1:
            prev = t
            continue
        runs.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = t
    runs.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(runs)


def render(template: str, mapping: dict) -> str:
    out = open(template).read()
    for k, v in mapping.items():
        out = out.replace(f"__{k}__", str(v))
    left = re.findall(r"__[A-Z_]+__", out)
    assert not left, f"unfilled placeholders in {template}: {left}"
    return out


def sbatch(path: str, dry: bool) -> str | None:
    cmd = ["sbatch", "--parsable", path]
    print("  $", " ".join(cmd))
    if dry:
        return None
    if shutil.which("sbatch") is None:
        sys.exit("sbatch not found: run this on the cluster login node (or use --dry-run)")
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"sbatch failed:\n{out.stderr}")
    return out.stdout.strip().split(";")[0]


def done_slots(outdir: str) -> set[int]:
    return {int(os.path.basename(p).split("slot")[1].split("_")[0])
            for p in glob.glob(os.path.join(outdir, "sample_slot*_seed*.h5"))}


def status(outdir: str, name: str) -> None:
    done = done_slots(outdir)
    rej = len(glob.glob(os.path.join(outdir, "rejected", "*.h5")))
    sub = os.path.join(outdir, "slurm", "submission.json")
    n = None
    if os.path.exists(sub):
        s = json.load(open(sub))
        n = s.get("n")
        print(f"last submission: job {s.get('array_job_id')} merge {s.get('merge_job_id')} "
              f"at {s.get('time')} (n={n}, nx={s.get('nx')}, chunk={s.get('chunk')})")
    print(f"{name}: {len(done)} passing samples" + (f" / {n}" if n else "") + f", {rej} rejected attempts")
    if n:
        missing = sorted(set(range(n)) - done)
        if missing:
            print(f"missing slots ({len(missing)}): {compress(missing)}")
    for f in ("dataset_summary.json",):
        p = os.path.join(outdir, f)
        if os.path.exists(p):
            print(f"{f}: present (merge ran)")
    if shutil.which("squeue"):
        q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-n", "cloth_gen,cloth_merge",
                            "-o", "%.12i %.10P %.10j %.3t %.11M %.6D %R"], capture_output=True, text=True)
        print(q.stdout.rstrip())
    logs = sorted(glob.glob(os.path.join(outdir, "slurm", "gen_*.err")), key=os.path.getmtime)
    errs = [l for l in logs if os.path.getsize(l) > 0]
    if errs:
        print(f"{len(errs)} task .err logs are non-empty; newest: {errs[-1]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=os.path.join(HERE, "cluster.env"))
    ap.add_argument("--n", type=int, help="total number of samples (slots 0..n-1)")
    ap.add_argument("--nx", type=int)
    ap.add_argument("--chunk", type=int, help="samples per array task")
    ap.add_argument("--material", choices=["cloth", "paper"],
                    help="sheet preset (default MATERIAL in cluster.env, else cloth)")
    ap.add_argument("--seed0", type=int)
    ap.add_argument("--name", help="dataset directory name under DATA_DIR (default ds<nx>_<n>)")
    ap.add_argument("--sample-args", help="override SAMPLE_ARGS (quote it)")
    ap.add_argument("--batch-args", help="override BATCH_ARGS: generate_batch.py flags such as "
                                         "'--loading pusher --table' (quote it; '' = none)")
    ap.add_argument("--partition")
    ap.add_argument("--account")
    ap.add_argument("--time")
    ap.add_argument("--mem")
    ap.add_argument("--max-concurrent", type=int)
    ap.add_argument("--gpus", help="GPU request, e.g. l40s:1 or 1 (override GPUS)")
    ap.add_argument("--jobs-per-task", type=int, help="concurrent samples per GPU (override JOBS_PER_TASK)")
    ap.add_argument("--exclude", help="nodes to avoid, comma-separated (override EXCLUDE_NODES)")
    ap.add_argument("--tasks", help="only these array tasks, e.g. 3,7-9 (default: tasks with missing slots)")
    ap.add_argument("--all-tasks", action="store_true", help="submit every task even if its slots are done")
    ap.add_argument("--max-tasks", type=int, default=0,
                    help="submit at most this many of the selected tasks (lowest ids first); 0 = all. "
                         "With --max-concurrent K and TIME per task this bounds the run to ~max_tasks/K waves")
    ap.add_argument("--no-merge", action="store_true", help="do not chain the merge/packaging job")
    ap.add_argument("--dry-run", action="store_true", help="render scripts and print commands only")
    ap.add_argument("--status", metavar="NAME", help="show progress of an existing dataset and exit")
    args = ap.parse_args()

    cfg = load_env(args.env)
    code_dir = cfg.get("CODE_DIR") or DEFAULT_CODE_DIR
    data_dir = cfg.get("DATA_DIR") or os.path.expanduser("~/scratch/cloth_data")

    if args.status:
        status(os.path.join(data_dir, args.status), args.status)
        return
    if not args.n:
        ap.error("--n is required (or use --status NAME)")

    nx = args.nx or int(cfg.get("NX", 256))
    chunk = args.chunk or int(cfg.get("CHUNK", 25))
    seed0 = args.seed0 if args.seed0 is not None else int(cfg.get("SEED0", 100000))
    material = args.material or cfg.get("MATERIAL") or "cloth"
    name = args.name or f"ds{nx}_{args.n}" + ("" if material == "cloth" else f"_{material}")
    sample_args = args.sample_args if args.sample_args is not None else cfg.get("SAMPLE_ARGS", "")
    batch_args = args.batch_args if args.batch_args is not None else cfg.get("BATCH_ARGS", "")
    account = args.account if args.account is not None else cfg.get("ACCOUNT", "")
    outdir = os.path.join(data_dir, name)
    logdir = os.path.join(outdir, "slurm")
    n_tasks = -(-args.n // chunk)

    # Decide which tasks to run: by default only those with missing slots (resume).
    if args.tasks:
        tasks = [t for t in parse_tasks(args.tasks) if t < n_tasks]
    elif args.all_tasks or not os.path.isdir(outdir):
        tasks = list(range(n_tasks))
    else:
        done = done_slots(outdir)
        tasks = [t for t in range(n_tasks)
                 if any(s not in done for s in range(t * chunk, min((t + 1) * chunk, args.n)))]
        print(f"resume: {len(done)} passing samples already in {outdir}")
    if args.max_tasks and len(tasks) > args.max_tasks:
        print(f"limiting to the first {args.max_tasks} of {len(tasks)} selected tasks (--max-tasks)")
        tasks = tasks[: args.max_tasks]
    if not tasks:
        print("nothing to do: all slots present. Use --all-tasks to force, or --status.")
        return
    array = compress(tasks)
    maxc = args.max_concurrent or int(cfg.get("MAX_CONCURRENT", 16))
    if maxc > 0:
        array += f"%{maxc}"

    # Guard: a previous submission with a different seed0/nx would mix datasets.
    sub_path = os.path.join(logdir, "submission.json")
    if os.path.exists(sub_path):
        prev = json.load(open(sub_path))
        for k, v in (("seed0", seed0), ("nx", nx), ("chunk", chunk), ("sample_args", sample_args),
                     ("material", material), ("batch_args", batch_args)):
            if prev.get(k, {"material": "cloth", "batch_args": ""}.get(k)) != v:
                sys.exit(f"{name} was submitted with {k}={prev.get(k)!r}, now {v!r}. "
                         f"Use a different --name or match the previous settings.")

    if not args.dry_run:
        os.makedirs(logdir, exist_ok=True)
    account_line = f"#SBATCH --account={account}" if account else ""
    exclude = args.exclude if args.exclude is not None else cfg.get("EXCLUDE_NODES", "")
    exclude_line = f"#SBATCH --exclude={exclude}" if exclude else ""
    jobs = args.jobs_per_task or int(cfg.get("JOBS_PER_TASK", 1) or 1)
    common = {"CODE_DIR": code_dir, "OUTDIR": outdir, "LOGDIR": logdir, "ACCOUNT_LINE": account_line,
              "EXCLUDE_LINE": exclude_line}
    gen = render(os.path.join(HERE, "gen_array.sbatch"), common | {
        "PARTITION": args.partition or cfg.get("PARTITION", "gpu"),
        "GPUS": args.gpus or cfg.get("GPUS", "1"), "CPUS": cfg.get("CPUS", "4"), "JOBS": jobs,
        "MEM": args.mem or cfg.get("MEM", "24G"), "TIME": args.time or cfg.get("TIME", "12:00:00"),
        "ARRAY": array, "CHUNK": chunk, "SEED0": seed0, "NX": nx, "SAMPLE_ARGS": sample_args,
        "MATERIAL": material, "BATCH_ARGS": batch_args})
    merge = render(os.path.join(HERE, "merge.sbatch"), common | {
        "CPU_PARTITION": cfg.get("CPU_PARTITION", "day"), "EXPECT_N": args.n})

    stamp = time.strftime("%Y%m%d_%H%M%S")
    gen_path = os.path.join(logdir, f"gen_{stamp}.sbatch")
    merge_path = os.path.join(logdir, f"merge_{stamp}.sbatch")
    print(f"dataset   : {outdir}")
    print(f"code      : {code_dir}")
    print(f"samples   : n={args.n}  nx={nx}  material={material}  seed0={seed0}  chunk={chunk}  -> {n_tasks} tasks, submitting {len(tasks)}")
    print(f"array     : --array={array}")
    print(f"batch args : {batch_args or '(none)'}")
    print(f"sample args: {sample_args or '(none)'}   gpus={args.gpus or cfg.get('GPUS', '1')}  "
          f"jobs/task={jobs}  partition={args.partition or cfg.get('PARTITION', 'gpu')}"
          + (f"  exclude={exclude}" if exclude else ""))
    if args.dry_run:
        print("\n--- rendered gen_array.sbatch ---")
        print(gen)
        print("--- commands (not run) ---")
        sbatch(gen_path, True)
        if not args.no_merge:
            print("  $ sbatch --parsable --dependency=afterany:<ARRAY_JOB_ID>", merge_path)
        return

    open(gen_path, "w").write(gen)
    open(merge_path, "w").write(merge)
    if not os.path.isfile(os.path.join(code_dir, ".venv", "bin", "python")):
        sys.exit(f"no venv at {code_dir}/.venv: run  bash cluster/setup_env.sh  first")
    job = sbatch(gen_path, False)
    merge_job = None
    if not args.no_merge:
        cmd = ["sbatch", "--parsable", f"--dependency=afterany:{job}", merge_path]
        print("  $", " ".join(cmd))
        out = subprocess.run(cmd, capture_output=True, text=True)
        merge_job = out.stdout.strip().split(";")[0] if out.returncode == 0 else None
        if out.returncode != 0:
            print("merge submission failed (array still running):", out.stderr)
    json.dump({"time": stamp, "n": args.n, "nx": nx, "material": material, "seed0": seed0, "chunk": chunk,
               "sample_args": sample_args, "batch_args": batch_args, "tasks": tasks, "array": array,
               "array_job_id": job, "merge_job_id": merge_job, "outdir": outdir},
              open(sub_path, "w"), indent=2)
    print(f"submitted array job {job}" + (f", merge job {merge_job}" if merge_job else ""))
    print(f"monitor : python3 cluster/submit.py --status {name}")
    print(f"logs    : {logdir}/gen_{job}_<task>.out")


if __name__ == "__main__":
    main()
