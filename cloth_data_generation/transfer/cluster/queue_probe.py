#!/usr/bin/env python3
"""Estimate queue wait per GPU type WITHOUT submitting jobs.

For each GPU type it asks Slurm's scheduler for an estimated start time via
`sbatch --test-only` (nothing is queued) and counts idle/allocated GPUs of that
type from sinfo, plus pending jobs in the partition. Results are printed and
appended to a JSONL log so repeated probes (e.g. hourly, see --watch) build a
picture of wait times by time of day.

  python3 cluster/queue_probe.py                       # all default types, 1 GPU, 2 h
  python3 cluster/queue_probe.py --time 12:00:00 --types a40,l40s,h100
  python3 cluster/queue_probe.py --watch 1800          # re-probe every 30 min
  python3 cluster/queue_probe.py --history             # summarize the log
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_types import DEFAULT_TYPES, GPU_TYPES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "bench", "queue_probe.jsonl")


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def est_start(partition: str, gtype: str, walltime: str, cpus: int, mem: str) -> dict:
    """sbatch --test-only prints: 'sbatch: Job N to start at YYYY-MM-DDTHH:MM:SS using ...'"""
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write("#!/bin/bash\nsleep 1\n")
        path = f.name
    try:
        p = subprocess.run(["sbatch", "--test-only", f"--partition={partition}", f"--gpus={gtype}:1",
                            f"--cpus-per-task={cpus}", f"--mem={mem}", f"--time={walltime}", path],
                           capture_output=True, text=True)
        msg = (p.stdout + p.stderr).strip()
        m = re.search(r"to start at (\S+)", msg)
        if not m:
            return {"est_start": None, "wait_min": None, "msg": msg[-200:]}
        start = dt.datetime.fromisoformat(m.group(1))
        wait = (start - dt.datetime.now()).total_seconds() / 60.0
        return {"est_start": start.isoformat(timespec="minutes"), "wait_min": max(0.0, round(wait, 1)), "msg": ""}
    finally:
        os.unlink(path)


def gpu_counts(partition: str, gtype: str) -> dict:
    """Idle / mixed / allocated node counts carrying this GPU type, and idle GPUs
    (from %G 'gpu:type:N' and the node state; 'mixed' nodes may have free GPUs,
    which %G alone cannot tell, so GPU-level free counts come from GresUsed)."""
    out = run(["sinfo", "-h", "-p", partition, "-N", "-o", "%N|%G|%t|%C"])
    nodes = {"idle": 0, "mixed": 0, "alloc": 0, "other": 0, "total": 0}
    names = []
    for line in out.splitlines():
        name, gres, state, _ = line.split("|")
        if f"gpu:{gtype}:" not in gres:
            continue
        nodes["total"] += 1
        key = state.rstrip("*~#$@^-")
        nodes[key if key in nodes else "other"] += 1
        names.append(name)
    free_gpus = None
    if names:
        s = run(["scontrol", "show", "node", ",".join(names)])
        tot = used = 0
        for m in re.finditer(r"Gres=gpu:(\w+):(\d+)", s):
            if m.group(1) == gtype:
                tot += int(m.group(2))
        for m in re.finditer(r"GresUsed=gpu:(\w+):(\d+)", s):
            if m.group(1) == gtype:
                used += int(m.group(2))
        if tot:
            free_gpus = tot - used
            nodes["gpus_total"] = tot
    nodes["gpus_free"] = free_gpus
    pend = run(["squeue", "-h", "-p", partition, "--state=PD", "-o", "%i"]).split()
    nodes["pending_jobs_partition"] = len(pend)
    return nodes


def probe(types: list[str], walltime: str, cpus: int, mem: str, partitions_override: str | None) -> list[dict]:
    rows = []
    now = dt.datetime.now().isoformat(timespec="minutes")
    for t in types:
        if t not in GPU_TYPES:
            print(f"unknown GPU type {t}; known: {list(GPU_TYPES)}", file=sys.stderr)
            continue
        part = partitions_override or GPU_TYPES[t][0]
        row = {"time": now, "gpu": t, "partition": part, "walltime": walltime}
        row.update(est_start(part, t, walltime, cpus, mem))
        row.update(gpu_counts(part, t))
        rows.append(row)
    return rows


def print_rows(rows: list[dict]) -> None:
    print(f"{'gpu':24s} {'partition':12s} {'est wait':>9s} {'free GPUs':>9s} {'idle/mixed/alloc nodes':>22s} {'pending':>8s}")
    for r in rows:
        w = "now" if r["wait_min"] == 0 else (f"{r['wait_min']:.0f} min" if r["wait_min"] is not None else "n/a")
        fg = "?" if r.get("gpus_free") is None else f"{r['gpus_free']}/{r.get('gpus_total', '?')}"
        print(f"{r['gpu']:24s} {r['partition']:12s} {w:>9s} {fg:>9s} "
              f"{r['idle']:>6d}/{r['mixed']:>5d}/{r['alloc']:>5d}        {r['pending_jobs_partition']:>8d}"
              + (f"   {r['msg']}" if r.get("msg") else ""))


def history() -> None:
    if not os.path.exists(LOG):
        sys.exit(f"no log at {LOG}")
    rows = [json.loads(l) for l in open(LOG)]
    by = {}
    for r in rows:
        by.setdefault(r["gpu"], []).append(r)
    print(f"{len(rows)} probes since {rows[0]['time']}")
    print(f"{'gpu':24s} {'probes':>6s} {'median wait':>11s} {'p90 wait':>9s} {'% immediate':>11s} {'median free GPUs':>16s}")
    for g, rs in by.items():
        w = sorted(r["wait_min"] for r in rs if r["wait_min"] is not None)
        if not w:
            continue
        med = w[len(w) // 2]
        p90 = w[min(len(w) - 1, int(0.9 * len(w)))]
        imm = 100.0 * sum(1 for x in w if x == 0) / len(w)
        fr = sorted(r["gpus_free"] for r in rs if r.get("gpus_free") is not None)
        frm = fr[len(fr) // 2] if fr else "?"
        print(f"{g:24s} {len(w):>6d} {med:>9.0f} m {p90:>7.0f} m {imm:>10.0f}% {str(frm):>16s}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--types", default=",".join(DEFAULT_TYPES))
    ap.add_argument("--time", default="02:00:00", help="walltime to ask for (affects backfill estimate)")
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--mem", default="24G")
    ap.add_argument("--partition", default=None, help="probe this partition for all types (e.g. scavenge_gpu)")
    ap.add_argument("--watch", type=int, default=0, help="repeat every N seconds")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()
    if args.history:
        history()
        return
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    while True:
        rows = probe(types, args.time, args.cpus, args.mem, args.partition)
        print(f"\n[{rows[0]['time'] if rows else ''}] walltime {args.time}")
        print_rows(rows)
        if not args.no_log:
            os.makedirs(os.path.dirname(LOG), exist_ok=True)
            with open(LOG, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
