#!/usr/bin/env python3
"""Combine benchmark run times, measured queue waits and probe history into
one table per GPU type, and estimate time-to-completion for a dataset.

  python3 cluster/bench_report.py                      # latest run
  python3 cluster/bench_report.py --run bench_20261028_0900 --n 1000 --nx 256 --concurrent 16

Columns per GPU type:
  wait        queue wait of the benchmark job itself (sacct Submit -> Start)
  probe wait  median scheduler estimate from queue_probe.py history, if any
  s/sample    sim wall time at each resolution (median over repeats, excluding
              the first repeat which includes kernel compilation)
  util        mean GPU utilization while the simulation was running
  mem         peak GPU memory
  billing     Slurm TRES billing weight of the job (fairshare cost per hour)
Estimate: time to finish N samples at --nx with K concurrent tasks
  = expected wait + ceil(N/K) * s_per_sample, also shown per billing-hour.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(os.path.dirname(HERE), "bench")


def sacct(job: str) -> dict:
    out = subprocess.run(["sacct", "-j", job, "-X", "-n", "-P", "--format=Submit,Start,End,Elapsed,State,AllocTRES,NodeList"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        return {}
    sub, start, end, elapsed, state, tres, node = out[0].split("|")[:7]
    row = {"state": state, "elapsed": elapsed, "node": node, "tres": tres}
    try:
        row["wait_min"] = (dt.datetime.fromisoformat(start) - dt.datetime.fromisoformat(sub)).total_seconds() / 60
    except Exception:
        row["wait_min"] = None
    m = re.search(r"billing=(\d+)", tres)
    row["billing"] = int(m.group(1)) if m else None
    return row


def probe_waits() -> dict:
    p = os.path.join(BENCH_DIR, "queue_probe.jsonl")
    by = {}
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            if r.get("wait_min") is not None:
                by.setdefault(r["gpu"], []).append(r["wait_min"])
    return {g: sorted(v)[len(v) // 2] for g, v in by.items()}


def fmt_min(x):
    if x is None:
        return "n/a"
    return "now" if x < 1 else (f"{x:.0f} m" if x < 120 else f"{x/60:.1f} h")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="bench run name (default: latest)")
    ap.add_argument("--n", type=int, default=1000, help="dataset size for the estimate")
    ap.add_argument("--nx", type=int, default=256, help="resolution for the estimate")
    ap.add_argument("--concurrent", type=int, default=16, help="concurrent array tasks (one GPU each)")
    ap.add_argument("--wait-source", choices=["job", "probe", "max"], default="max",
                    help="which wait estimate to use: the benchmark job's own, the probe median, or the larger")
    args = ap.parse_args()

    runs = sorted(d for d in glob.glob(os.path.join(BENCH_DIR, "bench_*")) if os.path.isdir(d))
    if not runs:
        sys.exit(f"no bench runs under {BENCH_DIR}")
    run_dir = os.path.join(BENCH_DIR, args.run) if args.run else runs[-1]
    manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
    probes = probe_waits()
    have_sacct = subprocess.run(["which", "sacct"], capture_output=True).returncode == 0

    table = {}
    nxs = set()
    # group job entries by GPU type (one-job-per-nx runs have keys like "h100/nx256")
    groups = {}
    for key, info in manifest["jobs"].items():
        groups.setdefault(info.get("gpu") or key.split("/")[0], []).append((key, info))
    for g, entries in groups.items():
        rows = []
        for key, _ in entries:
            rp = os.path.join(run_dir, key, "results.jsonl")
            if os.path.exists(rp):
                rows += [json.loads(l) for l in open(rp)]
        jp = os.path.join(run_dir, entries[0][0], "job.json")
        jobinfo = json.load(open(jp)) if os.path.exists(jp) else {}
        accts = [sacct(i["job_id"]) for _, i in entries if have_sacct and "job_id" in i]
        accts = [a for a in accts if a]
        acct = {}
        if accts:
            waits = sorted(a["wait_min"] for a in accts if a.get("wait_min") is not None)
            acct = dict(accts[0])
            acct["wait_min"] = waits[len(waits) // 2] if waits else None
            states = {a["state"] for a in accts}
            acct["state"] = states.pop() if len(states) == 1 else "mixed"
        per_nx = {}
        for r in rows:
            if r["rc"] != 0 or r["sim_wall_s"] is None:
                per_nx.setdefault(r["nx"], {"fail": True})
                nxs.add(r["nx"])
                continue
            conc = r.get("conc", 1)
            key = r["nx"] if conc == 1 else f"{r['nx']}x{conc}"
            d = per_nx.setdefault(key, {"t": [], "util": [], "mem": [], "conc": conc})
            # effective time per sample: concurrent rounds use the group's wall clock / processes
            eff = r["group_wall_s"] / conc if conc > 1 and r.get("group_wall_s") else r["sim_wall_s"]
            same = [x for x in rows if x["nx"] == r["nx"] and x.get("conc", 1) == conc]
            if (r["rep"] > 1 or len(same) == conc) and r.get("proc", 1) == 1:
                d["t"].append(eff)  # one entry per round, not per process
            d["util"].append(r.get("util_busy_mean_pct") or 0)
            d["mem"].append(r.get("mem_peak_mb") or 0)
            nxs.add(key)
        table[g] = {"acct": acct, "job": jobinfo, "per_nx": per_nx, "probe_wait": probes.get(g)}

    def nx_key(k):
        base, _, c = str(k).partition("x")
        return (int(base), int(c) if c else 1)

    def nx_num(k):
        return nx_key(k)[0]
    nxs = sorted(nxs, key=nx_key)
    print(f"run: {os.path.basename(run_dir)}   (sample args: {manifest.get('sample_args')})\n")
    hdr = f"{'gpu':22s} {'state':10s} {'wait':>7s} {'probe':>7s} {'bill':>5s} " + " ".join(f"{'nx'+str(n):>9s}" for n in nxs)
    print(hdr)
    if any("x" in str(n) for n in nxs):
        print("(nx256x3 = three samples at once on one GPU; its s/sample is wall clock / 3)")
    print(" " * 55 + " ".join(f"{'s/sample':>9s}" for _ in nxs))
    for g, t in table.items():
        a = t["acct"]
        cells = []
        for n in nxs:
            d = t["per_nx"].get(n)
            if not d:
                cells.append(f"{'-':>9s}")
            elif d.get("fail"):
                cells.append(f"{'FAIL':>9s}")
            else:
                cells.append(f"{sorted(d['t'])[len(d['t'])//2]:>9.0f}")
        print(f"{g:22s} {a.get('state','?')[:10]:10s} {fmt_min(a.get('wait_min')):>7s} {fmt_min(t['probe_wait']):>7s} "
              f"{str(a.get('billing') or '?'):>5s} " + " ".join(cells))
    print("\nsamples per 1-hour job (3600 / s_per_sample) / samples measured:")
    for g, t in table.items():
        cells = []
        for n in nxs:
            d = t["per_nx"].get(n)
            if not d or d.get("fail") or not d.get("t"):
                cells.append(f"{'-':>9s}")
            else:
                med = sorted(d["t"])[len(d["t"]) // 2]
                cells.append(f"{3600/med:6.1f}/{len(d['t']) * d.get('conc', 1):>2d}")
        print(f"{g:22s} " + " ".join(cells))
    print("\nGPU utilization (%) / peak memory (GB) while simulating:")
    for g, t in table.items():
        cells = []
        for n in nxs:
            d = t["per_nx"].get(n)
            cells.append(f"{'-':>9s}" if not d or d.get("fail") else f"{max(d['util']):3.0f}%/{max(d['mem'])/1024:4.1f}")
        print(f"{g:22s} " + " ".join(cells))

    # scaling exponent per GPU: t ~ particles^alpha
    print("\nscaling exponent alpha (time ~ particles^alpha; 1.0 = linear):")
    for g, t in table.items():
        pts = [(nx_num(n) ** 2, sorted(d["t"])[len(d["t"]) // 2]) for n, d in t["per_nx"].items()
               if d.get("t") and d.get("conc", 1) == 1]
        if len(pts) >= 2:
            xs = [math.log(p) for p, _ in pts]; ys = [math.log(s) for _, s in pts]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            alpha = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(1e-12, sum((x - mx) ** 2 for x in xs))
            print(f"  {g:22s} alpha = {alpha:.2f}")

    print(f"\nestimate: {args.n} samples at nx={args.nx}, {args.concurrent} concurrent tasks")
    print(f"{'gpu':22s} {'wait used':>9s} {'s/sample':>9s} {'run time':>9s} {'total':>9s} {'GPU-h':>7s} {'billing-h':>10s}")
    ranked = []
    for g, t in table.items():
        d = t["per_nx"].get(args.nx)
        if not d or not d.get("t"):
            # interpolate from the two nearest measured sizes using the fitted exponent if possible
            pts = sorted((nx_num(n), sorted(v["t"])[len(v["t"]) // 2]) for n, v in t["per_nx"].items()
                         if v.get("t") and v.get("conc", 1) == 1)
            if len(pts) < 2:
                print(f"{g:22s}  no data at nx={args.nx}")
                continue
            (n1, t1), (n2, t2) = pts[-2], pts[-1]
            alpha = math.log(t2 / t1) / math.log((n2 * n2) / (n1 * n1))
            s = t2 * ((args.nx * args.nx) / (n2 * n2)) ** alpha
            note = "*"
        else:
            s = sorted(d["t"])[len(d["t"]) // 2]
            note = ""
        waits = [w for w in (t["acct"].get("wait_min"), t["probe_wait"]) if w is not None]
        if args.wait_source == "job":
            wait = t["acct"].get("wait_min") or 0.0
        elif args.wait_source == "probe":
            wait = t["probe_wait"] or 0.0
        else:
            wait = max(waits) if waits else 0.0
        run_min = math.ceil(args.n / args.concurrent) * s / 60
        total = wait + run_min
        gpu_h = args.n * s / 3600
        bill = t["acct"].get("billing")
        ranked.append((total, g, wait, s, run_min, gpu_h, bill, note))
    for total, g, wait, s, run_min, gpu_h, bill, note in sorted(ranked):
        print(f"{g:22s} {fmt_min(wait):>9s} {s:>8.0f}{note} {fmt_min(run_min):>9s} {fmt_min(total):>9s} {gpu_h:>7.1f} "
              f"{(gpu_h * bill if bill else float('nan')):>10.0f}")
    if any(r[7] for r in ranked):
        print("* extrapolated from measured sizes with the fitted scaling exponent")
    print("\nSmallest 'total' wins for wall-clock; smallest 'billing-h' wins for fairshare cost.")


if __name__ == "__main__":
    main()
