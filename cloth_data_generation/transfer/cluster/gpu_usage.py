#!/usr/bin/env python3
"""Live GPU occupancy per GPU type across the whole cluster.

Reads `scontrol show node` (configured vs. used GPUs, node state) and
`squeue` (running jobs, their users and nodes; pending GPU requests) and
prints one row per GPU type:

  GPUs total / in use / free, nodes idle/mixed/alloc/down,
  running jobs and distinct users holding that type, pending jobs asking
  for it (typed requests only; untyped 'gpu:1' requests are listed separately).

  python3 cluster/gpu_usage.py
  python3 cluster/gpu_usage.py --watch 600       # refresh + log every 10 min
  python3 cluster/gpu_usage.py --history         # free-GPU statistics from the log
Works from a login node or an allocation; no job is submitted.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "bench", "gpu_usage.jsonl")


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def nodes_by_type() -> tuple[dict, dict]:
    """Return {gtype: {total, used, nodes: {state: n}, node_names}} and {node: gtype}."""
    text = sh(["scontrol", "show", "node", "-o"])
    types: dict = {}
    node_type: dict = {}
    for line in text.splitlines():
        name = re.search(r"NodeName=(\S+)", line)
        gres = re.search(r"\bGres=(\S+)", line)
        if not name or not gres or "gpu" not in gres.group(1):
            continue
        used = re.search(r"GresUsed=(\S+)", line)
        state = re.search(r"State=(\S+)", line)
        parts = re.search(r"Partitions=(\S+)", line)
        st = (state.group(1) if state else "UNKNOWN").split("+")[0].rstrip("*~#$@^-")
        for m in re.finditer(r"gpu:([A-Za-z0-9_]+):(\d+)", gres.group(1)):
            g, n = m.group(1), int(m.group(2))
            u = 0
            if used:
                mu = re.search(rf"gpu:{re.escape(g)}:(\d+)", used.group(1))
                u = int(mu.group(1)) if mu else 0
            d = types.setdefault(g, {"total": 0, "used": 0, "nodes": collections.Counter(), "node_names": set(),
                                     "partitions": set()})
            d["total"] += n
            d["used"] += u
            d["nodes"][st.lower()] += 1
            d["node_names"].add(name.group(1))
            if parts:
                d["partitions"].update(parts.group(1).split(","))
            node_type[name.group(1)] = g
    return types, node_type


def expand_nodelist(nl: str) -> list[str]:
    out = sh(["scontrol", "show", "hostnames", nl])
    return out.split()


def running_jobs(node_type: dict) -> dict:
    """{gtype: {"jobs": n, "users": set, "gpus": n}} from running jobs with GPU allocations."""
    out = sh(["squeue", "-h", "-t", "R", "-O", "JobID:20,UserName:30,tres-alloc:200,NodeList:200"])
    res: dict = {}
    for line in out.splitlines():
        jid, user, tres, nodelist = line[:20].strip(), line[20:50].strip(), line[50:250].strip(), line[250:].strip()
        if "gres/gpu" not in tres:
            continue
        typed = re.findall(r"gres/gpu:([A-Za-z0-9_]+)=(\d+)", tres)
        if typed:
            for g, n in typed:
                d = res.setdefault(g, {"jobs": 0, "users": set(), "gpus": 0})
                d["jobs"] += 1
                d["users"].add(user)
                d["gpus"] += int(n)
        else:
            m = re.search(r"gres/gpu=(\d+)", tres)
            n = int(m.group(1)) if m else 0
            # untyped allocation: attribute to the GPU type of the node(s)
            for node in expand_nodelist(nodelist) if nodelist else []:
                g = node_type.get(node)
                if g:
                    d = res.setdefault(g, {"jobs": 0, "users": set(), "gpus": 0})
                    d["jobs"] += 1
                    d["users"].add(user)
                    d["gpus"] += n
                    break
    return res


def pending_jobs() -> tuple[dict, int]:
    """Pending jobs per requested GPU type, and count of untyped GPU requests."""
    out = sh(["squeue", "-h", "-t", "PD", "-O", "JobID:20,tres-per-node:120,tres-per-job:120,Partition:30"])
    per: dict = collections.Counter()
    untyped = 0
    for line in out.splitlines():
        s = line[20:]
        if "gpu" not in s:
            continue
        typed = re.findall(r"gpu:([A-Za-z0-9_]+)", s)
        typed = [t for t in typed if not t.isdigit()]
        if typed:
            for g in set(typed):
                per[g] += 1
        else:
            untyped += 1
    return per, untyped


def snapshot() -> list[dict]:
    types, node_type = nodes_by_type()
    run = running_jobs(node_type)
    pend, untyped = pending_jobs()
    rows = []
    now = dt.datetime.now().isoformat(timespec="minutes")
    for g, d in sorted(types.items(), key=lambda kv: -kv[1]["total"]):
        r = run.get(g, {"jobs": 0, "users": set(), "gpus": 0})
        down = sum(v for k, v in d["nodes"].items() if k in ("down", "drain", "drained", "draining", "maint", "reserved"))
        rows.append({"time": now, "gpu": g, "total": d["total"], "used": d["used"], "free": d["total"] - d["used"],
                     "nodes_idle": d["nodes"].get("idle", 0), "nodes_mixed": d["nodes"].get("mixed", 0),
                     "nodes_alloc": d["nodes"].get("allocated", 0), "nodes_down": down,
                     "running_jobs": r["jobs"], "users": len(r["users"]), "pending_jobs": pend.get(g, 0),
                     "partitions": sorted(p for p in d["partitions"] if not p.startswith(("scavenge", "priority")))})
    rows.append({"time": now, "gpu": "_untyped_pending", "pending_jobs": untyped})
    return rows


def print_rows(rows: list[dict]) -> None:
    print(f"{'gpu':24s} {'total':>5s} {'in use':>6s} {'free':>5s} {'free%':>5s} | {'idle':>4s} {'mixed':>5s} {'alloc':>5s} {'down':>4s} | "
          f"{'jobs':>4s} {'users':>5s} {'pending':>7s}  partitions")
    for r in rows:
        if r["gpu"].startswith("_"):
            continue
        pct = 100.0 * r["free"] / r["total"] if r["total"] else 0
        print(f"{r['gpu']:24s} {r['total']:>5d} {r['used']:>6d} {r['free']:>5d} {pct:>4.0f}% | {r['nodes_idle']:>4d} {r['nodes_mixed']:>5d} "
              f"{r['nodes_alloc']:>5d} {r['nodes_down']:>4d} | {r['running_jobs']:>4d} {r['users']:>5d} {r['pending_jobs']:>7d}  "
              f"{','.join(r['partitions'])}")
    unt = [r for r in rows if r["gpu"] == "_untyped_pending"]
    if unt:
        print(f"\npending jobs requesting a GPU without a type: {unt[0]['pending_jobs']} (compete for every type)")
    best = sorted((r for r in rows if not r["gpu"].startswith("_") and r["total"]), key=lambda r: (-r["free"], r["pending_jobs"]))
    if best:
        print("most open right now: " + ", ".join(f"{r['gpu']} ({r['free']} free, {r['pending_jobs']} pending)" for r in best[:3]))


def history() -> None:
    if not os.path.exists(LOG):
        sys.exit(f"no log at {LOG}")
    rows = [json.loads(l) for l in open(LOG)]
    by: dict = {}
    for r in rows:
        if not r["gpu"].startswith("_"):
            by.setdefault(r["gpu"], []).append(r)
    print(f"{len(rows)} snapshots since {rows[0]['time']}")
    print(f"{'gpu':24s} {'snaps':>5s} {'median free':>11s} {'min free':>8s} {'% time >=1 free':>15s} {'median pending':>14s}")
    for g, rs in sorted(by.items(), key=lambda kv: -kv[1][0]["total"]):
        fr = sorted(r["free"] for r in rs)
        pd_ = sorted(r["pending_jobs"] for r in rs)
        avail = 100.0 * sum(1 for x in fr if x >= 1) / len(fr)
        print(f"{g:24s} {len(rs):>5d} {fr[len(fr)//2]:>11d} {fr[0]:>8d} {avail:>14.0f}% {pd_[len(pd_)//2]:>14d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", type=int, default=0, help="repeat every N seconds and append to the log")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()
    if args.history:
        history()
        return
    while True:
        rows = snapshot()
        print(f"\n[{rows[0]['time']}] GPU occupancy by type")
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
