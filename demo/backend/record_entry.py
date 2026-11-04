"""Record one live POST /simulate result as a static JobResult for the frontend's
"View the demo video's run" button (public/present/entry.json is the M16-a run from the video;
entry_<model>.json holds the same pushes through another model, added 2026-11-03 so the button
works whichever model is selected).

Usage:
    API_URL=https://...modal.run python record_entry.py --model M21c [--entry 10] [--seed 2] \
        [--out demo/frontend/public/present/entry_M21c.json]

The pushes come from the cache entry (default 10, the video's run) or --forces-json. ``--seed``
fixes the paper re-sim (needs the backend from 2026-11-03 evening or later; older backends ignore it).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
API = os.environ.get("API_URL", "http://localhost:8000")


def req(url: str, body: dict | None = None, timeout: float = 300):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                               method="POST" if data else "GET")
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.load(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M21c")
    ap.add_argument("--entry", type=int, default=10)
    ap.add_argument("--forces-json", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--api", default=API)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-s", type=float, default=900)
    a = ap.parse_args()
    out = a.out or os.path.join(_REPO, "demo", "frontend", "public", "present", f"entry_{a.model}.json")
    if a.forces_json:
        src = json.load(open(a.forces_json))["forces"]
    else:
        src = json.load(open(os.path.join(_HERE, "cache", f"entry_{a.entry:02d}.json")))["true_forces"]
    forces = [{k: f[k] for k in ("u", "v", "direction", "magnitude", "duration")} for f in src]
    t0 = time.time()
    print("warming /models ...", flush=True)
    print(req(f"{a.api}/models", timeout=600)["default"], f"({time.time()-t0:.0f} s)", flush=True)
    body = {"forces": forces, "model": a.model, **({"seed": a.seed} if a.seed is not None else {})}
    jid = req(f"{a.api}/simulate", body, timeout=600)["job_id"]
    print("job", jid, flush=True)
    t1, last = time.time(), None
    while time.time() - t1 < a.max_s:
        j = req(f"{a.api}/jobs/{jid}", timeout=60)
        if j["status"] != last:
            print(f"{time.time()-t1:6.1f} s  {j['status']}", flush=True)
            last = j["status"]
        if j["status"] in ("done", "failed"):
            break
        time.sleep(2)
    if j["status"] != "done":
        print("FAILED:", j.get("error")); sys.exit(1)
    res = j["result"]
    res["source"] = {"recorded": time.strftime("%Y-%m-%d %H:%M"), "entry": a.entry, "seed": a.seed, "api": a.api}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"))
    m = res["metrics"]["mean"]
    print(f"wrote {out} ({os.path.getsize(out)/1e6:.2f} MB): model {res.get('model')}, "
          f"{m['pos_cm']:.1f} cm / {m['dir_deg']:.0f} deg, sim {res['timings']['sim_s']:.0f} s, "
          f"adjustments {res.get('adjustments')}")


if __name__ == "__main__":
    main()
