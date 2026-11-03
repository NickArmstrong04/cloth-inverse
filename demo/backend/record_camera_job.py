"""Record one live camera-path job (POST /camera) as static JSON for the video's presentation mode.

Usage:
    python record_camera_job.py --entry 12 [--api https://...modal.run] \
        --out demo/frontend/public/present/camera.json

Polls GET /jobs/{id} every 2 s, keeps every intermediate response (so the replay can show the
stages in the order and with the artifacts the backend produced) and writes
{"job": <final job>, "timeline": [{"t": s, "status": ...}], "forces": [...]}.
Exit code 2 when SAM 3.1 or Muse Spark fell back (re-run after fixing the backend).
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
API = os.environ.get("API_URL", "http://localhost:8000")  # the deployed backend URL


def req(url: str, body: dict | None = None, timeout: float = 300):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                               method="POST" if data else "GET")
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.load(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", type=int, default=12)
    ap.add_argument("--api", default=API)
    ap.add_argument("--out", default=os.path.join(_REPO, "demo", "frontend", "public", "present", "camera.json"))
    ap.add_argument("--max-s", type=float, default=900)
    ap.add_argument("--seed", type=int, default=None, help="fix the re-sim + camera views (needs the 2026-11-03 backend)")
    ap.add_argument("--forces-json", default=None, help="a sweep camera_*.json or any {'forces': [...]} file instead of a cache entry")
    a = ap.parse_args()
    if a.forces_json:
        src = json.load(open(a.forces_json))
        forces = [{k: f[k] for k in ("u", "v", "direction", "magnitude", "duration")} for f in src["forces"]]
        if a.seed is None and src.get("seed") is not None:
            a.seed = int(src["seed"])
    else:
        hit = json.load(open(os.path.join(_HERE, "cache", f"entry_{a.entry:02d}.json")))
        forces = [{k: f[k] for k in ("u", "v", "direction", "magnitude", "duration")} for f in hit["true_forces"]]
    t0 = time.time()
    print("warming /models ...", flush=True)
    print(req(f"{a.api}/models", timeout=600)["default"], f"({time.time()-t0:.0f} s)", flush=True)
    body = {"forces": forces, **({"seed": a.seed} if a.seed is not None else {})}
    jid = req(f"{a.api}/camera", body, timeout=600)["job_id"]
    print("job", jid, flush=True)
    t1 = time.time()
    timeline, last = [], None
    while time.time() - t1 < a.max_s:
        j = req(f"{a.api}/jobs/{jid}", timeout=60)
        st = j["status"]
        if last != st:
            timeline.append({"t": round(time.time() - t1, 1), "status": st})
            print(f"{time.time()-t1:6.1f} s  {st}  stages={sorted((j.get('stages') or {}).keys())}", flush=True)
            last = st
        if st in ("done", "failed"):
            break
        time.sleep(2)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"job": j, "timeline": timeline, "forces": forces, "entry": a.entry, "seed": a.seed}, open(a.out, "w"))
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1e6:.2f} MB), status {j['status']}")
    if j["status"] != "done":
        print("FAILED:", j.get("error")); sys.exit(1)
    st = j.get("stages", {})
    bad = []
    if st.get("sam", {}).get("fallback"):
        bad.append("SAM 3.1 fell back to the renderer mask")
    if not st.get("narrate", {}).get("ok"):
        bad.append(f"Muse Spark verdict not ok: {st.get('narrate', {}).get('error')}")
    if bad:
        print("WARNING:", "; ".join(bad)); sys.exit(2)
    print("OK: SAM 3.1 live, Muse Spark ok")


if __name__ == "__main__":
    main()
