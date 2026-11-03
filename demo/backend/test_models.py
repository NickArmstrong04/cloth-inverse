"""Check of the model menu outside the web service: run app._run_paper (and, with --all, the
M16a cloth path) on one force pair and print the payload summary.

  SIM_PY=... python test_models.py [--model M21c|M20a3] [--all]
"""

from __future__ import annotations

import argparse
import json
import time

import app


FORCES = [{"u": 0.3, "v": 0.6, "direction": [0.9, -0.3, 0.3], "magnitude": 0.8, "duration": 0.5},
          {"u": 0.7, "v": 0.4, "direction": [0.0, 0.3, -0.95], "magnitude": 0.5, "duration": 0.4}]


def run(job_id: str, fn, *args) -> dict:
    with app._lock:
        app._jobs[job_id] = {"status": "queued", "progress": 0.0}
    t = time.time()
    fn(job_id, *args)
    j = app._jobs[job_id]
    print(f"{job_id}: {j['status']} in {time.time() - t:.1f} s")
    if j["status"] != "done":
        print("  error:", j.get("error"))
        return j
    r = j["result"]
    m = r["metrics"]["mean"]
    print(f"  model {r.get('model')} physics {r.get('physics')} grid {r['grid']} frames {r['frame_shape']}"
          f" sim {r['timings']['sim_s']:.1f} s infer {r['timings']['infer_s']:.2f} s")
    print(f"  pos {m['pos_cm']:.1f} cm dir {m['dir_deg']:.0f} deg impulse {m['impulse_rel']:.2f}"
          f" | reference {r['metrics']['reference']}")
    print("  adjustments:", r.get("adjustments"))
    print("  payload bytes", len(json.dumps(r)))
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M21c")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    print("available:", {k: app.model_available(k) for k in app.MODELS})
    run("paper1", app._run_paper, FORCES, a.model)
    if a.all:
        app._warm()
        run("cloth1", app._run, FORCES)


if __name__ == "__main__":
    main()
