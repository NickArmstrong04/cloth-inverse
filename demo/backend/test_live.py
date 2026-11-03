"""End-to-end check of the live camera path (camera.run_live) outside the web service.

  python test_live.py [--ref <rest test h5>] [--write-ref]

--write-ref stores the reference physics (sim_config + force footprint of a ds256_paper_rest sample)
in rest_reference.json, which the service loads.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import camera

REF_JSON = os.path.join(camera._HERE, "rest_reference.json")


def write_ref(h5: str) -> None:
    import h5py
    with h5py.File(h5, "r") as h:
        cfg = json.loads(h["sim_config"][()])
        fp = json.loads(h["force_params"][()])
    keep = {k: cfg[k] for k in ("sheet", "sim", "nx", "ny", "t_capture")}
    json.dump({"source": os.path.basename(h5), "sim_config": keep,
               "force_footprint": [{k: f[k] for k in ("sigma_cells", "radius_cells", "duration")} for f in fp[:2]]},
              open(REF_JSON, "w"), indent=1)
    print("wrote", REF_JSON)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None)
    ap.add_argument("--write-ref", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.write_ref:
        write_ref(a.ref)
    ref = json.load(open(REF_JSON))
    from eval_resim import Simulator
    t = time.time()
    work = tempfile.mkdtemp(prefix="camlive_")
    sim = Simulator(ref["sim_config"], workdir=work)
    model = camera.Model()
    print(f"warm {time.time() - t:.1f} s")
    forces = [{"u": 0.3, "v": 0.6, "direction": [0.9, -0.3, 0.3], "magnitude": 0.8, "duration": 0.5},
              {"u": 0.7, "v": 0.4, "direction": [0.0, 0.3, -0.95], "magnitude": 0.5, "duration": 0.4}]
    t = time.time()
    out = camera.run_live(model, sim, forces, ref["sim_config"], ref["force_footprint"], work, a.seed,
                          on_stage=lambda s, p: print(f"  [{time.time() - t:6.1f} s] {s}", flush=True))
    print(f"total {time.time() - t:.1f} s; timings", json.dumps({k: round(v, 1) for k, v in out["timings"].items()}))
    print("adjustments:", out["adjustments"])
    print("mask:", out["mask_source"], "| coverage", out["coverage"])
    for k in ("camera", "mesh"):
        m = out[k]["metrics"]["mean"]
        print(f"{k}: pos {m['pos_cm']:.1f} cm  dir {m['dir_deg']:.0f} deg  impulse {m['impulse_rel']:.2f}")
    print("payload bytes", len(json.dumps(out)))
    print("workdir", work)


if __name__ == "__main__":
    main()
