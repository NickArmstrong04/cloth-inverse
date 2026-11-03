"""Force-set / seed sweep for the video (Nick, 2026-11-03): find pushes on which the camera path
(photo -> SAM -> VGGT -> decoded grid -> M21-c) does exceptionally well, and record how the
perfect-information model (M16-a on the cloth mesh) does on the same pushes.

Per candidate (force set x seed), in this order:
  1. the mesh path: SimService (cloth physics, 256 grid, 30 snapshots) + M16-a, scored like
     make_cache.py; saved as cache-format ``entry_<set>_s<seed>.json`` (drop into cache/ or feed
     make_present.py);
  2. the camera path: camera.run_live (paper re-sim, photo pair, masks, VGGT, decode, grid, M21-c)
     with the stage artifacts recorded; saved as ``camera_<set>_s<seed>.json`` in the exact shape
     record_camera_job.py writes (so present/camera.json can be swapped directly);
  3. one row appended to ``results.jsonl`` and printed.

Usage (WSL, cloth venv, ~3 min per candidate; without META_API_KEY the masks are the renderer's):
    SIM_PY=~/cloth/.venv/bin/python SAM_PY=~/sam3env/bin/python \
    ~/cloth/.venv/bin/python demo/backend/sweep_live.py --out demo/backend/sweep --seeds 1 2 \
        [--sets A B C ...] [--skip-mesh] [--skip-camera]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "cloth_data_generation", "transfer"))

R2 = 1 / np.sqrt(2)
# "two simpler forces": opposing, in-plane (no tilt), well separated, moderate strength.
# The camera path turns any pair into its scrunch loading (each push heads toward the other,
# 0.2-2 N, 0.3-0.7 s) so these pass through unchanged; the mesh path (M16-a, free loading) sees
# them as they are.
SETS: dict[str, list[dict]] = {
    "A_horiz": [{"u": 0.25, "v": 0.5, "direction": [1, 0, 0], "magnitude": 0.6, "duration": 0.5},
                {"u": 0.75, "v": 0.5, "direction": [-1, 0, 0], "magnitude": 0.6, "duration": 0.5}],
    "B_vert": [{"u": 0.5, "v": 0.25, "direction": [0, 1, 0], "magnitude": 0.6, "duration": 0.5},
               {"u": 0.5, "v": 0.75, "direction": [0, -1, 0], "magnitude": 0.6, "duration": 0.5}],
    "C_diag": [{"u": 0.25, "v": 0.25, "direction": [R2, R2, 0], "magnitude": 0.6, "duration": 0.5},
               {"u": 0.75, "v": 0.75, "direction": [-R2, -R2, 0], "magnitude": 0.6, "duration": 0.5}],
    "D_gentle": [{"u": 0.25, "v": 0.5, "direction": [1, 0, 0], "magnitude": 0.4, "duration": 0.5},
                 {"u": 0.75, "v": 0.5, "direction": [-1, 0, 0], "magnitude": 0.4, "duration": 0.5}],
    "E_strong": [{"u": 0.25, "v": 0.5, "direction": [1, 0, 0], "magnitude": 1.0, "duration": 0.5},
                 {"u": 0.75, "v": 0.5, "direction": [-1, 0, 0], "magnitude": 1.0, "duration": 0.5}],
    "F_offset": [{"u": 0.3, "v": 0.4, "direction": [1, 0, 0], "magnitude": 0.7, "duration": 0.5},
                 {"u": 0.7, "v": 0.6, "direction": [-1, 0, 0], "magnitude": 0.7, "duration": 0.5}],
    # non-opposing "simple" pairs inside M16-a's free loading: both pushes straight into the sheet
    "H_down": [{"u": 0.3, "v": 0.5, "direction": [0, 0, -1], "magnitude": 0.5, "duration": 0.5},
               {"u": 0.7, "v": 0.5, "direction": [0, 0, -1], "magnitude": 0.5, "duration": 0.5}],
    "I_down": [{"u": 0.3, "v": 0.35, "direction": [0, 0, -1], "magnitude": 0.8, "duration": 0.5},
               {"u": 0.7, "v": 0.65, "direction": [0, 0, -1], "magnitude": 0.8, "duration": 0.5}],
    "G_tilt": [{"u": 0.25, "v": 0.5, "direction": [0.9, 0, 0.44], "magnitude": 0.6, "duration": 0.5},
               {"u": 0.75, "v": 0.5, "direction": [-0.9, 0, 0.44], "magnitude": 0.6, "duration": 0.5}],
}


def cache_set(entry: int) -> list[dict]:
    hit = json.load(open(os.path.join(_HERE, "cache", f"entry_{entry:02d}.json")))
    return [{k: f[k] for k in ("u", "v", "direction", "magnitude", "duration")} for f in hit["true_forces"]]


def mesh_path(svc, ckpt, forces: list[dict], seed: int) -> tuple[dict, dict]:
    import inference as inf
    from simulate import downsample_frames, encode_frames, validate_forces
    forces = validate_forces(forces)
    sim = svc.run(forces, nx=256, snapshots=30, seed=seed)
    pred, predicted = inf.predict_forces(ckpt, sim["final_pos"], sim["final_vel"])
    per_force, mean = inf.score_forces(ckpt, pred, forces)
    entry = {
        "frames": encode_frames(downsample_frames(sim["frames"], 64)),
        "true_forces": [dict(f, impulse=f["magnitude"] * f["duration"]) for f in forces],
        "predicted_forces": predicted,
        "reversal_forces": [dict(f, direction=[-x for x in f["direction"]]) for f in predicted],
        "metrics_mean": {k: float(mean[k]) for k in ("pos_cm", "dir_deg", "impulse_rel")},
        "timings": {"sim_s": sim["wall_s"]},
        "grid": 256,
        "seed": seed,
    }
    return entry, {"mean": entry["metrics_mean"], "per_force": [{k: float(p[k]) for k in ("pos_cm", "dir_deg", "impulse_rel")} for p in per_force]}


def camera_path(cam, model, sim, ref, forces: list[dict], seed: int, work: str) -> dict:
    stages: dict = {}
    timeline: list = []
    t0 = time.time()

    def on_stage(s, p, **artifacts):
        stages.update(artifacts)
        timeline.append({"t": round(time.time() - t0, 1), "status": s})

    out = cam.run_live(model, sim, forces, ref["sim_config"], ref["force_footprint"], work, seed, on_stage=on_stage)
    out["timings"]["total_s"] = time.time() - t0
    stages["predict"] = {"elapsed_s": out["timings"]["predict_s"]}
    stages["narrate"] = {"text": None, "ok": False, "model": None, "error": "skipped (local sweep)", "elapsed_s": 0.0}
    out["narration"] = stages["narrate"]
    timeline.append({"t": round(time.time() - t0, 1), "status": "done"})
    return {"job": {"status": "done", "progress": 1.0, "stages": stages, "result": out}, "timeline": timeline,
            "forces": forces, "seed": seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "sweep"))
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "checkpoints", "M16a"))
    ap.add_argument("--sets", nargs="*", default=None, help="names from SETS and/or cache:NN")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--skip-mesh", action="store_true")
    ap.add_argument("--skip-camera", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = a.sets or (list(SETS) + ["cache:10", "cache:12"])
    sets = {n: (cache_set(int(n.split(":")[1])) if n.startswith("cache:") else SETS[n]) for n in names}

    t = time.time()
    svc = ckpt = None
    if not a.skip_mesh:
        import inference as inf
        from simulate import SimService
        ckpt, _ = inf.load_checkpoint(a.ckpt)
        svc = SimService()
    cam = model = sim = ref = None
    if not a.skip_camera:
        import camera as cam_mod
        from eval_resim import Simulator
        cam = cam_mod
        ref = json.load(open(os.path.join(_HERE, "rest_reference.json")))
        work0 = tempfile.mkdtemp(prefix="sweep_")
        sim = Simulator(ref["sim_config"], workdir=work0)
        model = cam_mod.Model()
    print(f"warm {time.time() - t:.0f} s; {len(sets)} sets x {len(a.seeds)} seeds; "
          f"masks: {'SAM 3.1 hosted' if os.environ.get('META_API_KEY') else 'renderer (no META_API_KEY)'}", flush=True)
    rows_path = os.path.join(a.out, "results.jsonl")
    for name, forces in sets.items():
        tag = name.replace(":", "")
        for seed in a.seeds:
            row = {"set": name, "seed": seed, "forces": forces}
            t0 = time.time()
            try:
                if svc is not None:
                    entry, m16 = mesh_path(svc, ckpt, forces, seed)
                    json.dump(entry, open(os.path.join(a.out, f"entry_{tag}_s{seed}.json"), "w"))
                    row["m16"] = m16
                    row["mesh_s"] = round(time.time() - t0, 1)
            except Exception as e:  # the camera path still runs (paper physics is a different sim)
                row["mesh_error"] = f"{type(e).__name__}: {str(e)[-200:]}"
                traceback.print_exc()
            try:
                if cam is not None:
                    t1 = time.time()
                    work = os.path.join(sim.workdir if hasattr(sim, "workdir") else tempfile.gettempdir(), f"{tag}_s{seed}")
                    rec = camera_path(cam, model, sim, ref, forces, seed, work)
                    json.dump(rec, open(os.path.join(a.out, f"camera_{tag}_s{seed}.json"), "w"))
                    r = rec["job"]["result"]
                    row["camera"] = r["camera"]["metrics"]
                    row["m21_exact"] = r["mesh"]["metrics"]
                    row["coverage"] = r["coverage"]
                    row["adjustments"] = r["adjustments"]
                    row["mask_source"] = r["mask_source"]
                    row["camera_s"] = round(time.time() - t1, 1)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {str(e)[-300:]}"
                traceback.print_exc()
            with open(rows_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            c = row.get("camera", {}).get("mean", {})
            m = row.get("m16", {}).get("mean", {})
            x = row.get("m21_exact", {}).get("mean", {})
            cov = row.get("coverage", {})
            print(f"{name:9s} s{seed}  camera {c.get('pos_cm', float('nan')):5.1f} cm {c.get('dir_deg', float('nan')):5.1f} deg "
                  f"{c.get('impulse_rel', float('nan')):.2f} | M21-c exact {x.get('pos_cm', float('nan')):4.1f} cm {x.get('dir_deg', float('nan')):4.1f} deg "
                  f"| M16-a {m.get('pos_cm', float('nan')):4.1f} cm {m.get('dir_deg', float('nan')):4.1f} deg {m.get('impulse_rel', float('nan')):.2f} "
                  f"| seen {cov.get('camera_sees', float('nan')):.2f} decoded {cov.get('decoded', float('nan')):.2f} "
                  f"| {row.get('mesh_s', 0)}+{row.get('camera_s', 0)} s {row.get('mesh_error', '')} {row.get('error', '')}", flush=True)
    print("-> ", rows_path, flush=True)


if __name__ == "__main__":
    main()
