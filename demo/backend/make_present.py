"""Build the static presentation data for the demo video (plan 2026-11-03, step 1).

Takes one cache entry (``cache/entry_NN.json``) and rebuilds the exact ``JobResult`` payload the
app's ``/jobs/{id}`` returns for a live push (same scoring path as ``app._fallback``), so the
frontend's presentation mode (``?present=1``) can replay it with no backend.

Usage (WSL, cloth venv):
    /home/narmstrong/cloth/.venv/bin/python demo/backend/make_present.py --entry 12 \
        --ckpt demo/backend/checkpoints/M16a --out demo/frontend/public/present/entry.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "cloth_data_generation", "transfer"))

import inference as inf  # noqa: E402

REFERENCE = {"pos_cm": 1.0, "dir_deg": 4.3, "impulse_rel": 0.156}  # app.py REFERENCE (M16-a test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", type=int, default=12)
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "checkpoints", "M16a"))
    ap.add_argument("--out", default=os.path.join(_REPO, "demo", "frontend", "public", "present", "entry.json"))
    a = ap.parse_args()
    hit = json.load(open(os.path.join(_HERE, "cache", f"entry_{a.entry:02d}.json")))
    import base64, io
    import numpy as np
    n_frames = int(np.load(io.BytesIO(base64.b64decode(hit["frames"]))).shape[0])  # some cache runs have < 30
    ckpt, _ = inf.load_checkpoint(a.ckpt)
    forces = [{k: f[k] for k in ("u", "v", "direction", "magnitude", "duration")} for f in hit["true_forces"]]
    t0 = time.time()
    pred_raw = ckpt["encode_forces"](hit["predicted_forces"])
    per_force, mean = inf.score_forces(ckpt, pred_raw, forces)
    # Measured inference time of the model itself (this GPU), the app's footer shows it.
    import numpy as np
    pos = np.zeros((256, 256, 3), np.float32); pos[..., 0] = np.linspace(-0.25, 0.25, 256)[None]; pos[..., 1] = np.linspace(-0.25, 0.25, 256)[:, None]
    inf.predict_forces(ckpt, pos, np.zeros_like(pos))  # warm
    t0 = time.time()
    inf.predict_forces(ckpt, pos, np.zeros_like(pos))
    infer_s = time.time() - t0
    for k in ("pos_cm", "dir_deg", "impulse_rel"):
        assert abs(mean[k] - hit["metrics_mean"][k]) < 1e-4, (k, mean[k], hit["metrics_mean"][k])
    result = {
        "frames": hit["frames"],
        "frame_shape": [n_frames, 64, 64, 3],
        "true_forces": [dict(f, impulse=f["magnitude"] * f["duration"]) for f in forces],
        "predicted_forces": hit["predicted_forces"],
        "reversal_forces": hit["reversal_forces"],
        "metrics": {"per_force": [{k: f[k] for k in ("pos_cm", "dir_deg", "impulse_rel")} for f in per_force],
                    "mean": {k: mean[k] for k in ("pos_cm", "dir_deg", "impulse_rel")},
                    "reference": REFERENCE},
        "timings": {"sim_s": hit["timings"]["sim_s"], "infer_s": infer_s},
        "grid": hit["grid"],
        "cached": False,
        "model": "M16a",
        "physics": "cloth",
        "adjustments": [],
        "source": {"entry": a.entry, "seed": hit.get("seed")},
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(result, open(a.out, "w"))
    print(f"wrote {a.out}: entry {a.entry} seed {hit.get('seed')} mean {mean} ({os.path.getsize(a.out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
