"""Camera path for the demo: rest-state paper -> photo -> SAM 3.1 -> VGGT -> decoded grid -> M21-c.

The research chain (cloth_data_generation/inverse, RESULTS.md M21) run on one sheet:

1. ``photo.py --views 1 --stereo 20 --texture coded --gt``: a 20 deg stereo pair of the sheet
   wearing the coded print;
2. masks from hosted SAM 3.1 (``sam_api.write_masks``, prompt "fabric", key META_API_KEY);
   views SAM misses get the renderer's mask and are flagged (``mask_fallback``);
3. ``lift_stereo.py --stereo-only --sam``: VGGT on the pair + known camera motion, metric depth;
4. ``decode_print.py --base-only``: per-vertex correspondences from the print;
5. ``grid_from_views.py --corr decoded --scale-fix 0.15``: world-frame grid, harmonic fill of
   the hidden side, known-size scale check;
6. M21-c on the camera grid (``X_photo_dec``) and on the true mesh (``X_oracle``), both scored
   against the true pushes.

``package()`` turns step-5 outputs into the JSON the frontend draws; ``make_gallery.py`` calls it
on the 100-sheet hosted-SAM run, ``run_live()`` runs steps 1-6 on a fresh simulation.

Env: SIM_PY (python for photo / decode / grid, default this one), SAM_PY (python with VGGT,
default this one), CAMERA_CKPT (run dir with model.pt, default checkpoints/M21c),
META_API_KEY (hosted SAM 3.1; without it every view falls back to the renderer mask).
"""

from __future__ import annotations

import base64
import glob
import io
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
INVERSE = os.path.join(_REPO, "cloth_data_generation", "inverse")
TRANSFER = os.path.join(_REPO, "cloth_data_generation", "transfer")
for p in (INVERSE, TRANSFER):
    if p not in sys.path:
        sys.path.insert(0, p)

SIM_PY = os.environ.get("SIM_PY", sys.executable)
SAM_PY = os.environ.get("SAM_PY", sys.executable)
CAMERA_CKPT = os.environ.get("CAMERA_CKPT", os.path.join(_HERE, "checkpoints", "M21c"))
SCALE_FIX = "0.15"
NAMES_SHOWN = ("pos_cm", "dir_deg", "impulse_rel")

# M21-c on the 100-sheet hosted SAM 3.1 run, photo_dec + scale fix (RESULTS.md M21) and on the true
# mesh (clean test, 181 sheets): what a typical camera / perfect-information result looks like.
REFERENCE = {"camera": {"pos_cm": 7.4, "dir_deg": 17.0, "pos_cm_median": 6.5},
             "mesh": {"pos_cm": 3.4, "dir_deg": 7.3}}

# Rest-capture paper (ds256_paper_rest, scrunch loading): the training ranges of M21-c.
MAG_RANGE = (0.2, 2.0)
DUR_RANGE = (0.3, 0.7)
UV_RANGE = (0.05, 0.95)
MIN_SEP = 0.2
HEADING_JITTER_DEG = 30.0
TILT_MAX_DEG = 35.0


class Model:
    """M21-c (grid checkpoint, --hybrid --no-vel) via eval_grid.GridRun."""

    def __init__(self, run_dir: str = CAMERA_CKPT, device: str | None = None):
        import torch
        from eval_grid import GridRun
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.run = GridRun(run_dir, dev)

    def predict(self, X: np.ndarray, G: np.ndarray) -> np.ndarray:
        from data import _pool
        pool = X.shape[-1] // self.run.in_size
        if pool > 1:
            X = _pool(X.transpose(1, 2, 0), pool).transpose(2, 0, 1)
        return self.run.predict(X[None].astype(np.float32), G[None].astype(np.float32))[0]


def score(pred: np.ndarray, true_fp: list[dict]):
    """Per-force rows (best of the two pairings, metrics.pair_metrics cost) and their mean."""
    from data import encode_forces
    from metrics import NAMES, _err_one
    y = encode_forces(true_fp)
    best, best_cost = None, None
    for a, b in ((0, 1), (1, 0)):
        e = np.stack([_err_one(pred[0], y[a]), _err_one(pred[1], y[b])])
        cost = e[:, 0].sum() + e[:, 1].sum() / 10
        if best_cost is None or cost < best_cost:
            best, best_cost = e, cost
    per = [{n: float(best[k, i]) for i, n in enumerate(NAMES) if n in NAMES_SHOWN} for k in range(2)]
    mean = {n: float(best[:, i].mean()) for i, n in enumerate(NAMES) if n in NAMES_SHOWN}
    return per, mean


def forces_out(pred: np.ndarray) -> list[dict]:
    from data import decode_forces
    fs = decode_forces(pred)
    for f in fs:
        f["impulse"] = f["magnitude"] * f["duration"]
    return fs


def grid64_b64(X: np.ndarray) -> str:
    """(C>=3, 256, 256) centred positions -> (1, 64, 64, 3) float16 .npy, base64 (frontend parseNpyF16)."""
    p = X[:3].transpose(1, 2, 0)
    p = p - p.reshape(-1, 3).mean(0)
    n = p.shape[0]
    f = n // 64
    p = p.reshape(64, f, 64, f, 3).mean(axis=(1, 3))[None]
    buf = io.BytesIO()
    np.save(buf, p.astype(np.float16), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode()


def jpeg_b64(img: np.ndarray, q: int = 85) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def mask_overlay(photo: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """SAM mask over the photo: outside dimmed, inside tinted blue, outline white."""
    from scipy import ndimage as ndi
    out = photo.astype(np.float32) * 0.3
    tint = np.array([58, 160, 255], np.float32)
    out[mask] = photo[mask].astype(np.float32) * 0.55 + tint * 0.45
    edge = mask & ~ndi.binary_erosion(mask, iterations=2)
    out[edge] = 255
    return out.clip(0, 255).astype(np.uint8)


def package(model: Model, grids_npz: str, photo_png: str, mask_png: str | None, true_fp: list[dict],
            record: dict | None = None, mask_source: str = "SAM 3.1 (Meta Model API)") -> dict:
    """One camera-path result: images, the camera's grid and the true mesh, both predictions, metrics."""
    from PIL import Image
    z = np.load(grids_npz)
    photo = np.asarray(Image.open(photo_png).convert("RGB"))
    mask = (np.asarray(Image.open(mask_png).convert("L")) > 127) if mask_png and os.path.exists(mask_png) else None
    out = {"stem": str(z["stem"]), "true_forces": [dict(f, impulse=f["magnitude"] * f["duration"]) for f in true_fp],
           "images": {"photo": jpeg_b64(photo), "mask": jpeg_b64(mask_overlay(photo, mask)) if mask is not None else None},
           "mask_source": mask_source,
           "grids": {"camera": grid64_b64(z["X_photo_dec"]), "true": grid64_b64(z["X_oracle"]), "shape": [1, 64, 64, 3]}}
    for key, var in (("camera", "photo_dec"), ("mesh", "oracle")):
        pred = model.predict(z[f"X_{var}"], z[f"G_{var}"])
        per, mean = score(pred, true_fp)
        out[key] = {"predicted_forces": forces_out(pred), "metrics": {"per_force": per, "mean": mean}}
    r = record or {}
    out["coverage"] = {"camera_sees": r.get("visible_frac"), "decoded": r.get("dec_visible_frac"),
                       "scale_fix": r.get("scale_fix_dec"), "lift_err_visible_cm": r.get("err_vis_dec_cm")}
    out["reference"] = REFERENCE
    return out


# ---------------------------------------------------------------- live: simulate + chain

def scrunch_forces(forces: list[dict], ref_fp: list[dict]) -> tuple[list[dict], list[str]]:
    """User pushes -> the rest-paper training distribution: each push heads toward the other within
    +-30 deg, tilt within +-35 deg, 0.2-2 N, 0.3-0.7 s. Returns the forces actually simulated and
    a note per change."""
    if len(forces) != 2:
        raise ValueError(f"need exactly 2 pushes, got {len(forces)}")
    fs = [dict(f) for f in forces]
    for i, f in enumerate(fs):
        for k in ("u", "v", "direction", "magnitude", "duration"):
            if k not in f:
                raise ValueError(f"push {i + 1}: missing {k!r}")
        f["u"], f["v"] = (float(np.clip(f["u"], *UV_RANGE)), float(np.clip(f["v"], *UV_RANGE)))
    if np.hypot(fs[0]["u"] - fs[1]["u"], fs[0]["v"] - fs[1]["v"]) < MIN_SEP:
        raise ValueError(f"patches closer than {MIN_SEP} in (u, v)")
    notes, out = [], []
    for i, (a, b) in enumerate(((fs[0], fs[1]), (fs[1], fs[0]))):
        d = np.asarray(a["direction"], np.float64)
        d = d / max(np.linalg.norm(d), 1e-9)
        toward = np.arctan2(b["v"] - a["v"], b["u"] - a["u"])
        head = np.arctan2(d[1], d[0]) if np.hypot(d[0], d[1]) > 1e-6 else toward
        dh = (head - toward + np.pi) % (2 * np.pi) - np.pi
        lim = np.deg2rad(HEADING_JITTER_DEG)
        if abs(dh) > lim:
            notes.append(f"push {i + 1}: heading turned toward push {2 - i} (paper is creased by opposing pushes)")
            dh = float(np.clip(dh, -lim, lim))
        tilt = np.arcsin(np.clip(d[2], -1, 1))
        tmax = np.deg2rad(TILT_MAX_DEG)
        if abs(tilt) > tmax:
            notes.append(f"push {i + 1}: tilt limited to {TILT_MAX_DEG:.0f} deg")
            tilt = float(np.clip(tilt, -tmax, tmax))
        th = toward + dh
        mag = float(np.clip(a["magnitude"], *MAG_RANGE))
        dur = float(np.clip(a["duration"], *DUR_RANGE))
        if abs(mag - a["magnitude"]) > 1e-9:
            notes.append(f"push {i + 1}: magnitude set to {mag:.2f} N (range {MAG_RANGE[0]}-{MAG_RANGE[1]} N)")
        out.append({"u": a["u"], "v": a["v"],
                    "direction": [float(np.cos(tilt) * np.cos(th)), float(np.cos(tilt) * np.sin(th)), float(np.sin(tilt))],
                    "magnitude": mag, "duration": dur,
                    "sigma_cells": float(ref_fp[0]["sigma_cells"]), "radius_cells": float(ref_fp[0]["radius_cells"])})
    return out, notes


def _sh(cmd: list[str], log: str, timeout: float = 600) -> None:
    with open(log, "a") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        r = subprocess.run(cmd, cwd=INVERSE, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode != 0:
        tail = open(log).read()[-800:]
        raise RuntimeError(f"{os.path.basename(cmd[1])} failed (rc {r.returncode}): {tail}")


def masks(P: str, log: str) -> tuple[str, list[str]]:
    """Hosted SAM 3.1 for every view of the set; renderer mask for views it misses."""
    from PIL import Image
    src, missed = "SAM 3.1 (Meta Model API)", []
    if os.environ.get("META_API_KEY"):
        try:
            from sam_api import FALLBACK, write_masks
            write_masks(P, ["fabric"] + [p for p in FALLBACK if p != "fabric"])
        except Exception as e:  # API down: every view falls back below
            with open(log, "a") as fh:
                fh.write(f"SAM 3.1 API failed: {type(e).__name__}: {e}\n")
    os.makedirs(os.path.join(P, "sam3"), exist_ok=True)
    for png in sorted(glob.glob(os.path.join(P, "*_v[0-9]*.png"))):
        stem = os.path.basename(png)[:-4]
        mp = os.path.join(P, "sam3", f"{stem}__fabric.png")
        if not os.path.exists(mp):
            gt = np.load(os.path.join(P, f"{stem}_gt.npz"))
            Image.fromarray(gt["mask"].astype(np.uint8) * 255).save(mp)
            missed.append(stem)
    if missed:
        n = len(glob.glob(os.path.join(P, "*_v[0-9]*.png")))
        src = "renderer mask (SAM 3.1 unavailable)" if len(missed) == n else f"SAM 3.1, renderer mask on {len(missed)} of {n} views"
    return src, missed


def chain(P: str, h5: str, seed: int, log: str) -> dict:
    """Steps 1-5 on one sample h5; returns paths and timings."""
    t, T = time.time(), {}
    _sh([SIM_PY, "photo.py", "--h5", h5, "--views", "1", "--stereo", "20", "--texture", "coded", "--gt",
         "--seed", str(seed), "--out", P], log)
    T["render_s"] = time.time() - t; t = time.time()
    src, missed = masks(P, log)
    T["sam_s"] = time.time() - t; t = time.time()
    _sh([SAM_PY, "lift_stereo.py", P, "--stereo-only", "--sam"], log)
    T["vggt_s"] = time.time() - t; t = time.time()
    _sh([SIM_PY, "decode_print.py", "--photos", P, "--base-only"], log)
    T["decode_s"] = time.time() - t; t = time.time()
    G = os.path.join(P, "grids")
    _sh([SIM_PY, "grid_from_views.py", "--photos", P, "--corr", "decoded", "--scale-fix", SCALE_FIX, "--out", G], log)
    T["grid_s"] = time.time() - t
    man = json.load(open(os.path.join(P, "manifest.json")))
    base = [v for v in man["views"] if v["partner"] > v["view"]][0]
    stem = os.path.splitext(base["png"])[0]
    npz = os.path.join(G, f"{stem}_grids.npz")
    if not os.path.exists(npz):
        raise RuntimeError("the chain produced no grid for this view (decode found too few print cells)")
    recs = {r["stem"]: r for r in json.load(open(os.path.join(G, "grids.json")))["records"]}
    return {"npz": npz, "photo": os.path.join(P, base["png"]), "mask": os.path.join(P, "sam3", f"{stem}__fabric.png"),
            "record": recs.get(stem), "mask_source": src, "mask_fallback": missed, "timings": T}


def run_live(model: Model, sim, forces: list[dict], ref_cfg: dict, ref_fp: list[dict], workdir: str,
             seed: int, on_stage=lambda s, p: None) -> dict:
    """Simulate the user's pushes on rest-state paper (dataset physics), then the camera chain."""
    fs, notes = scrunch_forces(forces, ref_fp)
    os.makedirs(workdir, exist_ok=True)
    log = os.path.join(workdir, "chain.log")
    h5 = os.path.join(workdir, "sample_live.h5")
    settle = float(ref_cfg["t_capture"]) - max(float(f["duration"]) for f in ref_fp[:2])
    on_stage("simulating", 0.05)
    t = time.time()
    sim.run(fs, max(f["duration"] for f in fs) + settle, seed=seed, keep=h5)
    sim_s = time.time() - t
    on_stage("camera", 0.6)
    P = os.path.join(workdir, "photos")
    c = chain(P, h5, seed, log)
    on_stage("predicting", 0.95)
    out = package(model, c["npz"], c["photo"], c["mask"], fs, c["record"], c["mask_source"])
    out["timings"] = {"sim_s": sim_s, **c["timings"]}
    out["adjustments"] = notes
    out["mask_fallback"] = c["mask_fallback"]
    out["live"] = True
    return out
