"""Demo backend service: simulate -> inverse model -> arrows + metrics.

Runs unchanged under plain ``uvicorn`` (local, Baseten later) and behind
``modal_app.py`` on Modal. All Modal-specific code lives in ``modal_app.py``.

Endpoints:
    POST /simulate        {"forces": [{u,v,direction,magnitude,duration} x2], "model"?: name}
                          -> {"job_id"}. 422 on out-of-range forces or an unavailable model.
                          Models (GET /models): M16a (default; cloth at release, positions +
                          velocities, M16-a) and the rest-state paper models M20a3 (clean
                          grids, positions only) and M21c (trained on camera-chain grids); the
                          paper models run the paper physics (camera.py's Simulator, ~100 s)
                          with the pushes clamped to their training range (``adjustments``).
    GET  /jobs/{job_id}   {"status", "progress", "result" | "error", "stages"?}.
                          Polling every 500 ms is fine. Camera jobs carry ``stages``: one entry
                          per finished step (simulating, render, sam, vggt, decode, grid,
                          predict, narrate) with that step's images (base64 JPEG) and numbers;
                          ``status`` names the step now running.
    GET  /healthz          {"status", "warm"}.
    POST /camera          {"forces": [...] x2, "seed"?: int} -> {"job_id"}: the live camera path
                          (camera.py): the pushes on rest-state paper (dataset physics,
                          ~2 min), then photo -> SAM 3.1 -> VGGT -> decoded grid -> M21-c.
                          Same /jobs polling; result is a camera.package payload.

Result payload: 30 frames of vertex positions downsampled to 64x64 as a
float16 base64 string ``[30, 64, 64, 3]``; the full-resolution final state
stays server-side for the model. Plus ``true_forces``,
``predicted_forces``, ``reversal_forces`` (predicted, direction negated),
``metrics`` (``pos_cm`` / ``dir_deg`` / ``impulse_rel`` per force and mean,
with test-set reference values), ``timings`` (``sim_s``, ``infer_s``),
``grid`` and ``cached``.

Env: CKPT_DIR (default ``demo/backend/checkpoints/M16a``), GRID_NX (=256,
the step-2 decision), SNAPSHOTS (=30), CORS_ORIGINS (comma-separated,
default ``http://localhost:5173``).
"""

from __future__ import annotations

import os
import threading

import numpy as np
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import inference as inf
from simulate import SimService, downsample_frames, encode_frames, validate_forces
from cache import load_cache, nearest
import camera as cam

_HERE = os.path.dirname(os.path.abspath(__file__))

CKPT_ROOT = os.environ.get("CKPT_ROOT", os.path.join(_HERE, "checkpoints"))
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(CKPT_ROOT, "M16a"))

# The demo's model menu. No refinement (M22, several simulated guesses) by decision 2026-11-03.
# References: RESULTS.md M16 table (M16-a), M21 table "clean test" column (M20-a3, M21-c); the
# paper models were not scored on impulse there (None -> shown as a dash).
MODELS = {
    "M16a": {"label": "Positions + final velocities", "physics": "cloth",
             "blurb": "M16-a: the full mesh with its velocities at the instant the pushes end (cloth, zero gravity).",
             "reference": {"pos_cm": 1.0, "dir_deg": 4.3, "impulse_rel": 0.156}},
    "M20a3": {"label": "Positions only", "physics": "paper",
              "blurb": "M20-a3: a creased paper sheet settled to rest, exact geometry, no velocities (~2 min per push).",
              "reference": {"pos_cm": 2.8, "dir_deg": 6.6, "impulse_rel": None}},
    "M21c": {"label": "Pseudo camera", "physics": "paper",
             "blurb": "M21-c: same paper setting, trained on grids rebuilt from photos, the camera path's model (~2 min per push).",
             "reference": {"pos_cm": 3.4, "dir_deg": 7.3, "impulse_rel": None}},
}


def model_dir(name: str) -> str:
    return CKPT_DIR if name == "M16a" else os.path.join(CKPT_ROOT, name)


def model_available(name: str) -> bool:
    return name in MODELS and os.path.exists(os.path.join(model_dir(name), "model.pt"))
GRID_NX = int(os.environ.get("GRID_NX", "256"))
SNAPSHOTS = int(os.environ.get("SNAPSHOTS", "30"))
CORS_ORIGINS = [o.strip() for o in
                os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")
                if o.strip()]

# M16-a test-set means (RESULTS.md M16 table).
REFERENCE = {"pos_cm": 1.0, "dir_deg": 4.3, "impulse_rel": 0.156}

_executor = ThreadPoolExecutor(max_workers=1)  # one GPU: sims run in order
_sim_pool = ThreadPoolExecutor(max_workers=1)  # the sim itself, so it can time out
JOB_TIMEOUT_S = float(os.environ.get("JOB_TIMEOUT_S", "90"))
_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_svc: SimService | None = None
_ckpt = None
_entries: list = []
CAMERA_TIMEOUT_S = float(os.environ.get("CAMERA_TIMEOUT_S", "600"))
NARRATE_TIMEOUT_S = float(os.environ.get("NARRATE_TIMEOUT_S", "45"))
_cam: dict = {}  # model, sim, ref: warmed on the first /camera job
_paper_models: dict = {}  # name -> camera.Model for the paper models of the mesh path


def _warm():
    global _svc, _ckpt, _entries
    with _lock:
        if _svc is None:
            _svc = SimService()
        if _ckpt is None:
            _ckpt, _ = inf.load_checkpoint(CKPT_DIR)
        if not _entries:
            _entries = load_cache()


def _set(job_id: str, **kw):
    with _lock:
        _jobs[job_id].update(kw)


def _fallback(job_id: str, forces: list[dict], reason: str):
    """Nearest precomputed result, re-scored against the user's forces."""
    with _lock:
        entries = list(_entries)
    hit = nearest(entries, forces)
    if hit is None:
        _set(job_id, status="failed", error=reason + " (cache empty)")
        return
    t0 = time.time()
    pred_raw = _ckpt["encode_forces"](hit["predicted_forces"])
    per_force, mean = inf.score_forces(_ckpt, pred_raw, forces)
    true_echo = [dict(f, impulse=f["magnitude"] * f["duration"]) for f in forces]
    _set(job_id, status="done", progress=1.0, result={
        "frames": hit["frames"],
        "frame_shape": [30, 64, 64, 3],
        "true_forces": true_echo,
        "predicted_forces": hit["predicted_forces"],
        "reversal_forces": hit["reversal_forces"],
        "metrics": {"per_force": [
                        {k: f[k] for k in ("pos_cm", "dir_deg", "impulse_rel")}
                        for f in per_force],
                    "mean": {k: mean[k] for k in ("pos_cm", "dir_deg", "impulse_rel")},
                    "reference": REFERENCE},
        "timings": {"sim_s": hit["timings"]["sim_s"], "infer_s": time.time() - t0},
        "grid": hit["grid"],
        "cached": True,
    })


def _run(job_id: str, forces: list[dict]):
    try:
        _set(job_id, status="simulating", progress=0.05)
        try:
            sim = _sim_pool.submit(_svc.run, forces, GRID_NX, SNAPSHOTS).result(timeout=JOB_TIMEOUT_S)
        except Exception as e:
            _set(job_id, progress=0.8, status="predicting")
            _fallback(job_id, forces, f"sim unavailable ({type(e).__name__}: {e})")
            return
        _set(job_id, progress=0.8, status="predicting")
        t0 = time.time()
        pred, predicted = inf.predict_forces(_ckpt, sim["final_pos"], sim["final_vel"])
        infer_s = time.time() - t0
        per_force, mean = inf.score_forces(_ckpt, pred, forces)
        reversal = [dict(f, direction=[-x for x in f["direction"]]) for f in predicted]
        frames64 = downsample_frames(sim["frames"], 64)
        true_echo = [dict(f, impulse=f["magnitude"] * f["duration"]) for f in forces]
        _set(job_id, status="done", progress=1.0, result={
            "frames": encode_frames(frames64),
            "frame_shape": list(frames64.shape),
            "true_forces": true_echo,
            "predicted_forces": predicted,
            "reversal_forces": reversal,
            "metrics": {"per_force": [
                            {k: f[k] for k in ("pos_cm", "dir_deg", "impulse_rel")}
                            for f in per_force],
                        "mean": {k: mean[k] for k in ("pos_cm", "dir_deg", "impulse_rel")},
                        "reference": REFERENCE},
            "timings": {"sim_s": sim["wall_s"], "infer_s": infer_s},
            "grid": GRID_NX,
            "cached": False,
            "model": "M16a",
            "physics": "cloth",
            "adjustments": [],
        })
    except Exception as e:  # never leave a job hanging; cache fallback lands in step 6
        _set(job_id, status="failed", error=f"{type(e).__name__}: {e}")


def _paper_model(name: str):
    with _lock:
        m = _paper_models.get(name)
    if m is None:
        m = _cam["model"] if name == "M21c" else cam.Model(model_dir(name))
        with _lock:
            _paper_models[name] = m
    return m


def _run_paper(job_id: str, forces: list[dict], name: str):
    """Mesh path on the rest-state paper physics for M20a3 / M21c: the pushes clamped to the
    training range, the paper simulated to rest (30 snapshots for playback), the model on the
    exact final grid (positions only)."""
    import random
    import h5py
    import inference as inf2
    try:
        _set(job_id, status="simulating", progress=0.05)
        _warm_camera()
        model = _paper_model(name)
        ref, sim = _cam["ref"], _cam["sim"]
        fs, notes = cam.scrunch_forces(forces, ref["force_footprint"])
        settle = float(ref["sim_config"]["t_capture"]) - max(float(f["duration"]) for f in ref["force_footprint"][:2])
        t_cap = max(f["duration"] for f in fs) + settle
        h5 = os.path.join(_cam["work"], f"mesh_{job_id}.h5")
        sim.sim.n_snapshots = SNAPSHOTS
        t0 = time.time()
        fut = _sim_pool.submit(sim.run, fs, t_cap, random.randrange(10 ** 6), h5)
        grid, _vel = fut.result(timeout=CAMERA_TIMEOUT_S)
        sim_s = time.time() - t0
        with h5py.File(h5, "r") as h:
            frames = np.asarray(h["intermediate_grids"][()], np.float32)
        os.remove(h5)
        if len(frames) != SNAPSHOTS:  # resample by index so the player's 30-frame clock holds
            idx = np.linspace(0, len(frames) - 1, SNAPSHOTS).round().astype(int)
            frames = frames[idx]
        _set(job_id, progress=0.8, status="predicting")
        t0 = time.time()
        X, G = inf2.grid_input(grid, None)
        pred = model.predict(X, G)
        infer_s = time.time() - t0
        predicted = cam.forces_out(pred)
        per_force, mean = cam.score(pred, fs)
        reversal = [dict(f, direction=[-x for x in f["direction"]]) for f in predicted]
        frames64 = downsample_frames(frames, 64)
        _set(job_id, status="done", progress=1.0, result={
            "frames": encode_frames(frames64),
            "frame_shape": list(frames64.shape),
            "true_forces": [dict(f, impulse=f["magnitude"] * f["duration"]) for f in fs],
            "predicted_forces": predicted,
            "reversal_forces": reversal,
            "metrics": {"per_force": per_force, "mean": mean, "reference": MODELS[name]["reference"]},
            "timings": {"sim_s": sim_s, "infer_s": infer_s},
            "grid": int(grid.shape[0]),
            "cached": False,
            "model": name,
            "physics": "paper",
            "adjustments": notes,
        })
    except Exception as e:
        _set(job_id, status="failed", error=f"{name}: {type(e).__name__}: {str(e)[-300:]}")


def create_app() -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _executor.submit(_warm)
        yield

    app = FastAPI(title="cloth-demo", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/healthz")
    def healthz():
        with _lock:
            warm = _svc is not None and _ckpt is not None
        return {"status": "ok", "warm": warm}

    @app.get("/models")
    def models():
        return {"default": "M16a",
                "models": [{"name": k, **v, "available": model_available(k)} for k, v in MODELS.items()]}

    @app.post("/simulate")
    def simulate(body: dict):
        if "forces" not in body:
            raise HTTPException(422, "missing 'forces'")
        name = body.get("model") or "M16a"
        if name not in MODELS:
            raise HTTPException(422, f"unknown model {name!r}; one of {list(MODELS)}")
        if not model_available(name):
            raise HTTPException(422, f"model {name} has no checkpoint on this server")
        try:
            forces = validate_forces(body["forces"])
            if MODELS[name]["physics"] == "paper":
                cam.scrunch_forces(forces, [{"sigma_cells": 6.0, "radius_cells": 18.0}])
        except ValueError as e:
            raise HTTPException(422, str(e))
        job_id = uuid.uuid4().hex[:12]
        with _lock:
            _jobs[job_id] = {"status": "queued", "progress": 0.0}
        if MODELS[name]["physics"] == "paper":
            _executor.submit(_run_paper, job_id, forces, name)
        else:
            _executor.submit(_warm_and_run, job_id, forces)
        return {"job_id": job_id}

    @app.post("/camera")
    def camera(body: dict):
        if "forces" not in body:
            raise HTTPException(422, "missing 'forces'")
        try:
            cam.scrunch_forces(body["forces"], [{"sigma_cells": 6.0, "radius_cells": 18.0}])
        except (ValueError, TypeError, KeyError) as e:
            raise HTTPException(422, str(e))
        job_id = uuid.uuid4().hex[:12]
        with _lock:
            _jobs[job_id] = {"status": "queued", "progress": 0.0}
        seed = body.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise HTTPException(422, "'seed' must be an integer")
        _executor.submit(_run_camera, job_id, body["forces"], seed)
        return {"job_id": job_id}

    @app.get("/jobs/{job_id}")
    def job(job_id: str):
        with _lock:
            j = _jobs.get(job_id)
            if j is None:
                raise HTTPException(404, "unknown job_id")
            return {"status": j["status"], "progress": j["progress"],
                    **({"stages": j["stages"]} if "stages" in j else {}),
                    **({"result": j["result"]} if "result" in j else {}),
                    **({"error": j["error"]} if "error" in j else {})}

    return app


def _warm_camera():
    with _lock:
        ready = bool(_cam)
    if ready:
        return
    import json
    import tempfile
    from eval_resim import Simulator
    ref = json.load(open(os.path.join(_HERE, "rest_reference.json")))
    work = tempfile.mkdtemp(prefix="camera_")
    c = {"ref": ref, "work": work, "model": cam.Model(), "sim": Simulator(ref["sim_config"], workdir=work)}
    with _lock:
        _cam.update(c)


def _run_camera(job_id: str, forces: list[dict], seed: int | None = None):
    """``seed`` (optional, 2026-11-03) fixes the paper re-sim and the camera views so a recorded
    run reproduces (sweep_live.py picks it); random when absent."""
    import random
    import shutil
    t0 = time.time()
    try:
        _set(job_id, status="simulating", progress=0.02)
        _warm_camera()
        work = os.path.join(_cam["work"], job_id)
        def stage(s, p, **artifacts):
            with _lock:
                st = dict(_jobs[job_id].get("stages", {}))
                st.update(artifacts)
                _jobs[job_id].update(status=s, progress=p, stages=st)
        fut = _sim_pool.submit(cam.run_live, _cam["model"], _cam["sim"], forces, _cam["ref"]["sim_config"],
                               _cam["ref"]["force_footprint"], work,
                               seed if seed is not None else random.randrange(10 ** 6), stage)
        out = fut.result(timeout=CAMERA_TIMEOUT_S)
        out["timings"]["total_s"] = time.time() - t0
        shutil.rmtree(work, ignore_errors=True)
        # The prediction shows while Muse Spark writes the verdict (narrate.py; skipped without a key).
        stage("narrate", 0.95, predict={"elapsed_s": out["timings"]["predict_s"]})
        _set(job_id, result=out)
        t1 = time.time()
        try:
            import narrate
            n = narrate.verdict(out, timeout=NARRATE_TIMEOUT_S)
        except Exception as e:
            n = {"text": None, "ok": False, "error": f"{type(e).__name__}: {str(e)[-200:]}"}
        n["elapsed_s"] = time.time() - t1
        out["narration"] = n
        stage("done", 1.0, narrate=n)
        _set(job_id, result=out)
    except Exception as e:
        _set(job_id, status="failed", error=f"camera path: {type(e).__name__}: {str(e)[-400:]}")


def _warm_and_run(job_id: str, forces: list[dict]):
    _warm()
    _run(job_id, forces)


app = create_app()
