"""Demo backend service: simulate -> inverse model -> arrows + metrics.

Runs unchanged under plain ``uvicorn`` (local, Baseten later) and behind
``modal_app.py`` on Modal. All Modal-specific code lives in ``modal_app.py``.

Endpoints:
    POST /simulate        {"forces": [{u,v,direction,magnitude,duration} x2]}
                          -> {"job_id"}. 422 on out-of-range forces.
    GET  /jobs/{job_id}   {"status", "progress", "result" | "error"}.
                          Polling every 500 ms is fine.
    GET  /healthz          {"status", "warm"}.
    POST /camera          {"forces": [...] x2} -> {"job_id"}: the live camera path
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

CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(_HERE, "checkpoints", "M16a"))
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
_cam: dict = {}  # model, sim, ref: warmed on the first /camera job


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
        })
    except Exception as e:  # never leave a job hanging; cache fallback lands in step 6
        _set(job_id, status="failed", error=f"{type(e).__name__}: {e}")


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

    @app.post("/simulate")
    def simulate(body: dict):
        if "forces" not in body:
            raise HTTPException(422, "missing 'forces'")
        try:
            forces = validate_forces(body["forces"])
        except ValueError as e:
            raise HTTPException(422, str(e))
        job_id = uuid.uuid4().hex[:12]
        with _lock:
            _jobs[job_id] = {"status": "queued", "progress": 0.0}
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
        _executor.submit(_run_camera, job_id, body["forces"])
        return {"job_id": job_id}

    @app.get("/jobs/{job_id}")
    def job(job_id: str):
        with _lock:
            j = _jobs.get(job_id)
            if j is None:
                raise HTTPException(404, "unknown job_id")
            return {"status": j["status"], "progress": j["progress"],
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


def _run_camera(job_id: str, forces: list[dict]):
    import random
    import shutil
    t0 = time.time()
    try:
        _set(job_id, status="simulating", progress=0.02)
        _warm_camera()
        work = os.path.join(_cam["work"], job_id)
        stage = lambda s, p: _set(job_id, status=s, progress=p)
        fut = _sim_pool.submit(cam.run_live, _cam["model"], _cam["sim"], forces, _cam["ref"]["sim_config"],
                               _cam["ref"]["force_footprint"], work, random.randrange(10 ** 6), stage)
        out = fut.result(timeout=CAMERA_TIMEOUT_S)
        out["timings"]["total_s"] = time.time() - t0
        _set(job_id, status="done", progress=1.0, result=out)
        shutil.rmtree(work, ignore_errors=True)
    except Exception as e:
        _set(job_id, status="failed", error=f"camera path: {type(e).__name__}: {str(e)[-400:]}")


def _warm_and_run(job_id: str, forces: list[dict]):
    _warm()
    _run(job_id, forces)


app = create_app()
