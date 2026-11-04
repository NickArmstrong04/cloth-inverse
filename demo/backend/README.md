# Demo backend

FastAPI service: `POST /simulate` -> `GET /jobs/{id}`. Simulates the two
forces at 256 (step-2 decision), runs M16-a on the final state, returns
30 frames (64x64 float16 base64) plus true / predicted / reversal forces,
metrics, and timings.

## Local run (WSL, warm GPU)

The service imports the simulator in-process, so it runs in the sim env:

```bash
cd <repo>/demo/backend
<sim-venv>/bin/python -m uvicorn app:app --port 8000   # the venv from cloth_data_generation/transfer/README.md
```

`CKPT_DIR` defaults to `demo/backend/checkpoints/M16a`. `CORS_ORIGINS`
defaults to `http://localhost:5173` (comma-separated to add Vercel).

End to end from `curl` (first call warms Warp + the checkpoint, ~30 s):

```bash
curl -s -X POST localhost:8000/simulate -H 'Content-Type: application/json' \
  -d '{"forces": [{"u": 0.3, "v": 0.6, "direction": [0.2, -0.5, 0.84], "magnitude": 0.8, "duration": 0.5},
                  {"u": 0.7, "v": 0.4, "direction": [0.0, 0.3, -0.95], "magnitude": 0.5, "duration": 0.4}]}'
# -> {"job_id": "..."}
curl -s localhost:8000/jobs/<job_id>   # poll every 500 ms to "done"
```

`POST /camera` (same body) runs the camera path: the pushes re-simulated on paper, two photos,
SAM 3.1 (hosted, `META_API_KEY`), VGGT (`SAM_PY` = the sam3env python), print decode, grid,
M21-c, then Muse Spark (`narrate.py`). Its `GET /jobs/{id}` carries `stages` (one entry per
finished step with base64 JPEGs and numbers) while `status` names the running step. Smoke test
outside the service: `SIM_PY=... SAM_PY=... python test_live.py [--no-narrate]`.

## Modal deploy (one A10G; warm 08:00-23:00 ET on the judging days, scale to zero otherwise)

```bash
modal setup                                          # first time only
modal volume create cloth-demo-checkpoints
modal volume put cloth-demo-checkpoints demo/backend/checkpoints/M16a/model.pt M16a/model.pt
modal volume put cloth-demo-checkpoints demo/backend/checkpoints/M16a/metrics.json M16a/metrics.json
cd <repo> && modal deploy demo/backend/modal_app.py  # prints the https URL
```

Jobs are kept in the container's memory (`_jobs`), so the function is pinned to
`max_containers=1` with `@modal.concurrent(max_inputs=32)` (2026-11-03). Before that a
busy container made Modal start a second one, which knew no jobs and answered the
frontend's polling with `GET /jobs: 404`. Redeploy to apply.

Warm pool (2026-11-03). The decorator has `min_containers=1`, so a deploy is warm at once.
The scheduled `set_warm_pool` (every hour at xx:05 ET) then sets 1 inside `WARM_DAYS` x
`WARM_HOURS` (2026-11-04 to 11-09, 08:00-22:59 ET) and 0 outside, so the A10G at about
$1.10/h costs about $17/day instead of $26 (Modal pricing page, 2026-11-03; the Starter plan
includes $30/month of credit). Off-hours the first push pays the 1-2 min cold start and
`scaledown_window=1200` keeps the container 20 min after the last request. Check with
`modal app list` (one running container during warm hours) and `modal app logs cloth-demo`
(one `min_containers=...` line per hour). To force the pool by hand without a deploy:

```bash
modal run demo/backend/modal_app.py::set_warm_pool        # applies the schedule's value now
python -c "import modal; modal.Function.from_name('cloth-demo','web').autoscaler.update(min_containers=1)"
```

Then `CORS_ORIGINS=https://<vercel-app>.vercel.app,http://localhost:5173`
via `modal deploy` env (or `modal secret`), redeploy to apply.

Moving to Baseten later: point its entrypoint at `app:create_app`
(or `app:app`); no code change needed.

## Cache and fallback (step 6)

`demo/backend/cache/` holds 15 precomputed results (built by
`make_cache.py --ckpt ... --n 15 --seed0 200000`, one warm GPU, ~3 min).
A job whose sim fails or exceeds `JOB_TIMEOUT_S` (default 90 s) returns
the nearest cached result — smallest summed (u, v) distance, better
pairing wins — with `cached: true`. The cached prediction is re-scored
against the user's own forces, so the metrics stay truthful; the UI
shows a notice. The demo never ends on an empty screen.
