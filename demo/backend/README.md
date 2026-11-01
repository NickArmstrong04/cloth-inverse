# Demo backend

FastAPI service: `POST /simulate` -> `GET /jobs/{id}`. Simulates the two
forces at 256 (step-2 decision), runs M16-a on the final state, returns
30 frames (64x64 float16 base64) plus true / predicted / reversal forces,
metrics, and timings.

## Local run (WSL, warm GPU)

The service imports the simulator in-process, so it runs in the sim env:

```bash
cd /mnt/c/Users/nicka/Desktop/projects/meta/demo/backend
/home/narmstrong/cloth/.venv/bin/python -m uvicorn app:app --port 8000
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

## Modal deploy (one A10G, scales to zero when idle)

```bash
modal setup                                          # first time only
modal volume create cloth-demo-checkpoints
modal volume put cloth-demo-checkpoints demo/backend/checkpoints/M16a/model.pt M16a/model.pt
modal volume put cloth-demo-checkpoints demo/backend/checkpoints/M16a/metrics.json M16a/metrics.json
cd <repo> && modal deploy demo/backend/modal_app.py  # prints the https URL
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
