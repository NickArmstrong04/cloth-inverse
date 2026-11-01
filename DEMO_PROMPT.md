# Build the cloth-inverse live demo website

You are implementing a hackathon demo website for the repository
`https://github.com/NickArmstrong04/cloth-inverse.git`. Work on a new branch `demo` cut from
`photo-render`. All new code goes under a new top-level directory `demo/` (`demo/backend/`,
`demo/frontend/`). Do not modify the simulator (`cloth_data_generation/transfer/`) or the model
code (`cloth_data_generation/inverse/`) except through thin wrappers that import them.

Read these first, in this order:

1. `README.md` (repo overview) and `ROADMAP.md`
2. `cloth_results/RESULTS.md`, section "M16: six improvements to the two best models" (the model
   we ship and its numbers)
3. `cloth_results/DATASETS.md`, row `ds256_const` (the physics settings the model was trained on)
4. `cloth_data_generation/transfer/README.md` and `cloth_data_generation/transfer/cluster/README.md`
5. `cloth_data_generation/transfer/generate_sample.py` (the simulator CLI),
   `cloth_data_generation/transfer/generate_batch.py` (how forces were drawn; `sample_forces`,
   `MAG_RANGE`, `res_scale`), `cloth_data_generation/transfer/cloth_sim/forces.py` (`ForceParams`),
   `cloth_data_generation/transfer/cloth_sim/sheet.py` (sheet is 0.5 m x 0.5 m, centred at the
   origin, cell = 0.5/(nx-1) m)
6. `cloth_data_generation/inverse/train.py`, `data.py`, `models.py`, `geometry.py`,
   `eval_grid.py`, `eval_resim.py` (how a sample becomes model input and how predictions are scored)

## Context you will not find in the repo

- **Owner:** Nick (GitHub `NickArmstrong04`). Meta hackathon; judging criteria are below.
  Working software beats polish.
- **Where things run.** Nick's Windows workstation holds the repo at
  `C:\Users\nicka\Desktop\projects\meta`. The Python environment that runs the simulator and the
  models is in **WSL** (user `narmstrong`) at `/home/narmstrong/cloth/`: a copy of
  `cloth_data_generation/transfer/` with its own `.venv` (Warp, Newton, PyTorch, CUDA on an RTX
  4070 Ti SUPER) and the inverse code beside it at `/home/narmstrong/cloth/inverse/`. It is a
  copy, not a git checkout, so sync changed files from the Windows repo before running. A second
  env at `/home/narmstrong/sam3env` holds SAM 3 and VGGT (numpy 2.5). Run Python through WSL, never
  Windows Python. WSL `/tmp` is wiped between sessions. Long inline shell commands with nested
  quotes or heredocs fail in this setup: write scripts to files and run them by path.
- **Cluster (Yale Bouchet).** Repo clone: `/home/naa79/scratch_pi_sz622/naa79/cloth-inverse`.
  Datasets: `/home/naa79/scratch_pi_sz622/naa79/cloth_data/<name>`. The packaged split the model
  was trained on: `/home/naa79/scratch_pi_sz622/naa79/cloth_data/ds256_const_splits`
  (`train/`, `val/`, `test/` of HDF5 samples). Model runs:
  `/home/naa79/scratch_pi_sz622/naa79/cloth-inverse/cloth_data_generation/inverse/runs/<name>/`
  with `model.pt` and `metrics.json`. **You have no SSH access to the cluster** (Duo). Give Nick
  exact commands and paths; he runs them and pastes the output. Local tests use 1 to 10 samples;
  anything larger goes to the cluster.
- **Physics model in use (since 2026-10-29, dataset `ds256_const`):** constant force applied every
  substep for the whole duration, capture at release (no settle), 5 snapshots with velocities,
  free flight in zero gravity, no table. Forces drawn with `u, v` uniform in [0.05, 0.95],
  direction **uniform on the full sphere (a 3D unit vector, not in-plane)**, magnitude
  log-uniform 0.15 to 1.25 N, duration uniform 0.3 to 0.7 s per force, material separation of the
  two patches at least 0.2, patch `sigma_cells = 3.0 * res_scale`, `radius_cells = 9.0 * res_scale`
  with `res_scale = nx / 128` (so the patch footprint in metres is the same at every resolution).
  The VBD solver delivers only about 8% of the nominal impulse (a per-substep velocity clamp);
  labels are nominal F*t and the model is trained and scored on nominal labels, so stay consistent.
- **Timings:** a 256x256 attempt is about 10 to 20 s on a GPU once the process is warm; process
  start-up and Warp module load dominate in a cold process. Keep the sim process warm.
- **Logs:** `cloth_results/JOBS.md` tracks every cluster job; add a row if you ask Nick to run
  anything on the cluster. Do not edit `cloth_results/RESULTS.md`; demo measurements go in
  `demo/README.md`.
- **Conventions:** Meta tools first for perception steps (SAM 3, VGGT, DINOv3). Short factual
  commit messages saying what was verified and how. Ask before destructive actions.

## What the demo does

A user pushes a square cloth sheet with two forces and our model tells them which forces did it.

1. The user sees the 0.5 m sheet floating in a zero-gravity 3D scene with a **free camera**:
   orbit, pan and zoom with the mouse at every stage, no fixed view, so they can look at the
   sheet from any angle both when choosing where to push and later when inspecting the
   crumpling. They click two points on the sheet (raycast against the mesh, either face) to place
   the two force patches; the click gives the material coordinates `(u, v)`. For each patch they
   set a **3D direction** (a draggable arrow head on a small sphere gizmo, or azimuth and
   elevation sliders; the dataset directions cover the whole sphere so the UI must allow
   out-of-plane pushes), a magnitude slider (0.15 to 1.25 N) and a duration slider (0.3 to 0.7
   s). Enforce the training ranges and a minimum `(u, v)` separation of 0.2 between the patches.
2. They press "Push". The backend runs the real simulator with those forces and returns 30
   frames of the sheet deforming. The front end plays the frames in the same 3D view, with scrub
   and replay, camera still free.
3. The backend then runs the trained inverse model on the final simulated state (full mesh
   positions and velocities, the "rung 1" input, no camera path) and returns the predicted force
   locations, directions and impulses.
4. The front end overlays, on the deformed sheet, the user's true force arrows and the predicted
   arrows, then the "reversal": the predicted vectors negated, drawn at the predicted locations,
   as the pushes that would undo the deformation. Arrows only, no re-simulation of the reversal.
5. A metrics panel shows, per force and overall: position error in cm, direction error in
   degrees, impulse relative error, with the model's test-set means as reference values (M16-a:
   1.0 cm, 4.3 deg, 0.156 relative). Also the wall time of sim and inference. Compute the metrics
   exactly as `cloth_data_generation/inverse/metrics.py` does (including the two-force assignment).

## Model to ship: the full-information model, velocities included

The demo feeds the model **everything the simulator knows at capture**: the full 256x256 grid of
vertex positions and the full grid of vertex velocities at the instant the forces end, plus the
intrinsic strain and curvature features the hybrid input derives from them. This is the
project's "rung 1" setting and its best-performing model. Do **not** use the velocity-free
models (`--no-vel`, JOBS.md matrix M19) or the paper rest-capture track (M20,
`ds256_paper_rest`); those exist for the single-photo deployment, which is a different product.
Velocities carry most of the impulse information at capture (M16-a 0.9 to 1.0 cm with them,
3.5 cm without), and the demo has them for free because it runs the simulator itself.

The model is **M16-a**: `cnn_reg` with aux heads and the hybrid input, trained on `ds256_const`
for 300 epochs with `--model cnn_reg --epochs 300 --no-rot --amp --tta 0 --aux-w 1 --hybrid`
(the exact `run.sbatch` line is in `cloth_results/JOBS.md`, row M16-a, Slurm id 31411391).
Test-set numbers: 1.0 cm / 4.3 deg / 0.156 relative impulse (RESULTS.md M16 table; a local
16-sample re-score gave 0.9 cm / 4.8 deg). M16-h ties it within noise but is larger, so M16-a.
If a newer run on `ds256_const` with velocities beats M16-a in `RESULTS.md` when you start, ask
Nick before switching; the velocity-free M19 runs do not qualify.

Checkpoint, on the cluster only (it is not on the workstation; the WSL `smoke_runs/m16/`
folders are local smoke trainings, not this model):

```
/home/naa79/scratch_pi_sz622/naa79/cloth-inverse/cloth_data_generation/inverse/runs/M16a/model.pt
/home/naa79/scratch_pi_sz622/naa79/cloth-inverse/cloth_data_generation/inverse/runs/M16a/metrics.json
```

Ask Nick to copy both to `demo/backend/checkpoints/M16a/` (git-ignored). Sixteen test samples
of `ds256_const` already sit on the workstation in WSL at
`/home/narmstrong/cloth/ds256_const_test16/` (`sample_slot0029_seed100029.h5` and 15 more); use
them for verification, and ask for more from
`/home/naa79/scratch_pi_sz622/naa79/cloth_data/ds256_const_splits/test/` only if needed.

`model.pt` is a dict with `state`, `args` (the full train.py namespace), `scaler` (mean, std for
the targets), `xstd`, `gstd`. Reproduce the input pipeline from `data.py` and `train.py` exactly
(position channels, **velocity channels**, the hybrid strain/curvature channels, standardisation
with the stored `xstd`/`gstd`, `--no-rot`, no TTA) in `demo/backend/inference.py`. The simulator
output must therefore keep the final-state velocities: the HDF5 written by `generate_sample.py`
stores them (`intermediate_vels` for snapshots and the final velocity grid; check `cloth_sim/io.py`
for the dataset names), and the in-process path must hand the same arrays to the model. Verify
the wrapper by running the 16 local test samples and matching the metrics in `RESULTS.md` for
M16-a within noise before building anything else. `eval_grid.py --run ... --h5
...` is the reference implementation to compare against.

## Simulation wrapper and the grid-size decision

`demo/backend/simulate.py` wraps `cloth_data_generation/transfer/generate_sample.py`. Force
input is `--force-json <file>` with a list of dicts matching `ForceParams`:

```json
[{"u": 0.3, "v": 0.6, "direction": [0.2, -0.5, 0.84], "magnitude": 0.8, "duration": 0.5,
  "sigma_cells": 3.0, "radius_cells": 9.0}, {...}]
```

The dataset used `--nx 256 --ny 256` and `generate_batch.py` added `--radius`, `--margin`, `--gap`
scaled by `1/res_scale` and `sigma_cells 6.0 / radius_cells 18.0` (res_scale 2). Read lines
255 to 275 of `generate_batch.py` for the exact flags and reproduce them in the wrapper for any
`nx`. For the demo run at **`--nx 128 --ny 128`** (res_scale 1: `sigma_cells 3.0`,
`radius_cells 9.0`, radius 2.0 mm, margin 1.0 mm, gap 0.5 mm) with `--snapshots 30`, `--settle 0`,
the default constant-force options (no `--ramp`, `--force-every-substep` is the default), zero
gravity (no `--table`), `--device cuda:0`. Set `--t-capture` to the longer of the two durations,
which is the default. Prefer calling the Python API in `generate_sample.py` (`configure`,
`SimSession`) from a long-lived process over spawning the CLI, so Warp loads once.

The model expects a 256 grid. Upsample the 128 position and velocity grids to 256 on the indexed
grid (bilinear) before building the features. This is a domain shift, so **measure it first**:
simulate 20 random force configurations (draw them with `sample_forces` from `generate_batch.py`
with the right `res_scale`) at 128 and at 256 with the same seeds, run the model on both, and
report both error tables in `demo/README.md`. If the 128 path is more than roughly 2x worse than
the 256 path on position or direction, switch the demo to 256 and accept the longer wait. State
the decision and the numbers.

## Backend

- Python 3.11, FastAPI, deployed on **Modal** with one GPU (A10G or L4 class) and a warm container
  (`min_containers=1` / `keep_warm`) so Warp and the checkpoint load once. Keep all Modal-specific
  code in `demo/backend/modal_app.py`; the service itself (`demo/backend/app.py`) must run
  unchanged under plain `uvicorn` so it can move to Baseten later.
- Endpoints. `POST /simulate` takes
  `{"forces": [{"u","v","direction","magnitude","duration"}, {...}]}` (validated against the
  ranges above) and returns `{"job_id"}`. `GET /jobs/{job_id}` returns
  `{"status": "queued|simulating|predicting|done|failed", "progress": 0..1, "result": ...}`.
  Polling every 500 ms is fine.
- Result payload: 30 frames of vertex positions downsampled to a 64x64 grid for rendering, as a
  float16 base64 string with shape `[30, 64, 64, 3]`; the full-resolution final state stays
  server-side for the model. Plus `true_forces`, `predicted_forces` (same schema, with impulse
  `magnitude * duration` and the model's own fields), `reversal_forces` (predicted with direction
  negated), `metrics` (`pos_cm`, `dir_deg`, `impulse_rel` per force and mean), `timings`
  (`sim_s`, `infer_s`), `grid` (128 or 256), `cached` (bool).
- Reliability for the live demo: keep 10 to 20 precomputed results in `demo/backend/cache/`. If a
  GPU job fails or exceeds 90 s, return the nearest cached result (nearest in `(u, v)` of both
  patches) with `cached: true`, and the UI shows a small notice. The demo must never end in an
  empty screen.
- CORS for the Vercel origin and `http://localhost:5173`. `demo/backend/README.md` with deploy and
  local-run commands (local: WSL, `/home/narmstrong/cloth/.venv/bin/python`).
  `demo/backend/requirements.txt` pinned to the versions in
  `cloth_data_generation/transfer/requirements.txt` where they overlap.

## Frontend

- React 18 + Vite + TypeScript, `@react-three/fiber` and `@react-three/drei`, deployed on
  **Vercel**. Backend URL from `VITE_API_URL`. Project at `demo/frontend/`.
- Single page, minimal. The 3D scene fills most of the screen; a right-hand panel holds the force
  controls (two cards, one per force, each with its accent colour), the Push button, playback
  controls (play, pause, scrub, loop), and the metrics. A collapsible "How it works" section
  states the pipeline in five lines (simulate, observe the mesh, inverse model, predicted pushes,
  reversal) and a one-line impact pitch.
- Camera: `OrbitControls` with pan enabled, damping on, a "reset view" button. The app never
  animates or locks the camera; the user keeps control while the deformation plays so they can
  fly around the crumpling sheet. Clicking the sheet places a patch only in the placement step,
  so orbiting during playback never adds forces.
- Sheet rendering: `PlaneGeometry` 64x64 segments updated per frame from the payload,
  double-sided material, shaded by local stretch or height so the folds read on video, readable
  from behind and edge-on. Arrows: true forces solid in the force's accent colour, predicted
  dashed or translucent, reversal in a third style, with a legend. Arrow length proportional to
  impulse.
- Light and dark theme. Mouse only. The whole flow is doable in under a minute by a judge who has
  never seen it.

## Judging criteria the page must serve

| Criterion (weight) | What the page does for it |
|---|---|
| Impact (30%) | The "How it works" pitch names who benefits: robots handling garments, fabric and sheet goods that must plan or undo deformations; recovering unseen loads from observed shape. One sentence each, no marketing language. |
| Use of Meta AI tooling (25%) | Everything is PyTorch; the footer lists the stack honestly. Stretch goal, only after the core flow works end to end: a "what a camera would see" tile that renders the final state with `cloth_data_generation/inverse/render.py` or `photo.py` and shows the SAM 3 mask from `cloth_data_generation/inverse/segment.py` (runs in `sam3env`), clearly labelled as the camera path under construction, not as the model input. |
| Technical depth (20%) | Metrics panel with the reference numbers; sim and inference timings; a tooltip per metric saying how it is computed; the 128-vs-256 decision with numbers in `demo/README.md`. |
| Demo (25%) | Fast, reliable, visibly real: live progress during the sim, smooth 30-frame playback, cached fallback, no dead ends. Record a 60 to 90 s screen capture of the full flow at the end and save it as `demo/demo.mp4`. |

## Working rules

- Build in this order and stop to report after each step: (1) inference wrapper verified on the
  20 test samples, (2) sim wrapper and the 128-vs-256 measurement, (3) Modal backend running end
  to end from `curl`, (4) frontend against the live backend on localhost, (5) Vercel deploy,
  (6) cache and fallback, (7) stretch camera tile, (8) demo video.
- Whenever you need something from the cluster or the workstation that you cannot reach, give
  Nick the exact command and the exact source and destination paths.
- Commit after each step on branch `demo` with a message that states what was verified and how.
