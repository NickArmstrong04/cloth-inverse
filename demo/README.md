# Demo: decisions and measurements (2026-11-01 onward)

## 128-vs-256 grid decision (step 2)

20 random `free` force configurations (`sample_forces`, seeds 100000-100019), simulated at 128 and 256 with the same seeds, M16-a scored on both (128 upsampled bilinear to 256). Means; impulse-rel median in brackets.

| path | pos cm | dir deg | impulse rel (median) | sim wall s |
|---|---|---|---|---|
| 128 -> 256 | 3.47 | 25.8 | 0.867 (0.588) | 4.4 |
| 256 | 1.37 | 6.4 | 0.251 (0.174) | 11.9 |

Ratios 128/256: pos 2.53x, dir 4.01x. **Decision: grid 256.**

Failed seeds (1, excluded): 100011.

## Under the hood: the camera path as a step strip (2026-11-03)

Why: judges score the Meta tooling they can see run; the first camera UI (gallery + mesh/camera
comparison, 58588d3) was removed on 2026-11-03 as unclear. Decision: one tile per step,
filled in live after every Push, with the real intermediate image and the model named.

Steps and what each tile shows (`demo/backend/camera.py` `on_stage`, `GET /jobs/{id}.stages`):
1. re-simulate on paper (Newton/VBD; pushes clamped to the M21-c training range, notes shown);
2. two photos 20 deg apart with the coded print (view 1 shown);
3. SAM 3.1 via the Meta Model API: mask overlay, prompt, instance count, latency; amber badge when
   the renderer mask had to stand in;
4. VGGT: metric depth image of view 1, depth range, photometric scale;
5. decoded print: decoded vertices on the photo (camera-only, no truth), cells read;
6. lifted grid: seen-vs-filled map (64x64), seen fraction, known-size scale fix;
7. M21-c prediction from the camera grid next to the prediction from the exact sheet, with the
   typical numbers (7.4 cm / 17 deg vs 3.4 / 7);
8. Muse Spark verdict (`demo/backend/narrate.py`): three sentences from the rounded numbers only,
   every number checked (`t3_narrate.check`); fixed sentence when the check or the API fails.

The mesh path (M16-a) stays the headline; the camera job starts from a button under Push once the
mesh result lands (on demand since 2026-11-03; it used to start by itself) and polls every 2 s. The metrics table gets a "from the camera" row.

## Model menu (2026-11-03)

Menu labels (2026-11-03): "Positions + final velocities" (M16-a), "Positions only" (M20-a3),
"Pseudo camera" (M21-c).

Decision: the demo's model is the clean model with velocities, M16-a (cloth at release,
1.0 cm / 4.3 deg); a dropdown next to Push offers the two rest-state paper models, M20-a3 (clean
grids, positions only; 2.8 / 6.6 clean) and M21-c (trained on camera-chain grids; 3.4 / 7.3 clean).
No model with a second simulated guess (M22 refinement) in the demo.

How it runs (`app.py` `MODELS`, `_run_paper`, `GET /models`, `POST /simulate {"model"}`): the
paper models run the paper physics (`camera.py`'s Simulator from `rest_reference.json`, 30
snapshots for the playback, ~100 s per push) with the pushes clamped to their training range
(`scrunch_forces`; the cards and true arrows update to what was simulated and the table says what
changed), then the model on the exact final grid (positions only). Checkpoints resolve as
`CKPT_ROOT/<name>/model.pt` (`/vol` on Modal); a model without a checkpoint is listed as
"(checkpoint not deployed)" and refused with 422. M20-a3 comes from Bouchet `runs/M20a3/model.pt`.

## Presentation mode and the 3-minute video (2026-11-03)

`?present=1` (`demo/frontend/src/present.ts`): the app replays `public/present/entry.json` (one
cache entry rebuilt as a JobResult by `demo/backend/make_present.py`) and `public/present/camera.json`
(one recorded camera-path job, `demo/backend/record_camera_job.py`) with no backend call; the panel
and the step strip stay hidden until cued (`window.__present`, key bindings), so a scripted
browser can record the demo with no waits. Built for the submission video
(a separate `meta_video_generation/video/` folder, not in this repo); normal
visitors see no change.

Fixes made on the way: the player and the cache fallback assumed 30 frames per result, but six
cache entries hold 22-27 (the sheet vanished past the last frame); the frame count now comes from
the payload. A flat sheet rendered solid blue (zero height span); it is now neutral.
`narrate.py` rejected every Muse Spark verdict that mentioned "SAM 3.1" (3.1 read as an
unsupported number); 3.1 is now allowed. The live backend needs a `modal deploy` to pick that up.

Since 2026-11-03 the normal app also offers the same files behind a "View the demo video's run"
button next to Push (`App.tsx` `showDemoRun`): result shown at once with the pushes set and
locked, a "Show the camera path for this run" button replays the recorded camera job into the
step strip (~1 s per tile), no backend call either way. Judges skip the wait.

The button follows the model menu (2026-11-03 evening): with "Pseudo camera" or "Positions
only" selected it shows the same pushes through that model from `public/present/entry_M21c.json`
or `entry_M20a3.json`, recorded on the live backend with `demo/backend/record_entry.py --model
M21c --entry 10` (the paper re-sim, 30 frames, the model on the exact final grid, the clamped
pushes in `adjustments`). The notice above the table names the model and the recording date. The recorded camera strip
replays by itself after the result (2.5 s later, one tile per second) except for "Positions only",
which gets no "Under the hood" and no camera button in live runs either: the strip is the camera
path (photos, SAM 3.1, VGGT, M21-c) and that model takes no part in it (Nick, 2026-11-03).
The paper re-sim seed was random on the backend of that evening; `POST /simulate` takes an
optional `seed` since the same commit, so after the next deploy `--seed 2` re-records the paper
runs with the crease pattern the camera strip shows. Without the file the M16-a run is shown and
the menu switches to it, as before.

`public/present/camera.json` re-recorded 2026-11-03 against the live Modal backend
(`record_camera_job.py --forces-json <old camera.json> --seed 2 --api <modal url>`): SAM 3.1 live
(prompt "fabric", 1 instance, 2.2 s) and a Muse Spark verdict; the earlier file came from the local
sweep and carried the renderer-mask fallback ("SAM 3.1 unavailable") into the replay.

Video entry: cache entry 10 (its camera run decodes 397 / 1024 print cells, 40 % of the sheet
seen; entry 12 crumples harder but only 4 % decodes after the paper re-sim).

## Force set and seed for the video (2026-11-03)

Nick wanted pushes on which the camera path does exceptionally well, with the perfect-information
model scored on the same pushes. `demo/backend/sweep_live.py` runs both paths per (force set, seed)
locally in WSL (renderer mask in place of SAM 3.1, no Muse verdict; ~100 s per candidate), writes
`sweep/results.jsonl`, `entry_*.json` (cache format) and `camera_*.json` (the shape of
`present/camera.json`). Round 1: seven designed pairs + cache entries 10 and 12, seeds 1-2.
Round 2: cache entries 12, 08, 03, 02, 07, 13 and two straight-down pairs, seeds 3-8 (stopped
after cache 13 seed 3 once the choice was made).

Findings: opposing in-plane pairs (the "simple" designs) suit neither model: M16-a was trained
on free-direction pushes and either ignores a 0.4 N pair (89° error) or the cloth sim goes NaN at
1.0 N; the camera view drawn from the seed moves the camera-path error more than the forces do
(cache 10: 18.8 cm at seed 1, 6.8 cm at seed 2). **Decision (Nick): cache entry 12 at seed 2**:
camera path 2.2 cm / 15° with 53 % of the sheet seen (23 % decoded), M21-c on the exact paper
grid 1.2 cm / 11°, M16-a 1.4 cm / 5° / 0.18. Its strength estimate is off (x1.8), which the
video does not show. `POST /camera` now takes an optional `seed` and `record_camera_job.py
--entry 12 --seed 2` reproduces the run on Modal with SAM 3.1 live (after `modal deploy`).

| set | seed | camera pos / dir / strength | M21-c on the exact paper grid | M16-a (cloth mesh) | seen / decoded |
|---|---|---|---|---|---|
| A_horiz | 1 | 8.6 cm / 44° / 0.11 | 5.2 cm / 10° | 0.8 cm / 9° / 0.46 | 0.35 / 0.22 |
| A_horiz | 2 | 5.9 cm / 22° / 0.14 | 4.4 cm / 4° | 0.7 cm / 9° / 0.46 | 0.12 / 0.01 |
| B_vert | 1 | failed | failed | 0.7 cm / 5° / 0.33 | nan / nan |
| B_vert | 2 | failed | failed | failed | nan / nan |
| C_diag | 1 | 7.3 cm / 17° / 0.10 | 2.7 cm / 8° | 1.0 cm / 3° / 0.35 | 0.34 / 0.20 |
| C_diag | 2 | 9.1 cm / 29° / 0.34 | 3.7 cm / 8° | 1.8 cm / 3° / 0.28 | 0.07 / nan |
| D_gentle | 1 | 6.2 cm / 9° / 0.86 | 3.6 cm / 10° | 5.6 cm / 89° / 1.01 | 0.31 / 0.21 |
| D_gentle | 2 | 12.4 cm / 72° / 0.01 | 4.3 cm / 7° | 0.5 cm / 3° / 0.45 | 0.13 / nan |
| E_strong | 1 | failed | failed | failed | nan / nan |
| E_strong | 2 | 10.7 cm / 49° / 0.65 | 3.8 cm / 6° | 4.0 cm / 90° / 0.20 | 0.10 / 0.02 |
| F_offset | 1 | 12.8 cm / 41° / 0.11 | 4.7 cm / 4° | 1.7 cm / 5° / 0.43 | 0.41 / 0.26 |
| F_offset | 2 | 7.2 cm / 72° / 0.43 | 5.1 cm / 6° | 1.6 cm / 51° / 0.42 | 0.07 / nan |
| G_tilt | 1 | 4.9 cm / 21° / 0.53 | 3.6 cm / 13° | 3.3 cm / 53° / 0.05 | 0.27 / 0.18 |
| G_tilt | 2 | 3.3 cm / 20° / 0.33 | 3.7 cm / 9° | 2.4 cm / 41° / 0.28 | 0.20 / 0.14 |
| cache:10 | 1 | 18.8 cm / 25° / 0.19 | 2.0 cm / 11° | 0.9 cm / 4° / 0.30 | 0.27 / 0.19 |
| cache:10 | 2 | 6.8 cm / 19° / 0.08 | 2.3 cm / 11° | 0.9 cm / 4° / 0.30 | 0.28 / 0.20 |
| cache:12 | 1 | 4.1 cm / 15° / 1.81 | 1.2 cm / 11° | 1.4 cm / 5° / 0.18 | 0.55 / 0.49 |
| cache:12 | 2 | 2.2 cm / 15° / 1.80 | 1.2 cm / 11° | 1.4 cm / 5° / 0.18 | 0.53 / 0.23 |
| cache:12 | 3 | 6.7 cm / 14° / 0.64 | 1.2 cm / 11° | 1.4 cm / 5° / 0.18 | 0.39 / 0.19 |
| cache:12 | 4 | 12.2 cm / 12° / 1.05 | 1.2 cm / 11° | 1.4 cm / 5° / 0.18 | 0.18 / 0.09 |
| cache:12 | 5 | 4.2 cm / 9° / 0.83 | 1.2 cm / 11° | 1.4 cm / 5° / 0.18 | 0.41 / 0.23 |
| cache:12 | 6 | 8.1 cm / 13° / 0.94 | 1.2 cm / 11° | 1.4 cm / 5° / 0.18 | 0.23 / 0.10 |
| cache:12 | 7 | 6.9 cm / 11° / 1.34 | 1.2 cm / 11° | 1.4 cm / 5° / 0.18 | 0.08 / 0.02 |
| cache:12 | 8 | 6.9 cm / 8° / 1.01 | 1.2 cm / 11° | 1.4 cm / 5° / 0.18 | 0.43 / 0.34 |
| cache:08 | 3 | 10.4 cm / 25° / 0.46 | 2.6 cm / 19° | 0.2 cm / 2° / 0.23 | 0.27 / 0.22 |
| cache:08 | 4 | 6.6 cm / 20° / 0.54 | 2.7 cm / 16° | 0.2 cm / 2° / 0.23 | 0.53 / 0.51 |
| cache:08 | 5 | 6.5 cm / 16° / 0.55 | 3.0 cm / 18° | 0.2 cm / 2° / 0.23 | 0.47 / 0.35 |
| cache:08 | 6 | 14.4 cm / 26° / 0.33 | 3.4 cm / 17° | 0.2 cm / 2° / 0.23 | 0.22 / 0.08 |
| cache:08 | 7 | 9.8 cm / 87° / 0.47 | 2.5 cm / 17° | 0.2 cm / 2° / 0.23 | 0.25 / 0.17 |
| cache:08 | 8 | 5.1 cm / 43° / 0.43 | 4.0 cm / 20° | 0.2 cm / 2° / 0.23 | 0.57 / 0.55 |
| cache:03 | 3 | 15.7 cm / 54° / 0.52 | 5.1 cm / 14° | 0.6 cm / 3° / 0.29 | 0.20 / 0.03 |
| cache:03 | 4 | 6.1 cm / 15° / 0.59 | 4.3 cm / 11° | 0.6 cm / 3° / 0.29 | 0.74 / 0.56 |
| cache:03 | 5 | 3.7 cm / 15° / 0.78 | 4.1 cm / 11° | 0.6 cm / 3° / 0.29 | 0.60 / 0.51 |
| cache:03 | 6 | 4.6 cm / 18° / 0.51 | 3.9 cm / 11° | 0.6 cm / 3° / 0.29 | 0.28 / 0.14 |
| cache:03 | 7 | 12.2 cm / 27° / 1.04 | 5.2 cm / 13° | 0.6 cm / 3° / 0.29 | 0.49 / 0.37 |
| cache:03 | 8 | 7.6 cm / 10° / 0.55 | 4.8 cm / 12° | 0.6 cm / 3° / 0.29 | 0.57 / 0.46 |
| cache:02 | 3 | 12.3 cm / 11° / 0.56 | 5.1 cm / 10° | 0.9 cm / 2° / 0.25 | 0.20 / 0.11 |
| cache:02 | 4 | 8.3 cm / 20° / 0.40 | 5.5 cm / 14° | 0.9 cm / 2° / 0.25 | 0.33 / 0.21 |
| cache:02 | 5 | 2.7 cm / 15° / 0.64 | 2.9 cm / 18° | 0.9 cm / 2° / 0.25 | 0.24 / 0.09 |
| cache:02 | 6 | failed | failed | 0.9 cm / 2° / 0.25 | nan / nan |
| cache:02 | 7 | 13.7 cm / 58° / 0.34 | 6.5 cm / 16° | 0.9 cm / 2° / 0.25 | 0.12 / 0.01 |
| cache:02 | 8 | 4.1 cm / 17° / 1.20 | 3.1 cm / 15° | 0.9 cm / 2° / 0.25 | 0.39 / 0.21 |
| cache:07 | 3 | 14.3 cm / 12° / 1.31 | 9.9 cm / 11° | 0.5 cm / 5° / 0.31 | 0.60 / 0.48 |
| cache:07 | 4 | 12.8 cm / 13° / 1.63 | 9.9 cm / 11° | 0.5 cm / 5° / 0.31 | 0.34 / 0.28 |
| cache:07 | 5 | 11.7 cm / 10° / 1.16 | 9.9 cm / 11° | 0.5 cm / 5° / 0.31 | 0.44 / 0.28 |
| cache:07 | 6 | 17.0 cm / 18° / 1.18 | 9.9 cm / 11° | 0.5 cm / 5° / 0.31 | 0.42 / 0.19 |
| cache:07 | 7 | 17.0 cm / 21° / 1.26 | 9.9 cm / 11° | 0.5 cm / 5° / 0.31 | 0.16 / 0.04 |
| cache:07 | 8 | 9.6 cm / 7° / 1.08 | 9.9 cm / 11° | 0.5 cm / 5° / 0.31 | 0.61 / 0.57 |
| cache:13 | 3 | 4.0 cm / 41° / 0.48 | 1.9 cm / 15° | 1.3 cm / 2° / 0.65 | 0.15 / 0.01 |
