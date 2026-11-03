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

The mesh path (M16-a) stays the headline; the camera job starts when the mesh result lands and
polls every 2 s. The metrics table gets a "from the camera" row.

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

Video entry: cache entry 10 (its camera run decodes 397 / 1024 print cells, 40 % of the sheet
seen; entry 12 crumples harder but only 4 % decodes after the paper re-sim).
