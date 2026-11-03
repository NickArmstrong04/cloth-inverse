# Demo log (measurements; 2026-11-01)

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
comparison, 58588d3) was removed on 2026-11-03 as unclear. Decision (Nick): one tile per step,
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
