# Meta hackathon: improvements to the use of Meta tooling

Written 2026-11-01 against the judging rubric on the event page
(https://cerebralvalley.ai/e/meta-ai-developer-hackathon/details): Impact 30%, Use of Meta AI
Tooling 25%, Technical Depth & Execution 20%, Demo 25%, each 0-10, after a pass/fail Stage One
("reasonably used Meta Model API and/or Meta's open-weight models"). Submission due Wed 2026-11-04
21:00 ET with a three-minute video; judging 2026-11-05 to 11-09.

Self-assessment on 2026-11-01: ~6.1 / 10 weighted (Impact 6, Tooling 4, Depth 9, Demo 6).
Demo and impact are handled by the video (planned 2026-11-03). This file is about tooling.

## The gap

The research chain has done real work with Meta models (SAM 3 prompt study to IoU 0.985, VGGT
stereo lifting to 1.4 cm metric depth, DINOv2 ablation), but none of it runs in the deployed
product and none of it uses the models the event page names. The live demo runs the simulator and
our own CNN; SAM 3 appears only as a static PNG under "Camera path (under construction)". Judges
score what they can see run.

Models the event page names: Muse Spark series, Muse Wonder, Muse Image, Muse Voice Transcribe and
SAM 3.1 through the Meta Model API; open weights Muse Glimmer and Muse Spark Open; Muse Code.
Hosted `sam-3.1` takes an image plus a text prompt (or boxes / points) and returns masks over the
Responses API (`POST https://api.meta.ai/v1/responses`, Bearer key), ~$2.50 per 1,000 images.
Muse Spark takes image input and returns text through the same endpoint.

- Models: https://dev.meta.ai/docs/models
- SAM overview: https://dev.meta.ai/docs/sam/overview
- API docs: https://ai.developer.meta.com/docs/
- SAM 3.1 API example: https://github.com/anthonynolan/sam3

## Ranked plan (points gained per day of work)

| # | item | effort | rubric gain | status |
|---|---|---|---|---|
| 1 | Live camera path in the demo | 1-2 days | tooling 4 -> ~7; removes the Stage One question | todo |
| 2 | Muse Spark as the SAM re-prompting agent | 0.5 day | "beyond a basic integration"; $5K agentic bonus | todo |
| 3 | Muse Spark narration of the result | 2 h | visible use of the best-known Meta model | todo |
| 4 | One real photo through the chain | 1 day | the "models meet the real world" frame for the video | todo |
| 5 | SAM 3D Objects on Bouchet | 1 cluster job | breadth: four Meta models measured, not two | stretch |
| 6 | README "Meta models used" table | 1 h | judges find the tooling before RESULTS.md | todo |

### 1. Live camera path in the demo

After each simulation the backend has the final mesh. Add:

1. render a 20 deg stereo pair with `cloth_data_generation/inverse/photo.py --stereo 20`;
2. send both frames to hosted SAM 3.1 (prompt "fabric"); fall back to open-weight SAM 3
   (`segment.py`, sam3env) when the API fails;
3. VGGT on the Modal GPU for the metric lift (`lift_stereo.py`, known camera motion);
4. world-frame grid with `grid_from_views.py` (harmonic fill of hidden vertices);
5. the no-velocity model (M19/M20 checkpoint when available; M16-a `--no-vel` eval otherwise).

Show three tiles (photo, SAM mask, lifted grid) and two predictions side by side, mesh path and
photo path, each with its error. The photo-path error will be worse; say so on screen.

Build order: `demo/backend/camera.py` taking the final grid and returning photo, mask, lifted grid
and the photo-path prediction; verify on the 20 local test samples; then wire into the Modal
container and the `/jobs/{id}` payload; then the frontend tiles. Expect 20-30 s extra on a warm
container; keep the mesh path as the headline interaction.

### 2. Muse Spark as the SAM re-prompting agent

Finding from `cloth_results/photos/README.md`: detection depends on the phrase, not the threshold
("fabric" 47/48 views, "a crumpled sheet of fabric" 22/48, "paper" 0/48). Give Muse Spark the
photo, the SAM result (instances found, mask fraction) and the expected sheet fraction from the
camera, and let it accept, re-prompt with another phrase, or add a box prompt, up to three rounds.
Log the decisions in the result panel. Surfaces a SAM 3.1 behaviour worth showing Meta's team.

### 3. Muse Spark narration of the result

Feed it the photo, predicted forces, reversal forces and the metric table; it writes the
two-sentence plain-language verdict the tooltips currently try to express.

### 4. One real photo through the chain

Crumple a sheet of paper, two phone photos ~20 deg apart, SAM 3.1 + VGGT, show the lifted grid.
No trustworthy force number needed; the frame where a Meta model touches a real object is what the
video needs.

### 5. SAM 3D Objects on Bouchet (stretch)

Deferred earlier for VRAM (32 GB minimum). One L40S job on the existing 56 synthetic views gives
a single-image lift number next to VGGT's 4 cm. Only if the queue is kind. Needs a JOBS.md row.

### 6. README "Meta models used" table

One row per model: SAM 3.1 API, SAM 3 open weights, VGGT, DINOv2, Muse Spark (and SAM 3D Objects
if run), with where it runs, what it does, the measured number, and the date. Also state at the
top of the README that all work dates from 2026-10-27 (first commit) onward, inside the building
period 2026-10-27 to 11-04.

## Cautions

- Keep the open-weight SAM 3 path as the fallback when the API is down during judging; both count.
- Keep the mesh-path demo as the fast, reliable headline loop; the camera path is additive.
- Keep one Modal container warm during judging (2026-11-05 to 11-09); the scale-to-zero cold start
  is one to two minutes.
- Video must be three minutes (event rule), not the 60-90 s in `DEMO_PROMPT.md`.
