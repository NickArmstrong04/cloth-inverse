# Synthetic photos (rung 4b, step 1)

Test pictures of deformed sheets rendered from `final_grid` with
`cloth_data_generation/inverse/photo.py`, made 2026-10-30 for planning the SAM step.
PNGs are not tracked (`*.png`); each set's `manifest.json` is, and records per view the
source h5, camera (R: world -> camera, camera at the origin looking down -z, sheet centre at
(0, 0, -d)), the fraction of vertices visible, the mask fraction and the force labels.

What a picture is: one random camera on the sphere (uniform SO(3)) at 0.9-1.4 m, 40 deg
horizontal FOV, 512 px, pinhole perspective, z-buffer self-occlusion, printed two-tone
checker with a red (u, v) = (0, 0) corner and a dark u = 0 edge (the asymmetric cue), one
directional light, two-sided Lambertian shading, flat grey background. No cast shadows,
noise, blur or clutter: the information content of a photo, not its look.

| set | source (WSL) | grid | samples x views | command |
|---|---|---|---|---|
| `cloth256/` | `~/cloth/smoke_const/b256`, `~/cloth/smoke_const/batch8` | 256 | 12 x 4 | `python photo.py --h5 ~/cloth/smoke_const/b256 ~/cloth/smoke_const/batch8 --out ../../cloth_results/photos/cloth256 --views 4 --seed 0` |
| `paper256/` | `~/cloth/smoke_paper_256` (first 2) | 256 | 2 x 4 | `python photo.py --h5 ~/cloth/smoke_paper_256 --limit 2 --out ../../cloth_results/photos/paper256 --views 4 --seed 1` |
| `check/` | synthetic flat / edge-on / folded sheets | 256 | 4 x 1 | `python photo.py --check --out ../../cloth_results/photos/check` |

Run from `cloth_data_generation/inverse` inside the WSL venv (`~/cloth/.venv/bin/python`);
12 samples x 4 views take about 1 s on the RTX 4070 Ti SUPER.

Sanity numbers from `--check` (2026-10-30): flat sheet seen frontally fills 0.478 of the
frame (expected 0.472 for 0.5 m at 1 m, 40 deg) with every vertex visible; edge-on 0.012
visible; folded in half 0.500 visible from either side.

`--gt` also writes `<stem>_gt.npz` per view (gitignored): oracle `mask`, metric `depth`,
per-pixel `uv`, the visible-vertex flags `vis`, each vertex's pixel `vpix`, the camera
(`R`, `d`, `fov_deg`, `size`) and the force labels. The sets above were regenerated with
`--gt` (same seeds, identical PNGs).

## SAM 3 segmentation (rung 4b, step 2)

Environment: `cloth_data_generation/transfer/cluster/setup_sam3.sh` makes `~/sam3env`
(own venv: SAM 3 pins numpy < 2, the cloth venv runs numpy 2.5 for Newton) with torch
cu128 and `facebookresearch/sam3` editable; checkpoints are gated, so `hf auth login` once
with a token that has `facebook/sam3` access.

`inverse/segment.py` runs every prompt in `prompts.txt` over a photo set, keeps the
highest-scoring instance as "the sheet", and scores it against the oracle mask: IoU,
precision, recall, coverage (fraction of oracle-visible vertices inside the SAM mask, which
is what the point model loses) and K (instances found). `--box-oracle` adds a box prompt
from the oracle mask as the upper bound of prompting. Outputs per set under `sam3/`:
mask PNGs, `scores.json`, `contact_<prompt>.png` sheets (photo | oracle | SAM) and
`summary.md`. `prompts.txt` records each refinement round.

`inverse/package_views.py` turns a mask x the oracle depth into the rung-4a point-model
input: pixels inside the mask are unprojected with the same pinhole model, centred, and
written as `views/<stem>_pts_<oracle|sam3>.npz` (`pts (N,3)`, `g (6,)`, `y (2,7)` with
directions in the camera frame, camera). `--roundtrip` re-projects the oracle package and
compares masks: 0 mismatched pixels on all 60 views (2026-10-30).

### Results

(filled in after the first sweep)
