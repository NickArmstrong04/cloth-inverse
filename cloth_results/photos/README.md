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

### Results (2026-10-30, SAM 3 `facebook/sam3`, threshold 0.5 unless noted)

56 views (48 cloth, 8 paper), 15 prompts over two rounds plus the oracle box; full tables in
`cloth256/sam3/summary.md`, `paper256/sam3/summary.md`, threshold-0.3 runs in `sam3_t03/`.

| prompt | cloth: mean IoU / found / coverage | paper: mean IoU / found |
|---|---|---|
| **fabric** (winner, default) | **0.957 / 47 of 48 / 0.978** | 0.989 / 8 of 8 |
| a piece of cloth | 0.897 / 44 / 0.916 (0.957 / 47 at threshold 0.3) | 0.989 / 8 |
| a checkered cloth | 0.955 / 47 / 0.975 | 0.983 / 8 |
| cloth | 0.757 / 37 / 0.770 (0.876 / 43 at 0.3) | 0.989 / 8 |
| a piece of fabric | 0.916 / 45 / 0.937 | 0.989 / 8 |
| a folded cloth | 0.693 / 34 | 0.865 / 7 |
| a crumpled sheet of fabric / a crumpled cloth | 0.449 / 22, 0.287 / 14 | 0.370 / 3, 0.246 / 2 |
| textile, sheet, paper, a crumpled sheet of paper | 0.00-0.02 / 0-1 | 0.00-0.37 |
| [oracle box] (ceiling) | 0.977 / 48 / 0.999 | 0.989 / 8 |

Findings:
- **Segmentation is solved for these renders.** When SAM 3 finds the sheet, the mask is the oracle
  mask to within the pixel-splat boundary: median IoU 0.985-0.989 for every working prompt, the same
  as the oracle box. There are no partial or wrong masks; every failure is a zero detection (K = 0).
- **The prompt decides detection, not mask quality.** "fabric" misses one view in 48 (the
  near-edge-on `sample_slot0001_seed100001_v3`, 11% of vertices visible); "cloth" misses 11.
  Adjectives hurt: "crumpled" halves the detection rate, "folded" costs a quarter. Bare "sheet",
  "textile" and anything with "paper" find nothing: the checkered sheet reads as cloth, including the
  paper-material renders (which look the same apart from creases).
- **Lowering the threshold to 0.3 rescues "cloth" and "a piece of cloth" but not "fabric":** the
  remaining miss has no candidate at all (presence logit), so it is not a threshold problem.
- Round 3 was not needed: "fabric" is within 0.02 mean IoU and one view of the oracle-box ceiling.
  Texture-agnostic phrase for real photos: "fabric" (or "a piece of cloth" at threshold 0.3).

Packages (`views/`, `package_views.py --prompt fabric`): 48 + 8 SAM packages next to the oracle
ones. Points per view: SAM / oracle = 0.968 mean on cloth (0.985-0.993 on every found view; the one
miss gives an empty package) and 0.990 on paper; oracle median 33.6k (cloth) / 38.1k (paper) points
from 512 px views.
