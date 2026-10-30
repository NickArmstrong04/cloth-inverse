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

## VGGT lifting (rung 4b, step 3, 2026-10-30)

Meta's VGGT (`facebook/VGGT-1B`, installed by `setup_sam3.sh` into the SAM venv) predicts a
depth map and intrinsics from each single RGB photo; `inverse/lift.py` unprojects the pixels
inside the SAM "fabric" mask, scores against the oracle cloud at the same pixels, and writes
`views/<stem>_pts_vggt[_knownfov].npz` (unscaled cloud + `scale_to_oracle`). SAM 3D Objects
was ruled out on the workstation (32 GB VRAM minimum, conda + Kaolin + PyTorch3D stack); it is
a Bouchet L40S job if object-level reconstruction is wanted later.

| metric (median over views) | cloth, 47 views | paper, 8 views |
|---|---|---|
| depth relative error after one scale factor, median / p90 | 0.040 / 0.096 | 0.033 / 0.091 |
| scale factor to the oracle (monocular has none), median / log-spread | 1.12 / 0.13 | 1.11 / 0.13 |
| focal length, predicted / true | 0.88 | 0.86 |
| point RMSE, scale + centring only, VGGT intrinsics | 7.1 cm | 7.3 cm |
| point RMSE, scale + centring only, true intrinsics (`--known-fov`) | 6.6 cm | 6.6 cm |
| point RMSE after the best similarity (rotation + translation + scale) | **4.0 cm** | **4.9 cm** |
| same with true intrinsics | 4.3 cm | 4.6 cm |

Findings:
- **Lifting, not segmentation, is the cost of the perception stage.** SAM keeps 98-99% of the
  oracle points; VGGT places them with ~4 cm RMSE on a 50 cm sheet after pose and scale are
  fitted, i.e. at the fold scale. A 4% median depth error at ~1.1 m is ~4.5 cm, which is where
  the number comes from.
- **Scale is unknown and varies 13% (log std) view to view**, so the model input must be
  normalised (ROADMAP: by the known sheet size or `--view-norm`); a real photo has no oracle.
- **VGGT's own intrinsics are self-consistent**: forcing the true focal length lowers the
  centred error (7.1 -> 6.6 cm) but raises the similarity-aligned one (4.0 -> 4.3 cm), so
  use VGGT's intrinsics with its depth.
- Runtime: ~1 s per 512 px view on the 4070 Ti after a 10 s model load (one earlier run took
  22 s per view for no reproducible reason; a re-run was 1 s).

## Evaluating a rung-4a model on the packages (step 4)

`inverse/eval_views.py --run <run> --photos cloth256 paper256 --sources oracle sam3 vggt`
loads any `--view` checkpoint (point model `pointtf`, or `cnn_reg` / `dino_reg` on the
orthographic depth view, re-rendered from the package points with `render.render` at the
run's size / extent / splat) and reports the force metrics per source on the same views, so
oracle -> sam3 is the segmentation cost and sam3 -> vggt the lifting cost. The global
feature (world centroid in training) is zeroed; VGGT clouds are scaled by their oracle
factor under `--scale metric` (flagged) or normalised like `--view-norm` under `--scale norm`.

Plumbing check (2026-10-30, local 2-epoch smoke checkpoints on the 128 split, which sit at
chance): pointtf and cnn_reg both give ~10 cm / 75 deg on every source, i.e. the shim runs
and the sources are interchangeable to the model at chance level. The method number needs
the cluster checkpoints: M9i (cnn_reg, 8.5 cm / 25 deg on oracle views) and M9f / M9g
(pointtf); see JOBS.md M18. Packages for transfer: `views_packages.tgz` (75 MB, gitignored).

## Alignment report: truth vs perception per view (2026-10-30)

`inverse/align_report.py --photos cloth256 paper256 --out align` lines up, for every photo, what
the renderer knows (oracle mask, metric depth, visible vertices) against what the stack found (SAM 3
"fabric" mask; VGGT depth and cloud inside that mask, rebuilt per pixel from the saved package).
Outputs under `align/` (PNGs gitignored, `report.md` / `report.json` tracked): one panel per view
(photo | oracle mask | SAM mask | mask difference; oracle depth | VGGT depth at one fitted scale |
signed depth error | the clouds from above), `summary.png` (error vs obliqueness), `worst.png`.

| median over lifted views | cloth256 (47 of 48) | paper256 (8 of 8) |
|---|---|---|
| SAM mask IoU; recall / precision | 0.986; 0.990 / 0.999 | 0.989; 0.990 / 1.000 |
| visible-vertex coverage (min over views) | 1.000 (0.988) | 1.000 (0.998) |
| VGGT depth abs. error, median / p90 | 4.2 / 11.3 cm | 3.7 / 11.3 cm |
| VGGT depth bias | -0.8 cm | -0.5 cm |
| oracle depth range inside the mask | 44.7 cm | 40.2 cm |
| point RMSE, scale + centring / best similarity | 7.1 / 4.0 cm | 7.3 / 4.9 cm |

Reading: segmentation is solved on these renders (the only loss is the one near-edge-on view SAM
does not detect, and a 1-2 px silhouette band). Lifting is where the input departs from the truth:
the VGGT error is not noise but a smooth ramp along the sheet, i.e. a misjudged inclination /
depth compression, so it is largest on oblique views with a large depth range (worst view: 67 cm
range, 7 cm median / 25 cm p90 error, 13 cm RMSE even after the best similarity) and smallest on
frontal ones (1.5 cm median). Folds are reproduced; the global tilt is not. A per-view affine
(z-scale) fit would tell how much is pure depth compression; a second view or a known-size cue
would remove it. Depth error is uncorrelated with the visible fraction and grows with the depth
range (`summary.png`).

### M18 outcome (Bouchet, 2026-10-30): the models need the drift, not just the shape

eval_views on the smoke packages: M9i 8.2 cm / 73 deg (oracle) = 8.2 / 73 (SAM) vs 9.1 / 71
(VGGT); M9f / M9g 9.2-10.0 cm / 74-77 deg on every source. Like for like (64 views rendered on
the cluster from the ds256_free test split, oracle packages): M9i 9.2 / 56.7, M9f 9.0 / 60.7, M9g
8.8 / 59.8, while M9f scores 7.9 / 23.1 on the same split through train.py. So the shim loses
direction, and the cause is the global feature: under --no-vel train.py feeds g = the sheet's WORLD
centroid, its drift from the rest position, rotated into the camera frame; in the free-flight data
the sheet flies 15-108 cm along the first force (local test samples), a direct direction / impulse
cue, and the photo camera aims at the sheet so the drift is invisible. Plumbing for the oracle drift
is in place (photo.py --gt stores `centroid`, package_views writes `g_world`, `eval_views --g
oracle`) but the decisive rerun is parked: the focus moved to the perception alignment above.
Open design question: a fixed camera with a known rest position would make the drift observable
(the sheet's offset in the image).

## Shape-from-template, step 1: can known knowledge replace depth prediction? (2026-10-30)

Question (Nick): we know the node count, the rest grid and the sheet's conditions; can the mesh
be recovered from the SAM segmentation instead of predicting depth? `inverse/fit_template.py`
answers it with oracle correspondences (the upper bound): every pooled grid vertex is a free 3D
point; visible vertices are held to their true pixel (photo.py --gt `vpix`, half-pixel tolerance);
energy = isometry (all grid edges + diagonals at the flat rest length) + a weak bending term + a
silhouette term (hidden vertices must project inside the mask). NO depth input. Scored as the
absolute camera-frame error of the visible vertices against the true grid (no alignment), 64x64
template (`--pool 4`), 2000-3000 Adam iterations, ~20 s/view on the 4070 Ti.

| configuration (2 views unless stated) | visible-vertex error (mean, cm) | note |
|---|---|---|
| start at the true surface, true edge lengths | 0.12 | the energy model is right |
| start at truth + 2 cm noise, flat rest lengths | 1.2 | basin >= 2 cm even with the wrong prior |
| cold start (constant depth 1.15 m), flat lengths | 12-15 | stuck, 8-12 cm too near, 8% residual stretch |
| cold start, true lengths; or true-depth start | 5-15 | still stuck: non-convexity, not the prior |
| start from VGGT depth (oracle scale), flat lengths, all 55 lifted views | 4.9 -> 4.4 (median over views) | 76% of views improve, median ratio 0.95; depth error 4.0 -> 3.7 cm; good views 2.5 -> 2.3, bad views 7.8 -> 7.0 |
| start from VGGT depth at VGGT's own scale | 29 -> 29 | the template does not recover absolute scale |

Two facts decide it. (1) The simulated cloth is not inextensible: axis edges of the final grids are
stretched +8% on average, 17% RMS, p95 +42%, and the stretch is global, not confined to the force
patches (far from the forces RMS 7-34%; `tri_ke` 1000 at 0.2 kg/m^2 is a rubber sheet). The
isometry prior is therefore wrong by more than the depth error it is meant to fix. (2) Even with the
true edge lengths the problem is nearly affine at a 40 deg FOV with the sheet 1 m away: a uniformly
nearer, slightly tilted sheet is almost as isometric, so a cold start sinks into a wrong minimum and
absolute scale is unobservable. Conclusion: with a single view the template cannot replace monocular
depth; as a refinement on top of VGGT it buys ~10-15% and fixes nothing structural. What would
change the answer: a stiff (realistic) cloth in the simulator, which also matters for the data; a
known-size cue or second view for scale; and dense correspondences from a non-repeating printed
pattern (step 2). Hidden-vertex error stays at ~8 cm, as expected for the unseen half.
