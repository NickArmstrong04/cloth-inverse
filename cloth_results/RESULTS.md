# Inverse-model results

Task: from the final cloth state (material-indexed positions + velocities) predict the
two force vectors (position u,v; direction; magnitude; duration). Metrics are means over
the test split with optimal pairing of the two forces; position error is in cm on the
0.5 m sheet. "Chance" = predicting the training mean.

Since 2026-10-30 two more numbers are reported (commit after e1aa898): the absolute force error
in newtons (`mag N`, the unit the tactile-sensing papers use; force range 0.15-1.25 N) and the
re-simulation error (`eval_resim.py`): the predicted forces are simulated with the dataset's own
physics and the resulting sheet is compared with the true one, centred RMSE in m, reported as
`score = (pred - floor) / (chance - floor)` where floor = the true forces re-simulated (solver
noise) and chance = the constant predictor simulated. 0 = perfect, 1 = chance. It is the only
metric that scores the two forces jointly (M15-r jobs).

## 2026-10-29 evening: architecture levers on ds256_free, matrix M11

Same data and protocol as M8/M9 (cnn_reg width 32, bf16, TTA off, 2,453 train). New levers from
the architecture review: rotation-invariant inputs (strain + curvature in material coordinates,
directions predicted in the local tangent frame, `--intrinsic`), a dense force-field map as the
primary head (`cnn_field`), a flow-matching head that samples 8 candidate force pairs (`--flow`),
shaded renders (`--view-shade`) and an unfrozen DINOv2. Intrinsic runs are graded on the strict
metric (no D4 equivalence); M8d/M9 rung-2 numbers use the lenient one. Jobs in `JOBS.md` (M11).

| rung | input / lever | run | test pos / dir | train pos / dir | note |
|---|---|---|---|---|---|
| 1 | baseline + aux (from M9) | M9a | 5.7 / 14.4 | 4.8 / - | reference |
| 1 | intrinsic inputs + aux | M11b | 6.2 / 37.8 | 5.7 / 22.4 | position -0.5 cm, direction x2.6 worse |
| 1 | dense field as primary head | M11c | 9.8 / 41.4 | 10.4 / 38.5 | underfits; rejected |
| 1 | + flow head, regression output | M11f | 5.6 / 14.5 | 4.3 / 11.4 | the extra head costs nothing |
| 1 | flow head, best of 8 candidates (oracle) | M11f | **3.7 / 17.0** | | mean of 8: 6.9 cm; re-sim selection pending |
| 2 | baseline (lenient metric, from M8) | M8d | 6.9 / 21.7 | 6.6 / - | train = test |
| 2 | intrinsic inputs + aux, strict metric | M11a | **5.8** / 38.4 | 5.1 / 16.0 | rung-1 position at unknown pose |
| 2 | same, lenient 16-way metric | M11a2 | 7.0 / 37.5 | 7.0 / 36.4 | D4 equivalence is wrong for an invariant model |
| 2 | dense field head, 1000 ep | M11d | running | | val loss flat since ep 500 |
| 2 | dense field + intrinsic, 1000 ep | M11e | running | | train loss 0.28 vs val 2.2: first rung-2 run that overfits |
| 4a | shaded view, scratch CNN | M11g | 8.8 / 24.4 | 8.6 / 24.1 | = depth view (M9i 8.5 / 25.0) |
| 4a | shaded view, DINOv2-S/14 unfrozen | M11h | 9.5 / 25.5 | 9.3 / 25.5 | = frozen (M9h 9.2); still behind scratch |
| 3 | Sonata fine-tune | M11j/k | not run | | package not installed on the cluster |

Reading:
- **Pose was the position problem at rung 2.** With invariant inputs the unknown-pose model
  reaches 5.8 cm on a stricter metric than M8d's 6.9, the same level as the known-pose model
  (M9a 5.7). It also fits train (5.1) for the first time, and the field+intrinsic run (M11e) has
  train loss an order of magnitude below val. Rung 2 is no longer optimisation-limited; it is now
  on the same data slope as rung 1.
- **Direction collapsed to 38 deg on every intrinsic run, at both rungs.** The intrinsic path
  zeroed the global vector (centroid drift) to avoid leaking the pose. That vector is the direction
  cue: a push moves the whole sheet along the force, and the baseline reads direction mostly from
  that drift (train direction 16 deg vs val 39 says the deformation pattern alone does not
  generalise to direction). Fix in M12: the drift enters as its components in each cell's local
  frame, which is rotation-invariant, so nothing of the pose leaks.
- **The lenient D4 metric must not be used on invariant models** (M11a2): their directions are
  world vectors mapped through local frames, so the material-coordinate symmetry no longer
  commutes with the direction transform, and the 16-way min over wrong equivalents confuses
  training (train 7.0 cm).
- **The dense-field primary head loses by 4 cm** (M11c, and M11d's val loss stalled). Peak
  decoding (NMS + windowed soft-argmax) is not a better estimator than the global regression; the
  field stays as the aux loss, which is where it helped.
- **The posterior is wide: an oracle over 8 flow samples is 1.9 cm better than the point
  estimate** (3.7 vs 5.6 cm) while the sample mean is worse (6.9). That gap is what re-simulation
  selection can recover; the selector job failed only because it was queued before the candidate
  file existed (M12-f2).
- **Shading and fine-tuned DINOv2 do not move rung 4a.** The scratch CNN on a shaded view equals
  the depth view, and unfrozen DINOv2 equals frozen, 0.7 cm behind scratch. The single-view loss
  (8.5-8.8 vs 5.8-6.9) is about seeing one side, not about the feature extractor.
- Magnitude / impulse at chance everywhere (velocity clamp), unchanged.

## 2026-10-29: ds256_free (256x256 free flight, 2,453 train / 306 val / 306 test), matrices M8-M9

cnn_reg width 32 / pointtf d128 L4 as in the ladder; bf16, TTA off, D4-symmetric metric when the
pose is unknown. Test position / direction; "train" = train-split position error. 1,100-sample rows
are a random subset of the same train split (`--n-train`). Jobs and commands in `JOBS.md` (M8, M9).

| rung | input / lever | run | test pos / dir | train pos | note |
|---|---|---|---|---|---|
| 1 | positions, known pose, 2,453 samples | M8a, M8f (seed 2) | 6.4 / 15.9, 6.1 / 15.0 | 5.3, 4.9 | seed spread 0.3 cm / 0.9 deg |
| 1 | 1,100-sample subset | M8b | 7.2 / 17.2 | 5.4 | 2.2x data = -0.8 cm |
| 1 | old ds768_night pooled to 256 (1,100 samples) | M8c | 5.2 / 18.0 | 4.7 | the new data is harder, not just smaller |
| 1 | + aux force-field loss | M9a | **5.7 / 14.4** | 4.8 | -0.7 cm / -1.5 deg vs M8a, 2x the seed spread |
| 1 | + post-force frames (4x samples) | M9b | 6.7 / 17.8 | 5.9 | worse; val bottomed at ep ~100 |
| 1 | + aux + frames | M9c | 6.3 / 14.7 | 4.3 | frames cost 0.6 cm vs aux alone |
| 1 | frozen DINOv2-S/14 on the grid image, + aux | M9e | 8.8 / 20.2 | 6.2 | generic features hurt by 2.4 cm |
| 2 | unknown pose, 2,453 samples, 1000 ep | M8d | 6.9 / 21.7 | 6.6 | train = test |
| 2 | 1,100 subset, 1000 ep | M8e | 7.0 / 22.3 | 6.9 | same as full data |
| 2 | pointtf + per-point aux, 1000 ep | M9d | 7.6 / 22.3 | 7.2 | 0.6 cm behind the CNN |
| 4a | one orthographic depth+mask view, scratch CNN, + aux | M9i | 8.5 / 25.0 | 8.5 | first single-view number |
| 4a | same view, frozen DINOv2 | M9h | 9.2 / 28.2 | 9.0 | pretrained hurts here too |
| 4a | visible points only, pointtf, 1000 ep | M9f | eval crashed; recovery M9-f2 | | |

Reading:
- **Rung 1 is still on the data slope, but it is shallow.** 1,100 -> 2,453 samples buys 0.8 cm
  against a 0.3 cm seed spread, roughly 0.8 cm per doubling. The overnight refill (M10) is worth
  running but will not reach the old 4.5 cm on its own.
- **The free-flight 256 dataset is harder than ds768_night.** The same model with 1,100 samples
  gets 5.2 cm on the old data pooled to 256 (M8c) and 7.2 cm on the new data (M8b). Candidate
  causes: native 256 simulation vs 768 pooled (coarser dynamics, not just coarser input), and the
  --force-every-substep impulse fix changing how much the final state encodes. Worth a paired
  check before the next big dataset (same seeds, both resolutions).
- **Dense supervision is the lever.** Predicting the force-field map alongside the global target
  (M9a) is the only change that beats the baseline by more than the seed spread, worth about one
  doubling of data at no simulation cost. It also makes train/test closer (4.8 vs 5.7). Next: apply
  it on the unknown-pose line (M9-j) and confirm with a second seed (M9-k).
- **Frames are not extra data.** Post-force intermediate states carry the same label as their
  parent and the model overfits them (M9b val rose after ep 100). Rejected unless re-labelled with
  a time-conditioned target.
- **Frozen DINOv2 hurts, on the grid and on the rendered view.** ImageNet features do not describe
  wrinkle geometry at this scale; the from-scratch CNN wins by 2.4 cm (grid) and 0.7 cm (view).
  Fine-tuning the backbone (`--dino-finetune`) is the remaining variant; low priority.
- **Unknown pose is still optimisation-limited, not data-limited.** 2.2x data moves rung 2 by
  0.1 cm (M8e -> M8d) while train = test everywhere. The aux loss and longer schedules are the
  levers there, not simulation.
- **First rung-4a number: 8.5 cm / 25 deg** from a single orthographic depth view with unknown
  pose (M9i), 1.6 cm / 3 deg behind rung 2 (M8d). That is the cost of seeing one side with an
  oracle visible-point mask; the SAM + monocular-depth version (4b) comes on top of it.
- Magnitude / impulse at chance everywhere (velocity clamp, DATASETS.md), unchanged.

## 2026-10-28: observation ladder (run 2) + matrices M1-M3 on ds768_night

All cnn_reg width 32 unless stated, 300 epochs, bf16, TTA off, D4-symmetric metric when the pose is
unknown. Test mean position error (cm) / direction error (deg); "train" = position error on the
training split (gap to test = overfitting). Jobs in `cloth_results/JOBS.md`.

| rung | input | run(s) | test pos / dir | train pos | note |
|---|---|---|---|---|---|
| 0 | grid + vel, known frame | ladder, M1a, M1b (seeds 0,1,2) | **3.9 / 12.8**, 3.9 / 13.7, 4.2 / 13.3 | 3.4 | seed spread 0.3 cm / 0.5 deg |
| 1 | grid, positions only | ladder | 4.5 / 15.9 | 2.2 | overfits (train 2.2) |
| 1 | width 48 / 64 | M1e / M1f | 4.5 / 16.1, 4.4 / 15.3 | 2.3 / 1.4 | wider = more memorisation, same test |
| 1 | at 384 / 192 (pool 2 / 4) | M2a / M2b | 4.5 / 16.4, 5.0 / 17.2 | 3.9 / 4.1 | resolution barely matters |
| 2 | + unknown pose (SO(3)) | ladder, M1c, M1d | 8.5 / 32.8, 8.7 / 32.8, 9.3 / 34.8 | 8.6-9.6 | underfits (train = test) |
| 2 | at 192, **1000 epochs** | M2c | **5.9 / 24.1** | 6.2 | 3x epochs closes a third of the gap; still underfits |
| 2 | at 768, 1000 epochs | M1g | running | | |
| 3 | point cloud 2048 pts, unknown pose | ladder (pointtf d128 L4) | 8.3 / 31.1 | 8.8 | TTA8 hurts: 8.4 / 37.0 |
| 3 | 1024 / 8192 pts | M3a / M3b | 8.3 / 30.7, 8.2 / 30.0 | 8.8 / 8.6 | points do not matter |
| 3 | d256 / 8 layers | M3c / M3d | 8.1 / 32.1, 8.3 / 29.8 | 8.6 / 8.7 | model size does not matter |
| 3b | 4096 pts, d256, L6, 1000 epochs | ladder | running (val loss still falling at ep 875) | | |

Reading:
- **Known-pose rungs are data-limited.** Rung 1 reaches 2 cm on train and 4.5 cm on test, and
  wider models only memorise harder (M1f: 1.4 cm train, 4.4 test). Magnitude is pure memorisation
  (train 0.04 relative, test 0.66). More samples (ds512 tonight) is the lever here, not capacity.
- **Unknown-pose rungs are optimisation-limited, not data-limited.** Train = test at 8.5 cm for
  every model size, point count and representation. Three times the epochs at 192 resolution
  (M2c) drops it to 5.9 cm / 24 deg with train still = test, so the curve has not flattened.
  Rung 2 at 768 and 1000 epochs (M1g) and rung 3b will say how far this goes.
- **Representation is free.** Grid CNN and point transformer agree at every unknown-pose setting
  (8.1-8.7 cm). The point cloud is the representation for the crash direction; no penalty.
- **Resolution is cheap.** 384 costs ~0.3 cm, 192 ~0.7 cm at rung 1. Use pooled inputs for sweeps
  (7 min per 300 epochs on any GPU) and 768 only for the final number.
- Full SO(3) pose uncertainty is pessimistic for a bench: the support plane is known there. A
  yaw-only rung (`--yaw-only`, rotation about the sheet normal) is the realistic middle rung; M5.
- Magnitude / impulse at chance everywhere (velocity clamp, DATASETS.md); the stability sweep
  (J-002) decides whether the next dataset can fix this.

## 2026-10-28: ds768_night (768x768, 878 train / 110 val / 110 test), trained on Bouchet L40S

300 epochs, batch 16, bf16, full resolution, ~10 min per model. Code: `cloth_data_generation/inverse`,
job `cluster/train.sbatch`. Metrics JSON: cluster `inverse/runs/ds768_night_splits_*/`.

| model | pos cm | dir deg | mag rel | dur s | impulse rel |
|---|---|---|---|---|---|
| chance (mean) | 14.3 | 93.6 | 0.53 | 0.095 | 0.60 |
| ridge on hand features | 7.7 | 21.8 | 0.53 | 0.081 | 0.58 |
| cnn_reg, dihedral aug (3.86M) | **4.8** | **13.7** | 0.54 | 0.067 | 0.56 |
| cnn_heat, dihedral aug (4.45M) | 5.4 | 14.9 | 0.57 | 0.068 | 0.64 |
| cnn_heat, dihedral + SO(3) aug, TTA8 | 5.6 | 16.5 | 0.53 | 0.072 | 0.56 |

Observations: train ≈ val ≈ test for all CNNs (underfitting, data-limited, not overfitting).
Magnitude and impulse are at chance for every model: the VBD per-substep displacement clamp
caps every sample at 0.38 m/s, so the final state carries almost no magnitude information
(see DATASETS.md). SO(3) augmentation hurts; the fixed initial plane is a valid prior.

## 2026-10-27: dataset100 (128x128, 80/10/10), trained on workstation RTX 4070 Ti SUPER

| model | pos cm | dir deg | mag rel |
|---|---|---|---|
| chance (mean) | 15.1 | 103 | 0.58 |
| ridge | 15.8 | 46 | 0.62 |
| cnn_reg | 15.7 | 50 | 0.58 |
| cnn_heat, dihedral aug, 1500 ep | 11.6 | 36.6 | 0.59 |
| cnn_heat, SO(3) aug, 1500 ep, TTA8 | 12.9 | 30.4 | 0.56 |

Re-simulating cnn_heat's test predictions gave 0.27 m RMSE shape error vs a 0.02 m chaos floor.
A no-augmentation run memorised train (0.5 cm, 2 deg) with test at chance: code sound, data-limited.

## Impulse (F*t) as a single target: M13 runs (2026-10-30)

Question: should force and duration be merged into one impulse (or energy) target? `inverse/impulse_report.py`
on the five finished M13 runs (808 paired test forces; M13-e still training). Median relative error:

| run | force | duration | impulse F*t | corr(eF, et) | R2 log F |
|---|---|---|---|---|---|
| M13-a rung-1 baseline | 0.33 | 0.045 | 0.35 | +0.08 | 0.22 |
| M13-b rung-1 + velocities | 0.33 | 0.029 | 0.32 | -0.13 | 0.24 |
| M13-c rung-2 baseline | 0.35 | 0.060 | 0.37 | +0.15 | 0.21 |
| M13-d rung-2 intrinsic + drift | 0.18 | 0.043 | 0.20 | +0.06 | 0.69 |
| M13-f rung-1 intrinsic + drift | 0.19 | 0.051 | 0.21 | +0.03 | 0.65 |
| constant predictor | 0.46 | 0.19 | 0.47 | | 0 |

Reading: duration is solved by every model (3-6% vs 19% floor) and the force/duration errors are
uncorrelated, so there is no trade-off for an impulse target to absorb; impulse error equals force
error. Energy (~J^2) would only square the force error. The result to report is force magnitude,
where the intrinsic inputs halve the error (0.33 -> 0.18, R2 0.2 -> 0.7) at both rungs. Velocities
help duration only (M13-b vs M13-a). Labels are nominal F*t, not delivered momentum.

## M14: force-magnitude levers on ds256_const (2026-10-30)

Base M13-f (rung 1, intrinsic + drift, 300 ep). Median relative force error / position / direction on
the 808 paired test forces (`impulse_report.py`, `test/tta0` lines):

| run | change | force | pos cm | dir deg | dur s |
|---|---|---|---|---|---|
| M13-f | base | 0.192 | 4.3 | 18.1 | 0.051 |
| M14-e | seed 2 | 0.189 | 4.3 | 18.7 | 0.051 |
| M14-a | + max pooling | 0.200 | 4.7 | 19.3 | 0.058 |
| M14-b | + w_mag 4 | 0.197 | 5.4 | 21.8 | 0.058 |
| M14-c | + strain scale 100 | 0.194 | 4.3 | 18.7 | 0.053 |
| M14-d | all three | 0.173 | 5.3 | 24.0 | 0.077 |
| M14-f | width 64 | 0.193 | 4.4 | 19.3 | 0.048 |
| M14-g | all three + velocities | 0.165 | 5.5 | 24.6 | 0.072 |
| M13-e | rung 2, intrinsic + velocities, 1000 ep | **0.157** | 4.1 | 14.7 | 0.039 |
| M13-b | rung 1 baseline + velocities | 0.44 | **1.7** | **5.1** | 0.029 |

Reading: none of the three levers beats the seed band (0.003 on the median) without costing
position and direction; the combination trades 2 points of force for 1 cm and 6 deg. Width 64 does
nothing, so capacity is not the limit. The one real lever is the release velocity field: M13-e cuts
force error to 0.157 with no cost (vs M13-d 0.183 without velocities). Best per field is split
across models: the raw-position baseline with velocities (M13-b) has 1.7 cm / 5.1 deg but 0.44 on
force; the intrinsic model has 0.16 on force but 4 cm / 15 deg. Taking position/direction from
M13-b and magnitude from M13-e gives 1.7 cm / 5.1 deg / 0.16 at rung 1 today. Next lever: a hybrid
input (raw positions + intrinsic channels) at rung 1, and the identifiability floor (M14-0) to see
how far 0.16 is from what the data allows.

## M15 / M16-a: paper material and the hybrid input (2026-10-30)

Same recipe as M13-b (rung 1, raw positions + release velocities, cnn_reg + aux, 300 ep). Test mean
position / direction, median relative force error, MAE in N; train position in brackets.

| run | data | change | pos cm | dir deg | force (rel) | force MAE N | resim 50 (centred RMSE m) |
|---|---|---|---|---|---|---|---|
| M13-b | cloth, 2,300 train | base | 1.7 | 5.1 | 0.33 | 0.199 | pred 0.030 * |
| M15-b | cloth, 1,356 train | size control | 1.7 (1.4) | 5.4 | 0.39 | 0.236 | pred 0.031 * |
| M15-a | **paper**, 1,356 train | material | 3.2 (1.9) | 6.8 | 0.36 | 0.228 | pred 0.031, floor 0.001, chance 0.111 * |
| M16-a | cloth, 2,300 train | **+ hybrid input** | **1.0 (0.8)** | **4.3** | **0.156** | **0.109** | pred 0.028, floor < 1e-4, chance 0.119 * |
| M13-e | cloth | rung 2, intrinsic + vel, 1000 ep | 4.1 | 14.7 | 0.157 | 0.114 | invalid (frame bug), re-run pending |
| chance | | | 14 | 94 | 0.46 | 0.247 | 0.119 |

\* provisional: re-simulated with the 128 force footprint (see below); the corrected runs are pending.

Reading:
- **The hybrid input is the first lever that improves every field at once.** Raw channels plus the
  intrinsic strain/curvature features: position 1.7 -> 1.0 cm, direction 5.1 -> 4.3 deg, force 0.33 ->
  0.156 (R2 on linear force 0.11 -> 0.70). It matches the best magnitude model (M13-e, 0.157) while
  keeping the raw model's localisation, so the per-field split of M14 is closed: M16-a is the best
  single model at rung 1 on cloth. Train 0.8 vs test 1.0 cm: on the data slope, not memorising.
- **Paper costs 1.5 cm and 1.4 deg at equal data** (M15-a vs M15-b), with a wider train/test gap
  (1.9 -> 3.2 cm vs 1.4 -> 1.7): creases make the mapping less smooth and the 1,356 samples go less
  far. Magnitude is at chance on both materials with the raw recipe, as on cloth before; the hybrid
  recipe on paper is the next row (M17).
- **Re-simulation: the solver is not chaotic at 256, and two bugs made the first numbers wrong.**
  The true forces perturbed by 1e-5 relative move the cloth sheet less than 0.1 mm (floor prints
  0.0000) and the paper sheet 1 mm, so re-sim can in principle discriminate predictions far below a
  centimetre. The earlier "floor 0.022-0.031 m" and every predicted-force re-sim so far used the
  128-resolution force footprint (sigma 3 / radius 9 cells) where the 256 data were generated with
  6 / 18: the same force on a quarter of the area. And the rung-2 runs (`--eval-rotated`) saved their
  predicted directions in the rotated evaluation frame, so their re-sims scored below chance
  (M16-d/e/f3 1.06-1.11, M13-e NaN) while their (u, v) metrics were right. Both are fixed
  (`with_footprint`, `to_file_frame`, `fix_rotated_preds.py`); the corrected re-sims are queued
  (JOBS.md M16 round 3).


## M16: six improvements to the two best models (2026-10-30)

Rung 1 = R1 recipe (M13-b: raw positions + velocities, cnn_reg + aux, 300 ep, `--no-rot`); rung 2 =
unknown pose (`--eval-rotated`), R2 = intrinsic features, 1000 ep. Test mean position / direction,
median relative force error, MAE in N over const MAE (the label-scale-free column), R2 on linear force,
and the re-simulation error of the predicted forces (50 test samples, centred RMSE in m; chance 0.126,
floor < 1e-4). All ds256_const (2,300 train).

| run | rung | change | pos cm | dir deg | force (rel) | MAE/const | R2 lin | re-sim m |
|---|---|---|---|---|---|---|---|---|
| M13-b | 1 | R1 base | 1.7 | 5.1 | 0.327 | 0.81 | 0.11 | 0.023 |
| M16-a | 1 | + hybrid input | **1.0** | **4.3** | **0.156** | **0.44** | **0.70** | **0.018** |
| M16-b | 1 | + two-snapshot input (acceleration channels) | 1.6 | 5.2 | 0.359 | 0.88 | -0.03 | 0.025 |
| M16-c | 1 | + patch-conditioned magnitude head | 1.5 | 5.2 | 0.325 | 0.80 | 0.18 | 0.024 |
| M16-h | 1 | hybrid + snapshot + patch head | 1.2 | 4.5 | 0.155 | 0.47 | 0.65 | 0.023 |
| M16-f | 1 | R1, delivered-momentum labels | 1.9 | 4.8 | 0.125 | 0.71 | 0.29 | (0.046) * |
| M16-f2 | 1 | R1, scaled labels | 1.7 | 5.6 | 0.216 | 0.88 | 0.19 | (0.044) * |
| M13-e | 2 | R2 base (intrinsic) | 4.1 | 14.7 | 0.157 | 0.46 | 0.66 | 0.046 |
| M16-d | 2 | **R1 + Kabsch canonicalisation**, 300 ep | **1.9** | **6.0** | 0.382 | 0.89 | -0.02 | **0.024** |
| M16-e | 2 | R2 + local-frame relative position | 4.0 | 14.1 | 0.156 | 0.46 | 0.67 | 0.048 |
| M16-f3 | 2 | R2, scaled labels | 3.8 | 13.9 | 0.129 | 0.50 | 0.48 | (0.059) * |
| M16-g | 2 | R2 + flow head, 8 candidates: mean-of-8 / best-of-8 | 4.0 / **0.65** | 15.9 / 18.5 | 0.25 | | | |
| M16-g2 | 2 | re-sim selection among the 8 (50 samples): selected / mean-of-8 / oracle | **0.84** / 1.17 / 0.67 | **15.3** / 26.1 / 17.7 | 0.27 | | | |

\* magnitudes in delivered units (~8% of nominal) applied as nominal force: not comparable.

Reading:
- **One lever did the magnitude work: the hybrid input.** Giving the encoder the raw standardised
  channels and the intrinsic strain/curvature features at once brings force error from 0.33 to 0.156
  relative (R2 0.11 -> 0.70) and improves position and direction as well. The two other magnitude
  ideas, an acceleration channel from the stored snapshot and a head that re-reads the feature map
  around the predicted contact, do nothing alone and nothing on top of the hybrid (M16-h equals
  M16-a within noise and loses 0.2 cm). The snapshot test files (S3F) can be retired.
- **The unknown pose is recoverable in closed form.** Aligning the indexed grid to its rest grid
  (Kabsch) and running the plain rung-1 recipe gives 1.9 cm / 6.0 deg at rung 2, where the
  invariant-feature models stop at 4 cm / 14 deg after three times the epochs. The intrinsic path's
  position deficit was never about missing channels (M16-e changed nothing); it is the loss of the
  global layout that any hand-made invariant throws away. Magnitude under Kabsch is at the raw level
  (0.38), so the obvious next run is Kabsch + hybrid: the rung-2 model should then match M16-a.
- **Label inconsistency is a minor part of the magnitude error.** Training on the delivered momentum
  instead of the nominal F*t lowers the raw model's MAE/const from 0.81 to 0.71; the hybrid input with
  nominal labels reaches 0.44. The scaled variant is worse, and the intrinsic model gains nothing from
  either. The solver's ~8% delivery is a dataset property worth fixing at generation time, not a
  modelling lever.
- **Re-simulation selection works.** The flow head's 8 candidates contain a 0.65 cm answer where the
  mean candidate is 1.17 cm off; picking the candidate whose re-simulation best matches the observed
  sheet gives 0.84 cm and 15.3 deg, i.e. 60% of the position gap to the oracle and a better direction
  than the position-oracle itself. This is the first component that uses the simulator at test time,
  and it pays on the weakest model (intrinsic, 4 cm mean prediction); M17 runs it on the best one.
- **Re-sim agrees with the parameter metrics and adds one thing:** it ranks the intrinsic models
  (0.046-0.048 m) well below the raw and Kabsch ones (0.023-0.025) although their force error is
  better, because a 14 deg direction error moves the sheet more than a 0.2 relative magnitude error.
  Direction is the field that matters physically; magnitude the least.

## Rung 4b, step 2: SAM 3 on synthetic photos (2026-10-30, branch photo-render)

Synthetic photos of 12 cloth + 2 paper 256-grid samples, 4 random pinhole views each
(`inverse/photo.py`, `cloth_results/photos/README.md`); SAM 3 text prompts scored against the
renderer's oracle mask; SAM mask x oracle depth packaged in the rung-4a point-model format.

| prompt | cloth mean IoU | cloth found | coverage of visible vertices | paper mean IoU / found |
|---|---|---|---|---|
| **fabric** | **0.957** | **47 / 48** | 0.978 | 0.989 / 8 of 8 |
| a piece of cloth (thr 0.3) | 0.957 | 47 / 48 | 0.978 | 0.989 / 8 |
| cloth | 0.757 | 37 / 48 | 0.770 | 0.989 / 8 |
| a crumpled sheet of fabric | 0.449 | 22 / 48 | 0.458 | 0.370 / 3 |
| paper, sheet, textile | 0.00 | 0-1 / 48 | 0.00 | 0.00 / 0 |
| oracle box (ceiling) | 0.977 | 48 / 48 | 0.999 | 0.989 / 8 |

Reading:
- **On clean renders the segmentation cost is zero when the sheet is detected**: median IoU 0.985,
  equal to the oracle box; the lost points are only the splat boundary (SAM / oracle point count
  0.985-0.993 per found view). The whole error budget of this step is detection: one zero-detection
  in 48 for "fabric" (the near-edge-on view with 11% of vertices visible).
- **The phrase matters more than the threshold**: adjectives ("crumpled", "folded") and the word
  "paper" suppress detection; "fabric" is the texture-agnostic default for real photos.
- This is the plumbing number, not the method number (ROADMAP standing caveat): the perception stage
  feeds a model whose rung-4a error is 8.5 cm / 25 deg (M9i). Next: monocular depth (SAM 3D Objects
  or a depth model) in place of oracle depth, then the rung-4a checkpoint on both packages.

## Rung 4b, step 3: VGGT single-image lifting (2026-10-30, branch photo-render)

Same 56 views; SAM "fabric" mask x VGGT (facebook/VGGT-1B) depth, scored against the oracle
cloud at the same pixels (`inverse/lift.py`; tables in `cloth_results/photos/README.md`).

| lift | cloth point RMSE | paper point RMSE | note |
|---|---|---|---|
| oracle depth (step 2) | 0 | 0 | SAM keeps 98-99% of oracle points |
| VGGT, scale + centring fitted | 7.1 cm | 7.3 cm | depth rel. error median 4%, p90 10% |
| VGGT, best similarity (pose + scale) | **4.0 cm** | **4.9 cm** | what a pose-invariant model sees |

Reading:
- **The perception cost sits in the 3D lift, at the fold scale (~4 cm on a 50 cm sheet)**, not
  in segmentation. Monocular scale is unknown and spreads 13% between views.
- The rung-4a model's own error is 8.5 cm / 25 deg (M9i, oracle visible points), so the
  proof-of-concept number will be bounded by the model before the lift; both need work.
- Next: a loader shim for the `views/*_pts_*.npz` packages (normalise by sheet size), then the
  rung-4a checkpoint on oracle vs SAM vs VGGT packages on the same views (cluster run).

## Rung 4b, step 4: the shim on the cluster checkpoints, and the alignment report (2026-10-30)

M18 (JOBS.md): M9i on the smoke packages 8.2 cm / 73 deg (oracle) = 8.2 / 73 (SAM) vs 9.1 / 71
(VGGT); point models 9-10 cm / 75 deg everywhere. Like for like on 64 views of the ds256_free test
split: M9i 9.2 / 57, M9f 9.0 / 61 (M9f's own test score 7.9 / 23). The loss is not domain shift
and not the perception: train.py's --view path also feeds g = the sheet's world centroid (its drift
from the rest position, rotated into the camera frame), which the shim zeroes. In free flight the
sheet drifts 15-108 cm along the first force (local test samples, cos 0.5-0.97 with the force), so
the rung-4a numbers (8.5 / 25) rest on a cue a photo aimed at the sheet does not contain. Position
stays near the model's own score, direction collapses. Follow-ups: the oracle-drift rerun (`eval_views
--g oracle`, plumbing in place) to quantify it; a fixed-camera protocol where the drift is observable;
or models trained with g zeroed, the honest rung-4a number.

Alignment report (`inverse/align_report.py`, photos/README.md): per view, SAM 3 masks match the
truth at IoU 0.986 median with vertex coverage 1.000 (min 0.988); VGGT depth is off by 4.2 cm median
/ 11.3 cm p90 over a 45 cm depth range, as a smooth ramp along the sheet (inclination / depth
compression), worst on oblique views (up to 13 cm RMSE after the best similarity) and ~1.5 cm on
frontal ones. Segmentation is solved on these renders; single-view lifting is the input error.

## Rung 4b: shape-from-template with oracle correspondences (2026-10-30)

Can the known template (node count, rest edge lengths, inextensibility) recover the mesh from the
segmentation instead of predicting depth? `fit_template.py` (photos/README.md): with oracle
correspondences, no depth input, a 64x64 template and an isometry + bending + silhouette energy, the
fit reproduces the truth only when started near it (0.1 cm from the truth, 1.2 cm from 2 cm noise);
from a cold start it sticks 8-15 cm too near. Started from VGGT it improves the visible-vertex error
from 4.9 to 4.4 cm median (76% of 55 views improve, median ratio 0.95) and cannot recover scale.
Root causes: the simulated cloth stretches 17% RMS (p95 42%, global) so the inextensibility prior is
wrong, and the single-view problem is near-affine at this FOV. Verdict: the template refines VGGT by
10-15%; it does not replace depth. Levers: a stiff cloth in the simulator (realism issue in its own
right), a scale cue, a second view, and printed non-repeating texture for real correspondences.

## Rung 4b: depth error decomposition and stereo (2026-10-31)

The single-view VGGT error is a per-view depth scale, offset and tilt: fitting those four numbers
to the oracle takes the median depth error from 4.1 to 1.5 cm, equal to the full affine ceiling
(photos/README.md). The known template cannot supply them (4-dof template fit 9.4 cm; 6.2 cm even
with the true edge lengths: 1% of scale is 1 cm at 1 m, and the simulated cloth is 8% stretched).
A stereo pair with a 20 deg swing (`photo.py --stereo`, `lift_stereo.py`) fed to VGGT removes the
tilt: 4.8 -> 1.2 cm median depth error at oracle scale, point RMSE 7.1 -> 3.1 cm; the residual is
scale, and VGGT's predicted baseline is 10% off, so the metric step needs a model that takes the
known intrinsics and camera motion (MapAnything, next) or the phone's pose. Protocol decision
implied: two photos per capture, SAM on both.

## Rung 4b: metric depth from two photos + SAM + known camera motion (2026-10-31)

`lift_stereo.py`: VGGT on a 20 deg stereo pair gives the shape (1.2 cm median at oracle scale);
the scale comes from the known relative camera pose by a 1-D photometric consistency search over
VGGT's depth pair (window 1.4x around VGGT's own baseline, which is 10% off): scale error 0.6%
median / 8.6% p90, metric depth error 1.42 cm median, point RMSE 3.19 cm inside the SAM mask with
no oracle input (photos/README.md). 10 deg pairs: 1.78 cm. MapAnything with true intrinsics and
pose: 19-24% scale error, shape no better than single-view VGGT; dropped. Capture protocol for
rung 5: two photos ~20 deg apart with a known camera motion; a non-repeating print removes the
remaining scale aliases.
