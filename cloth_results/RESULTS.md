# Inverse-model results

Task: from the final cloth state (material-indexed positions + velocities) predict the
two force vectors (position u,v; direction; magnitude; duration). Metrics are means over
the test split with optimal pairing of the two forces; position error is in cm on the
0.5 m sheet. "Chance" = predicting the training mean.

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
