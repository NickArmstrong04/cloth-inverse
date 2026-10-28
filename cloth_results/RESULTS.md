# Inverse-model results

Task: from the final cloth state (material-indexed positions + velocities) predict the
two force vectors (position u,v; direction; magnitude; duration). Metrics are means over
the test split with optimal pairing of the two forces; position error is in cm on the
0.5 m sheet. "Chance" = predicting the training mean.

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
