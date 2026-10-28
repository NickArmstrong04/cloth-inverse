# Inverse-model results

Task: from the final cloth state (material-indexed positions + velocities) predict the
two force vectors (position u,v; direction; magnitude; duration). Metrics are means over
the test split with optimal pairing of the two forces; position error is in cm on the
0.5 m sheet. "Chance" = predicting the training mean.

## 2026-10-28: matrices M1-M3 on ds768_night (cloth_results/JOBS.md), 300 epochs, TTA off, D4-symmetric metric where pose is unknown

| run | rung / setting | pos cm | dir deg | train pos cm |
|---|---|---|---|---|
| M1a, M1b | rung 0 (grid + vel, known frame), seeds 1, 2 | 3.9, 4.2 | 13.7, 13.3 | 3.3, 3.9 |
| M1c, M1d | rung 2 (grid, no vel, unknown pose), seeds 1, 2 | 8.7, 9.3 | 32.8, 34.8 | 8.9, 9.6 |
| M2a | rung 1 at 384 (pool 2) | 4.5 | 16.4 | 3.9 |
| M2b | rung 1 at 192 (pool 4) | 5.0 | 17.2 | 4.1 |
| M3a | rung 3 (point cloud 1024 pts, no vel, unknown pose) | 8.3 | 30.7 | 8.8 |

Reading (ladder run 1 for reference: rung 0 = 4.8 / 13.7, rung 1 = 4.3 / 14.3):
- Seed-to-seed spread is ~0.4 cm / 0.5 deg at rung 0 and ~0.6 cm / 2 deg at rung 2. Differences
  smaller than that between rungs or settings are noise.
- **Unknown pose is the expensive rung**: 4 cm -> 9 cm, 13 deg -> 33 deg, and train ~ test, so the
  SO(3)-augmented task is not learned with this model/data rather than overfit.
- Going from the grid to an unordered point cloud (rung 2 -> 3) costs nothing: the 1024-point
  transformer matches the pose-unknown CNN. The representation is not the bottleneck; pose is.
- Resolution: 384 loses ~0.3 cm / 2 deg vs 768, 192 loses ~0.7 cm / 3 deg. A 512 dataset is
  sufficient for the grid model; the pooled runs start to show a train/test gap (3.9 vs 4.5).
- Magnitude / impulse still at chance in every run (velocity clamp, DATASETS.md).

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
