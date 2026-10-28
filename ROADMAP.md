# Roadmap: inverting forces from deformed geometry

## Stage 1 (current): show that an ML model can invert the force on a simulated cloth

Status 2026-10-28: done for position and direction with privileged inputs
(material-indexed grid + velocities + known frame): 4.8 cm / 13.7 deg on held-out
768x768 samples (`cloth_results/RESULTS.md`). Magnitude/impulse are at chance because
the simulator clamps velocities (`cloth_results/DATASETS.md`); stability sweep pending.

## Future directions (recorded 2026-10-28)

1. **Replicate on paper.** Physically crumple/push real paper with instrumented,
   repeatable loading; observe the result (scan / multi-view) and recover the applied
   forces. Needs: no material correspondence, no velocities, unknown pose, partial
   observation, plastic material (creases persist).
2. **Image + SAM.** From a photo, segment the cloth/paper with SAM, impute the 3D state,
   and report the forces that produced it (and the inverse that would undo it). Needs:
   single-view observation, rendering-based training (domain randomization), monocular
   depth/shape imputation.
3. **Proof of concept on a more complex system** (e.g. a car crash or another system where
   loading history must be recovered from deformed geometry). Needs: representation that
   is not a grid (FEM meshes / point clouds), contact-impulse style labels, a different
   simulator; the pipeline pattern (sim -> QC -> dataset -> inverse model -> re-sim check)
   carries over.

## Plan to make Stage 1 conducive to 1-3 (the observation ladder)

Remove input privileges one rung at a time, in simulation, measuring the loss at each rung
against the privileged ceiling. Keep the privileged model as upper bound / teacher.

| rung | what the model sees | flag / model | status |
|---|---|---|---|
| 0 | indexed grid, positions + velocities, known frame | `cnn_reg` (baseline) | 4.8 cm / 13.7 deg |
| 1 | indexed grid, positions only | `--no-vel` | built, to run |
| 2 | + unknown rigid pose | `--no-vel` + SO(3) aug + `--eval-rotated` | built, to run |
| 3 | unordered point cloud (no grid), positions only, unknown pose | `--model pointtf` | built, to run |
| 4 | rendered depth / RGB views with occlusion, domain randomization | to build | |
| 5 | real observations (paper scan / SAM on photos) | | |

Finding (2026-10-28, ladder run 1): velocities add nothing (rung 1 = 4.3 cm / 14.3 deg vs
rung 0 = 4.8 / 13.7): the force information is in the shape. Under an unknown pose the
material coordinates (u, v) are only defined up to the square sheet's 8-element symmetry
group (90-degree turns and flipping over are proper rotations), so loss and metrics must be
D4-symmetric (`--sym`, automatic with `--eval-rotated`); the first rung-2/3 numbers (15.7 /
16.0 cm) were graded against an unlearnable target.

Decisions still open (before the next large generation run):
- Symmetry breaking for real experiments: the sheet needs an asymmetric cue (marked corner,
  rectangle, printed pattern) if absolute (u, v) is wanted; otherwise report position up to D4.
- Label space: predict **impulse** (direction x magnitude x duration) instead of magnitude
  and duration separately; impulse is what the shape determines and what transfers to crash.
- Loading model: shape the simulated force application to mirror what a bench rig can apply
  and measure (pushers/grippers with recorded force), so sim and real share labels.
- Material: paper is plastic; Newton cloth is elastic. Decide whether cloth data informs the
  paper experiment or whether a crease-capable model/simulator is needed.
- Physics: fix the velocity clamp (substeps) so magnitude becomes identifiable.
