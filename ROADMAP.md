# Roadmap: inverting forces from deformed geometry

The story: (1) successful prediction in a virtual environment, for cloth and for paper;
(2) ingest a real-world scene with SAM or similar Meta tools and state how to undo the
deformation; (3) a broader or more technical system. Decided 2026-10-28.

## Data generation for that story (the bench profile, 2026-10-28)

What the first datasets lacked for the real-world target, and what changed:

| gap | fix | where |
|---|---|---|
| free-floating, zero-gravity sheet | sheet on a table under gravity, friction 0.5 | `--table` (sheet.py) |
| forces pull from any direction | two pusher-like pushes, 10-60 deg into the table, never a pull | `--loading pusher` (generate_batch.py) |
| cloth only | cloth and paper datasets with the same seeds (paired loading) | `--material`, same `seed0` |
| 768 grid, ~5 min a sample | 256 grid (resolution cost ~0.3-0.7 cm, RESULTS.md), ~9x fewer particles | `NX=256` in cluster.env |
| magnitude unlearnable (velocity clamp) | the clamp scales with the self-contact margin, 3x looser at 256; verify on the smoke test, else raise substeps | stability sweep J-002 |
| magnitude x duration labels | predict impulse (what the shape determines) | loader/metrics already report impulse |

Still open: observation rung 4 (rendered depth/RGB with occlusion, domain randomisation),
which is what the SAM stage consumes.

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
| 1 | indexed grid, positions only | `--no-vel` | 4.5 cm / 15.9 deg (overfits: data-limited) |
| 1.5 | + unknown in-plane heading, known support plane | `--yaw-only` | built (M5) |
| 2 | + unknown rigid pose | `--no-vel` + SO(3) aug + `--eval-rotated` | 8.5 cm / 32.8 deg at 300 ep; 5.9 / 24.1 at 192 res, 1000 ep (underfits: epochs-limited) |
| 3 | unordered point cloud (no grid), positions only, unknown pose | `--model pointtf` | 8.3 cm / 31.1 deg = rung 2 (representation is free) |
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

## Queued work

- ds512_v1 (free-floating, 512) is dropped: resolution barely matters and the free-floating
  scenario is not the target. Next generation run is the bench profile, cloth and paper
  paired (`cloth_results/JOBS.md` matrix M6): smoke on gpu_devel, then 3000 slots each.
- M5 (unknown-pose cost, yaw-only rung) still runs on ds768_night; its conclusions carry over.
