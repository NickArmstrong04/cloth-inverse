# Datasets generated so far

| name | where | grid | samples (passing) | physics flags | generated | notes |
|---|---|---|---|---|---|---|
| dataset100 | workstation WSL `~/cloth/dataset100`; splits in `~/cloth/dataset` and `cloth_results/dataset` (80/10/10) | 128x128 | 100 | original once-per-frame force write (≈53% impulse), deterministic mode, seeds 2000-2099 | 2026-10-27 | 24 rejected attempts (all stretch > 2). First inverse models trained on this: direction ~31°, position/magnitude at chance (`cloth_results/inverse_v0`). |
| ds768_night | Bouchet `/nfs/roberts/scratch/pi_sz622/naa79/cloth/cloth_data/ds768_night` | 768x768 | 1,115 of 3,000 slots (final, 2026-10-28; all arrays finished, 888 rejected attempts) | `--force-every-substep`, non-deterministic, seeds 100000+slot, piercing QC skipped, margin/gap scaled to 0.167/0.083 mm | 2026-10-28 overnight, L40S + A5000 via `gpu` and `scavenge_gpu` arrays | ~57 MB per sample (~62 GB). 860 rejected attempts: 611 stretch (median 4.0x, i.e. real tearing) + 158 NaN, ~44% failure rate. Passing set is marginal: stretch 50/90/99th pct = 1.37/1.83/1.99, 35% above 1.5. Peak speed is 0.38 m/s in EVERY sample: the VBD per-substep displacement clamp (tied to the self-contact margin) caps velocity; at 768 the scaled margin makes the cap ~0.4 m/s, so force patches cannot move and the membrane tears. Stability sweep (`cluster/stability.py`, substeps 60/90/150, iterations 40) submitted 2026-10-28 to find a stable setting. |
| ds256_free | Bouchet `.../cloth_data/ds256_free` | 256x256 | ~3,065 of 6,000 slots (arrays 31181234 + 31220691, stopped by user 2026-10-29 at ~51% to test data vs model, JOBS.md M8; 76% pass rate) | free-flight cloth, zero gravity, `--force-every-substep`, radius/margin/gap scaled to 1/0.5/0.25 mm, seeds 100000+slot, piercing QC on | 2026-10-28/29, any `gpu` card, 2 workers per task | ~8 MB per sample (24 GB). Merge 2026-10-29: attempts needed 1/2/3 = 2001/292/74; QC stretch_max median 1.22, max 2.0 (768 set: median 1.37, 35% above 1.5); peak speed 0.60-1.15 m/s (768 set: 0.38 in every sample), i.e. the velocity clamp no longer pins the patches at this margin. Sim wall only 43 s mean (29-115 s) per sample but 106 GPU-hours total = ~2 min of GPU per passing sample: ~80% of task time is outside the solver (process start-up, Warp load, CPU piercing QC, HDF5 write), see JOBS.md M7 notes. |
| ds256_const | Bouchet `.../cloth_data/ds256_const` (generated 2026-10-29 night, JOBS.md J-005; array 31345274 + scavenge) | 256x256 | 3000 slots (overnight 2026-10-29) | constant-force model: no ramp, force every substep (full impulse), capture at release (settle 0), 5 snapshots with velocities; free flight, magnitudes 0.15-1.25 N (halved), durations 0.3-0.7 s; radius/margin/gap scaled; seeds 100000+slot; piercing QC on | 2026-10-29 overnight | Cluster run: 3,269 attempts, 1,109 rejected (34%; stretch 2.0-2.5 bulk, tail to 37.9, ~130 also piercing); final count after merge. Local 256 smoke: 4/4 passed, stretch 1.23-1.86, ~20 s per sample with 2 workers per GPU, QC 1.5 s. Not comparable to ds256_free: no settle, so the sheet is captured mid-stroke (taut band or early fold) rather than after 0.7 s of ballistic folding. |
| ds256_const_paper | Bouchet `.../cloth_data/ds256_const_paper` (planned, JOBS.md J-006; `cluster/paper.sh`) | 256x256 | 3000 slots planned | as ds256_const but `--material paper` (`--yield-curvature 50 --edge-ke 200 --crease-soften 0.1 --settle 0` pinned in SAMPLE_ARGS): plastic bending hinges with crease softening (a yielded hinge keeps 10% of edge_ke, so folds localize), tri_ke 1e4, edge_ke 200, mass 0.08 kg/m^2, stretch gate 1.5; same seeds (100000+slot) so every slot is the cloth sample's paired force configuration | | Workstation smoke 2026-10-30 (creasing preset, 256): 4/4 slots pass first attempt, 150-655 hinges past 0.5 rad per sample, stretch 1.13-1.49, vmax 0.69-0.79 m/s, none NaN, ~17 s per sample. Without softening (first smoke) the sheets passed but did not crease (2-6% of hinges yielded, almost all < 0.4 rad). Creases are stored per sample (`plastic/edge_rest_angle`), so the relaxed, folded state can be re-simulated from the capture. |

Packaging: `cluster/merge_summaries.py --indir <dir>` then `package_splits.py --indir <dir> --outdir <dir>_splits` (8:1:1; test split carries final state only). Slots are independent, so a partially finished dataset is valid; re-running `submit.py` with the same `--name` fills missing slots.

## Bench-profile smoke tests on the workstation (2026-10-28, 256 grid, RTX 4070 Ti SUPER)

Scenario for the real-world story: sheet on a table under gravity (`--table`), two pusher-like
forces (`--loading pusher`). 2-4 samples per round, scratch only, not a dataset.

| round | loading | cloth | paper | lesson |
|---|---|---|---|---|
| 1 | tangential pushes, random heading, 10-60 deg down, 0.3-2.5 N | 0.5/0.85 N: flat, stretched to 0.9 m (pulled apart), 115 s; 2/1.2 N: torn | 0.5/0.85 N: slid and rotated, 3 mm lift; 2/1.2 N: NaN | outward pushes drag the sheet apart; >1 N tears a friction-anchored sheet; ground plane holds (z >= 0 in healthy runs); peak speed 0.7-0.8 m/s, so the 256 margin lifts the 0.38 m/s clamp seen at 768 |
| 2 | pushes toward the sheet centre, 0-30 deg down, 0.1-1 N, 0.1-cell z-jitter | 0.19/0.31 N: torn, 7 cm through the table, 85k contacts; 0.77/0.43 N: healthy, 2 cm crumple | 0.19/0.31 N: NaN at 0.45 s; 0.77/0.43 N: perfectly flat, no creases, slid | a single push stronger than table friction (0.1 N paper, 0.25 N cloth) slides the sheet instead of compressing it; the samples that did crumple against the table stressed VBD self-contact (same failure as 44% of the 768 run) |
| 3 | the two pushes oppose each other (two-hands scrunch), 0-20 deg down, 0.1-1 N, 60 substeps | 0.11/0.53 N: NaN at 0.72 s, 230k contacts, 2 m/s | 0.21/0.20 N: NaN at 0.4 s | the sheet now buckles against the table and VBD self-contact explodes even at 60 substeps. Cause found: the particle (self-contact) radius was never scaled with resolution: 2 mm = 50% of the 128 spacing, but 100% of the 256 cell and 3 cells at 768, so any compressed region has every neighbour in permanent contact. `generate_batch.py` now scales the radius with margin and gap (`--radius`). This likely also explains the 768 tearing and the velocity clamp (DATASETS.md above) |
| 4 | as round 3, radius scaled to 1 mm, 30 substeps | NaN at 0.75 s (162k contacts, 1.1 m/s) | NaN at 0.58 s | later and slower than round 3, so the radius matters but is not the root cause; round 5 separates ground contact from gravity stacking |
| 5 | cloth seed 1 as round 4, (a) no table, (b) table with gravity 0 | (a) survives: stretch 6.1 / 0.52, 33 cm high; (b) survives: stretch 6.0 / 0.39, 19 cm high, 139 s | | the ground plane itself is fine; **gravity pressing the crumpled layers onto the table is what sends VBD to NaN**. Also: opposing 0.1-0.5 N pushes already over-stretch the cloth (6x = torn), the same tearing mode as the free datasets; the cloth preset (tri_ke 1e3) is far more stretchable than real cloth |

**Status of the bench profile (2026-10-28 evening):** mechanically complete (table, gravity,
friction, opposing pusher forces, paired materials, 256 default, radius/margin/gap scaled,
QC height/compression diagnostics) but **not yet simulable**: 0 of 6 table+gravity samples
survived. The solver settings for a sheet crumpled against a table (substeps, table contact
stiffness, membrane stiffness, iterations) must come from a sweep on the cluster, not from
single workstation samples: `cloth_results/JOBS.md` matrix M6-0. Until it passes, production
stays on hold; `cluster.env` carries the bench profile as the target.

Per-sample wall time at 256, 30 substeps: 90-120 s when the sheet stays smooth, ~300 s when
it crumples (contact cost); 60 substeps roughly doubles it.

## Rest capture sweep (2026-10-31, workstation, 128 grid, `transfer/rest_sweep.py`)

Question (plan "concurrent-sprouting-gizmo", step 2): does the settled, still-image state of the
paper sheet keep the force information of the release state? Settle machinery and the
literature it follows: `transfer/REPORT.md`, "Capture at rest". Every run is loading as in the
datasets, then a settle phase to a cap of 4 s with a rest criterion (sheet RMS speed < 3 mm/s
for 10 frames). Outputs in WSL `~/cloth/smoke_runs/{rest_sweep128,mag_sweep128,sens128}`
(`sweep.md`, `sweep.json`, renders).

**1. The dataset's own force draws leave no record.** Six of M16-a's test draws (ds256_const
forces, 0.15-0.55 N, uniform-sphere directions, patch rescaled to 128) on the paper preset:

| settle | at rest by 4 s | plane RMS release -> rest [cm] | compaction release -> rest | yielded hinges |
|---|---|---|---|---|
| quasistatic | 0/6 (vrms 0.03 m/s at 4 s) | 4.6 -> 4.6 | 0.92 -> 0.96 | 0 (median; 2 of 6 draws ~55) |
| drag 3/s | 6/6, median 3.8 s | 4.6 -> 0.05 | 0.92 -> 1.00 | 0 |
| drag 10/s | 3/6, 2.8 s | 4.6 -> 0.5 | 0.92 -> 1.00 | 0 |
| drag 30/s | 1/6 | 4.6 -> 2.0 | 0.92 -> 0.99 | 0 |
| cloth, drag 10/s | 4/6, 2.9 s | 4.3 -> 0.6 | 1.02 -> 1.00 | (elastic) |

Four of the six draws never yield a single hinge, the other two ~55 (of 48,641): at these
magnitudes with random directions the paper bends smoothly, and once the free flight is
damped it relaxes flat (render: `renders/render_drag10_sample_slot0029_seed100029.png`). The
quasistatic numbers are not "memory": that mode had not converged (REPORT.md). The two draws
that did yield are the ones with a compressive component between the patches; the two with
fully in-plane but non-opposing forces (0.3-0.5 N) yielded nothing. The ds256_const_paper
dataset (J-006, captured at release) therefore owes its creases largely to the un-scaled
bending stiffness at 256 (REPORT.md caveat: 256 bends ~2x softer) and to being read under
load; with `--bend-scale-res` and a settle it would come out flat.

**2. Compression creases; magnitude barely matters above 0.25 N.** Two geometries (a: the two
patches of smoke slot 0, one near-in-plane pull and one tilted push; b: opposing pushes at
u = 0.2 / 0.8), both patches at the same magnitude, duration 0.5 s, drag 10/s, resolution-
consistent bending:

| yield / damage | geometry | 0.25 N | 0.5 N | 1 N | 1.5 N | 2 N | 2.5 N |
|---|---|---|---|---|---|---|---|
| 50 /m, step 10% | a: plane RMS rest [cm] / sharp hinges / compaction | 3.2 / 175 / 0.83 | 3.5 / 249 / 0.83 | 3.8 / 298 / 0.81 | 4.3 / 290 / 0.81 | 4.6 / 304 / 0.82 | 4.8 / 307 / 0.82 |
| 50 /m, step 10% | b | 1.7 / 289 / 0.97 | 0.7 / 113 / 0.99 | 2.0 / 256 / 0.97 | 2.0 / 413 / 0.97 | 1.7 / 220 / 0.98 | 1.9 / 193 / 0.98 |
| 20 /m, step 10% | a | 4.7 / 1145 / 0.73 | 4.8 / 1161 / 0.75 | 5.1 / 1196 / 0.74 | 5.3 / 1040 / 0.76 | 5.1 / 1107 / 0.75 | 5.8 / 1002 / 0.76 |
| 20 /m, step 10% | b | 3.3 / 1473 / 0.82 | 3.5 / 1267 / 0.83 | 3.7 / 1318 / 0.80 | 3.6 / 1417 / 0.83 | 3.9 / 1791 / 0.79 | 3.7 / 1619 / 0.82 |
| 50 /m, graded gamma 2 | a | 1.9 / 127 / 0.82 | 3.4 / 127 / 0.83 | 5.1 / 127 / 0.85 | 5.8 / 127 / 0.86 | 6.1 / 127 / 0.87 | 6.1 / 127 / 0.88 |
| 50 /m, graded gamma 2 | b | 0.4 / 97 / 1.00 | 0.8 / 89 / 1.00 | 0.7 / 66 / 1.00 | 1.5 / 114 / 0.99 | 0.5 / 21 / 1.00 | 0.6 / 38 / 1.00 |

(sharp = hinges with |rest angle| > 0.5 rad; compaction = mean current / rest distance over
random material pairs, 1 = unfolded; stretch at release 1.6-2.2 in every run and 1.00-1.05 at
rest with yield 50, 1.05-1.14 with yield 20.)

Readings: (i) once the loading compresses the sheet, 0.25 N already creases it (175-290 sharp
hinges) and the rest shape keeps 3-5 cm of out-of-plane deformation at compaction 0.8; the
crease count saturates above ~1 N, so magnitude will be read from crease depth and extent,
not count. (ii) Yield 20 /m keeps nearly all of the release deformation (compaction 0.73-0.83)
with 4-5x the hinges (diffuse creasing, REPORT.md yield sweep) and a residual stretch. (iii)
Graded damage (Narain's 1/(1 + gamma delta), gamma 2) localises everything into one line
(exactly 127 sharp hinges = one row of the grid for geometry a) and lets the opposing-push
sheets relax flat: not adopted; the step softening stays the preset. (iv) The release stretch
of 1.6-2.2 is elastic over-stretch of VBD under load, not tearing (it relaxes to ~1.02), so the
QC gate now applies to the rest grid and the release grid only gets an explosion guard at
twice the limit. (v) Settle time to the rest criterion: 2.2-3.3 s at drag 10/s (10 of 12).

**Decision for the rest dataset** (`cluster/paper.sh`, `PROFILE=rest`): opposing "scrunch"
loading (`generate_batch.py --loading scrunch`: each patch pushes toward the other, +-30 deg
heading jitter, +-35 deg out-of-plane tilt, 0.2-2 N log-uniform), yield 50 /m with the 10%
step, resolution-consistent bending, air drag 10/s, rest criterion vrms < 3 mm/s, cap 4 s.

**End-to-end smoke of the profile** (`PROFILE=rest bash cluster/paper.sh smoke 4 128`): 4/4
pass first try, 0.2-1.8 N draws, rest in 2.5-2.9 s (one at the 4 s cap), rest plane RMS
3.3-4.0 cm, 130-360 sharp hinges, rest stretch 1.02-1.17 with release stretch 1.29-1.78
(renders `smoke_paper_rest_128/*.png` in the session scratchpad). One 256 sample with slot 0's
draws: 285 s wall (270 frames; capture at release is ~80 s), 2,267 yielded hinges, crease
total constant from 1.5 s on while the RMS speed plateaus at ~0.008 m/s (3,000+ self-contacts
creeping under friction), so the 0.003 m/s criterion never fires at 256. Profile defaults
therefore: REST_VMAX 0.01 (fires at ~2.6 s at 256), cap 3 s, i.e. ~220 s per 256 sample,
~190 GPU-hours for 3,000 slots before retries.

**3. Identifiability survives the settle (yield 50), and yield 20 is chaotic.** Finite
differences through the simulator on three creasing loads (geometry a at 1 N and 0.5 N,
geometry b at 1 N; drag 10/s): RMS change of the grid after the best rigid alignment, release
-> rest, per label perturbation of force 1 (floor = magnitude x 1.0001):

| load | floor | magnitude x1.1 | direction +5 deg | position +1 cm | duration x1.05 |
|---|---|---|---|---|---|
| a, 1 N, yield 50 | - | 0.13 -> 0.17 cm | 1.2 -> 1.6 | 0.7 -> 2.7 | 1.0 -> 1.2 |
| b, 1 N, yield 50 | 0.02 -> 0.03 | 1.0 -> 2.1 | 2.3 -> 2.0 | 1.8 -> 3.1 | 1.8 -> 1.6 |
| a, 0.5 N, yield 50 | 0.000 -> 0.000 | 0.21 -> 0.20 | 1.2 -> 1.1 | 0.8 -> 0.6 | 1.0 -> 0.6 |
| a, 1 N, yield 20 | 0.02 -> **0.52** | 0.6 -> 2.0 | 1.9 -> 3.2 | 1.2 -> 2.3 | 0.8 -> 1.9 |
| b, 1 N, yield 20 | 0.7 -> **1.75** | 1.3 -> 3.3 | 1.9 -> 4.1 | 2.2 -> 3.1 | 1.8 -> 2.4 |

With yield 50 every label moves the rest shape at least as much as it moved the release shape
(ratios 0.6-3.8) and 10-100x above the floor: the creased rest state is at least as informative
as the state the current models read, and position / direction / duration are strong signals
(0.6-3 cm per unit step). Magnitude is the weak label at the single-patch geometry (0.13-0.2 cm
per 10%, consistent with the crease count saturating above 1 N) and strong for opposing pushes
(1-2 cm per 10%). With yield 20 a 1e-4 perturbation moves the rest state by 0.5-1.75 cm: the
diffusely creasing sheet is chaotic and its rest shape would not be learnable; yield 50 it is.
