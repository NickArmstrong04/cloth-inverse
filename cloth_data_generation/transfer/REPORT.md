# Cloth-first prototype: report

## Environment (measured inside WSL)

- WSL 2.6.1.0, kernel 6.6.87.2-1, distro Ubuntu-20.04 (20.04.6 LTS).
  Compromise: task asked for 22.04/24.04; this machine has 20.04 whose
  system Python 3.8 is too old for Newton (>= 3.10). Workaround: `uv`
  fetched CPython 3.12.16 for the venv. No other issue encountered.
- CPU: 28 threads (i7-14700K). RAM 15 GiB (14 free). Disk 1007 GiB
  (918 free). No Linux NVIDIA driver installed (correct: GPU comes from
  the Windows driver).
- GPU (from `nvidia-smi` in WSL): NVIDIA GeForce RTX 4070 Ti SUPER,
  16376 MiB, driver 610.88, CUDA UMD 13.3. Warp sees
  `cuda:0 sm_89, 16 GiB, mempool enabled`. No `nvcc` (not needed).

## Simulator

- newton==1.7.0 on warp-lang==1.18.0 (full freeze: requirements_full.txt).
  Solver: `newton.solvers.SolverVBD` (vertex-block dynamics) with
  `particle_enable_self_contact=True`. Official example
  `newton.examples cloth_hanging --solver vbd --device cuda:0 --viewer null`
  runs on CUDA: 99.3 FPS at 64x32 (297 frames / 2.99 s benchmark).
- APIs verified against installed sources, not memory: `warp.sim` is gone
  (Newton replaces it); VBD requires `builder.color()` before `finalize()`;
  VBD detects self-contact internally (external `collide()` only feeds
  particle-shape contacts, whose counter stays 0 here).

## Sheet and parameters

- 0.5 x 0.5 m flat sheet, 128x128 particles (dim 127x127, cell 3.937 mm),
  triangulated by `add_cloth_grid`, centered at origin in z=0 plane.
  No fixed particles, no ground, gravity [0,0,0]. Material map:
  flat index p = j*nx + i; saved grids are (v=j rows, u=i cols, xyz).
- Mass per area 0.2 kg/m^2 (light shirting) -> 3.05e-6 kg/particle.
- VBD stiffness from Newton's cloth example: tri_ke=tri_ka=1e3,
  tri_kd=1e2, edge_ke=1.0 (example used 10 at 10 cm cells; 1.0 at 4 mm
  cells permits wrinkling without instability), edge_kd=0.
- Self-contact margin 1.0 mm (~25% spacing), gap 0.5 mm, buffers 64/128.
  Tuning: 0.8 mm tunnels under violent local acceleration; 4 mm
  (50% spacing) makes contact response explode the sheet 2x.
- Time stepping: 60 fps frames, 30 substeps -> dt = 1/1800 s = 0.556 ms,
  20 VBD iterations. Stable: no NaNs in any run.

## Forces

- Two patches, Gaussian weights sigma=3 cells, cutoff 9 cells, normalized
  so the vector sum equals magnitude x direction exactly. Material-indexed.
- First sample: F1 (u,v)=(0.12,0.35) dir (+x,+z tilt) 1.0 N / 0.5 s;
  F2 (0.88,0.65) dir (-x,-z tilt) 1.0 N / 0.5 s. Opposing z seeds a
  twist/S-fold. Smoothstep ramp on/off over 15% of duration (avoids jerk).
- Capture: T = max duration + 0.7 s settle = 1.2 s (lets force-shutoff
  transients decay while folds persist), plus 5 evenly spaced intermediates.

## First sample (sample_seed0.h5)

- Twisted roll with wrinkles, max|z| = 0.26 m, stretch ~1.0 (peak ~1.15
  at patch, no tearing), KE 4.5e-3 J at capture, 995 self-contact pairs.
- Self-intersections at capture: 0 piercing pairs / 18230 candidates
  (edge-piercing test in float64; unit-tested on synthetic crossing /
  touch / coplanar / stepped cases).
- Wall time ~18 s (23 s incl. one-time kernel compile), GPU 3.20 -> 3.55 GB.

## Reproducibility and sensitivity

- Same seed twice (default mode): max 1.6 cm, mean 2.2 mm per-point diff.
  GPU atomic nondeterminism is real and comparable to small force changes.
- Deterministic mode (`--deterministic`, RUN_TO_RUN): two runs bit-exact
  (0.0), at ~10x cost (167-179 s, 5.8 GB). Recommended for dataset labels.
- +1% magnitude: F1 -> mean 5.7 mm (max 1.3 cm); F2 -> mean 3.2 cm
  (max 12.6 cm). Folding amplifies perturbations asymmetrically; F2 sits
  nearer a fold-branch point.

## Batch feasibility

- Newton supports multi-world batching (`ModelBuilder.replicate()`), but
  per-world force routing and per-world contact logging were not verified
  in this budget, so the batch uses 8 parallel single-sample processes
  (instances trivially non-interacting), one HDF5 per sample.
- 32 samples, seeds 1000-1031, random forces (u1 in [0.05,0.3],
  u2 in [0.7,0.95], v in [0.2,0.8], |F| log-uniform 0.5-2 N,
  duration 0.3-0.7 s, T = duration + 0.7 s): 32/32 ok, no NaNs,
  total wall 577.4 s -> 18.0 s per sample at 8 jobs. GPU peaked ~6.1 GB.
  Caveat: per-worker sample time under 8-way contention was 120-155 s
  vs ~18 s solo, so 8-way parallelism gave no net speedup over
  sequential on this single GPU (577 s ~= 32 x 18 s). For real dataset
  generation use 1-4 concurrent workers or sequential; the one-file-per-
  sample layout already shards trivially across GPUs/machines.
- Per-sample stats across batch: frames 64-84 (duration-dependent
  capture), contacts_max up to 117k, all nan=None. One HDF5 per sample
  (+ per-sample force JSON), batch_summary.json manifest.

## Compromises (nothing silent)

1. Ubuntu 20.04 instead of 22.04/24.04 (uv Python 3.12 workaround).
2. edge_ke=1.0, not example 10 (resolution-dependent bending feel).
3. Force ramp envelope added (not in spec; physical, documented).
4. Contact log = VBD internal self-contact VT+EE pair counts
   (pipeline soft_contact_count is always 0 without rigid shapes).
5. Intersection metric counts true edge piercings only (resting contact
   excluded); whole-triangle containment would be missed (negligible).
6. No gravity, no ground, self-collision ON, external forces only --
   problem definition unchanged.

## Paper sheet: plastic creasing (2026-10-28)

Goal: the same two-force scenario on a paper-like sheet whose response is
not fully reversible. `--material paper` (`cloth_sim/sheet.py` `MATERIALS`):
mass 0.08 kg/m^2, tri_ke = tri_ka = 1e4 (10x cloth), edge_ke 50 (5x), plus
plastic bending hinges (`cloth_sim/plasticity.py`). After every substep, any
interior edge bent more than a yield angle away from its rest angle drags the
rest angle along (perfect plasticity, rate 1). VBD reads
`model.edge_rest_angle` on every solve, so the solver itself is untouched.
Yield is a curvature (default 50 /m, i.e. 2 cm radius) converted to an angle
with the cell spacing, so creases yield at the same load at every resolution.
Overhead is a single edge kernel per substep (19.5 s vs 19.3 s per sample at 128).

Yield sweep, default forces, 128x128, 3 s settle (`--force-every-substep`):

| yield [1/m] | edges yielded | creases > 0.5 rad | crease share in top 5% edges | compaction at force end -> t=3.5 s |
|---|---|---|---|---|
| cloth (elastic) | - | - | - | 0.90 -> 1.01 |
| 0 (elastic paper) | 0 | 0 | - | 0.88 -> 1.00 |
| 20 | 45% | 31 | 0.37 (diffuse) | 0.82 -> 0.96 |
| **50 (default)** | 12% | 100 | 0.75 (localized lines) | 0.89 -> 0.95 |
| 100 | 0.5% | 0 | - | 0.88 -> 1.00 |

Compaction = mean over random material pairs of current / rest distance
(1 = unfolded). Elastic sheets spring back to flat within ~0.6 s of the forces
ending; creased paper stays folded. Paper stretch stays ~1.04-1.07 (cloth
1.65), so the paper QC stretch limit is 1.3 instead of 2.0.

Crease localization (2026-10-30, constant-force model, capture at release, 256,
dataset slots 0 and 3 of ds256_const_paper): with the preset above only 2-6% of
hinges yield and 95% of them by less than 0.4 rad, i.e. the sheet bends smoothly
and shows no crease. Perfect plasticity alone does not localize: an unyielded
hinge is as stiff as a yielded one, so curvature spreads. `plastic_soften` drops a
yielded hinge to that fraction of `edge_ke` (VBD reads `edge_bending_properties`
every solve, so it is a per-edge array update in the same kernel as the rest
angle). Sweep (yield 50 /m):

| edge_ke | soften | hinges > 0.5 rad (slot 0 / 3) | p95 yielded angle | stretch_max |
|---|---|---|---|---|
| 50 | 1 (none) | 71 / 136 | 0.22 / 0.41 | 1.29 / 1.07 |
| 50 | 0.1 | 1930 / 986 | 1.46 / 2.26 | 1.31 / **4.01** (torn) |
| **200** | **0.1** | 655 / 441 | 1.44 / 2.73 | 1.33 / 1.13 |
| 200 | 0.02 | 799 / 621 | 1.91 / 2.76 | **2.02** / 1.13 |
| 500 | 0.05 | 580 / 482 | 1.82 / 2.60 | 1.36 / **3.61** |

edge_ke 200 with soften 0.1 is the preset now (slots 1, 2: 175 / 193 hinges past
0.5 rad, stretch 1.31 / 1.19). Weaker sheets or stronger softening tear at the
crease. Healthy paper under load reads stretch 1.1-1.35 at the force patches, so
the paper QC gate is 1.5 (was 1.3, set on settled sheets). Known artifact: on the
regular grid the creases run along the mesh axes (crease maps in
`cloth_results/paper_crease_maps_sweep.png`); diagonal folds would need an
irregular or diagonal-edge mesh.

Caveats: (1) the momentum/velocity-cap issue of the cloth runs applies
unchanged (peak speed 0.67 m/s at 256), so force magnitude is still weakly
identifiable; (2) material constants are VBD units tuned for crease
behaviour, not measured paper; (3) the end state still rings during the
settle (compaction oscillates by ~0.03), so the capture time matters more
than for cloth; (4) rest angles are clamped to +-2.8 rad because VBD does not
wrap the atan2 branch cut, so a crease folded completely flat is
slightly under-represented.

Dataset-distribution batch (`generate_batch.py --material paper`, random
forces, seeds 5000+): 128x128 8/8 passed first try (~60 s each single, ~20 s
amortised at 4 workers); 256x256 4/4 slots passed, with 1 retry (stretch 1.57,
torn), 75-84 s of simulation each, plus the CPU piercing QC. Stretch 1.02-1.08,
peak speed 1.4-1.6 m/s (128) / 0.66-0.73 m/s (256). At capture time paper
compaction is 0.77-0.94 with 125-977 sharp (>0.5 rad) crease edges per sample
at 128. At 256 there are more creased edges (233-1289) because the existing
pipeline does not scale edge_ke with resolution, so bending is effectively
softer at higher resolution; this affects cloth equally.

## Capture at rest: the still-image state (2026-10-31)

Why: every `ds256_const*` sample is captured at the instant the force is released, when
most of the impulse is still velocity. A photo cannot show velocity, and the best model's
0.9 cm rests on it (3.5 cm without, photos README Phase A). The crumpling literature never
reads a mid-motion state: Andrejevic & Rycroft (2021) solve the crumpled sheet
quasistatically (0 = F, implicit DAE) and switch to dynamics only for snap-throughs;
Ren et al. (2018) / Chen et al. (2019), the closest inverse problem to ours (load location,
magnitude and duration from the permanent deformation of a beam / shell), run the dynamics
for 50-100 s after a 10-30 ms impact so the network sees the "final stable deformation".
What survives at rest is the plastic record: for an elastic sheet it is whatever friction
locks in, for paper it is the creases (Gottesman et al. 2018: the crease network is a state
variable set by the deepest compaction, not the history).

What changed (loading phase untouched, so release states stay comparable):

| knob | where | meaning |
|---|---|---|
| `--settle T` | existing | now the cap of the settle phase |
| `--settle-mode dynamic|quasistatic` | `SimConfig.settle_mode` | quasistatic = velocities zeroed before every frame after the last force ends (overdamped relaxation) |
| `--drag k` | `SimConfig.drag` | isotropic air drag in free flight, v *= exp(-k dt) per frame (Andrejevic's b_iso) |
| `--rest-vmax v --rest-metric vrms|vmax --rest-frames n` | `SimConfig.rest_*` | stop once the sheet's RMS speed sqrt(2 KE / M) (or the fastest particle) stays below v for n frames; `t_rest` recorded |
| `release_grid` / `release_vel` | h5 | the state at the end of the force window, stored next to `final_grid` (the rest state) |
| `--self-mu` | `SheetConfig.self_contact_mu` | self-contact friction; Newton's VBD already applied `model.soft_contact_mu` = 0.5 to self-contact, now recorded in the meta (`self_contact_mu`) |
| `--bend-scale-res` | `SheetConfig.bend_scale_resolution` | edge_ke x (128-grid cell / cell): VBD's hinge energy is 0.5 k L (theta - rest)^2 while a plate's is 0.5 B L / h_mean (theta - rest)^2 (discrete shells), so k stands for B / h and must scale with 1 / cell; fixes the "256 is softer" caveat above |
| `--damage-gamma g` | `SheetConfig.plastic_damage_gamma` | Narain et al. 2013 graded weakening ke / (1 + g delta), delta += excess / yield per yield event, instead of the 10% step at first yield |
| QC | `validate.qc_sample_h5` | stretch gate on max(release, rest): a tear at release relaxes at rest and would pass otherwise |

Snapshots stay at fifths of the force window (the fifth one is the release) and the
capture is appended, so loaders that take the last snapshot as the capture still work.
The default path (`--settle 0`) is unchanged and bit-exact run to run (slot 0 of
`smoke_paper_128` regenerated twice: max |dq| = 0); the stored smoke files differ from a
regeneration only because they were made with the older preset (edge_ke 50, no softening).

Smoke, slot 0 of `smoke_paper_128` (128, paper preset, settle cap 3 s, 21 s per run vs 3 s
at release):

| settle | vrms at 3.5 s [m/s] | plane RMS release -> rest [cm] | stretch release -> rest | crease total release -> rest [rad] |
|---|---|---|---|---|
| dynamic, no drag | 0.53 (still ringing) | 5.1 -> 1.5 | 1.56 -> 1.02 | 215 -> 298 -> 58 |
| dynamic, drag 3/s | 0.002 | 5.1 -> 4.0 | 1.56 -> 1.00 | 215 -> 282 -> 150 |
| quasistatic | 0.010 | 5.1 -> 5.2 | 1.56 -> 1.05 | 215 -> 226 |
| quasistatic, gamma 0.5 | 0.019 | 4.6 -> 4.2 | 1.52 -> 1.01 | 48 -> 47 |
| cloth, quasistatic | 0.026 | 4.1 -> 5.0 | 2.33 -> 1.14 | (elastic) |

Readings: (1) **free flight erases the crease memory**: with perfect plasticity and no
hardening the ringing sheet folds further (215 -> 298 rad) and then yields back as it
unfolds (-> 58 rad); drag 3/s loses a third, the quasistatic settle keeps everything. A
settle must be damped, and the capture-at-release choice of the current datasets was the
right one for an undamped solver. (2) The membrane stretch relaxes to ~1.0 at rest (paper
is near-inextensible; the 1.1-1.5 at release is VBD under-convergence under load, as the
VBD paper warns for stiff materials with few iterations), while the out-of-plane
deformation (plane RMS 4-5 cm) stays: the rest state is the crease pattern, as it should be.
(3) The quasistatic mode converges only algebraically (vrms 0.16 -> 0.01 m/s over 3 s):
velocities are zeroed per frame but build again over the 30 substeps, so it is a creeping
relaxation; drag 3/s reaches vrms 0.003 at 3.3 s. The rest criterion therefore reads vrms
(energy-weighted; vmax is held up by single contact flickers) and the drag is swept
(3, 10, 30/s) against quasistatic in `rest_sweep.py`. (4) Graded damage at gamma 0.5 makes
far fewer and weaker creases than the 10% step (47 vs 226 rad): the step is a much
stronger weakening; gamma needs its own sweep (2 is in `rest_sweep.py`) before it
replaces the preset. (5) Elastic cloth under a damped settle does not flatten within 3 s
either (plane RMS 5 cm): self-contact friction 0.5 and the folds' local minima hold it; the
sensitivity test decides whether that shape still encodes the force.

`rest_sweep.py` (step 2 of the plan) runs these conditions on real force draws and adds
the model-free identifiability test: finite differences of the rest grid and the release
grid to each label of force 1 (magnitude x1.1, direction +5 deg, position +1 cm, duration
x1.05), ratio rest / release = the fraction of the shape's sensitivity that survives the
settle. Results: `cloth_results/DATASETS.md`, "Rest capture sweep".

Sweep on real force draws and a magnitude ladder (DATASETS.md, "Rest capture sweep"): with
the datasets' uniform-sphere directions at 0.15-0.55 N the paper does not crease at all (4 of
6 draws: zero yielded hinges) and relaxes flat once damped; what creases paper is in-plane
compression between the two patches, and then 0.25 N is already enough (175-290 sharp
hinges, 3-5 cm of rest deformation, compaction 0.8). Hence `generate_batch.py --loading
scrunch` (opposing pushes with +-35 deg tilt, 0.2-2 N) for the rest dataset, drag 10/s
(rest in 2.2-3.3 s), yield 50 /m with the step softening (graded damage at gamma 2
localised into a single grid line and let the sheet relax flat), and the QC stretch gate on
the rest grid (the release stretch of 1.6-2.2 is VBD's elastic over-stretch under load, it
relaxes to ~1.02; the release grid keeps an explosion guard at twice the limit).
