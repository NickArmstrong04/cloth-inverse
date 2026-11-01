# CLAUDE.md: cloth-inverse

Recover the forces that crumpled a thin sheet (cloth or paper) from its deformed geometry,
heading toward real sheets seen in one photo. Owner: Nick (GitHub `NickArmstrong04`).
Read `README.md`, `ROADMAP.md`, then `cloth_results/{DATASETS,RESULTS,JOBS}.md` before
proposing work; those three files are the project's memory and must stay current.

## Where this session runs

Two homes for the same repo:

- **Bouchet (Yale YCRC, Slurm).** Repo `/nfs/roberts/scratch/pi_sz622/naa79/cloth/cloth-inverse`,
  datasets `/nfs/roberts/scratch/pi_sz622/naa79/cloth/cloth_data/<name>` (`DATA_DIR` in
  `cloth_data_generation/transfer/cluster/cluster.env`), model runs
  `cloth_data_generation/inverse/runs/<id>/` (`model.pt`, `metrics.json`). Simulator venv:
  `cloth_data_generation/transfer/.venv` (`bash cluster/setup_env.sh`; `--with-torch` for
  training). Policy: Python runs inside an allocation, never on the login node
  (`bash cluster/interactive.sh [gpu]`, or `sbatch cluster/util.sbatch <cmd>` for one-shot
  work that must not be cut off). Node `a1125u05n01` has a bad GPU (excluded in cluster.env).
  `scavenge_gpu` is preemptible but safe: array tasks resume. If you are the session on
  Bouchet you can run `squeue`, `sacct`, `sbatch` and read logs yourself.
  Layout since 2026-10-31: repo and data both live under `.../naa79/cloth/` (were siblings
  directly under `/home/naa79/scratch_pi_sz622/naa79/`); archived `gen_*.sbatch` and
  `submission.json` under `cloth_data/*/slurm/` still show the old paths (history, not
  read on resume). Slurm logs live in `transfer/logs/<date>/`. Claude sessions run in a
  container that does not mount `~/.local/share/uv`, so the venv python only works from a
  normal shell or job on the cluster.
- **Nick's Windows workstation.** Repo at `C:\Users\nicka\Desktop\projects\meta`; the
  simulator and models run in WSL at `~/cloth` (a copy of `transfer/` with its own `.venv`,
  RTX 4070 Ti SUPER); SAM 3 / VGGT live in `~/sam3env`. That session has no cluster access
  (Duo MFA), so it commits + pushes and Nick runs commands on Bouchet and pastes output.

Branch of record: `photo-render` (ahead of `main`). Commit messages: short, factual, say
what was verified and how. Ask before anything destructive. Never commit HDF5 samples,
checkpoints or `runs/`.

## How the work is organised

- **Datasets are made by Slurm job arrays** (`cluster/submit.py`, `cluster/gen_array.sbatch`):
  global slots and seeds (`seed = SEED0 + slot`), QC gate with retries, re-submitting the same
  command resumes (only tasks with missing slots run). Paper datasets go through
  `cluster/paper.sh {smoke|devel|full|status|merge|compare}` with env overrides
  (`PROFILE`, `N`, `NX`, `TIME`, ...). Then `merge` packages 8:1:1 splits to
  `<name>_splits`.
- **Training** is one GPU per config: `sbatch cluster/run.sbatch <splits> <train.py flags>
  --out runs/<id>`; results in `metrics.json`; `cluster/summarize_runs.py`, `cluster/jobs.sh`
  for status. Nick wants several experiments queued in parallel, not one at a time.
- **Request only what a job needs, especially wall time** (Nick, 2026-10-31): a test or check
  that should take ~5 min gets `--time=00:05:00`, not a template's 1-2 h default, so it
  backfills and starts sooner. Size production `TIME`/`MEM` from measured sacct
  Elapsed/MaxRSS plus modest headroom.
- **Every cluster job gets a row in `cloth_results/JOBS.md`** (id, Slurm id, status, why,
  command, result), grouped in matrices (M1 ... M20, J-00x for datasets). Update the status
  column from `squeue`/`jobs.sh` output. Findings go to `RESULTS.md` (models) and
  `DATASETS.md` (data, physics sweeps); simulator-side physics notes to
  `cloth_data_generation/transfer/REPORT.md`.
- Local smoke tests: 1 to 10 samples. Volume is the cluster's job.
- Perception steps: Meta tools first (SAM 3, VGGT, SAM 3D, DINOv3).

## Physics and data model (state on 2026-10-31)

- Sheet 0.5 m square, 256x256 grid in the datasets (128 for smoke), free-floating in zero
  gravity, Newton 1.7.0 VBD solver with self-contact (friction 0.5). Two Gaussian force
  patches in material coordinates, each a constant force for its duration (0.3-0.7 s).
  Labels per force: `(u, v, direction, magnitude, duration)`; the model predicts both forces
  as an unordered pair. Materials: `cloth` (elastic) and `paper` (plastic hinges: yield 50 /m,
  edge_ke 200, yielded hinges softened to 10%; creases are permanent).
- **Capture-at-release datasets** (`ds256_const`, `ds256_const_paper` = J-005/J-006): the
  sample is the state at the instant the force ends, so most of the impulse is still velocity.
  Best model there, M16-a (hybrid input + velocities): 0.9 cm / 4.8 deg / 0.16 rel. force.
  The VBD solver delivers only ~8% of the nominal impulse (velocity clamp); labels are nominal.
- **Decision 2026-10-31: the deployment input is one still image, so velocities are out of
  the model input** (`--no-vel`, JOBS.md M19) **and the photo-track dataset is paper settled
  to rest** (JOBS.md M20, `cluster/paper.sh PROFILE=rest` -> `ds256_paper_rest`): opposing
  "scrunch" pushes 0.2-2 N (`generate_batch.py --loading scrunch`; free-direction loads at the
  old magnitudes leave no creases and the sheet relaxes flat), air drag 10/s after release,
  stop when the sheet's RMS speed < 0.01 m/s (cap 3 s), release state kept as `release_grid`
  next to the rest state `final_grid`, resolution-consistent bending (`--bend-scale-res`).
  ~220 s per 256 sample, so `PROFILE=rest` submits 5 h array tasks. Rationale and sweeps:
  `transfer/REPORT.md` "Capture at rest", `DATASETS.md` "Rest capture sweep". Known
  rejects: graded crease damage (one crease line, sheet relaxes flat), yield 20 /m (chaotic),
  undamped free flight (yields the creases back).
- Loader notes: `train.py --grid final|release`, `--no-vel --hybrid` is the photo-track recipe;
  `eval_grid.py` scores a checkpoint on grids built outside `train.py` (photo chain).

## Photo chain (branch photo-render, `cloth_data_generation/inverse/`)

`photo.py` renders a sample (coded print available), `segment.py` (SAM 3), `lift_stereo.py`
(VGGT on a 20 deg pair + known camera motion -> 1.4 cm metric cloud), `decode_print.py`
(per-vertex correspondences), `grid_from_views.py` (world-frame grid, harmonic fill of hidden
vertices), `degrade_split.py` (photo-like copies of a split for training), `eval_views.py`,
`eval_grid.py`. Status and numbers: `RESULTS.md` sections "Rung 4b" and
`cloth_results/photos/README.md`.

## Conventions when writing for Nick

Lead with the number or outcome, state what was verified and how, give pasteable command
blocks with absolute paths, and record every decision with its date in the markdown files
above rather than only in chat.
