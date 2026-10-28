# Cloth deformation sample pipeline (Newton/Warp prototype)

One end-to-end GPU sample: flat 128x128 cloth sheet + two force vectors
-> deformed state at fixed capture time, saved material-indexed to HDF5.

## 1. Prerequisites (inside WSL2 Ubuntu)

- WSL2 + Ubuntu 20.04 (this prototype; 22.04/24.04 also fine), Windows
  NVIDIA driver >= 545. Do NOT install a Linux display driver in WSL.
- GPU check (must show your NVIDIA card; if it fails, update the Windows
  driver and stop):

```bash
wsl --version            # WSL 2.x, kernel 6.x
nvidia-smi               # GPU model, VRAM, driver, CUDA
nproc; free -h; df -h ~
```

Reference machine: WSL 2.6.1, Ubuntu 20.04.6, 28 threads, 15 GiB RAM,
RTX 4070 Ti SUPER 16 GiB, driver 610.88, CUDA UMD 13.3.

## 2. Environment setup

```bash
mkdir -p ~/cloth && cd ~/cloth
curl -LsSf astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv        # system Python 3.8 is too old
uv pip install --python .venv/bin/python "newton[examples]" \
  h5py matplotlib imageio pillow
```

System Python on Ubuntu 20.04 is 3.8; Newton needs >= 3.10, so `uv`
fetches CPython 3.12 for the venv. No local CUDA toolkit needed
(Warp ships its own; driver provides CUDA).

Verify on CUDA (expect `cuda:0 ... RTX ...`, and ~100 FPS example):

```bash
.venv/bin/python -c 'import warp as wp; wp.init(); print(wp.get_device())'
.venv/bin/python -m newton.examples cloth_hanging --solver vbd \
  --device cuda:0 --viewer null --num-frames 60
```

## 3. Exact package versions

newton==1.7.0, warp-lang==1.18.0, warp-nn==0.4.1, numpy==2.5.4,
h5py==3.16.0, matplotlib==3.11.2, imageio==2.38.1, pillow==13.0.0.
Full freeze (117 packages): `requirements_full.txt`.
Solver: `newton.solvers.SolverVBD` with particle self-contact.

## 4. How to run

```bash
cd ~/cloth
# one sample, default forces (writes sample_seed0.h5 + force JSON is built in)
.venv/bin/python generate_sample.py --seed 0 --out sample_seed0.h5
# custom forces
.venv/bin/python generate_sample.py --seed 1 --force-json my_forces.json --out s1.h5
# validate: renders + maps + animation + self-intersection count
.venv/bin/python validate_sample.py sample_seed0.h5 --outdir val_sample0
# analyses
.venv/bin/python analyze_repro.py --ref A.h5 --rerun B.h5 --sens1 C.h5 --sens2 D.h5
# batch of 32 with random forces, 8 parallel workers
.venv/bin/python generate_batch.py --n 32 --jobs 8 --outdir batch
# dataset: 100 deterministic samples, QC gate + retries, 2 workers
.venv/bin/python generate_batch.py --n 100 --jobs 2 --seed0 2000 \
  --outdir dataset100 --sample-args "--deterministic"
# resume an interrupted dataset run (keeps passing files, skips attempts
# already in dataset100/rejected; per-slot progress in progress.jsonl)
.venv/bin/python generate_batch.py --n 100 --jobs 2 --seed0 2000   --outdir dataset100 --sample-args=--deterministic --resume
# package 8:1:1 splits (test keeps final state only)
.venv/bin/python package_splits.py --indir dataset100 --outdir dataset
```

Note: pass forwarded flags as `--sample-args=--deterministic` (with `=`);
argparse otherwise consumes the flag itself.

Key flags for `generate_sample.py`: `--nx/--ny` (default 128),
`--t-capture` (default: max force duration + `--settle 0.7`),
`--magnitude` (override), `--edge-ke`, `--substeps`, `--iterations`,
`--margin/--gap` (self-contact), `--ramp` (force envelope, default 0.15),
`--deterministic` (bit-exact repeats, ~10x slower), `--device`,
`--material {cloth,paper}` (paper: stiff sheet with plastic creases,
`--yield-curvature`, `--plastic-rate`; also on `generate_batch.py` and
`cluster/submit.py`, see `cluster/README.md` section 3b').

## 5. Layout

- `cloth_sim/sheet.py` — sheet construction, (i,j) material mapping
- `cloth_sim/forces.py` — Gaussian-patch force vectors
- `cloth_sim/simulate.py` — VBD time loop + per-frame log
- `cloth_sim/io.py` — HDF5 save/load
- `cloth_sim/validate.py` — renders, maps, piercing-count metric
- `generate_sample.py`, `generate_batch.py`, `validate_sample.py`,
  `analyze_repro.py`, `make_force_video.py` (dense re-run + GIF with live
  force arrows; see `sample_seed0_forces.gif`)

## 6. HDF5 contents

`final_grid` (128,128,3), `final_vel` (128,128,3),
`intermediate_grids` (K,128,128,3) + `intermediate_times`,
`force_params` (JSON), `sim_config` (JSON), `seed`, `sim_log`
(kinetic_energy, max_speed, soft_contacts per frame + frame_dt),
package versions as attributes. Grid convention: axis0 = v (j rows),
axis1 = u (i cols); flat particle p = j*nx + i.
