# cloth-inverse

Recovering the forces that deformed a thin sheet (cloth or paper) from its deformed
geometry, with the aim of doing it for real sheets observed with a camera.

The story, in three stages (details and status in `ROADMAP.md`):

1. **Virtual.** Simulate a sheet on a table pushed by two pusher-like forces (NVIDIA
   Newton / Warp, VBD solver), train an inverse model to recover where the pushes were
   applied, in which direction and with what impulse, and verify by re-simulation.
   Privileges are removed one rung at a time (the observation ladder) to measure what
   a camera-based observation will cost.
2. **Real.** Segment a real cloth/paper sheet from a photo (SAM or similar), impute its
   3D state, run the inverse model, and state the forces that would undo the deformation.
3. **Harder system.** Same pipeline on a system where loading history must be
   recovered from deformed geometry (FEM meshes / point clouds, contact-impulse labels).

Layout:

- `cloth_data_generation/transfer/` — simulator pipeline: `generate_sample.py`,
  `generate_batch.py` (QC gate, retries, resume, job-array aware, `--loading`,
  `--table`), `cloth_sim/`, `package_splits.py`, `tools/`. See its `README.md`.
- `cloth_data_generation/transfer/cluster/` — Slurm job arrays for data generation and
  single-GPU training jobs on Bouchet (`submit.py`, `run.sbatch`, `ladder.sbatch`,
  `summarize_runs.py`, `watchq.sh`). See its `README.md`.
- `cloth_data_generation/inverse/` — inverse models (ridge baseline, grid CNNs, point
  transformer), training with symmetry-aware losses, re-simulation evaluation.
- `cloth_results/` — `DATASETS.md` (what was generated), `RESULTS.md` (what the models
  achieve), `JOBS.md` (every cluster job and matrix), reports. HDF5 samples are not
  tracked.

Cluster quick start (Bouchet, everything under the PI scratch):

```bash
cd /home/naa79/scratch_pi_sz622/naa79
git clone https://github.com/NickArmstrong04/cloth-inverse.git
cd cloth-inverse/cloth_data_generation/transfer
bash cluster/interactive.sh          # Python runs inside an allocation, not on the login node
bash cluster/setup_env.sh --with-torch
python3 cluster/submit.py --n 4 --chunk 2 --name smoke256_bench --time 00:30:00 --partition gpu_devel
python3 cluster/submit.py --status smoke256_bench
```
