# cloth-inverse

Data generation and inverse modelling for a cloth sheet scrunched by two
force vectors (NVIDIA Newton / Warp VBD simulator).

- `cloth_data_generation/transfer/` — simulator pipeline: `generate_sample.py`,
  `generate_batch.py` (QC gate, retries, resume, job-array aware), `cloth_sim/`,
  `package_splits.py`. See its `README.md` and `REPORT.md`.
- `cloth_data_generation/transfer/cluster/` — Slurm job-array submission for
  Bouchet (`submit.py`, `setup_env.sh`, templates). See its `README.md`.
- `cloth_data_generation/inverse/` — first inverse models (baselines, CNNs),
  training and re-simulation evaluation.
- `cloth_results/` — reports and metrics from the workstation runs (HDF5
  samples are not tracked; see `cloth_results/inverse_v0/` for metrics).

Cluster quick start:

```bash
cd ~/project && git clone https://github.com/NickArmstrong04/cloth-inverse.git
ln -s ~/project/cloth-inverse/cloth_data_generation/transfer ~/project/cloth
cd ~/project/cloth && bash cluster/setup_env.sh
python3 cluster/submit.py --n 4 --nx 128 --chunk 2 --name smoke128 --time 00:30:00
```
