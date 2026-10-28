# Cloth data generation on Bouchet (Slurm job arrays)

Generates N independent samples as a Slurm job array: each task simulates
`CHUNK` samples on one GPU, one HDF5 per sample, with the same QC gate and
retry logic as the workstation pipeline. A chained CPU job merges the
per-task summaries and packages 8:1:1 train/val/test splits.

Slots and seeds are global (`seed = SEED0 + slot`), so the dataset does not
depend on how the array is cut, and re-submitting resumes: finished slots
are skipped, rejected attempts are not re-simulated.

## 1. Get the code onto the cluster

```bash
# first time
cd ~/project
git clone https://github.com/NickArmstrong04/cloth-inverse.git
ln -s ~/project/cloth-inverse/cloth_data_generation/transfer ~/project/cloth
# later updates
cd ~/project/cloth-inverse && git pull
```

`~/project/cloth` is the CODE_DIR the scripts auto-detect (the `.venv` and
`.warp_cache` created inside it are git-ignored).

## 2. One-time environment setup (on the login node)

```bash
cd ~/project/cloth
bash cluster/setup_env.sh
```

Installs `uv`, creates `.venv` (Python 3.12) with the pinned stack
(newton 1.7.0, warp-lang 1.18.0, h5py, ...). No CUDA module needed: Warp
ships its CUDA runtime; GPU nodes provide the driver.

Then check the Slurm names in `cluster/cluster.env` against the cluster:

```bash
sinfo -o "%P %a %l %G %D" | sort -u      # partition names, GPU gres, limits
sacctmgr show assoc user=$USER format=account,partition -P   # your accounts
```

Edit `PARTITION`, `CPU_PARTITION`, `ACCOUNT`, `TIME` if they differ.

## 3. Smoke test, then the real run

```bash
# tiny: 4 samples at 128x128, 2 tasks x 2, short time limit
python3 cluster/submit.py --n 4 --nx 128 --chunk 2 --name smoke128 --time 00:30:00
python3 cluster/submit.py --status smoke128          # progress, queue, missing slots
cat ~/scratch/cloth_data/smoke128/slurm/gen_*_0.out  # task log

# preview the real thing without submitting
python3 cluster/submit.py --n 1000 --dry-run

# 1000 samples at 256x256: 40 tasks x 25 samples, <=16 running at once
python3 cluster/submit.py --n 1000 --name ds256_1000
```

Useful flags: `--nx`, `--chunk`, `--seed0`, `--time`, `--mem`,
`--max-concurrent`, `--sample-args "--force-every-substep --deterministic"`,
`--tasks 3,7-9` (re-run specific tasks), `--all-tasks`, `--no-merge`.

Re-running `submit.py` with the same `--name` only submits tasks whose slots
are incomplete (timeouts, node failures). It refuses to mix settings: a name
is bound to its `seed0`, `nx`, `chunk` and sample args.

## 4. Outputs

```
~/scratch/cloth_data/ds256_1000/
  sample_slot00000_seed100000.h5 ...   passing samples (QC gate: no NaN,
  forces_slot*_seed*.json              stretch<=2, no gross piercing)
  rejected/                            failed attempts (kept for inspection)
  batch_summary_00000.json ...         one per task
  batch_summary.json                   merged (input to package_splits.py)
  dataset_summary.json                 QC stats, retries, GPU hours, missing slots
  progress_*.jsonl                     per-task incremental log
  slurm/                               rendered sbatch files, submission.json, logs
~/scratch/cloth_data/ds256_1000_splits/{train,val,test}/ + dataset_manifest.json
```

Scratch is purged after 60 days. When a dataset is final:

```bash
rsync -a ~/scratch/cloth_data/ds256_1000_splits ~/project/cloth_datasets/
```

## 5. What changes with resolution

`generate_batch.py --nx 256` keeps the physics comparable to the 128 reference:
the Gaussian force patch stays the same physical size (sigma 3 -> 6 cells,
cutoff 9 -> 18 cells) and the self-contact margin/gap shrink with the cell
(1.0/0.5 mm -> 0.5/0.25 mm, i.e. the same 25% of particle spacing that was
tuned at 128). `--no-res-scaling` disables this. Expect roughly 4x the
particles and several times the wall time per sample versus 128; the smoke
test tells you the real per-sample time on the cluster GPUs, then size
`CHUNK` so a task finishes well inside `TIME`.

## 6. Physics caveats carried over from the workstation study (2026-10-27)

- `--force-every-substep` (default in `cluster.env`) fixes a buffer bug that
  delivered only half the force impulse. Keep it on for new datasets.
- The VBD solver at these mass/stiffness settings transfers only ~7% of the
  nominal impulse at the dataset's force magnitudes (80% at 20x smaller
  forces) and does not conserve momentum. Force magnitude and position were
  not identifiable by the first inverse models on the 100-sample set. If the
  goal is inferring force vectors, consider validating a momentum-conserving
  configuration (solver choice, dt, mass-to-stiffness ratio) before spending
  the full GPU budget. `sim_log/momentum` is now saved in every HDF5 so this
  can be checked on the smoke test: compare `|sum(mass*vel)|` at the force
  end time with `|magnitude*duration*0.85|` summed over the two forces.

## 7. Monitoring and troubleshooting

```bash
python3 cluster/submit.py --status ds256_1000
squeue -u $USER
sacct -j <ARRAY_JOB_ID> --format=JobID,State,Elapsed,MaxRSS,ExitCode | head
tail -n 20 ~/scratch/cloth_data/ds256_1000/slurm/gen_<JOBID>_<TASK>.err
```

- `CUDA driver` / `no device` errors in a task: the partition has no GPU or
  `--gpus` was not honoured; check `nvidia-smi` output at the top of the log.
- Out-of-memory (host): raise `MEM` (the piercing QC check at 256x256 is
  CPU/RAM heavy).
- Tasks time out: lower `CHUNK` or raise `TIME`; re-submit to resume.
