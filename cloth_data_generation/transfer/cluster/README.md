# Cloth data generation on Bouchet (Slurm job arrays)

Generates N independent samples as a Slurm job array: each task simulates
`CHUNK` samples on one GPU, one HDF5 per sample, with the same QC gate and
retry logic as the workstation pipeline. A chained CPU job merges the
per-task summaries and packages 8:1:1 train/val/test splits.

Slots and seeds are global (`seed = SEED0 + slot`), so the dataset does not
depend on how the array is cut, and re-submitting resumes: finished slots
are skipped, rejected attempts are not re-simulated.

## 1. Get the code onto the cluster

Everything lives under `/home/naa79/scratch_pi_sz622/naa79`:

```bash
cd /home/naa79/scratch_pi_sz622/naa79
git clone https://github.com/NickArmstrong04/cloth-inverse.git
cd cloth-inverse/cloth_data_generation/transfer        # this is CODE_DIR
# later updates
cd /home/naa79/scratch_pi_sz622/naa79/cloth-inverse && git pull
```

Datasets go to `/home/naa79/scratch_pi_sz622/naa79/cloth_data/<name>/`
(`DATA_DIR` in `cluster.env`). The `.venv` and `.warp_cache` created inside
CODE_DIR are git-ignored.

## 2. One-time environment setup

Bouchet policy: run Python inside an allocation, not on the login node.
Housekeeping: `bash cluster/tidy.sh` (dry run) / `--apply` moves finished training logs to
`logs/<date>/`, deletes failed-attempt logs, warning-only `.err` files, generation task logs of
finished arrays, crashed run dirs and `__pycache__`; never touches queued/running jobs or data.
One-shot steps (repackaging splits, `check_splits.py`, model-weight downloads)
go through `sbatch cluster/util.sbatch <command...>` so they cannot be cut off
when an interactive allocation ends; use `salloc` only for quick manual tries.
`sbatch`, `squeue`, `sacct` and `sbatch --test-only` all work from inside an
allocation, so every command below is run from one:

```bash
bash cluster/interactive.sh          # CPU allocation (devel, 2 h) for setup / submit / probe / report
bash cluster/interactive.sh gpu      # 1 GPU on gpu_devel, to try generate_sample.py by hand
```

Then:

```bash
cd /home/naa79/scratch_pi_sz622/naa79/cloth-inverse/cloth_data_generation/transfer
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

Current Bouchet GPU partitions: `gpu` (a40/a5000/l40s/rtx_5000_ada, default
here), `gpu_h100`, `gpu_h200`, `gpu_devel` (smoke tests), `scavenge_gpu`
(preemptible). CPU: `day` (24 h) for the merge job. Override per run with
`--partition gpu_h100`, or edit `cluster.env`.

## 3. Smoke test, then the real run

(Inside `bash cluster/interactive.sh`; `submit.py` only renders scripts and
calls `sbatch`, the simulations themselves run as batch jobs.)

```bash
# tiny: 4 samples at 128x128, 2 tasks x 2, short time limit, devel GPU partition
python3 cluster/submit.py --n 4 --nx 128 --chunk 2 --name smoke128 --time 00:30:00 --partition gpu_devel
python3 cluster/submit.py --status smoke128          # progress, queue, missing slots
cat /home/naa79/scratch_pi_sz622/naa79/cloth_data/smoke128/slurm/gen_*_0.out  # task log

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

## 3b. Overnight production recipe (768x768 on L40S, maximum throughput)

`cluster.env` now carries the 768 profile: `GPUS=l40s:1`, `CHUNK=10`
(~1 h per task), `TIME=02:00:00`, `JOBS_PER_TASK=2`, piercing QC off above
256 (`generate_batch.py --pierce-qc auto`). Submit two arrays over disjoint
task ranges of the SAME dataset: a normal one on `gpu` and a preemptible one
on `scavenge_gpu` (tasks are `--requeue`d and resume, so preemption only
costs the sample in flight).

```bash
# 3000 slots = 300 tasks of 10. Tasks 0-99 on gpu (L40S), 100-299 on scavenge_gpu.
python3 cluster/submit.py --n 3000 --name ds768_night --tasks 0-99 --max-concurrent 40 --no-merge
python3 cluster/submit.py --n 3000 --name ds768_night --tasks 100-299 --partition scavenge_gpu \
    --max-concurrent 80 --no-merge
python3 cluster/submit.py --status ds768_night
```

In the morning, whatever finished is the dataset (slots are independent):

```bash
python3 cluster/submit.py --status ds768_night            # passing count, missing slots
scancel -u $USER -n cloth_gen                             # stop the rest if you are done
.venv/bin/python cluster/merge_summaries.py --indir /home/naa79/scratch_pi_sz622/naa79/cloth_data/ds768_night
.venv/bin/python package_splits.py --indir /home/naa79/scratch_pi_sz622/naa79/cloth_data/ds768_night \
    --outdir /home/naa79/scratch_pi_sz622/naa79/cloth_data/ds768_night_splits
```

Sizing: one L40S makes ~12 samples/h at 768 single-process (~306 s each incl.
setup), more with `JOBS_PER_TASK=2`. Each 768 sample is ~50 MB on disk (final
state, velocities, 5 intermediates), so 2000 samples is ~100 GB.

## 3b'. Paper sheets (plastic creasing, not fully reversible)

Same two-force scenario, `--material paper` sheet (`cloth_sim/sheet.py`
`MATERIALS`): lighter (0.08 kg/m^2), 10x stiffer in plane, 5x stiffer in
bending, and bends sharper than a yield curvature (50 /m) become permanent
creases (`cloth_sim/plasticity.py`). Cloth and elastic sheets spring back to
flat once the forces stop; creased paper stays folded, so the end state no
longer determines the forces uniquely. Same pipeline, QC (stretch limit 1.3),
resume and merge; datasets get a `_paper` suffix.

```bash
# smoke: 4 samples at 128 on the devel partition
python3 cluster/submit.py --n 4 --nx 128 --chunk 2 --material paper --name smoke128_paper \
    --time 00:30:00 --partition gpu_devel
# production at 768 (same profile as the cloth run) -> cloth_data/ds768_3000_paper
python3 cluster/submit.py --n 3000 --material paper --max-concurrent 40
```

Each paper HDF5 also carries a `plastic/` group (hidden state, withheld from
the test split): `edge_rest_angle` and `edge_indices` (final crease angle per
mesh edge), `crease_map` (ny, nx), and per-frame `crease_total` /
`edges_yielded`. Tune the crease with `--sample-args="--force-every-substep
--yield-curvature 35"` (lower = creases more easily; 0 = elastic paper).

## 3d. Bench profile (256 grid, table + gravity, pusher forces, cloth and paper)

The real-world story (ROADMAP.md) needs a sheet on a table pushed by a rig,
observed by a camera. The bench profile in `cluster.env` generates that:
`NX=256` (384 lost ~0.3 cm and 192 ~0.7 cm against 768 in the inverse
model, RESULTS.md, while a 256 sample is ~9x fewer particles than 768),
`BATCH_ARGS=--loading pusher --table` (two pusher-like forces, 10-60 deg
into the table, never a pull; gravity 9.81 and a ground plane with friction
0.5), `--force-every-substep`. Forces depend only on the seed, so a cloth
and a paper dataset submitted with the same `seed0` are **paired**: the same
loading on both materials.

```bash
# smoke: 4 samples of each material on the devel partition
python3 cluster/submit.py --n 4 --chunk 2 --name smoke256_bench --time 00:30:00 --partition gpu_devel
python3 cluster/submit.py --n 4 --chunk 2 --material paper --name smoke256_bench_paper --time 00:30:00 --partition gpu_devel
# production, paired: same n and seed0, any GPU type
python3 cluster/submit.py --n 3000 --name ds256_bench --gpus 1 --max-concurrent 40
python3 cluster/submit.py --n 3000 --name ds256_bench_paper --material paper --gpus 1 --max-concurrent 40
python3 cluster/submit.py --status ds256_bench
```

The merge job packages each dataset into `<name>_splits`. Legacy free-floating
datasets are reproduced with `--batch-args ''` and `--nx 768` (or 128).

## 3c. Training the inverse model on the cluster (full resolution)

```bash
bash cluster/setup_env.sh --with-torch                        # once (adds PyTorch to the venv)
sbatch cluster/train.sbatch /home/naa79/scratch_pi_sz622/naa79/cloth_data/ds768_night_splits 300
tail -f train_<jobid>.out
```

`train.sbatch` runs the baselines, then the heatmap CNN and the regression CNN
with dihedral augmentation, then the heatmap CNN with 3D-rotation augmentation,
all at native 768x768 with bf16 autocast. It asks for 160 GB host RAM and one
L40S: the dataset is held as float32 tensors (~14 MB per sample) and moved to
the GPU; pass `--data-on-cpu` as an extra arg if the GPU runs out of memory,
or `--pool 2` to train on block-averaged 384 grids. Metrics and test
predictions land in `../inverse/runs/<splits>_<model>_*/`.

## 4. Outputs

```
/home/naa79/scratch_pi_sz622/naa79/cloth_data/ds256_1000/
  sample_slot00000_seed100000.h5 ...   passing samples (QC gate: no NaN,
  forces_slot*_seed*.json              stretch<=2, no gross piercing)
  rejected/                            failed attempts (kept for inspection)
  batch_summary_00000.json ...         one per task
  batch_summary.json                   merged (input to package_splits.py)
  dataset_summary.json                 QC stats, retries, GPU hours, missing slots
  progress_*.jsonl                     per-task incremental log
  slurm/                               rendered sbatch files, submission.json, logs
/home/naa79/scratch_pi_sz622/naa79/cloth_data/ds256_1000_splits/{train,val,test}/ + dataset_manifest.json
```

Check the retention policy of `scratch_pi_sz622` with your group; if it is
purged like personal scratch, copy finished `*_splits` directories to a
persistent location.

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

## 7. Choosing a GPU type: queue wait vs speed

The scarce resource is usually wall-clock time to a finished dataset, which is
queue wait plus run time. A slower GPU that starts now often beats an H100
that waits hours. Three tools (all under `cluster/`, outputs in `bench/`):

```bash
# 1. How long would a 1-GPU job wait right now, per GPU type? (no job submitted:
#    uses Slurm's own estimate via `sbatch --test-only` + free-GPU counts)
python3 cluster/queue_probe.py --time 12:00:00
python3 cluster/queue_probe.py --time 12:00:00 --watch 1800 &   # log every 30 min for a day
python3 cluster/queue_probe.py --history                        # median / p90 wait per type

# 1b. Who is using what right now? GPUs total / in use / free per type, running
#     jobs and distinct users holding each type, pending requests per type.
python3 cluster/gpu_usage.py
python3 cluster/gpu_usage.py --watch 600 &     # log every 10 min; --history summarizes

# 2. How fast is each GPU type at each resolution? One job per type, same forces,
#    nx = 64..384 by default, 2 repeats (the first includes kernel compilation).
python3 cluster/bench.py --dry-run
python3 cluster/bench.py --types a5000,rtx_5000_ada,a40,l40s,h100,h200
squeue -u $USER -n cloth_bench

# 2b. Resolution ceiling: one 5-minute job per size, one sample each; the report
#     marks sizes that did not finish as TIMEOUT or OUT_OF_MEMORY.
python3 cluster/bench.py --types l40s --nx 128,192,256,320,384,448,512,640,768,896,1024 \
    --one-job-per-nx --time 00:05:00 --budget-min 5 --repeats 1 --run ceiling_l40s

# 3. Combine: real queue wait of each bench job (sacct), probe history, run time
#    per sample, GPU utilization, peak memory, Slurm billing weight, and the
#    estimated time to finish N samples at a resolution with K concurrent tasks.
python3 cluster/bench_report.py --n 1000 --nx 256 --concurrent 16
```

The report ranks GPU types by total time (wait + run) and also shows
fairshare cost in billing-hours, so the "sweet spot" is explicit. GPU types
and partitions are listed in `cluster/gpu_types.py`; `--partition scavenge_gpu`
on `bench.py` / `queue_probe.py` tests the preemptible pool. Peak memory per
resolution tells you which GPUs can run 384 or 512 grids at all.

Then run production with e.g. `submit.py --partition gpu --max-concurrent 24`
(the `gpu` partition has four GPU types; add `#SBATCH --gpus=l40s:1`-style
targeting by setting `GPUS=l40s:1` in `cluster.env` if one type wins clearly).

## 8. Monitoring and troubleshooting

```bash
python3 cluster/submit.py --status ds256_1000
squeue -u $USER
sacct -j <ARRAY_JOB_ID> --format=JobID,State,Elapsed,MaxRSS,ExitCode | head
tail -n 20 /home/naa79/scratch_pi_sz622/naa79/cloth_data/ds256_1000/slurm/gen_<JOBID>_<TASK>.err
```

- `CUDA driver` / `no device` errors in a task: the partition has no GPU or
  `--gpus` was not honoured; check `nvidia-smi` output at the top of the log.
- Out-of-memory (host): raise `MEM` (the piercing QC check at 256x256 is
  CPU/RAM heavy).
- Tasks time out: lower `CHUNK` or raise `TIME`; re-submit to resume.
