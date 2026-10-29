# Job log (Bouchet)

One row per submitted job or job matrix. Fill in the Slurm job id at submission, update
`status` when `bash cluster/jobs.sh` is run, and link the result row in RESULTS.md /
DATASETS.md when done. Status: `queued` `running` `done` `failed` `cancelled` `planned`.

Conventions: `S` = `/home/naa79/scratch_pi_sz622/naa79/cloth_data/ds768_night_splits`;
all `sbatch` commands run from `cloth_data_generation/transfer` on the cluster; single
training configs use `cluster/run.sbatch` (one GPU per job, `--out runs/<id>` so the
metrics directory matches the row). Snapshot the queue with `bash cluster/jobs.sh` and
paste the output into a chat with Claude to update this file. Per-run metrics in one line each: `python3 cluster/summarize_runs.py 'M1*' 'M2*' 'M3*'`.

| id | job id(s) | submitted | status | what / why | command | result |
|---|---|---|---|---|---|---|
| J-001 | arrays `cloth_gen` | 2026-10-27 night | done | ds768_night: 3000 slots at 768, L40S + scavenge | README 3b | 1,115 samples, DATASETS.md |
| J-002 | | 2026-10-28 | running | stability sweep substeps 60/90/150 on failed force configs; find a setting where the velocity clamp does not tear the sheet (magnitude identifiability) | `cluster/stability.py --dataset ds768_night` then `--report` | |
| J-003 | 31144541 (earlier attempts 31134453, 31139874, 31140657 cancelled) | 2026-10-28 | running (rung 3b) | ladder run 2, rungs 0,1,2,3,3b sequentially (D4-symmetric metric, TTA off), one GPU | `sbatch cluster/ladder.sbatch $S 300` | rungs 0/1/2/3 = 3.9/12.8, 4.5/15.9, 8.5/32.8, 8.3/31.1 (cm/deg); RESULTS.md |
| J-004 | | | dropped | ds512_v1 free-floating generation; superseded by the bench profile (M6) after the resolution result | | |

Submitted 2026-10-28 ~13:00: M1 (31153218-24), M2 (31153226-28), M3 (31153239-42), M1a-d ran on L40S; the other ten were resubmitted ~13:30 as GPU-agnostic jobs (31155587-96) after an in-queue scontrol edit cancelled them (sacct: CANCELLED, never assigned). GPU-agnostic placement worked: runs landed on RTX 5000 Ada (32 GB, data on CPU), A40, A5000 (24 GB, data on CPU) and L40S.

## Matrix M1: noise floor and capacity (submit alongside the ladder, ~10-20 min each)

Why: ladder rungs differ by ~0.5 cm; without seed repeats we cannot tell a rung effect
from training noise. All CNNs underfit (train = val = test), so capacity and epochs are
the cheapest levers before more data.

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M1-a | 31153218 | done | rung 0 seed 1 | `sbatch --job-name=M1a cluster/run.sbatch $S --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --seed 1 --out runs/M1a` | test 3.9 cm / 13.7 deg (train 3.3 / 11.4) |
| M1-b | 31153219 | done | rung 0 seed 2 | `sbatch --job-name=M1b cluster/run.sbatch $S --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --seed 2 --out runs/M1b` | test 4.2 cm / 13.3 deg (train 3.9 / 12.1) |
| M1-c | 31153220 | done | rung 2 seed 1 | `sbatch --job-name=M1c cluster/run.sbatch $S --model cnn_reg --epochs 300 --amp --tta 0 --no-vel --eval-rotated --seed 1 --out runs/M1c` | test 8.7 cm / 32.8 deg (train 8.9 / 33.9) |
| M1-d | 31153221 | done | rung 2 seed 2 | `sbatch --job-name=M1d cluster/run.sbatch $S --model cnn_reg --epochs 300 --amp --tta 0 --no-vel --eval-rotated --seed 2 --out runs/M1d` | test 9.3 cm / 34.8 deg (train 9.6 / 36.6) |
| M1-e | 31155587 (resubmitted; 31153222 lost to scontrol TresPerJob update) | done | rung 1, width 48 | `sbatch --job-name=M1e cluster/run.sbatch $S --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --width 48 --out runs/M1e` | test 4.5 cm / 16.1 deg (train 2.3): overfits, no gain; 25 min on A40 |
| M1-f | 31155588 (resubmitted; 31153223 lost to scontrol TresPerJob update) | done | rung 1, width 64 | `sbatch --job-name=M1f cluster/run.sbatch $S --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --width 64 --out runs/M1f` | test 4.4 cm / 15.3 deg (train 1.4): overfits harder, no gain; 28 min |
| M1-g | 31155589 (resubmitted; 31153224 lost to scontrol TresPerJob update) | done | rung 2, 1000 epochs | `sbatch --job-name=M1g cluster/run.sbatch $S --model cnn_reg --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --out runs/M1g` | test 7.8 cm / 26.5 deg (train 8.1 / 26.3, val 7.8 / 22.3): 3x epochs at 768 gives 8.5 -> 7.8 cm; M2c (192, 1000 ep) is still better at 5.9 |

## Matrix M2: how much resolution is needed (direction 2: images / depth; and the 512 run)

Why: if block-averaged 384 or 192 grids lose nothing, image-based observation does not need
768 and the 512 dataset is sufficient; training also gets 4-16x faster. First run of each
pool builds a new tensor cache (`*_pool2.pt`, `*_pool4.pt`), so submit one job per pool
first, then more.

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M2-a | 31155590 (resubmitted; 31153226 lost to scontrol TresPerJob update) | done | rung 1 at 384 (pool 2) | `sbatch --job-name=M2a cluster/run.sbatch $S --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --pool 2 --out runs/M2a` | test 4.5 cm / 16.4 deg (train 3.9 / 12.4), 7 min on RTX 5000 Ada |
| M2-b | 31155591 (resubmitted; 31153227 lost to scontrol TresPerJob update) | done | rung 1 at 192 (pool 4) | `sbatch --job-name=M2b cluster/run.sbatch $S --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --pool 4 --out runs/M2b` | test 5.0 cm / 17.2 deg (train 4.1 / 13.3), 7 min |
| M2-c | 31155592 (resubmitted; 31153228 lost to scontrol TresPerJob update) | done | rung 2 at 192 (pool 4), 1000 epochs (cheap at this size) | `sbatch --job-name=M2c cluster/run.sbatch $S --model cnn_reg --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --pool 4 --out runs/M2c` | test 5.9 cm / 24.1 deg (train 6.2): 3x epochs = big gain at rung 2; 27 min on A5000 (data on CPU) |

## Matrix M3: point-cloud model (direction 3: non-grid representations)

Why: rung 3 is the representation that carries over to FEM meshes / crash. One config in
the ladder is not a sweep; points, width and depth are the knobs.

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M3-a | 31155593 (resubmitted; 31153239 lost to scontrol TresPerJob update) | done | pointtf 1024 pts, d128, 4 layers | `sbatch --job-name=M3a cluster/run.sbatch $S --model pointtf --epochs 300 --amp --tta 0 --no-vel --eval-rotated --bs 16 --lr 3e-4 --n-points 1024 --out runs/M3a` | test 8.3 cm / 30.7 deg (train 8.8 / 35.2, val 8.6 / 29.3): underfits; 6 min on L40S |
| M3-b | 31155594 (resubmitted; 31153240 lost to scontrol TresPerJob update) | done | pointtf 8192 pts, d128, 4 layers, bs 8 | `sbatch --job-name=M3b cluster/run.sbatch $S --model pointtf --epochs 300 --amp --tta 0 --no-vel --eval-rotated --bs 8 --lr 3e-4 --n-points 8192 --out runs/M3b` | test 8.2 cm / 30.0 deg (train 8.6); 29 min |
| M3-c | 31155595 (resubmitted; 31153241 lost to scontrol TresPerJob update) | done | pointtf 2048 pts, d256, 4 layers | `sbatch --job-name=M3c cluster/run.sbatch $S --model pointtf --epochs 300 --amp --tta 0 --no-vel --eval-rotated --bs 16 --lr 3e-4 --pt-dim 256 --out runs/M3c` | test 8.1 cm / 32.1 deg (train 8.6); 10 min |
| M3-d | 31155596 (resubmitted; 31153242 lost to scontrol TresPerJob update) | done | pointtf 2048 pts, d128, 8 layers | `sbatch --job-name=M3d cluster/run.sbatch $S --model pointtf --epochs 300 --amp --tta 0 --no-vel --eval-rotated --bs 16 --lr 3e-4 --pt-layers 8 --out runs/M3d` | test 8.3 cm / 29.8 deg (train 8.7); 10 min |

## Matrix M4: paper data (direction 1: plastic creasing)

Why: paper is the physical target. A small 256 set tells us the QC pass rate (stretch
limit 1.3) and whether creases make the inverse ambiguous before spending 768 budget.
Runs on scavenge so it never competes with the ladder or ds512.

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M4-a | | planned | ds256_paper: 400 slots at 256, paper material, scavenge | `python3 cluster/submit.py --n 400 --nx 256 --chunk 20 --material paper --name ds256_paper --partition scavenge_gpu --time 02:00:00 --max-concurrent 20` | |
| M4-b | | planned | train rung 1 on ds256_paper once packaged | `sbatch --job-name=M4b cluster/run.sbatch /home/naa79/scratch_pi_sz622/naa79/cloth_data/ds256_paper_splits --model cnn_reg --epochs 600 --no-rot --amp --tta 0 --no-vel --out runs/M4b` | |

## Matrix M5: where the unknown-pose cost comes from (after M1-M3, 2026-10-28)

Why: rung 2/3 underfit regardless of model size and representation but improve a lot with epochs
(M2c), and full SO(3) is pessimistic for a bench with a known table plane. Three questions: how far
does longer training go, how much does a known support plane (yaw-only pose) recover, and does
the answer hold for the point cloud. Pooled inputs where possible (cheap, any GPU).

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M5-a | | planned | rung 1.5 (yaw-only pose) at 768, 300 ep | `sbatch --job-name=M5a cluster/run.sbatch $S --model cnn_reg --epochs 300 --amp --tta 0 --no-vel --eval-rotated --yaw-only --out runs/M5a` | |
| M5-b | | planned | rung 1.5 at 192, 1000 ep (pair with M2c) | `sbatch --job-name=M5b --mem=60G cluster/run.sbatch $S --model cnn_reg --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --yaw-only --pool 4 --out runs/M5b` | |
| M5-c | | planned | rung 2 at 192, 3000 ep (epoch curve) | `sbatch --job-name=M5c --mem=60G cluster/run.sbatch $S --model cnn_reg --epochs 3000 --amp --tta 0 --no-vel --eval-rotated --pool 4 --out runs/M5c` | |
| M5-d | | planned | rung 2 at 384, 1000 ep | `sbatch --job-name=M5d --mem=60G cluster/run.sbatch $S --model cnn_reg --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --pool 2 --out runs/M5d` | |
| M5-e | | planned | rung 3 default pointtf, 1000 ep (epochs alone, vs 3b's bigger model) | `sbatch --job-name=M5e cluster/run.sbatch $S --model pointtf --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --bs 16 --lr 3e-4 --out runs/M5e` | |
| M5-f | | planned | rung 3 pointtf, yaw-only pose, 1000 ep | `sbatch --job-name=M5f cluster/run.sbatch $S --model pointtf --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --yaw-only --bs 16 --lr 3e-4 --out runs/M5f` | |
| M5-g | | planned | stability sweep report (J-002) | `python3 cluster/stability.py --dataset ds768_night --report` | |

## Matrix M6: bench-profile data generation (the new story, 2026-10-28)

Why: the real-world target is a sheet on a table pushed by a rig and seen by a camera.
The bench profile (`cluster.env`: 256 grid, `--table`, `--loading pusher`) generates that
for cloth and paper with paired seeds. Smoke first (per-sample time sets CHUNK/TIME), then
production on any GPU type. Replaces the queued free-floating ds512 run (J-004, dropped).

Workstation smoke rounds 1-5 (DATASETS.md): the profile is mechanically complete but every
sample that crumpled against the table under gravity went NaN. So the order is: M6-a/b smoke
(expect failures; they supply force configs and timing), then **M6-0**, the solver sweep on
those configs, then production with the winning flags in `SAMPLE_ARGS`.

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M6-0 | | planned | solver sweep for the bench scenario: 7 settings x 8 configs (cloth), one job per setting | `D=/home/naa79/scratch_pi_sz622/naa79/cloth_data; python3 cluster/stability.py --dataset smoke256_bench --forces-dir $D/smoke256_bench --all-configs --base-flags="--table" --settings "base=;sub60=--substeps 60;sub120=--substeps 120;ke1e4=--contact-ke 1e4;sub60ke1e4=--substeps 60 --contact-ke 1e4;it40=--iterations 40;stiff=--tri-ke 1e4 --substeps 60" --time 02:00:00` then `--report` | |
| M6-0p | | planned | same sweep on paper configs | as M6-0 with `--dataset smoke256_bench_paper --forces-dir $D/smoke256_bench_paper --material paper --run stability_bench_paper` | |
| M6-a | | planned | smoke: 4 cloth samples, bench profile, devel GPU (also feeds M6-0) | `python3 cluster/submit.py --n 4 --chunk 2 --name smoke256_bench --time 00:30:00 --partition gpu_devel` | |
| M6-b | | planned | smoke: 4 paper samples, same seeds | `python3 cluster/submit.py --n 4 --chunk 2 --material paper --name smoke256_bench_paper --time 00:30:00 --partition gpu_devel` | |
| M6-c | | blocked on M6-0 | production cloth, 3000 slots, any GPU; add the winning sweep flags to SAMPLE_ARGS first | `python3 cluster/submit.py --n 3000 --name ds256_bench --max-concurrent 40` | |
| M6-d | | blocked on M6-0p | production paper, 3000 slots, paired seeds | `python3 cluster/submit.py --n 3000 --name ds256_bench_paper --material paper --max-concurrent 40` | |
| M6-e | | planned | ladder on ds256_bench (rungs 0-3) once packaged | `sbatch cluster/ladder.sbatch /home/naa79/scratch_pi_sz622/naa79/cloth_data/ds256_bench_splits 300` | |
| M6-f | | planned | ladder on ds256_bench_paper | `sbatch cluster/ladder.sbatch /home/naa79/scratch_pi_sz622/naa79/cloth_data/ds256_bench_paper_splits 300` | |

Check on the smoke output before production (`python3 cluster/submit.py --status smoke256_bench`
and the task logs): QC pass rate, per-sample wall time, peak speed (should exceed the 0.38 m/s
clamp seen at 768), and that the sheet stayed on the table (final z >= -particle radius).

## Matrix M7: free-flight cloth at 256, volume (decided 2026-10-28 evening)

Why: the bench (table) scenario is parked until a solver sweep passes; the free-floating
scenario is proven and the inverse model is data-limited at known pose (RESULTS.md). 256 is
~9x cheaper per sample than 768 at a 0.3-0.7 cm cost. Radius/margin/gap scale with the cell
(new), so this is a new dataset, not an extension of ds768_night. Smoke one task first for the
QC pass rate and timing, then fill 6000 slots on gpu + scavenge (expect ~50-60% to pass).

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M7-a | 31178959 | TIMEOUT (1 h) | smoke: one task of 40 samples on gpu | `python3 cluster/submit.py --n 40 --name ds256_free --time 01:00:00 --no-merge` then `python3 cluster/submit.py --status ds256_free` | first 4 slots ok on attempt 1 (RTX 5000 Ada); ~25 of 40 slots done in 1 h. seff: CPU eff 48.7% (4 cores), MaxRSS 1.7 GB, so 4 cores / 12G are right-sized (could go 2 cores / 6G) |
| M7-b | 31181234 (31179012, 31179200 cancelled before start; resubmitted 15:47 with 12G / 45 min) | timed out, partial | production: slots 40-2999, tasks 1-74 on gpu (scavenge half M7-c skipped to save space) | `python3 cluster/submit.py --n 6000 --name ds256_free --tasks 1-74 --max-concurrent 40 --no-merge` | 16:35: 964 passing / 6000, 304 rejects (76% pass). Each 45-min task finished only 12-16 of its 40 slots (~5 min per attempt per worker, 2 workers): chunk 40 needs ~2.5 h, not 45 min. 70 non-empty .err logs (check). Remaining 5036 slots: M7-b2 |
| M7-b2 | 31220691 (tasks 0-149, %40) | running (submitted 2026-10-28 ~20:40) | fill the 5036 missing slots: same chunk (name-bound), 2.5 h per task, 40 concurrent; ~4 waves of 40 tasks, expect done ~10 h later | check 1 (evening): 1190 passing, refill 345 ok / 4 rejected, only 12 tasks running at once (gpu partition availability), early tasks finish in <1 h (few missing slots), full chunks from task ~75 need ~2 h; at 12 concurrent expect ~1 day. Option: move tasks 100-149 to scavenge_gpu (scancel them first). Check 2 (2026-10-29 morning): 3064 passing / 969 rejected (76%), 24 GB; tasks 60 COMPLETED, 11 TIMEOUT, 12 running, ~67 pending; timed-out tasks need one more refill pass after the array clears. **Stopped by user 2026-10-29** (pending tasks cancelled at ~3100 passing) to test data vs model first (M8); refill later if M8 says more data helps | `python3 cluster/submit.py --n 6000 --name ds256_free --chunk 40 --time 02:30:00 --max-concurrent 40 --no-merge` (defaults to tasks with missing slots; `--chunk 40` required because cluster.env now defaults to 20 and chunk is bound to the dataset name) | |
| M7-c | | skipped (space) | production: tasks 75-149 on scavenge (preemptible, requeued) | `python3 cluster/submit.py --n 6000 --name ds256_free --tasks 75-149 --partition scavenge_gpu --max-concurrent 60 --no-merge` | |
| M7-d | | planned | package when enough passed | `.venv/bin/python cluster/merge_summaries.py --indir $D/ds256_free && .venv/bin/python package_splits.py --indir $D/ds256_free --outdir $D/ds256_free_splits` | |
| M7-e | | planned | ladder on it (rungs 0-3, any GPU) | `sbatch cluster/ladder.sbatch $D/ds256_free_splits 300` | |

`D=/home/naa79/scratch_pi_sz622/naa79/cloth_data`. The M7-a run must be submitted first with the
same name: `submit.py` binds the name to nx/seed0/chunk/args and later calls only fill missing slots.

## Submitting a whole matrix at once

## Matrix M8: is it the data or the model? (2026-10-29)

Question: ds768_night (1,115 marginal samples) gave rung 0/1 about 4 cm and rung 2 about 6-9 cm,
with known-pose rungs overfitting (data-limited). ds256_free has ~3,100 cleaner samples at 256.
Three comparisons separate the effects: (1) same model on 3x the data, (2) same model on a
1,100-sample subset of the new data (`--n-train 1100`, new flag) to isolate dataset quality from
dataset size, (3) the old data pooled to 256 (`--pool 3`) so resolution is matched.
Reading: if (1) beats (2) clearly, data volume matters and the refill is worth finishing; if (2)
already beats (3), the cleaner 256 physics is the gain; if all three tie, the model/targets are
the limit and more data will not help.

`S2=/home/naa79/scratch_pi_sz622/naa79/cloth_data/ds256_free_splits`, `S=.../ds768_night_splits`.

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M8-0 | | done 2026-10-29 | package ds256_free (M7-d) | `D=/home/naa79/scratch_pi_sz622/naa79/cloth_data; .venv/bin/python cluster/merge_summaries.py --indir $D/ds256_free && .venv/bin/python package_splits.py --indir $D/ds256_free --outdir $D/ds256_free_splits` | |
| M8-a | 31295878 (attempts 31280929 val NaN, 31290287 val split empty) | queued 2026-10-29 13:16 | rung 1 (positions, known pose), all ~2,500 train samples | `sbatch --job-name=M8a --mem=60G cluster/run.sbatch $S2 --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --out runs/M8a` | |
| M8-b | 31295879 (attempts 31280930 val NaN, 31290288 val split empty) | queued 2026-10-29 13:16 | rung 1, 1,100-sample subset (size control) | `sbatch --job-name=M8b --mem=60G cluster/run.sbatch $S2 --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --n-train 1100 --out runs/M8b` | |
| M8-c | 31280931 | done | rung 1, old data pooled to 256 (resolution control; compare M1e/M2a) | `sbatch --job-name=M8c --mem=60G cluster/run.sbatch $S --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --pool 3 --out runs/M8c` | test 5.2 cm / 18.0 deg (train 4.7 / 14.6, val 4.9 / 16.6), 10 min. Pool 3 of the old data costs 0.7 cm vs M1e/M2a (4.5) |
| M8-d | 31295880 (attempts 31280932 val NaN, 31290289 val split empty) | queued 2026-10-29 13:16 | rung 2 (unknown pose), all data, 1000 epochs (compare M2c 5.9 cm, M1g) | `sbatch --job-name=M8d --mem=60G cluster/run.sbatch $S2 --model cnn_reg --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --out runs/M8d` | |
| M8-e | 31295881 (attempts 31280933 val NaN, 31290290 val split empty) | queued 2026-10-29 13:16 | rung 2, 1,100-sample subset | `sbatch --job-name=M8e --mem=60G cluster/run.sbatch $S2 --model cnn_reg --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --n-train 1100 --out runs/M8e` | |
| M8-f | 31295882 (attempts 31280934 val NaN, 31290291 val split empty) | queued 2026-10-29 13:16 | rung 1 seed repeat (noise floor at this size) | `sbatch --job-name=M8f --mem=60G cluster/run.sbatch $S2 --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --seed 2 --out runs/M8f` | |

Gather: `python3 cluster/summarize_runs.py 'M8*'`.

**2026-10-29 11:30 outcome.** The five runs on `ds256_free_splits` (M8a/b/d/e/f) all trained to the last
epoch with a finite train loss but `val nan` at every eval, so no checkpoint was saved and the final
eval exited 1 (`run_3128093x.out`: `... val nan best inf`). M8c (old data) completed. The val split of
ds256_free contains at least one sample with non-finite positions/velocities/targets; the merge QC
(stretch / vmax / piercing) does not check finiteness and `package_splits.py` copies whatever passed.
Fix committed: `train.py` drops non-finite samples per split with a warning and fails early with a
clear message if the val loss is never finite; `cluster/check_splits.py <splits_dir>` names the bad
files. Resubmit M8a/b/d/e/f unchanged after `git pull` (same commands as above). Wasted: ~3.1 GPU-hours.
Resubmitted with M9/M10 on 2026-10-29 12:05 (31290287-99): **all 13 failed within seconds**,
`no files in ds256_free_splits/val`. The splits directory was a half-finished package (train
present and cached, val empty, no manifest): `package_splits.py` rmtree's the output first and
writes train, val, test, manifest in that order, so an interrupted run (interactive 31289968,
exit 127, ~12:00) leaves exactly this state. Fix: re-run M8-0 packaging to completion, check
the three counts and the manifest, then resubmit the same lines (third attempt).
Repackaged 2026-10-29 ~13:30: 2,453 / 306 / 306 (3,065 passing; 184 missing + 33 failed slots
remain for a refill). `check_splits.py`: one non-finite val sample, `sample_slot1519_seed101519.h5`
(final_grid and final_vel), confirming the first failure; train.py drops it, and
`merge_summaries.py` now rejects non-finite samples so the next package excludes it. DINOv2
repo + weights cached in ~/.cache/torch/hub.


## Matrix M9: beating the data limit without more simulation (2026-10-29)

Three levers implemented in `inverse/` (commit after 793c85a), all on `ds256_free_splits`:
(1) **dense supervision** `--aux-w 1`: the model also predicts the applied force field on
the grid (per-cell direction of both Gaussian patches, ~65k labels per sample; per point for
pointtf, per patch for dino_reg); (3) **stored frames** `--frames post`: every train sample's
intermediate states after the forces ended become extra samples (about 4 per sample, zero
simulation cost); (4) **pretrained features** `--model dino_reg`: frozen DINOv2 ViT-S/14 with a
small head, on the material-grid image or on a rendered view. Also new: `--view` (rung 4a, one
orthographic camera: depth+mask image for image models, visible points only for pointtf) and
`--init <model.pt>` (curriculum). Rung 1 baseline to beat: M8a (and M1e/M2a 4.5 cm on the old data).

Before the dino jobs, once on the login node (internet; caches repo + weights in ~/.cache/torch/hub):
`.venv/bin/python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"`.
Frames jobs train on ~4x the samples per epoch: expect ~4x the wall time of M8a (use `--time`).

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M9-a | 31295883 (attempt 31290292 val split empty) | queued 2026-10-29 13:16 | rung 1 + aux force-field loss (vs M8a) | `sbatch --job-name=M9a --mem=60G cluster/run.sbatch $S2 --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --aux-w 1 --out runs/M9a` | |
| M9-b | 31295884 (attempt 31290293 val split empty) | queued 2026-10-29 13:16 | rung 1 + post-force frames (4x samples) | `sbatch --job-name=M9b --mem=100G --time=08:00:00 cluster/run.sbatch $S2 --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --frames post --out runs/M9b` | |
| M9-c | 31295885 (attempt 31290294 val split empty) | queued 2026-10-29 13:16 | rung 1 + aux + frames | `sbatch --job-name=M9c --mem=100G --time=08:00:00 cluster/run.sbatch $S2 --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --aux-w 1 --frames post --out runs/M9c` | |
| M9-d | 31295886 (attempt 31290295 val split empty) | queued 2026-10-29 13:16 | rung 3 pointtf + per-point aux, 1000 ep (vs M3/M5e) | `sbatch --job-name=M9d --mem=60G cluster/run.sbatch $S2 --model pointtf --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --bs 16 --lr 3e-4 --aux-w 1 --out runs/M9d` | |
| M9-e | 31295887 (attempt 31290296 val split empty) | queued 2026-10-29 13:16 | rung 1 with frozen DINOv2 on the grid image (+aux) | `sbatch --job-name=M9e --mem=60G cluster/run.sbatch $S2 --model dino_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --lr 3e-4 --aux-w 1 --out runs/M9e` | |
| M9-f | 31295888 (attempt 31290297 val split empty) | queued 2026-10-29 13:16 | rung 4a: pointtf on the visible points of one view, from scratch | `sbatch --job-name=M9f --mem=60G cluster/run.sbatch $S2 --model pointtf --epochs 1000 --amp --tta 0 --no-vel --view --bs 16 --lr 3e-4 --aux-w 1 --out runs/M9f` | |
| M9-g | | after M9-d | rung 4a pointtf initialised from rung 3 (curriculum) | `sbatch --job-name=M9g --mem=60G cluster/run.sbatch $S2 --model pointtf --epochs 1000 --amp --tta 0 --no-vel --view --bs 16 --lr 3e-4 --aux-w 1 --init runs/M9d/model.pt --out runs/M9g` | |
| M9-h | 31295889 (attempt 31290298 val split empty) | queued 2026-10-29 13:16 | rung 4a: DINOv2 on the rendered depth image (+aux) | `sbatch --job-name=M9h --mem=60G cluster/run.sbatch $S2 --model dino_reg --epochs 300 --amp --tta 0 --no-vel --view --lr 3e-4 --aux-w 1 --out runs/M9h` | |
| M9-i | 31295890 (attempt 31290299 val split empty) | queued 2026-10-29 13:16 | rung 4a: from-scratch CNN on the rendered depth image (control for M9-h) | `sbatch --job-name=M9i --mem=60G cluster/run.sbatch $S2 --model cnn_reg --epochs 300 --amp --tta 0 --no-vel --view --aux-w 1 --out runs/M9i` | |

Reading: M9-a/b/c vs M8a say what each lever buys at rung 1; M9-e vs M8a is the pretrained-feature
test; M9-f/g/h/i are the first rung-4a numbers (gap to rung 3 = cost of seeing one side);
M9-h vs M9-i is pretrained vs scratch on images. Gather: `python3 cluster/summarize_runs.py 'M8*' 'M9*'`.

## Matrix M10: more data, in case M8 says it still helps (2026-10-29)

Four GPUs for ~8 h on the ds256_free refill. The chunk is bound to the dataset name (40 slots,
~2.5 h per task), so "4 jobs x 8 h" is an array of 12 tasks at 4 concurrent (3 waves). Expected
yield: ~480 attempts, ~360 passing samples (+12%). To grow the set materially, raise
`--max-concurrent` back to 40 and drop `--max-tasks`.

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M10-a | | cancelled (pending tasks) 2026-10-29 | refill ds256_free: 12 tasks, 4 at a time, 2.5 h each; submitted then cancelled to keep GPUs on the M8/M9 model runs (+360 samples is not decisive; regenerate overnight at 40 concurrent if M8 says data helps) | `python3 cluster/submit.py --n 6000 --name ds256_free --chunk 40 --time 02:30:00 --max-concurrent 4 --max-tasks 12 --no-merge` | |
| M10-b | | after M10-a | re-merge and re-package (new splits; M8/M9 runs used the 2026-10-29 package) | `D=/home/naa79/scratch_pi_sz622/naa79/cloth_data; .venv/bin/python cluster/merge_summaries.py --indir $D/ds256_free && .venv/bin/python package_splits.py --indir $D/ds256_free --outdir $D/ds256_free_splits` | |

Note: M10-b replaces the splits directory (and their tensor caches); submit it only after the
M8/M9 training jobs have finished loading, and record the new sample count in DATASETS.md.

```bash
bash cluster/interactive.sh      # CPU allocation
cd /home/naa79/scratch_pi_sz622/naa79/cloth-inverse && git pull && cd cloth_data_generation/transfer
S=/home/naa79/scratch_pi_sz622/naa79/cloth_data/ds768_night_splits
# paste the sbatch lines of the matrix, then:
squeue -u $USER -o "%.10i %.14j %.8T %.10M %R"
```

Training jobs are GPU-agnostic (`--gpus=1`, any card; `cluster/gpu_adapt.sh` adds `--data-on-cpu` on cards under 40 GB). Each full-resolution job asks for 160 GB host RAM; if jobs sit pending on memory, add
`--mem=100G` and `--data-on-cpu` (pool 2/4 jobs need far less, `--mem=60G` is enough).
Extra throughput: `--partition=scavenge_gpu` on seed repeats (cheap to rerun if preempted).
