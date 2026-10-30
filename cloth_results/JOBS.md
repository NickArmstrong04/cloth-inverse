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
| J-005 | 31345274 (`gpu`, tasks 0-29); scavenge array id pending | 2026-10-29 night | generated (merge pending) | ds256_const: first dataset under the constant-force model (no ramp, every substep, capture at release, 5 snapshots with velocities), free-flight cloth at 256, magnitudes halved to 0.15-1.25 N (full impulse tore 1/3 of attempts at 0.3-2.5 N). Generation is ~9x faster per attempt (CUDA graph, solver reuse, vectorized QC). 3000 slots, seeds 100000+slot (same force configs per slot as ds256_free apart from the magnitude draw) | `python3 cluster/submit.py --n 3000 --name ds256_const --tasks 0-29 --max-concurrent 30 --no-merge` then `--tasks 30-59 --partition scavenge_gpu --max-concurrent 30 --no-merge`; morning: `merge_summaries.py` + `package_splits.py` | 2026-10-30 morning: 3,269 attempts, 1,109 rejected (34%), all QC (no NaN, no capture failures): stretch 2.0-2.5 is the bulk (elastic stretch under load at release), tail to 37.9 (real tearing), ~130 also failed piercing (>100 pairs). Passing set biased to gentler draws; recalibrate the stretch gate / magnitude range from `rejected/` before the next generation run. Final counts after merge |
| J-006 | | planned (2026-10-30) | planned | ds256_const_paper: the ds256_const scenario on a creasing paper sheet (`--material paper`, yield 50 /m, plastic hinges, near-inextensible, mass 0.08). Same seeds and slots as ds256_const (slot k = identical force draw), captured at release with settle 0, so the two datasets are paired. Workstation smoke 2026-10-30 under the constant-force model: 128: 2/2 slots pass (4 attempts), 256: 4/4 slots pass (6 attempts, 2 rejected by stretch 1.49/1.93 > 1.3), no NaN at full impulse, stretch of passing samples 1.07-1.29, vmax 0.63-0.74 m/s at 256, ~24 s per sample at 2 workers. The paper stretch gate (1.3) was set on settled sheets (~1.05); under load at release paper reaches 1.2-1.3 in healthy samples, so expect a ~30% rejection rate like cloth | `bash cluster/paper.sh devel` (4 samples on gpu_devel), then `bash cluster/paper.sh full` (tasks 0-29 gpu, 30-59 scavenge_gpu, no merge); after the arrays `bash cluster/paper.sh merge`, then `bash cluster/paper.sh compare` for paired stats vs ds256_const | |

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
| M8-a | 31295878 (attempts 31280929 val NaN, 31290287 val split empty) | done 2026-10-29 (~1 h) | rung 1 (positions, known pose), all ~2,500 train samples | `sbatch --job-name=M8a --mem=60G cluster/run.sbatch $S2 --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --out runs/M8a` | test 6.4 cm / 15.9 deg (val 6.5 / 15.3). New data is harder than the old 768 data pooled to 256 (M8c 5.2): same model, 2.2x the samples, 1.2 cm worse |
| M8-b | 31295879 (attempts 31280930 val NaN, 31290288 val split empty) | done 2026-10-29 | rung 1, 1,100-sample subset (size control) | `sbatch --job-name=M8b --mem=60G cluster/run.sbatch $S2 --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --n-train 1100 --out runs/M8b` | test 7.2 cm / 17.2 deg (val 7.3 / 17.3). 1,100 -> 2,453 samples buys 0.8 cm / 1.3 deg (M8a); seed noise is ~0.3 cm (M8f), so still data-limited but with a shallow slope |
| M8-c | 31280931 | done | rung 1, old data pooled to 256 (resolution control; compare M1e/M2a) | `sbatch --job-name=M8c --mem=60G cluster/run.sbatch $S --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --pool 3 --out runs/M8c` | test 5.2 cm / 18.0 deg (train 4.7 / 14.6, val 4.9 / 16.6), 10 min. Pool 3 of the old data costs 0.7 cm vs M1e/M2a (4.5) |
| M8-d | 31295880 (attempts 31280932 val NaN, 31290289 val split empty) | done 2026-10-29 (1 h 19) | rung 2 (unknown pose), all data, 1000 epochs (compare M2c 5.9 cm, M1g) | `sbatch --job-name=M8d --mem=60G cluster/run.sbatch $S2 --model cnn_reg --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --out runs/M8d` | test 6.9 cm / 21.7 deg (val 6.8 / 21.8, train 6.6). Rung 2 on all 2,453 samples is no better than on 1,100 (M8e 7.0 / 22.3): unknown pose is not data-limited, as on the old data |
| M8-e | 31295881 (attempts 31280933 val NaN, 31290290 val split empty) | done 2026-10-29 | rung 2, 1,100-sample subset | `sbatch --job-name=M8e --mem=60G cluster/run.sbatch $S2 --model cnn_reg --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --n-train 1100 --out runs/M8e` | test 7.0 cm / 22.3 deg (val 7.5 / 21.8); rung 2 at 1,100 samples. Compare M8d when done |
| M8-f | 31295882 (attempts 31280934 val NaN, 31290291 val split empty) | done 2026-10-29 | rung 1 seed repeat (noise floor at this size) | `sbatch --job-name=M8f --mem=60G cluster/run.sbatch $S2 --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --seed 2 --out runs/M8f` | test 6.1 cm / 15.0 deg (val 6.4 / 14.8). Seed spread vs M8a: 0.3 cm / 0.9 deg |

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
| M9-a | 31295883 (attempt 31290292 val split empty) | done 2026-10-29 | rung 1 + aux force-field loss (vs M8a) | `sbatch --job-name=M9a --mem=60G cluster/run.sbatch $S2 --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --aux-w 1 --out runs/M9a` | **test 5.7 cm / 14.4 deg** (val 5.8 / 14.9). Aux force-field loss beats M8a by 0.7 cm / 1.5 deg, twice the seed spread: dense supervision is worth about as much as doubling the data |
| M9-b | 31295884 (attempt 31290293 val split empty) | done 2026-10-29 (1 h 37) | rung 1 + post-force frames (4x samples) | `sbatch --job-name=M9b --mem=100G --time=08:00:00 cluster/run.sbatch $S2 --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --frames post --out runs/M9b` | test 6.7 cm / 17.8 deg (val 6.6 / 16.7). Post-force frames hurt vs M8a (6.4 / 15.9): val bottomed at ep ~100 then rose. The frames are not new information about the force, just re-labelled copies |
| M9-c | 31295885 (attempt 31290294 val split empty) | done 2026-10-29 (1 h 40) | rung 1 + aux + frames | `sbatch --job-name=M9c --mem=100G --time=08:00:00 cluster/run.sbatch $S2 --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --aux-w 1 --frames post --out runs/M9c` | test 6.3 cm / 14.7 deg (val 6.5 / 15.2). Aux + frames: better than M8a but 0.6 cm worse than aux alone (M9a). Frames lever rejected |
| M9-d | 31295886 (attempt 31290295 val split empty) | done 2026-10-29 (1 h 57) | rung 3 pointtf + per-point aux, 1000 ep (vs M3/M5e) | `sbatch --job-name=M9d --mem=60G cluster/run.sbatch $S2 --model pointtf --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --bs 16 --lr 3e-4 --aux-w 1 --out runs/M9d` | test 7.6 cm / 22.3 deg (val 7.7 / 22.8, train 7.2). pointtf + per-point aux at rung 2, 0.6 cm behind the CNN M8d; still train = test (underfits) |
| M9-e | 31295887 (attempt 31290296 val split empty) | done 2026-10-29 | rung 1 with frozen DINOv2 on the grid image (+aux) | `sbatch --job-name=M9e --mem=60G cluster/run.sbatch $S2 --model dino_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --lr 3e-4 --aux-w 1 --out runs/M9e` | test 8.8 cm / 20.2 deg (val 8.7 / 20.0). Frozen DINOv2 on the grid image is 2.4 cm worse than the scratch CNN (M8a): generic image features do not help on the material-grid representation |
| M9-f | 31295888 (attempt 31290297 val split empty) | **failed at final eval** after 1000 ep (1 h 11); model.pt saved | rung 4a: pointtf on the visible points of one view, from scratch | `sbatch --job-name=M9f --mem=60G cluster/run.sbatch $S2 --model pointtf --epochs 1000 --amp --tta 0 --no-vel --view --bs 16 --lr 3e-4 --aux-w 1 --out runs/M9f` | exit 1 at the eval: `predict` built the dummy target on the data device (CPU on this <40 GB card, `--data-on-cpu`) while R sat on the GPU. Fixed in train.py (device = g.device) + new `--eval-only`. Recover with M9-f2 (no retraining) |
| M9-f2 | (id pending) | submitted 2026-10-29 ~16:30 | `sbatch --job-name=M9f2 --mem=60G --time=00:30:00 cluster/run.sbatch $S2 --model pointtf --epochs 1000 --amp --tta 0 --no-vel --view --bs 16 --lr 3e-4 --aux-w 1 --eval-only --init runs/M9f/model.pt --out runs/M9f` | |
| M9-j | (id pending) | submitted 2026-10-29 ~16:30 | `sbatch --job-name=M9j --mem=60G cluster/run.sbatch $S2 --model cnn_reg --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --aux-w 1 --out runs/M9j` | |
| M9-k | (id pending) | submitted 2026-10-29 ~16:30 | `sbatch --job-name=M9k --mem=60G cluster/run.sbatch $S2 --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --aux-w 1 --seed 2 --out runs/M9k` | |
| M9-g | (id pending) | submitted 2026-10-29 ~16:30 | `sbatch --job-name=M9g --mem=60G cluster/run.sbatch $S2 --model pointtf --epochs 1000 --amp --tta 0 --no-vel --view --bs 16 --lr 3e-4 --aux-w 1 --init runs/M9d/model.pt --out runs/M9g` | |
| M9-h | 31295889 (attempt 31290298 val split empty) | done 2026-10-29 | rung 4a: DINOv2 on the rendered depth image (+aux) | `sbatch --job-name=M9h --mem=60G cluster/run.sbatch $S2 --model dino_reg --epochs 300 --amp --tta 0 --no-vel --view --lr 3e-4 --aux-w 1 --out runs/M9h` | test 9.2 cm / 28.2 deg (val 9.2 / 29.1). Frozen DINOv2 on the rendered depth view: worse than the scratch CNN control (M9i) |
| M9-i | 31295890 (attempt 31290299 val split empty) | done 2026-10-29 | rung 4a: from-scratch CNN on the rendered depth image (control for M9-h) | `sbatch --job-name=M9i --mem=60G cluster/run.sbatch $S2 --model cnn_reg --epochs 300 --amp --tta 0 --no-vel --view --aux-w 1 --out runs/M9i` | test 8.5 cm / 25.0 deg (val 8.6 / 23.5). First rung-4a number: scratch CNN on one orthographic depth+mask view, unknown pose. Compare with M8d (rung 2, full data) for the cost of seeing one side; beats both DINO runs |

Reading: M9-a/b/c vs M8a say what each lever buys at rung 1; M9-e vs M8a is the pretrained-feature
test; M9-f/g/h/i are the first rung-4a numbers (gap to rung 3 = cost of seeing one side);
M9-h vs M9-i is pretrained vs scratch on images. Gather: `python3 cluster/summarize_runs.py 'M8*' 'M9*'`.

**2026-10-29 16:05 outcome (12 of 13 done, M9-f eval crashed, recoverable).** Full reading in
`RESULTS.md` (2026-10-29 section). Short form: data still helps at rung 1 (7.2 -> 6.4 cm for 2.2x,
seed spread 0.3) but not at rung 2 (7.0 -> 6.9); the new free-flight data is ~1.2 cm harder than
the old data; the aux force-field loss is the one lever that works (5.7 cm); frames and frozen
DINOv2 hurt; first rung-4a numbers 8.5 cm / 25 deg (CNN on a depth view). Next: M9-f2, M9-g, M9-j,
M9-k, and the overnight refill (M10) since rung 1 is still on the data slope.

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

## Matrix M11: architecture levers (2026-10-29 evening)

Why: rung 2 is optimisation-limited (train = test at 6-8 cm for every size, representation and
data amount) and the aux force-field loss was the only lever that moved rung 1. The architecture
review (chat 2026-10-29) traced both to the model: SO(3) invariance learned from augmentation on
2.4k samples, global pooling discarding location, and an MSE head averaging a multimodal posterior.
Five levers, all in `inverse/` (geometry.py, models.py, train.py, select_resim.py; smoke-tested in
WSL on the 128 splits, 2 epochs each, all configurations run end to end):

1. `--intrinsic`: strain + curvature in material coordinates as the input (rotation-invariant
   by construction; direction predicted in the local tangent frame and mapped back). Verified
   numerically: features change by 1e-6 under a random rotation, directions rotate exactly.
   Note: with intrinsic inputs the D4 relabeling is off by default (the metric is stricter than
   M8d/M9j's, so a win is a clean win); add `--sym on` to grade on the same lenient metric.
2. `--model cnn_field`: the dense force field is the primary output (presence map + field map,
   decoded by two NMS peaks with a windowed soft-argmax); the (2, 7) regression is a light
   second term. Permutation / D4 never enter the dense losses. `--aux-w` is forced to >= 1.
3. `--flow --flow-k K`: conditional flow-matching head sampling K candidate force pairs;
   metrics report best-of-K (oracle) and mean-of-K; `select_resim.py` picks among them by
   re-simulation (`cluster/resim.sbatch`). The regression head is trained alongside.
4. `--model sonata_reg`: Sonata (CVPR 2025, self-supervised PTv3) fine-tuned on the point cloud,
   rungs 3 / 4a. UNTESTED (needs `bash cluster/setup_sonata.sh` on the login node first).
5. `--view-shade` + `--dino-finetune`: rung 4a with a shaded render (depth, mask, Lambertian
   shade) and the DINO backbone unfrozen; DINOv3 via `--dino-repo facebookresearch/dinov3
   --dino-arch dinov3_vits16 --dino-weights <pth>` (gated weights, request access on HF).

Baselines to beat: rung 1 M9a 5.7 / 14.4 (seed spread 0.3 cm); rung 2 M8d 6.9 / 21.7 (sym metric),
M9j pending; rung 4a M9i 8.5 / 25.0, M9h (frozen DINO) 9.2 / 28.2.

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M11-a | 31317948 (attempt 31315352: old train.py, unpushed commit) | done 2026-10-29 (1 h 11) | rung 2, intrinsic inputs + aux, 1000 ep (the main test: does built-in invariance remove the optimisation limit? vs M9j / M8d) | `sbatch --job-name=M11a --mem=60G cluster/run.sbatch $S2 --model cnn_reg --intrinsic --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --aux-w 1 --out runs/M11a` | test **5.8 cm** / 38.4 deg (val 5.9 / 39.1, train 5.1 / 16.0), strict metric. Position at rung 2 reaches the rung-1 level (M9a 5.7) and beats M8d 6.9 on a stricter metric: pose was the position problem. Direction collapsed (M9a 14 deg): the intrinsic path zeroed the centroid drift, which is the direction cue (fixed in M12) |
| M11-a2 | 31317949 (attempt 31315354) | done 2026-10-29 (1 h 42) | same, graded on the lenient D4 metric for a like-for-like with M8d | `sbatch --job-name=M11a2 --mem=60G cluster/run.sbatch $S2 --model cnn_reg --intrinsic --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --aux-w 1 --sym on --out runs/M11a2` | test 7.0 / 37.5 (val 7.4 / 39.3, train 7.0 / 36.4). With the 16-way D4 equivalence the invariant model cannot even fit train; the lenient metric is wrong for an invariant model (its directions are world vectors, the D4 element is applied in material coordinates). Grade intrinsic runs strictly |
| M11-b | 31317950 (attempt 31315355) | done 2026-10-29 (25 min) | rung 1, intrinsic + aux, 300 ep (does invariance cost anything at known pose? vs M9a) | `sbatch --job-name=M11b --mem=60G cluster/run.sbatch $S2 --model cnn_reg --intrinsic --epochs 300 --no-rot --amp --tta 0 --no-vel --aux-w 1 --out runs/M11b` | test 6.2 / 37.8 (val 5.9 / 38.2, train 5.7 / 22.4). Known pose: +0.5 cm vs M9a 5.7 and direction 14 -> 38 deg, same drift-cue loss as M11-a |
| M11-c | 31317951 (attempt 31315356) | done 2026-10-29 (52 min) | rung 1, dense field as primary head, 300 ep (vs M9a) | `sbatch --job-name=M11c --mem=60G cluster/run.sbatch $S2 --model cnn_field --epochs 300 --no-rot --amp --tta 0 --no-vel --aux-w 1 --out runs/M11c` | test 9.8 / 41.4 (val 10.1, train 10.4). Dense-field decoding (NMS peaks + soft-argmax) is far behind the regression head at known pose (M9a 5.7 / 14.4) and underfits (train 10.4). Dropped as a primary head; keep the field as aux loss |
| M11-d | 31317952 (attempt 31315357) | running 2026-10-29 (ep 875/1000 at 2 h 51; val loss flat at 2.60 since ep ~500) | rung 2, dense field head, 1000 ep (vs M9j) | `sbatch --job-name=M11d --mem=60G cluster/run.sbatch $S2 --model cnn_field --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --aux-w 1 --out runs/M11d` | pending; expect ~M11-c level |
| M11-e | 31317953 (attempt 31315358) | running 2026-10-29 (ep 875/1000; train loss 0.28 vs val 2.21, best 2.03 at ep ~500) | rung 2, dense field head + intrinsic inputs, 1000 ep (levers 1+2) | `sbatch --job-name=M11e --mem=60G cluster/run.sbatch $S2 --model cnn_field --intrinsic --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --aux-w 1 --out runs/M11e` | pending; first rung-2 run that overfits (train << val), i.e. the invariant input removed the optimisation limit |
| M11-f | 31317954 (attempt 31315359) | done 2026-10-29 (21 min) | rung 1, flow head (8 candidates) + aux, 300 ep: oracle best-of-8 vs mean-of-8 says how multimodal the posterior is | `sbatch --job-name=M11f --mem=60G cluster/run.sbatch $S2 --model cnn_reg --flow --flow-k 8 --epochs 300 --no-rot --amp --tta 0 --no-vel --aux-w 1 --out runs/M11f` | regression head test 5.6 / 14.5 (= M9a, flow head costs nothing). Flow candidates: best-of-8 **3.7 cm / 17.0 deg**, mean-of-8 6.9 cm (val 3.8 / 7.3). Oracle selection is 1.9 cm better than the point estimate: the posterior is wide/multimodal and the re-sim selector (M11-f2) is worth running |
| M11-f2 | 31317955 failed (submitted together with M11-f, ran before test_candidates.json existed; exit 1 in 1 s) | resubmit (M12-f2) | re-sim selection among M11-f's candidates, first 50 test samples (50 x 8 sims, ~5 h) | `sbatch --job-name=M11f2 cluster/resim.sbatch $S2 runs/M11f --limit 50 --k 8` |  |
| M11-g | 31317956 (attempt 31315360) | done 2026-10-29 (30 min) | rung 4a, scratch CNN on the shaded view (control for M11-h; vs M9i) | `sbatch --job-name=M11g --mem=60G cluster/run.sbatch $S2 --model cnn_reg --epochs 300 --amp --tta 0 --no-vel --view --view-shade --aux-w 1 --out runs/M11g` | test 8.8 / 24.4 (val 8.9, train 8.6). Shaded view = depth+mask view (M9i 8.5 / 25.0): shading adds nothing for the scratch CNN |
| M11-h | 31317957 (attempt 31315361) | done 2026-10-29 (33 min) | rung 4a, DINOv2-S/14 UNFROZEN on the shaded view (vs M9h frozen 9.2, M9i 8.5) | `sbatch --job-name=M11h --mem=60G cluster/run.sbatch $S2 --model dino_reg --dino-finetune --epochs 300 --amp --tta 0 --no-vel --view --view-shade --lr 1e-4 --aux-w 1 --out runs/M11h` | test 9.5 / 25.5 (val 9.5, train 9.3). Unfrozen DINOv2-S/14 on the shaded view: 0.7 cm behind the scratch CNN (M11-g), no better than frozen (M9h 9.2). Pretrained image features do not help at this data size even fine-tuned |
| M11-i | | optional | rung 4a, DINOv3 ViT-S/16 unfrozen (after the gated weights are in `~/.cache/dinov3/`; `--view-size 224` is a multiple of 16) | `sbatch --job-name=M11i --mem=60G cluster/run.sbatch $S2 --model dino_reg --dino-repo facebookresearch/dinov3 --dino-arch dinov3_vits16 --dino-weights ~/.cache/dinov3/dinov3_vits16_pretrain_lvd1689m.pth --dino-finetune --epochs 300 --amp --tta 0 --no-vel --view --view-shade --lr 1e-4 --aux-w 1 --out runs/M11i` | |
| M11-j | 31317961 failed: `ModuleNotFoundError: No module named 'sonata'` (attempt 31315362 old train.py) | needs `bash cluster/setup_sonata.sh` on the login node first | rung 3, Sonata fine-tuned on the full point cloud (vs M9d 7.6 / 22.3, pointtf) | `sbatch --job-name=M11j --mem=60G cluster/run.sbatch $S2 --model sonata_reg --epochs 300 --amp --tta 0 --no-vel --eval-rotated --bs 8 --lr 1e-4 --aux-w 1 --out runs/M11j` |  |
| M11-k | 31317962 failed: same | after setup_sonata.sh | rung 4a, Sonata on the visible points (vs M9f2/M9g) | `sbatch --job-name=M11k --mem=60G cluster/run.sbatch $S2 --model sonata_reg --epochs 300 --amp --tta 0 --no-vel --view --bs 8 --lr 1e-4 --aux-w 1 --out runs/M11k` |  |

Submit (CPU allocation, from `cloth_data_generation/transfer`, after `git pull`):

```bash
bash cluster/interactive.sh
cd /home/naa79/scratch_pi_sz622/naa79/cloth-inverse && git pull && cd cloth_data_generation/transfer
S2=/home/naa79/scratch_pi_sz622/naa79/cloth_data/ds256_free_splits
# M11-a..h: paste the sbatch lines above
# Sonata (once, on the login node, needs internet; then M11-j/k):
bash cluster/setup_sonata.sh
squeue -u $USER -o "%.10i %.14j %.8T %.10M %R"
```

Reading: M11-a vs M9j is the decisive number (if rung 2 drops to the rung-1 level, pose was the
whole optimisation problem); M11-b says whether intrinsic inputs cost anything at known pose;
M11-c/d say whether the dense head beats the regression head; M11-f's best-of-8 vs mean-of-8 gap
is the multimodality measurement and M11-f2 says how much of it re-sim selection recovers;
M11-g/h repeat the pretrained-vs-scratch test with a fair input. Gather:
`python3 cluster/summarize_runs.py 'M11*'`.

### M11 outcome (2026-10-29 evening, 9 of 13 done)

Collected with `bash cluster/collect.sh M11`. Position: intrinsic inputs take rung 2 from 6.9 to
5.8 cm on a stricter metric (M11-a), the rung-1 level, and M11-e is the first unknown-pose run with
train loss far below val. Direction: every intrinsic run sits at 38 deg because `IntrinsicInput`
zeroed the global vector (centroid drift), which is how the baseline reads the force direction (a
push moves the whole sheet). Flow head: best-of-8 is 1.9 cm better than the point estimate at rung 1.
Dense-field primary head, shaded views and unfrozen DINOv2 all lose. Full table in RESULTS.md.

## Matrix M12: intrinsic inputs + drift channel (2026-10-29 night)

Change (commit after 5f41ae1): the intrinsic path now also receives the centroid drift (and mean
velocity when `--no-vel` is off) as its components in every cell's local frame (`F^T g`, 3 channels
per vector), so the drift cue is kept and the input stays rotation-invariant (checked: features
move 1e-4 under a random rotation, directions equivariant to 2e-6). Smoke-tested locally on the
128 splits (cnn_reg / cnn_field + flow / with velocity). Also resubmits the re-sim selector with a
dependency and the Sonata rows after the install.

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M12-a | | planned | rung 2, intrinsic + drift channel + aux, 1000 ep (vs M11-a 5.8 / 38.4; target: position 5.8, direction back to ~20 deg) | `sbatch --job-name=M12a --mem=60G cluster/run.sbatch $S2 --model cnn_reg --intrinsic --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --aux-w 1 --out runs/M12a` | |
| M12-b | | planned | rung 1, intrinsic + drift, 300 ep (vs M11-b 6.2 / 37.8, M9a 5.7 / 14.4) | `sbatch --job-name=M12b --mem=60G cluster/run.sbatch $S2 --model cnn_reg --intrinsic --epochs 300 --no-rot --amp --tta 0 --no-vel --aux-w 1 --out runs/M12b` | |
| M12-c | | planned | rung 2, intrinsic + drift + flow head (8 candidates), 1000 ep: rung-2 candidates for re-sim selection | `sbatch --job-name=M12c --mem=60G cluster/run.sbatch $S2 --model cnn_reg --intrinsic --flow --flow-k 8 --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --aux-w 1 --out runs/M12c` | |
| M12-d | | planned | rung 2, intrinsic + drift, with velocities (the mean velocity enters the same way; vs M12-a) | `sbatch --job-name=M12d --mem=60G cluster/run.sbatch $S2 --model cnn_reg --intrinsic --epochs 1000 --amp --tta 0 --eval-rotated --aux-w 1 --out runs/M12d` | |
| M12-e | | planned | rung 2, intrinsic + drift, seed 2 (spread) | `sbatch --job-name=M12e --mem=60G cluster/run.sbatch $S2 --model cnn_reg --intrinsic --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --aux-w 1 --seed 2 --out runs/M12e` | |
| M12-f2 | | planned | re-sim selection on M11-f's candidates (file now exists), first 50 test samples, 8 candidates each (~5 h) | `sbatch --job-name=M12f2 cluster/resim.sbatch $S2 runs/M11f --limit 50 --k 8` | |
| M12-c2 | | after M12-c | re-sim selection on the rung-2 candidates | `sbatch --job-name=M12c2 --dependency=afterok:<M12c id> cluster/resim.sbatch $S2 runs/M12c --limit 50 --k 8` | |
| M12-j | | after setup_sonata.sh | rung 3, Sonata fine-tuned (M11-j rerun) | `sbatch --job-name=M12j --mem=60G cluster/run.sbatch $S2 --model sonata_reg --epochs 300 --amp --tta 0 --no-vel --eval-rotated --bs 8 --lr 1e-4 --aux-w 1 --out runs/M12j` | |

Submit (from `cloth_data_generation/transfer`):

```bash
cd /home/naa79/scratch_pi_sz622/naa79/cloth-inverse && git pull && cd cloth_data_generation/transfer
S2=/home/naa79/scratch_pi_sz622/naa79/cloth_data/ds256_free_splits
grep -c n_glob ../inverse/models.py     # non-zero = the drift-channel commit is in
# M12-a..e, M12-f2: paste the sbatch lines above; M12-c2 once M12-c has an id
bash cluster/setup_sonata.sh && sbatch ... M12j
bash cluster/collect.sh M12             # when done; paste the output
```

Reading: M12-a vs M11-a is the direction test (position should stay at 5.8, direction should return
to the baseline 20 deg or better); M12-b vs M9a says whether the invariant model now matches the
world-frame model at known pose; M12-c + M12-c2 is the first rung-2 re-sim selection.

## Matrix M13: the M12 recipes on the constant-force data (2026-10-30)

ds256_const (J-005) is the first dataset under the constant-force model: captured at release with
the force still acting, velocities stored, no settle. It is not comparable to ds256_free, so the
matrix carries its own rung-1/rung-2 baselines (M9a / M9j recipes) next to the intrinsic+drift
runs planned as M12. Two new questions the data makes possible: the release velocity field now
encodes the impulse directly, so runs with velocities (no `--no-vel`) are in; and the captured
shape is a taut band or early fold rather than a settled fold, so the baseline error itself says
whether the new model is easier or harder to invert. `--frames post` is useless here (every
snapshot is at or before t_end); `--frames all` would use them but is not in this matrix.

S3=/home/naa79/scratch_pi_sz622/naa79/cloth_data/ds256_const_splits

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M13-a | | planned | rung 1 baseline, cnn_reg + aux, 300 ep (M9a recipe; vs M9a 5.7 / 14.4 on ds256_free) | `sbatch --job-name=M13a --mem=60G cluster/run.sbatch $S3 --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --aux-w 1 --out runs/M13a` | |
| M13-b | | planned | rung 1 baseline WITH release velocities (vs M13-a: how much the velocity field adds now that it is the impulse) | `sbatch --job-name=M13b --mem=60G cluster/run.sbatch $S3 --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --aux-w 1 --out runs/M13b` | |
| M13-c | | planned | rung 2 baseline, cnn_reg + aux, 1000 ep (M9j recipe) | `sbatch --job-name=M13c --mem=60G cluster/run.sbatch $S3 --model cnn_reg --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --aux-w 1 --out runs/M13c` | |
| M13-d | | planned | rung 2, intrinsic + drift + aux, 1000 ep (M12-a recipe; the main run) | `sbatch --job-name=M13d --mem=60G cluster/run.sbatch $S3 --model cnn_reg --intrinsic --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --aux-w 1 --out runs/M13d` | |
| M13-e | | planned | rung 2, intrinsic + drift, with velocities (M12-d recipe; vs M13-d) | `sbatch --job-name=M13e --mem=60G cluster/run.sbatch $S3 --model cnn_reg --intrinsic --epochs 1000 --amp --tta 0 --eval-rotated --aux-w 1 --out runs/M13e` | |
| M13-f | | planned | rung 1, intrinsic + drift, 300 ep (M12-b recipe; vs M13-a) | `sbatch --job-name=M13f --mem=60G cluster/run.sbatch $S3 --model cnn_reg --intrinsic --epochs 300 --no-rot --amp --tta 0 --no-vel --aux-w 1 --out runs/M13f` | |
| M13-g | | planned | rung 2, intrinsic + drift + flow head (8 candidates), 1000 ep (M12-c recipe; candidates for re-sim) | `sbatch --job-name=M13g --mem=60G cluster/run.sbatch $S3 --model cnn_reg --intrinsic --flow --flow-k 8 --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --aux-w 1 --out runs/M13g` | |
| M13-g2 | | after M13-g | re-sim selection on M13-g's candidates, first 50 test samples (resim.sbatch must use the constant-force flags, which are now the generate_sample defaults) | `sbatch --job-name=M13g2 --dependency=afterok:<M13g id> cluster/resim.sbatch $S3 runs/M13g --limit 50 --k 8` | |

Submit (CPU allocation, from `cloth_data_generation/transfer`, after the merge + package of J-005):

```bash
cd /home/naa79/scratch_pi_sz622/naa79/cloth-inverse && git pull && cd cloth_data_generation/transfer
S3=/home/naa79/scratch_pi_sz622/naa79/cloth_data/ds256_const_splits
ls $S3/train/*.h5 | wc -l; ls $S3/test/*.h5 | wc -l      # ~2300 / ~290 expected
# M13-a..g: paste the sbatch lines above; M13-g2 once M13-g has an id
squeue -u $USER -o "%.10i %.14j %.8T %.10M %R"
bash cluster/collect.sh M13             # when done; paste the output
```

Reading: M13-a vs M9a is the dataset-difficulty number (same recipe, same size, new force model);
M13-b vs M13-a is the value of the release velocity; M13-d vs M13-c repeats the M11-a test on this
data and, with the drift channel in, should hold direction near the baseline; M13-e vs M13-d says
whether the intrinsic model uses the velocity; M13-g's best-of-8 gap is the multimodality measurement.

## Matrix M14: force magnitude on the constant-force data (2026-10-30)

Finding (RESULTS.md, impulse analysis): on the M13 runs duration is solved (3-6% vs 19% floor) and
force magnitude is the whole error; impulse error = force error, so no target change helps. The
intrinsic input halves force error (0.33 -> 0.18) at both rungs, so the lever is how the magnitude
cue (strain at the patch) reaches the head. Three flag-gated changes (commit after 89bb08d; defaults
unchanged): `--pool-max` adds per-channel max pooling of the final map to the mean pooling (the patch
is sub-cell at the 8 px map, mean pooling dilutes its peak 64x); `--w-mag W` weights the
log-magnitude loss term; `--strain-scale S` scales the strain channels (default 10, so 1% strain is
0.1 before the learnable affine). Base recipe is M13-f (rung 1, intrinsic + drift, 300 ep: 0.19
force error), which matched the 1000-ep rung-2 run, so it is the cheap test bed. M14-e is the seed
spread without which 1-2 point gains are unreadable. `inverse/identifiability.py` (new) gives the
data's own floor: oracle magnitude error from peak strain at the true patch and from the logged
momentum, plus the delivered-momentum fraction and its spread.

S3=/home/naa79/scratch_pi_sz622/naa79/cloth_data/ds256_const_splits

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M14-0 | login node | planned | identifiability floor on the test split (CPU, ~1 min) | `../transfer/.venv/bin/python identifiability.py $S3/test` | |
| M14-a | | planned | M13-f + max pooling | `sbatch --job-name=M14a --mem=60G cluster/run.sbatch $S3 --model cnn_reg --intrinsic --epochs 300 --no-rot --amp --tta 0 --no-vel --aux-w 1 --pool-max --out runs/M14a` | |
| M14-b | | planned | M13-f + magnitude loss weight 4 | `sbatch --job-name=M14b --mem=60G cluster/run.sbatch $S3 --model cnn_reg --intrinsic --epochs 300 --no-rot --amp --tta 0 --no-vel --aux-w 1 --w-mag 4 --out runs/M14b` | |
| M14-c | | planned | M13-f + strain scale 100 | `sbatch --job-name=M14c --mem=60G cluster/run.sbatch $S3 --model cnn_reg --intrinsic --epochs 300 --no-rot --amp --tta 0 --no-vel --aux-w 1 --strain-scale 100 --out runs/M14c` | |
| M14-d | | planned | M13-f + all three | `sbatch --job-name=M14d --mem=60G cluster/run.sbatch $S3 --model cnn_reg --intrinsic --epochs 300 --no-rot --amp --tta 0 --no-vel --aux-w 1 --pool-max --w-mag 4 --strain-scale 100 --out runs/M14d` | |
| M14-e | | planned | M13-f seed 2 (seed spread control) | `sbatch --job-name=M14e --mem=60G cluster/run.sbatch $S3 --model cnn_reg --intrinsic --epochs 300 --no-rot --amp --tta 0 --no-vel --aux-w 1 --seed 2 --out runs/M14e` | |
| M14-f | | planned | M13-f width 64 (capacity control) | `sbatch --job-name=M14f --mem=60G cluster/run.sbatch $S3 --model cnn_reg --intrinsic --epochs 300 --no-rot --amp --tta 0 --no-vel --aux-w 1 --width 64 --out runs/M14f` | |
| M14-g | | planned | M14-d with release velocities (pairs with M13-e vs M13-f) | `sbatch --job-name=M14g --mem=60G cluster/run.sbatch $S3 --model cnn_reg --intrinsic --epochs 300 --no-rot --amp --tta 0 --aux-w 1 --pool-max --w-mag 4 --strain-scale 100 --out runs/M14g` | |
| M14-h | | after M14-d | rung 2 (M13-d recipe) + the winning flags, 1000 ep | `sbatch --job-name=M14h --mem=60G cluster/run.sbatch $S3 --model cnn_reg --intrinsic --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --aux-w 1 <winning flags> --out runs/M14h` | |

Submit (from `cloth_data_generation/transfer`):

```bash
cd /home/naa79/scratch_pi_sz622/naa79/cloth-inverse && git pull
S3=/home/naa79/scratch_pi_sz622/naa79/cloth_data/ds256_const_splits
cd cloth_data_generation/inverse && ../transfer/.venv/bin/python identifiability.py $S3/test; cd ../transfer
# M14-a..g: paste the sbatch lines above
squeue -u $USER -o "%.10i %.14j %.8T %.10M"
# when done: cd ../inverse && ../transfer/.venv/bin/python impulse_report.py runs/M13f runs/M13e runs/M14a runs/M14b runs/M14c runs/M14d runs/M14e runs/M14f runs/M14g
```

Reading: M14-e minus M13-f is the noise band; a lever counts only if it beats M13-f by more than that.
M14-0's strain-oracle error is the floor a magnitude reader can reach from shape alone; if the
delivered-momentum fraction has a wide IQR, velocity readers (M13-b/e, M14-g) are capped by it.
