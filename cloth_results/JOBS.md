# Job log (Bouchet)

One row per submitted job or job matrix. Fill in the Slurm job id at submission, update
`status` when `bash cluster/jobs.sh` is run, and link the result row in RESULTS.md /
DATASETS.md when done. Status: `queued` `running` `done` `failed` `cancelled` `planned`.

Conventions: `S` = `/home/naa79/scratch_pi_sz622/naa79/cloth_data/ds768_night_splits`;
all `sbatch` commands run from `cloth_data_generation/transfer` on the cluster; single
training configs use `cluster/run.sbatch` (one GPU per job, `--out runs/<id>` so the
metrics directory matches the row). Snapshot the queue with `bash cluster/jobs.sh` and
paste the output into a chat with Claude to update this file.

| id | job id(s) | submitted | status | what / why | command | result |
|---|---|---|---|---|---|---|
| J-001 | arrays `cloth_gen` | 2026-10-27 night | done | ds768_night: 3000 slots at 768, L40S + scavenge | README 3b | 1,115 samples, DATASETS.md |
| J-002 | | 2026-10-28 | running | stability sweep substeps 60/90/150 on failed force configs; find a setting where the velocity clamp does not tear the sheet (magnitude identifiability) | `cluster/stability.py --dataset ds768_night` then `--report` | |
| J-003 | 31144541 (earlier attempts 31134453, 31139874) | 2026-10-28 | running | ladder run 2, rungs 0,1,2,3,3b sequentially (D4-symmetric metric, TTA off), one GPU | `sbatch cluster/ladder.sbatch $S 300` | ROADMAP ladder table |
| J-004 | | | planned | ds512_v1 generation tonight (~8 pm): 2000 slots at 512, L40S + scavenge; resolution comparison and a second dataset | ROADMAP "Queued work" | |

Submitted 2026-10-28 ~13:00: M1 (31153218-24), M2 (31153226-28), M3 (31153239-42), M1a-d ran on L40S; the other ten were resubmitted ~13:30 as GPU-agnostic jobs (31155587-96) after an in-queue scontrol edit dropped them.

## Matrix M1: noise floor and capacity (submit alongside the ladder, ~10-20 min each)

Why: ladder rungs differ by ~0.5 cm; without seed repeats we cannot tell a rung effect
from training noise. All CNNs underfit (train = val = test), so capacity and epochs are
the cheapest levers before more data.

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M1-a | 31153218 | done | rung 0 seed 1 | `sbatch --job-name=M1a cluster/run.sbatch $S --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --seed 1 --out runs/M1a` | |
| M1-b | 31153219 | done | rung 0 seed 2 | `sbatch --job-name=M1b cluster/run.sbatch $S --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --seed 2 --out runs/M1b` | |
| M1-c | 31153220 | done | rung 2 seed 1 | `sbatch --job-name=M1c cluster/run.sbatch $S --model cnn_reg --epochs 300 --amp --tta 0 --no-vel --eval-rotated --seed 1 --out runs/M1c` | |
| M1-d | 31153221 | done | rung 2 seed 2 | `sbatch --job-name=M1d cluster/run.sbatch $S --model cnn_reg --epochs 300 --amp --tta 0 --no-vel --eval-rotated --seed 2 --out runs/M1d` | |
| M1-e | 31155587 (resubmitted; 31153222 lost to scontrol TresPerJob update) | running | rung 1, width 48 | `sbatch --job-name=M1e cluster/run.sbatch $S --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --width 48 --out runs/M1e` | |
| M1-f | 31155588 (resubmitted; 31153223 lost to scontrol TresPerJob update) | running | rung 1, width 64 | `sbatch --job-name=M1f cluster/run.sbatch $S --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --width 64 --out runs/M1f` | |
| M1-g | 31155589 (resubmitted; 31153224 lost to scontrol TresPerJob update) | running | rung 2, 1000 epochs | `sbatch --job-name=M1g cluster/run.sbatch $S --model cnn_reg --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --out runs/M1g` | |

## Matrix M2: how much resolution is needed (direction 2: images / depth; and the 512 run)

Why: if block-averaged 384 or 192 grids lose nothing, image-based observation does not need
768 and the 512 dataset is sufficient; training also gets 4-16x faster. First run of each
pool builds a new tensor cache (`*_pool2.pt`, `*_pool4.pt`), so submit one job per pool
first, then more.

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M2-a | 3115559x (resubmitted; 31153226 lost to scontrol TresPerJob update) | done | rung 1 at 384 (pool 2) | `sbatch --job-name=M2a cluster/run.sbatch $S --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --pool 2 --out runs/M2a` | |
| M2-b | 3115559x (resubmitted; 31153227 lost to scontrol TresPerJob update) | done | rung 1 at 192 (pool 4) | `sbatch --job-name=M2b cluster/run.sbatch $S --model cnn_reg --epochs 300 --no-rot --amp --tta 0 --no-vel --pool 4 --out runs/M2b` | |
| M2-c | 31155592 (resubmitted; 31153228 lost to scontrol TresPerJob update) | running | rung 2 at 192 (pool 4), 1000 epochs (cheap at this size) | `sbatch --job-name=M2c cluster/run.sbatch $S --model cnn_reg --epochs 1000 --amp --tta 0 --no-vel --eval-rotated --pool 4 --out runs/M2c` | |

## Matrix M3: point-cloud model (direction 3: non-grid representations)

Why: rung 3 is the representation that carries over to FEM meshes / crash. One config in
the ladder is not a sweep; points, width and depth are the knobs.

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M3-a | 3115559x (resubmitted; 31153239 lost to scontrol TresPerJob update) | done | pointtf 1024 pts, d128, 4 layers | `sbatch --job-name=M3a cluster/run.sbatch $S --model pointtf --epochs 300 --amp --tta 0 --no-vel --eval-rotated --bs 16 --lr 3e-4 --n-points 1024 --out runs/M3a` | |
| M3-b | 31155594 (resubmitted; 31153240 lost to scontrol TresPerJob update) | running | pointtf 8192 pts, d128, 4 layers, bs 8 | `sbatch --job-name=M3b cluster/run.sbatch $S --model pointtf --epochs 300 --amp --tta 0 --no-vel --eval-rotated --bs 8 --lr 3e-4 --n-points 8192 --out runs/M3b` | |
| M3-c | 31155595 (resubmitted; 31153241 lost to scontrol TresPerJob update) | running | pointtf 2048 pts, d256, 4 layers | `sbatch --job-name=M3c cluster/run.sbatch $S --model pointtf --epochs 300 --amp --tta 0 --no-vel --eval-rotated --bs 16 --lr 3e-4 --pt-dim 256 --out runs/M3c` | |
| M3-d | 31155596 (resubmitted; 31153242 lost to scontrol TresPerJob update) | running | pointtf 2048 pts, d128, 8 layers | `sbatch --job-name=M3d cluster/run.sbatch $S --model pointtf --epochs 300 --amp --tta 0 --no-vel --eval-rotated --bs 16 --lr 3e-4 --pt-layers 8 --out runs/M3d` | |

## Matrix M4: paper data (direction 1: plastic creasing)

Why: paper is the physical target. A small 256 set tells us the QC pass rate (stretch
limit 1.3) and whether creases make the inverse ambiguous before spending 768 budget.
Runs on scavenge so it never competes with the ladder or ds512.

| id | job id | status | what | command | result |
|---|---|---|---|---|---|
| M4-a | | planned | ds256_paper: 400 slots at 256, paper material, scavenge | `python3 cluster/submit.py --n 400 --nx 256 --chunk 20 --material paper --name ds256_paper --partition scavenge_gpu --time 02:00:00 --max-concurrent 20` | |
| M4-b | | planned | train rung 1 on ds256_paper once packaged | `sbatch --job-name=M4b cluster/run.sbatch /home/naa79/scratch_pi_sz622/naa79/cloth_data/ds256_paper_splits --model cnn_reg --epochs 600 --no-rot --amp --tta 0 --no-vel --out runs/M4b` | |

## Submitting a whole matrix at once

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
