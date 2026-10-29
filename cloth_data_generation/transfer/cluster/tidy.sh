#!/bin/bash
# Tidy the cluster working tree: Slurm logs of finished jobs, generation task logs, caches.
# Dry run by default (prints what it would do); add --apply to do it.
#   bash cluster/tidy.sh            # show
#   bash cluster/tidy.sh --apply    # do it
#   bash cluster/tidy.sh --apply --keep-days 14   # also tar logs/ folders older than 14 days
#
# What it does (nothing touches a queued or running job, and nothing touches data):
#   transfer/{run,ladder,train,util}_<jobid>.{out,err}
#     - job finished OK  -> moved to transfer/logs/<YYYY-MM-DD>/ (JOBS.md references these)
#     - job FAILED/CANCELLED/TIMEOUT and the log is not a training result -> deleted
#       (failed attempts are already explained in JOBS.md)
#     - empty .err files, and .err files holding only torch/xFormers warnings -> deleted
#   <DATA_DIR>/<dataset>/slurm/gen_<array>_<task>.{out,err}
#     - deleted once the task is no longer queued/running (slots resume from the h5 files,
#       submission.json and the .sbatch copies stay)
#   inverse/runs/<name>/ with neither metrics.json nor model.pt -> deleted (crashed before saving)
#   __pycache__ directories under the code tree -> deleted
#   logs/<date>/ older than --keep-days -> logs/<date>.tar.gz (default: never)
set -euo pipefail
APPLY=0; KEEP_DAYS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1 ;;
    --keep-days) KEEP_DAYS="$2"; shift ;;
    *) echo "usage: $0 [--apply] [--keep-days N]"; exit 1 ;;
  esac; shift
done
HERE="$(cd "$(dirname "$0")/.." && pwd)"          # transfer/
INV="$(dirname "$HERE")/inverse"
# shellcheck disable=SC1091
source "$HERE/cluster/cluster.env"
DATA_DIR="${DATA_DIR:?DATA_DIR missing in cluster.env}"
cd "$HERE"

say() { printf '%s\n' "$*"; }
do_rm() { if [[ $APPLY -eq 1 ]]; then rm -rf -- "$@"; fi; for f in "$@"; do say "  rm   $f"; done; }
do_mv() { local dst="$1"; shift; if [[ $APPLY -eq 1 ]]; then mkdir -p "$dst"; mv -- "$@" "$dst/"; fi; for f in "$@"; do say "  mv   $f -> $dst/"; done; }

# Job ids still active: never touch their logs.
ACTIVE="$(squeue -u "$USER" -h -o '%i' | sed 's/_.*//' | sort -u | tr '\n' ' ')"
is_active() { [[ " $ACTIVE " == *" $1 "* ]]; }
# One sacct call for every job id we are about to look at.
state_of() { sacct -j "$1" -X -n -o State 2>/dev/null | head -n 1 | tr -d ' ' | sed 's/+$//'; }

say "## training logs in $HERE"
shopt -s nullglob
for out in run_*.out ladder_*.out train_*.out util_*.out; do
  id="${out##*_}"; id="${id%.out}"; err="${out%.out}.err"
  if is_active "$id"; then say "  keep $out (active)"; continue; fi
  st="$(state_of "$id")"
  done_ok=0; grep -q -e "run done" -e "ladder done" -e "train done" -e "util done" "$out" && done_ok=1
  if [[ $done_ok -eq 1 ]]; then
    day="$(date -r "$out" +%Y-%m-%d)"
    do_mv "logs/$day" "$out"
    if [[ -f "$err" ]]; then
      if [[ ! -s "$err" ]] || ! grep -v -q -e "UserWarning" -e "warnings.warn" -e "self.tf = nn.TransformerEncoder" -e "^ *$" "$err"; then
        do_rm "$err"           # empty, or only the usual warnings
      else
        do_mv "logs/$day" "$err"
      fi
    fi
  else
    say "  job $id state=${st:-unknown}: not a finished result"
    do_rm "$out"; [[ -f "$err" ]] && do_rm "$err"
  fi
done

say "## generation task logs under $DATA_DIR/*/slurm"
for d in "$DATA_DIR"/*/slurm; do
  [[ -d "$d" ]] || continue
  n=0; keep=0
  for f in "$d"/gen_*.out "$d"/gen_*.err; do
    base="${f##*/gen_}"; arr="${base%%_*}"
    if is_active "$arr"; then keep=$((keep+1)); continue; fi
    n=$((n+1)); [[ $APPLY -eq 1 ]] && rm -f -- "$f"
  done
  say "  $d: ${n} task logs removed, ${keep} kept (array still active)"
done

say "## crashed run directories under $INV/runs"
for r in "$INV"/runs/*/; do
  [[ -f "$r/metrics.json" || -f "$r/model.pt" ]] && continue
  do_rm "${r%/}"
done

say "## caches"
find "$(dirname "$HERE")" -type d -name __pycache__ -not -path '*/.venv/*' | while read -r d; do do_rm "$d"; done

if [[ $KEEP_DAYS -gt 0 && -d logs ]]; then
  say "## log folders older than $KEEP_DAYS days -> tar.gz"
  find logs -mindepth 1 -maxdepth 1 -type d -mtime +"$KEEP_DAYS" | while read -r d; do
    say "  tar  $d.tar.gz"
    if [[ $APPLY -eq 1 ]]; then tar -czf "$d.tar.gz" -C logs "$(basename "$d")" && rm -rf -- "$d"; fi
  done
fi
[[ $APPLY -eq 1 ]] || say "(dry run: add --apply to make these changes)"
