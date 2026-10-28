#!/bin/bash
# Snapshot of every cloth job (queued, running, finished today) for cloth_results/JOBS.md.
# Run inside an allocation: bash cluster/jobs.sh [since, default today]
SINCE="${1:-$(date +%Y-%m-%d)}"
echo "## queue ($(date '+%Y-%m-%d %H:%M'))"
squeue -u "$USER" -o "%.10i %.14j %.12P %.8T %.10M %.12l %.6D %R" | sed 's/^/    /'
echo "## sacct since $SINCE"
sacct -u "$USER" -S "$SINCE" -X -o JobID%12,JobName%14,Partition%12,State%12,Elapsed%10,ExitCode%8,NodeList%14 | sed 's/^/    /'
echo "## last line of each training log"
for f in "$(dirname "$0")"/../ladder_*.out "$(dirname "$0")"/../run_*.out "$(dirname "$0")"/../train_*.out; do
  [[ -f "$f" ]] && echo "    $(basename "$f"): $(tail -n 1 "$f")"
done
echo "## metrics written"
ls -1t "$(dirname "$0")"/../../inverse/runs/*/metrics.json 2>/dev/null | head -20 | sed 's/^/    /'
