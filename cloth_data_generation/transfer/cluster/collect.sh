#!/bin/bash
# One pasteable report of the latest submission: state, final metrics and any crash per job.
#   bash cluster/collect.sh [prefix=M11] [since=today YYYY-MM-DD]
# Reads run_<jobid>.out/.err (cluster/run.sbatch) and resim_<jobid>.* (cluster/resim.sbatch)
# for every job of that name prefix in sacct, then prints summarize_runs.py for runs/<prefix>*.
PREFIX="${1:-M11}"
SINCE="${2:-$(date +%Y-%m-%d)}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
echo "## collect $PREFIX since $SINCE  ($(date '+%Y-%m-%d %H:%M'), $(git log --oneline -1 | cut -c1-60))"
sacct -u "$USER" -S "$SINCE" -X -n -o JobID%12,JobName%14,State%12,Elapsed%10,ExitCode%8,NodeList%14 \
  | awk -v p="$PREFIX" '$2 ~ "^"p' | sort -k2,2 -k1,1n | while read -r jid name state elapsed exit node; do
  echo
  echo "### $name  job $jid  $state  $elapsed  exit $exit  $node"
  f=""
  for c in "run_$jid" "resim_$jid"; do [[ -f "$c.out" ]] && f="$c"; done
  if [[ -z "$f" ]]; then echo "    (no log file)"; continue; fi
  head -n 1 "$f.out" | cut -c1-200 | sed 's/^/    /'
  if [[ -s "$f.err" ]]; then
    n=$(grep -vc "warn\|Warning\|UserWarning\|^ *$" "$f.err")
    echo "    err: $(wc -l < "$f.err") lines ($n non-warning)"
    grep -v "warn\|Warning\|UserWarning\|^ *$" "$f.err" | tail -n 12 | cut -c1-200 | sed 's/^/    ! /'
  fi
  if [[ "$state" == COMPLETED ]]; then
    grep -E "best val loss|/tta|flow:|resim|selected|oracle|mean_metrics|score|frac_beats" "$f.out" | tail -n 16 | sed 's/^/    /'
  else
    tail -n 2 "$f.out" | sed 's/^/    /'
  fi
done
echo
echo "## summarize_runs $PREFIX*"
python3 "$HERE/cluster/summarize_runs.py" "$PREFIX*"
for r in "$HERE/../inverse/runs/$PREFIX"*/resim/resim_metrics.json; do
  [[ -f "$r" ]] && echo "## $(basename "$(dirname "$(dirname "$r")")") resim (eval_resim.py)" && python3 -c "
import json,sys; s=json.load(open(sys.argv[1]))['summary']; g=lambda k: s.get(k,{}).get('rmse_centered_m',float('nan'))
print('    n', s['n'], 'failed', s.get('n_pred_failed'), '| centred RMSE m: pred %.4f floor %.4f chance %.4f | score %.2f (median %.2f) beats chance %.0f%%' % (
    g('pred'), g('floor'), g('chance'), s.get('score',float('nan')), s.get('score_median',float('nan')), 100*s.get('frac_beats_chance',float('nan'))))" "$r"
done
for r in "$HERE/../inverse/runs/$PREFIX"*/resim_select/resim_select.json; do
  [[ -f "$r" ]] && echo "## $(basename "$(dirname "$(dirname "$r")")") resim_select" && python3 -c "
import json,sys; d=json.load(open(sys.argv[1]))
for k in ('selected_metrics','oracle_metrics','mean_metrics'):
    print('   ', k, json.dumps(d.get(k)))" "$r"
done
