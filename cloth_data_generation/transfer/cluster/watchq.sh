# Source this file (source cluster/watchq.sh) and run  watchq [NAME_PATTERN ...]
# Waits until every queued/running job whose name matches a pattern is gone, mailing once
# per pattern as it clears and once more when all are clear. Patterns are shell globs
# matched against the job name; no pattern = all your jobs (original behaviour).
#   watchq                     # every job
#   watchq 'M1*' 'M2*' 'M3*'   # one mail per matrix as it finishes, one when all three are done
#   watchq cloth_ladder        # exact name
# Run it inside tmux/screen on the login node so it survives logout:  tmux new -s watchq
watchq() {
    local user="naa79"
    local email="nick.armstrong@yale.edu"
    local patterns=("$@") out pat remaining=() still=() cleared=()
    [ ${#patterns[@]} -eq 0 ] && patterns=('*')
    remaining=("${patterns[@]}")

    _wq_match() {  # job names on stdin, glob $1 -> prints the matching names
        local p="$1" n
        while read -r n; do case "$n" in $p) echo "$n";; esac; done
    }
    _wq_report() {  # today's sacct rows whose JobName matches glob $1
        local p="$1" line
        sacct -u "$user" -S "$(date +%Y-%m-%d)" -X -n -o JobID%12,JobName%14,State%12,Elapsed%10,ExitCode%8 \
          | while read -r line; do set -- $line; case "$2" in $p) echo "$line";; esac; done
    }

    while true; do
        # If squeue errors out, wait and retry instead of treating it as "empty"
        if ! out=$(squeue -h -u "$user" -o "%j" 2>/dev/null); then sleep 60; continue; fi
        still=(); cleared=()
        for pat in "${remaining[@]}"; do
            if [ -n "$(printf '%s\n' "$out" | _wq_match "$pat")" ]; then still+=("$pat"); else cleared+=("$pat"); fi
        done
        for pat in "${cleared[@]}"; do
            {
              echo "No queued or running jobs match '$pat' as of $(date) on $(hostname)."
              echo; echo "Finished today (JobID JobName State Elapsed ExitCode):"; _wq_report "$pat"
              echo; echo "Still waiting on: ${still[*]:-none}"
            } | mail -s "Slurm: '$pat' jobs clear" "$email"
            echo "$(date): '$pat' clear, mailed."
        done
        remaining=("${still[@]}")
        [ ${#remaining[@]} -eq 0 ] && break
        clear; date; echo "waiting on: ${remaining[*]}"
        squeue -u "$user" -o "%.10i %.14j %.8T %.10M %R"
        sleep 30
    done
    echo "All watched jobs (${patterns[*]}) finished at $(date)."
    echo "All watched Slurm jobs (${patterns[*]}) for $user finished at $(date) on $(hostname)." \
        | mail -s "Slurm queue clear: ${patterns[*]}" "$email"
}
