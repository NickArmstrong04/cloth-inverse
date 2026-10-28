#!/bin/bash
# Request an interactive allocation for running the Python tooling (Bouchet
# policy: do not run Python on login nodes).
#   bash cluster/interactive.sh          # CPU: devel partition, 2 h, 4 cores, 16G
#   bash cluster/interactive.sh gpu      # 1 GPU on gpu_devel, 2 h (to try generate_sample.py by hand)
#   bash cluster/interactive.sh gpu h100 # a specific GPU type from gpu_devel
set -euo pipefail
case "${1:-cpu}" in
  cpu) exec salloc --partition=devel --cpus-per-task=4 --mem=16G --time=02:00:00 ;;
  gpu) GT="${2:-}"; exec salloc --partition=gpu_devel --gpus="${GT:+$GT:}1" --cpus-per-task=4 --mem=24G --time=02:00:00 ;;
  *) echo "usage: $0 [cpu|gpu [gpu_type]]"; exit 1 ;;
esac
