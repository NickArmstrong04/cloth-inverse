# Sourced by the training sbatch scripts after the GPU is allocated. Prints the card and
# sets GPU_EXTRA: extra train.py flags so the same job runs on any GPU type.
#   >= 40 GB (L40S, A40, H100, H200): dataset tensors live on the GPU (fastest)
#   <  40 GB (A5000 24 GB, RTX 5000 Ada 32 GB): --data-on-cpu, batches are copied per step
# Pass --data-on-cpu yourself to force it; duplicates are harmless.
GPU_EXTRA=""
GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 1)"
GPU_MB="$(echo "$GPU_INFO" | grep -oE '[0-9]+ *MiB' | grep -oE '[0-9]+')"
echo "gpu: ${GPU_INFO:-unknown}"
if [[ "$GPU_MB" =~ ^[0-9]+$ ]] && (( GPU_MB < 40000 )); then
  GPU_EXTRA="--data-on-cpu"
  echo "gpu under 40 GB: adding $GPU_EXTRA"
fi
