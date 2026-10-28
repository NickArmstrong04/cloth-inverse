"""Bouchet GPU types -> partition, plus optional cost weights (shared by the
queue probe, the benchmark and the report). Edit when the cluster changes.
Source: sinfo -o "%P %G" on 2026-10-27."""

GPU_TYPES = {
    # type: (partition, approx VRAM GB)
    "a5000": ("gpu", 24),
    "rtx_5000_ada": ("gpu", 32),
    "a40": ("gpu", 48),
    "l40s": ("gpu", 48),
    "h100": ("gpu_h100", 80),
    "h200": ("gpu_h200", 141),
    "rtx_pro_6000_blackwell": ("gpu_rtx6000", 96),
    "b200": ("gpu_b200", 180),
}

# Default sweep: one of each tier you might realistically use.
DEFAULT_TYPES = ["a5000", "rtx_5000_ada", "a40", "l40s", "h100", "h200"]

# Alternative partitions that also carry these GPUs (preemptible / priority).
# The probe can report them with --partitions scavenge_gpu,...
