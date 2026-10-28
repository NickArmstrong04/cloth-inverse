#!/bin/bash
# One-time environment setup, run ON the cluster login node from the copied
# transfer/ directory:
#     bash cluster/setup_env.sh
# Creates <CODE_DIR>/.venv with the pinned simulator stack using uv. No conda
# or CUDA module needed: Warp ships its own CUDA runtime and only needs the
# node's NVIDIA driver (present on GPU nodes, absent on login nodes).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster.env"
CODE_DIR="${CODE_DIR:-$(dirname "$HERE")}"
DATA_DIR="$(eval echo "$DATA_DIR")"

mkdir -p "$DATA_DIR"
cd "$CODE_DIR"
echo "code: $CODE_DIR"
echo "data: $DATA_DIR"

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "installing uv into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
fi
# Same pins as the workstation prototype (requirements.txt).
uv pip install --python .venv/bin/python \
  "newton==1.7.0" "warp-lang==1.18.0" "warp-nn==0.4.1" \
  "numpy==2.5.4" "h5py==3.16.0" "matplotlib==3.11.2" "imageio==2.38.1" \
  "pillow==13.0.0" "scipy==1.18.1"

if [ "${1:-}" = "--with-torch" ]; then
  echo "== installing PyTorch (CUDA 12.8 wheels, ~2.5 GB) for the inverse models"
  uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
  .venv/bin/python -c "import torch; print('torch', torch.__version__)"
fi

echo "== import check (CPU only; the GPU is exercised by the first job)"
.venv/bin/python -c "import warp, newton, h5py, numpy; print('warp', warp.__version__, 'newton', newton.__version__, 'numpy', numpy.__version__)"
echo "setup done. Next: python3 cluster/submit.py --n 1000 --dry-run"
