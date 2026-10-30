#!/bin/bash
# SAM 3 environment for the rung-4b segmentation step (inverse/segment.py).
# Separate venv: SAM 3 pins numpy < 2 while the cloth venv runs numpy 2.5 for Newton.
# Usage: bash setup_sam3.sh [venv_dir] [repo_dir]      (workstation WSL or a Bouchet node)
# Afterwards, once, with your Hugging Face token (facebook/sam3 access): hf auth login
set -e
VENV=${1:-$HOME/sam3env}
REPO=${2:-$HOME/sam3}
# interpreter: $PYTHON, else python3.12 on PATH, else a uv-managed 3.12 (the workstation), else python3
PY=${PYTHON:-$(command -v python3.12 || true)}
[ -n "$PY" ] || PY=$(command -v uv >/dev/null && uv python find 3.12 2>/dev/null || true)
[ -n "$PY" ] || PY=$(ls $HOME/.local/share/uv/python/cpython-3.12*/bin/python3.12 2>/dev/null | tail -1)
[ -n "$PY" ] || PY=python3
echo "python: $($PY --version)  venv: $VENV  repo: $REPO"
[ -d "$VENV" ] || $PY -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
[ -d "$REPO" ] || git clone -q https://github.com/facebookresearch/sam3.git "$REPO"
pip install -q -e "$REPO"
pip install -q pillow matplotlib einops pycocotools opencv-python-headless scikit-image scipy   # imported by sam3 at runtime but missing from its dependency list
python - <<'EOF'
import torch, numpy, sam3, timm
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "numpy", numpy.__version__, "timm", timm.__version__)
print("sam3 at", sam3.__file__)
EOF
echo "done. next: 'source $VENV/bin/activate && hf auth login' (token with facebook/sam3 access), then"
echo "  python cloth_data_generation/inverse/segment.py --smoke"
