#!/bin/bash
# One-off, on the LOGIN node (internet): install the Sonata / Point Transformer V3 package into
# the training venv and cache the pretrained checkpoint (HF facebook/sonata -> ~/.cache/sonata).
# Sonata (CVPR 2025, facebookresearch/sonata) is an encoder-only PTv3 pretrained by
# self-distillation on 140k point clouds; --model sonata_reg fine-tunes it (JOBS.md M11-e).
# Package mode needs spconv and torch-scatter built for the venv's torch/CUDA; flash-attn is
# optional (the wrapper passes enable_flash=False if it is missing).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TORCH=$($PY -c "import torch; print(torch.__version__.split('+')[0])")
CUDA=$($PY -c "import torch; print(torch.version.cuda.replace('.', ''))")
echo "torch $TORCH cuda $CUDA"
$PY -m pip install "spconv-cu${CUDA:0:3}" || $PY -m pip install spconv-cu120
$PY -m pip install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH}+cu${CUDA}.html"
$PY -m pip install huggingface_hub timm
$PY -m pip install "git+https://github.com/facebookresearch/sonata.git"
$PY - <<'EOF'
import sonata
m = sonata.model.load("sonata", repo_id="facebook/sonata")
print("sonata loaded:", sum(p.numel() for p in m.parameters()) / 1e6, "M params; cached in ~/.cache/sonata")
EOF
