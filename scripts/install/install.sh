#!/usr/bin/env bash
# Create a virtual environment with everything needed for TEMPURA inference and benchmark evaluation.
# Usage: bash scripts/install/install.sh [python_executable]   (default: python3.12)
set -euo pipefail
PY=${1:-python3.12}
cd "$(dirname "$0")/../.."
$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# torch build for CUDA 12.8; pick another index-url (e.g. cu126) if your driver requires it.
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# FlashAttention-2 is optional; comment out if the wheel does not build on your machine.
pip install flash-attn --no-build-isolation || echo "flash-attn not installed; SDPA attention will be used"
python -c "import torch, transformers, decord; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
echo "Done. Activate with: source .venv/bin/activate"
