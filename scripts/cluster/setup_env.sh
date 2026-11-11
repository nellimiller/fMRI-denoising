#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found. Load the cluster's uv module or install uv in your user account." >&2
    exit 1
fi

# Load the cluster's Python/CUDA modules here if required, for example:
# module load python/3.13 cuda

uv sync --locked --python 3.13

uv run --no-sync python -c '
import sys
import torch
if sys.version_info < (3, 13):
    raise SystemExit(f"Python 3.13+ is required, found {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
cuda_runtime = torch.version.cuda or "CPU-only"
print(f"PyTorch CUDA runtime: {cuda_runtime}")
print(f"CUDA available in this shell: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
'

echo "Environment is ready at $REPO_DIR/.venv"
echo "Run the training job on an allocated GPU node to verify CUDA availability."
