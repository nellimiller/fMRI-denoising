#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found. Load or install uv using your cluster's instructions." >&2
    exit 1
fi

# Load the cluster's Python/CUDA modules here if required, for example:
# module load python/3.12 cuda

uv sync --locked --python 3.12

uv run --no-sync python -c '
import torch
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA build: {torch.version.cuda}")
print(f"CUDA available in this shell: {torch.cuda.is_available()}")
if torch.version.cuda != "12.4":
    raise SystemExit(f"Expected PyTorch CUDA 12.4, found {torch.version.cuda!r}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
'

echo "Environment is ready at $REPO_DIR/.venv"
echo "Run this from an allocated GPU node to verify CUDA availability there."
