#!/usr/bin/env bash
#SBATCH --job-name=fmri-nsd-pilot
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
cd "$REPO_DIR"

: "${NSD_PROCESSED:?Set NSD_PROCESSED to the processed NSD manifest directory}"
: "${RUN_DIR:?Set RUN_DIR to persistent output storage}"

if [[ ! -d "$NSD_PROCESSED" ]]; then
    echo "Processed NSD directory does not exist: $NSD_PROCESSED" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found. Load or install uv using your cluster's instructions." >&2
    exit 1
fi

# Load the cluster's Python/CUDA modules here if required, for example:
# module load python/3.12 cuda

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/visualizations"

uv run --no-sync python -c '
import torch
if torch.version.cuda != "12.4":
    raise SystemExit(f"Expected PyTorch CUDA 12.4, found {torch.version.cuda!r}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this Slurm job.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
'

uv run --no-sync python -c '
import os
from src.data.nsd_dataset import NSDDataset

root = os.environ["NSD_PROCESSED"]
frames = int(os.environ.get("PILOT_TIME_FRAMES", "8"))
dataset = NSDDataset(root, time_frames=frames)
groups = {(str(entry["subject"]), str(entry["run"])) for entry in dataset.entries}
if len(groups) < 2:
    raise SystemExit(
        f"Need at least two subject/run groups for train/validation; found {len(groups)}."
    )
sample = dataset[0]
print(f"Windows: {len(dataset)}; groups: {len(groups)}; sample: {tuple(sample.shape)}")
'

uv run --no-sync python -m src.train_nsd \
    --data_dir "$NSD_PROCESSED" \
    --time_frames "${PILOT_TIME_FRAMES:-8}" \
    --batch_size 1 \
    --epochs "${PILOT_EPOCHS:-1}" \
    --num_workers "${PILOT_NUM_WORKERS:-2}" \
    --precision "${PILOT_PRECISION:-16-mixed}" \
    --checkpoint_dir "$RUN_DIR/checkpoints" \
    --visualization_dir "$RUN_DIR/visualizations"
