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

read -r -a SUBJECTS <<< "${PILOT_SUBJECTS:-subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08}"
if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
    echo "PILOT_SUBJECTS must contain at least one subject ID." >&2
    exit 1
fi
export PILOT_SUBJECTS="${SUBJECTS[*]}"
VALIDATION_ARGS=()
if [[ -n "${PILOT_VALIDATION_SUBJECTS:-}" ]]; then
    read -r -a VALIDATION_SUBJECTS <<< "$PILOT_VALIDATION_SUBJECTS"
    VALIDATION_ARGS=(--validation_subjects "${VALIDATION_SUBJECTS[@]}")
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found. Load the cluster's uv module or add its install directory to PATH." >&2
    exit 1
fi

# Load the cluster's Python/CUDA modules here if required, for example:
# module load python/3.13 cuda

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/visualizations" "$RUN_DIR/logs"

uv run --no-sync python -c '
import torch
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this Slurm job.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
'

uv run --no-sync python -c '
import os
from pathlib import Path
from src.data.nsd_dataset import NSDDataset

root = os.environ["NSD_PROCESSED"]
frames = int(os.environ.get("PILOT_TIME_FRAMES", "8"))
crop = int(os.environ.get("PILOT_CROP_SIZE", "32"))
subjects = os.environ["PILOT_SUBJECTS"].split()
available = {path.parent.name for path in Path(root).glob("*/manifest.json")}
missing = sorted(set(subjects) - available)
if missing:
    raise SystemExit(
        f"Missing processed manifests for {missing} under {root}. "
        "Preprocess every requested subject before submitting the job."
    )
dataset = NSDDataset(
    root,
    subjects=subjects,
    time_frames=frames,
    crop_size=(crop, crop, crop),
)
groups = {(str(entry["subject"]), str(entry["run"])) for entry in dataset.entries}
loaded_subjects = {str(entry["subject"]) for entry in dataset.entries}
if loaded_subjects != set(subjects):
    raise SystemExit(
        f"Requested subjects {sorted(subjects)}, but manifests contain "
        f"training windows for {sorted(loaded_subjects)}."
    )
if len(groups) < 2:
    raise SystemExit(
        f"Need at least two subject/run groups for train/validation; found {len(groups)}."
    )
sample = dataset[0]
print(
    f"Subjects: {sorted(loaded_subjects)}; windows: {len(dataset)}; "
    f"groups: {len(groups)}; sample: {tuple(sample.shape)}"
)
'

uv run --no-sync python -m src.train_nsd \
    --data_dir "$NSD_PROCESSED" \
    --subjects "${SUBJECTS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    --time_frames "${PILOT_TIME_FRAMES:-8}" \
    --batch_size 1 \
    --epochs "${PILOT_EPOCHS:-1}" \
    --num_workers "${PILOT_NUM_WORKERS:-2}" \
    --precision "${PILOT_PRECISION:-16-mixed}" \
    --crop_size "${PILOT_CROP_SIZE:-32}" "${PILOT_CROP_SIZE:-32}" "${PILOT_CROP_SIZE:-32}" \
    --patch_size "${PILOT_PATCH_SIZE:-4}" \
    --embed_dim "${PILOT_EMBED_DIM:-24}" \
    --checkpoint_dir "$RUN_DIR/checkpoints" \
    --log_dir "$RUN_DIR/logs" \
    --visualization_dir "$RUN_DIR/visualizations" \
    --use_wandb \
    --project "${PILOT_WANDB_PROJECT:-7T-fMRI-Denoising}" \
    --name "nsd-8subject-one-epoch"
