#!/usr/bin/env bash
#SBATCH --job-name=fmri-nsd-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
cd "$REPO_DIR"
export PATH="$HOME/.local/bin:$PATH"

: "${NSD_PROCESSED:?Set NSD_PROCESSED to the full-window processed NSD directory}"
: "${RUN_DIR:?Set RUN_DIR to persistent output storage}"

if [[ ! -d "$NSD_PROCESSED" ]]; then
    echo "Processed NSD directory does not exist: $NSD_PROCESSED" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found. Load the cluster's uv module or add its install directory to PATH." >&2
    exit 1
fi

read -r -a TRAIN_SUBJECTS <<< "${TRAIN_SUBJECTS:-subj01 subj02 subj03 subj04 subj05 subj06}"
read -r -a VALIDATION_SUBJECTS <<< "${VALIDATION_SUBJECTS:-subj07}"
if [[ ${#TRAIN_SUBJECTS[@]} -eq 0 || ${#VALIDATION_SUBJECTS[@]} -eq 0 ]]; then
    echo "Training and validation subject lists must both be non-empty." >&2
    exit 1
fi

for validation_subject in "${VALIDATION_SUBJECTS[@]}"; do
    if [[ " ${TRAIN_SUBJECTS[*]} " == *" $validation_subject "* ]]; then
        echo "Subject $validation_subject cannot be in both training and validation." >&2
        exit 1
    fi
done

DATA_SUBJECTS=("${TRAIN_SUBJECTS[@]}" "${VALIDATION_SUBJECTS[@]}")
export DATA_SUBJECTS="${DATA_SUBJECTS[*]}"

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

root = Path(os.environ["NSD_PROCESSED"])
subjects = os.environ["DATA_SUBJECTS"].split()
available = {path.parent.name for path in root.glob("*/manifest.json")}
missing = sorted(set(subjects) - available)
if missing:
    raise SystemExit(f"Missing subject manifests under {root}: {missing}")

dataset = NSDDataset(
    root,
    subjects=subjects,
    time_frames=int(os.environ.get("TRAIN_TIME_FRAMES", "40")),
)
loaded = {str(entry["subject"]) for entry in dataset.entries}
if loaded != set(subjects):
    raise SystemExit(f"Requested subjects {sorted(subjects)}, got windows for {sorted(loaded)}")
groups = {(str(entry["subject"]), str(entry["run"])) for entry in dataset.entries}
if len(groups) < 2:
    raise SystemExit(f"Need at least two subject/run groups; found {len(groups)}")

sample = dataset[0]
print(
    f"Subjects: {sorted(loaded)}; windows: {len(dataset)}; "
    f"subject/run groups: {len(groups)}; sample shape: {tuple(sample.shape)}"
)
'

CROP_ARGS=()
if [[ -n "${TRAIN_CROP_SIZE:-}" ]]; then
    CROP_ARGS=(--crop_size "$TRAIN_CROP_SIZE" "$TRAIN_CROP_SIZE" "$TRAIN_CROP_SIZE")
fi

RUN_NAME="${TRAIN_WANDB_NAME:-nsd-full-40frame-${SLURM_JOB_ID:-run}}"

echo "Training subjects: ${TRAIN_SUBJECTS[*]}"
echo "Validation subjects: ${VALIDATION_SUBJECTS[*]}"
echo "Held-out subjects not used in this run remain available for final evaluation."

uv run --no-sync python -m src.train_nsd \
    --data_dir "$NSD_PROCESSED" \
    --subjects "${DATA_SUBJECTS[@]}" \
    --validation_subjects "${VALIDATION_SUBJECTS[@]}" \
    --time_frames "${TRAIN_TIME_FRAMES:-40}" \
    --batch_size "${TRAIN_BATCH_SIZE:-1}" \
    --epochs "${TRAIN_EPOCHS:-100}" \
    --lr "${TRAIN_LR:-1.5e-4}" \
    --num_workers "${TRAIN_NUM_WORKERS:-4}" \
    --seed "${TRAIN_SEED:-1234}" \
    --precision "${TRAIN_PRECISION:-16-mixed}" \
    --patch_size "${TRAIN_PATCH_SIZE:-14}" \
    --embed_dim "${TRAIN_EMBED_DIM:-96}" \
    "${CROP_ARGS[@]}" \
    --checkpoint_dir "$RUN_DIR/checkpoints" \
    --log_dir "$RUN_DIR/logs" \
    --visualization_dir "$RUN_DIR/visualizations" \
    --use_wandb \
    --project "${WANDB_PROJECT:-7T-fMRI-Denoising}" \
    --name "$RUN_NAME"
