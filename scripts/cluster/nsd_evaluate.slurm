#!/usr/bin/env bash
#SBATCH --job-name=fmri-nsd-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
cd "$REPO_DIR"
export PATH="$HOME/.local/bin:$PATH"

: "${NSD_PROCESSED:?Set NSD_PROCESSED to the processed NSD directory}"
: "${CHECKPOINT:?Set CHECKPOINT to the trained best Lightning checkpoint}"
: "${EVAL_DIR:?Set EVAL_DIR to persistent evaluation output storage}"

if [[ ! -d "$NSD_PROCESSED" ]]; then
    echo "Processed NSD directory does not exist: $NSD_PROCESSED" >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint does not exist: $CHECKPOINT" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found. Load the cluster's uv module or add its install directory to PATH." >&2
    exit 1
fi

read -r -a EVAL_SUBJECTS_ARRAY <<< "${EVAL_SUBJECTS:-subj08}"
if [[ ${#EVAL_SUBJECTS_ARRAY[@]} -eq 0 ]]; then
    echo "EVAL_SUBJECTS must contain at least one subject ID." >&2
    exit 1
fi

VISUAL_SPLIT_ARGS=()
if [[ -n "${EVAL_TRAIN_VISUAL_SUBJECTS:-}" ]]; then
    read -r -a EVAL_TRAIN_VISUAL_ARRAY <<< "$EVAL_TRAIN_VISUAL_SUBJECTS"
    VISUAL_SPLIT_ARGS+=(--train_visual_subjects "${EVAL_TRAIN_VISUAL_ARRAY[@]}")
fi
if [[ -n "${EVAL_VALIDATION_VISUAL_SUBJECTS:-}" ]]; then
    read -r -a EVAL_VALIDATION_VISUAL_ARRAY <<< "$EVAL_VALIDATION_VISUAL_SUBJECTS"
    VISUAL_SPLIT_ARGS+=(--validation_visual_subjects "${EVAL_VALIDATION_VISUAL_ARRAY[@]}")
fi

mkdir -p "$EVAL_DIR"

uv run --no-sync python -c '
import torch
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this Slurm job.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
'

EVAL_CROP_ARGS=()
if [[ -n "${EVAL_CROP_SIZE:-}" ]]; then
    read -r -a EVAL_CROP_ARRAY <<< "$EVAL_CROP_SIZE"
    if [[ ${#EVAL_CROP_ARRAY[@]} -ne 3 ]]; then
        echo "EVAL_CROP_SIZE must contain exactly three integers: X Y Z." >&2
        exit 1
    fi
    EVAL_CROP_ARGS=(--crop_size "${EVAL_CROP_ARRAY[@]}")
fi

echo "Evaluating checkpoint: $CHECKPOINT"
echo "Held-out subjects: ${EVAL_SUBJECTS_ARRAY[*]}"

uv run --no-sync python -m src.evaluate_nsd \
    --data_dir "$NSD_PROCESSED" \
    --checkpoint "$CHECKPOINT" \
    --subjects "${EVAL_SUBJECTS_ARRAY[@]}" \
    --output_dir "$EVAL_DIR" \
    --time_frames "${EVAL_TIME_FRAMES:-40}" \
    --num_workers "${EVAL_NUM_WORKERS:-4}" \
    --mask_repeats "${EVAL_MASK_REPEATS:-1}" \
    --seed "${EVAL_SEED:-1234}" \
    --visual_examples_per_subject "${EVAL_VISUAL_EXAMPLES_PER_SUBJECT:-3}" \
    "${VISUAL_SPLIT_ARGS[@]}" \
    "${EVAL_CROP_ARGS[@]}"
