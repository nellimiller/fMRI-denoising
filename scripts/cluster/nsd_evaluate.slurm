#!/usr/bin/env bash
#SBATCH --job-name=fmri-nsd-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=12:00:00

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/cluster/pipeline.env"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$RUN_DIR" "$CKPT_DIR" "$EVAL_DIR" "$SLURM_LOG_DIR"
cp "$REPO_DIR/scripts/cluster/pipeline.env" "$RUN_DIR/pipeline.env"

if [[ ! -d "$NSD_PROC_DIR" || ! -f "$CHECKPOINT" ]]; then
    echo "Processed data or checkpoint is missing: $NSD_PROC_DIR / $CHECKPOINT" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found. Load the cluster's uv module or add it to PATH." >&2
    exit 1
fi

read -r -a TEST_SUBJECTS_ARRAY <<< "$TEST_SUBJECTS"
read -r -a VALIDATION_SUBJECTS_ARRAY <<< "$VAL_SUBJECTS"
read -r -a TRAIN_SUBJECTS_ARRAY <<< "$TRAIN_SUBJECTS"
CROP_ARGS=()
if [[ -n "$CROP_SIZE" ]]; then
    read -r -a CROP_VALUES <<< "$CROP_SIZE"
    CROP_ARGS=(--crop_size "${CROP_VALUES[@]}")
fi

uv run --no-sync python -m src.evaluate_nsd \
    --data_dir "$NSD_PROC_DIR" \
    --checkpoint "$CHECKPOINT" \
    --subjects "${TEST_SUBJECTS_ARRAY[@]}" \
    --validation_subjects "${VALIDATION_SUBJECTS_ARRAY[@]}" \
    --train_visual_subjects "${TRAIN_SUBJECTS_ARRAY[@]:0:1}" \
    --output_dir "$EVAL_DIR" \
    --time_frames "$TIME_FRAMES" \
    --num_workers "$NUM_WORKERS" \
    --K "$EVAL_K" \
    --inference_mode "$EVAL_INFERENCE_MODE" \
    --mask_repeats "$EVAL_MASK_REPEATS" \
    --max_windows_per_subject "$EVAL_MAX_WINDOWS_PER_SUBJECT" \
    --visual_examples_per_subject "$EVAL_VISUAL_EXAMPLES_PER_SUBJECT" \
    --temporal_patch_size "$EVAL_TEMPORAL_PATCH_SIZE" \
    --detrend "$DETREND" \
    --tr "$PROC_TR" \
    --seed "$SEED" \
    "${CROP_ARGS[@]}"
