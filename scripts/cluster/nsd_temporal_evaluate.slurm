#!/usr/bin/env bash
#SBATCH --job-name=fmri-nsd-temporal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/cluster/pipeline.env"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$RUN_DIR" "$CKPT_DIR" "$EVAL_DIR" "$TEMPORAL_EVAL_DIR" "$SLURM_LOG_DIR"
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
CROP_ARGS=()
if [[ -n "$CROP_SIZE" ]]; then
    read -r -a CROP_VALUES <<< "$CROP_SIZE"
    CROP_ARGS=(--crop_size "${CROP_VALUES[@]}")
fi

uv run --no-sync python -m src.evaluate_temporal \
    --data_dir "$NSD_PROC_DIR" \
    --checkpoint "$CHECKPOINT" \
    --subjects "${TEST_SUBJECTS_ARRAY[@]}" \
    --output_dir "$TEMPORAL_EVAL_DIR" \
    --time_frames "$TIME_FRAMES" \
    --max_windows_per_subject "$TEMPORAL_MAX_WINDOWS_PER_SUBJECT" \
    --plots_per_subject "$TEMPORAL_PLOTS_PER_SUBJECT" \
    --temporal_patch_size "$EVAL_TEMPORAL_PATCH_SIZE" \
    --seed "$SEED" \
    --K "$EVAL_K" \
    --detrend "$DETREND" \
    --tr "$PROC_TR" \
    "${CROP_ARGS[@]}"
