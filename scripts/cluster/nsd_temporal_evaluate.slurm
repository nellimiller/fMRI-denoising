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
export PATH="$HOME/.local/bin:$PATH"

: "${NSD_PROCESSED:?Set NSD_PROCESSED to the processed NSD directory}"
: "${CHECKPOINT:?Set CHECKPOINT to the trained Lightning checkpoint}"
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

read -r -a SUBJECTS <<< "${EVAL_SUBJECTS:-subj08}"
TEMPORAL_EVAL_DIR="${TEMPORAL_EVAL_DIR:-$EVAL_DIR/temporal}"
mkdir -p "$TEMPORAL_EVAL_DIR"

EVAL_CROP_ARGS=()
if [[ -n "${EVAL_CROP_SIZE:-}" ]]; then
    read -r -a EVAL_CROP_ARRAY <<< "$EVAL_CROP_SIZE"
    if [[ ${#EVAL_CROP_ARRAY[@]} -ne 3 ]]; then
        echo "EVAL_CROP_SIZE must contain exactly three integers: X Y Z." >&2
        exit 1
    fi
    EVAL_CROP_ARGS=(--crop_size "${EVAL_CROP_ARRAY[@]}")
fi

uv run --no-sync python -c '
import torch
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this Slurm job.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
'

uv run --no-sync python -m src.evaluate_temporal \
    --data_dir "$NSD_PROCESSED" \
    --checkpoint "$CHECKPOINT" \
    --subjects "${SUBJECTS[@]}" \
    --output_dir "$TEMPORAL_EVAL_DIR" \
    --time_frames "${EVAL_TIME_FRAMES:-40}" \
    --max_windows_per_subject "${TEMPORAL_MAX_WINDOWS_PER_SUBJECT:-12}" \
    --plots_per_subject "${TEMPORAL_PLOTS_PER_SUBJECT:-3}" \
    --seed "${EVAL_SEED:-1234}" \
    --K "${EVAL_K:-4}" \
    --detrend "${EVAL_DETREND:-run}" \
    "${EVAL_CROP_ARGS[@]}"
