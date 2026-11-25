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
source "$REPO_DIR/scripts/cluster/pipeline.env"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$RUN_DIR" "$CKPT_DIR" "$EVAL_DIR" "$SLURM_LOG_DIR" "$LOG_DIR" "$VIS_DIR"
cp "$REPO_DIR/scripts/cluster/pipeline.env" "$RUN_DIR/pipeline.env"

if [[ ! -d "$NSD_PROC_DIR" ]]; then
    echo "Processed NSD directory does not exist: $NSD_PROC_DIR" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found. Load the cluster's uv module or add it to PATH." >&2
    exit 1
fi

read -r -a SUBJECTS <<< "$TRAIN_SUBJECTS $VAL_SUBJECTS"
read -r -a VALIDATION_SUBJECTS <<< "$VAL_SUBJECTS"

CROP_ARGS=()
if [[ -n "$CROP_SIZE" ]]; then
    read -r -a CROP_VALUES <<< "$CROP_SIZE"
    CROP_ARGS=(--crop_size "${CROP_VALUES[@]}")
fi
ABLATION_ARGS=()
[[ "$TRAIN_FREE_NOISE_HEAD" == "1" ]] && ABLATION_ARGS+=(--free_noise_head)
[[ "$TRAIN_SOFT_BANDPASS" == "1" ]] && ABLATION_ARGS+=(--soft_bandpass)
WANDB_ARGS=()
if [[ "$USE_WANDB" == "1" ]]; then
    WANDB_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE")
    if [[ -n "$WANDB_ENTITY" ]]; then
        WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
    fi
fi

uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(f"GPU: {torch.cuda.get_device_name(0)}")'

uv run --no-sync python -m src.train_nsd \
    --data_dir "$NSD_PROC_DIR" \
    --subjects "${SUBJECTS[@]}" \
    --validation_subjects "${VALIDATION_SUBJECTS[@]}" \
    --time_frames "$TIME_FRAMES" \
    --batch_size "$BATCH_SIZE" \
    --epochs 1 \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS" \
    --precision "$PRECISION" \
    --seed "$SEED" \
    "${CROP_ARGS[@]}" \
    --tr "$PROC_TR" \
    --patch_size "$PATCH_SIZE" \
    --embed_dim "$EMBED_DIM" \
    --mask_ratio "$MASK_RATIO" \
    --weight_decay "$WEIGHT_DECAY" \
    --rec_loss_weight "$TRAIN_REC_LOSS_WEIGHT" \
    --tissue_loss_weight "$TRAIN_TISSUE_LOSS_WEIGHT" \
    --band_loss_weight "$TRAIN_BAND_LOSS_WEIGHT" \
    --ortho_loss_weight "$TRAIN_ORTHO_LOSS_WEIGHT" \
    --n_comp "$N_COMP" \
    --erosion_iterations "$EROSION_ITERATIONS" \
    --band_low "$BAND_LOW" \
    --band_high "$BAND_HIGH" \
    --detrend "$DETREND" \
    --decoder_skips "$TRAIN_DECODER_SKIPS" \
    --prior_warmup_steps "$TRAIN_PRIOR_WARMUP_STEPS" \
    --prior_warmup_epochs "$TRAIN_PRIOR_WARMUP_EPOCHS" \
    --min_clean_var_frac "$TRAIN_MIN_CLEAN_VAR_FRAC" \
    --w_min_var "$TRAIN_W_MIN_VAR" \
    "${ABLATION_ARGS[@]}" \
    --skip_final_eval \
    --run_dir "$RUN_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    --log_dir "$LOG_DIR" \
    --visualization_dir "$VIS_DIR" \
    "${WANDB_ARGS[@]}" \
    --name "$RUN_ID"
