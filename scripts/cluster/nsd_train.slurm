#!/usr/bin/env bash
#SBATCH --job-name=fmri-nsd-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=48:00:00

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

read -r -a TRAIN_SUBJECTS_ARRAY <<< "$TRAIN_SUBJECTS"
read -r -a VALIDATION_SUBJECTS_ARRAY <<< "$VAL_SUBJECTS"
read -r -a TEST_SUBJECTS_ARRAY <<< "$TEST_SUBJECTS"
ALL_SUBJECTS=("${TRAIN_SUBJECTS_ARRAY[@]}" "${VALIDATION_SUBJECTS_ARRAY[@]}")

CROP_ARGS=()
if [[ -n "$CROP_SIZE" ]]; then
    read -r -a CROP_VALUES <<< "$CROP_SIZE"
    if [[ ${#CROP_VALUES[@]} -ne 3 ]]; then
        echo "CROP_SIZE must contain exactly three integers." >&2
        exit 1
    fi
    CROP_ARGS=(--crop_size "${CROP_VALUES[@]}")
fi
FLAG_ARGS=()
[[ "$TRAIN_FREE_NOISE_HEAD" == "1" ]] && FLAG_ARGS+=(--free_noise_head)
[[ "$TRAIN_SOFT_BANDPASS" == "1" ]] && FLAG_ARGS+=(--soft_bandpass)
EARLY_STOPPING_ARGS=()
if [[ "$EARLY_STOPPING_PATIENCE" -gt 0 ]]; then
    EARLY_STOPPING_ARGS=(--early_stopping_patience "$EARLY_STOPPING_PATIENCE")
fi
RESUME_ARGS=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    RESUME_ARGS=(--resume --resume_checkpoint "$RESUME_CHECKPOINT")
fi
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
    --subjects "${ALL_SUBJECTS[@]}" \
    --validation_subjects "${VALIDATION_SUBJECTS_ARRAY[@]}" \
    --test_subjects "${TEST_SUBJECTS_ARRAY[@]}" \
    --time_frames "$TIME_FRAMES" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS" \
    --gradient_clip_val "$GRADIENT_CLIP_VAL" \
    --seed "$SEED" \
    --precision "$PRECISION" \
    --tr "$PROC_TR" \
    "${CROP_ARGS[@]}" \
    --patch_size "$PATCH_SIZE" \
    --embed_dim "$EMBED_DIM" \
    --mask_ratio "$MASK_RATIO" \
    --weight_decay "$WEIGHT_DECAY" \
    --rec_loss_weight "$TRAIN_REC_LOSS_WEIGHT" \
    --tissue_loss_weight "$TRAIN_TISSUE_LOSS_WEIGHT" \
    --band_loss_weight "$TRAIN_BAND_LOSS_WEIGHT" \
    --ortho_loss_weight "$TRAIN_ORTHO_LOSS_WEIGHT" \
    --n_comp "$N_COMP" \
    --band_low "$BAND_LOW" \
    --band_high "$BAND_HIGH" \
    --erosion_iterations "$EROSION_ITERATIONS" \
    --full_res_channels "$FULL_RES_CHANNELS" \
    --detrend "$DETREND" \
    --eval_K "$EVAL_K" \
    --final_eval_windows_per_subject "$EVAL_MAX_WINDOWS_PER_SUBJECT" \
    --temporal_patch_size "$EVAL_TEMPORAL_PATCH_SIZE" \
    --decoder_skips "$TRAIN_DECODER_SKIPS" \
    --prior_warmup_steps "$TRAIN_PRIOR_WARMUP_STEPS" \
    "${FLAG_ARGS[@]}" \
    "${EARLY_STOPPING_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    --run_dir "$RUN_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    --log_dir "$LOG_DIR" \
    --visualization_dir "$VIS_DIR" \
    --eval_dir "$EVAL_DIR" \
    "${WANDB_ARGS[@]}" \
    --name "$RUN_ID"
