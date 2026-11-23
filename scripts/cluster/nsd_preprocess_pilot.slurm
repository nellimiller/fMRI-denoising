#!/usr/bin/env bash
#SBATCH --job-name=fmri-nsd-preprocess
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/cluster/pipeline.env"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$RUN_DIR" "$CKPT_DIR" "$EVAL_DIR" "$SLURM_LOG_DIR" "$NSD_PROC_DIR"
cp "$REPO_DIR/scripts/cluster/pipeline.env" "$RUN_DIR/pipeline.env"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found. Load the cluster's uv module or add it to PATH." >&2
    exit 1
fi
if [[ ! -d "$NSD_RAW_DIR/ppdata" || ! -d "$NSD_RAW_DIR/timeseries" ]]; then
    echo "NSD_RAW_DIR must contain ppdata/ and timeseries/: $NSD_RAW_DIR" >&2
    exit 1
fi

read -r -a SUBJECTS <<< "$PROC_SUBJECTS"
existing_manifest="$(find "$NSD_PROC_DIR" -name manifest.json -print -quit 2>/dev/null || true)"
if [[ -n "$existing_manifest" ]]; then
    echo "Found an existing manifest under $NSD_PROC_DIR; use a fresh PROC_ID." >&2
    exit 1
fi

echo "Preprocessing ${SUBJECTS[*]} at TR=${PROC_TR}s; window=${PROC_WINDOW}, stride=${PROC_STRIDE}"
for subject in "${SUBJECTS[@]}"; do
    uv run --no-sync python -m src.data_preprocessing.nsd.preprocess \
        --raw-root "$NSD_RAW_DIR" \
        --output-root "$NSD_PROC_DIR" \
        --subject "$subject" \
        --window-length "$PROC_WINDOW" \
        --stride "$PROC_STRIDE" \
        --tr "$PROC_TR"
done

find "$NSD_PROC_DIR" -mindepth 2 -maxdepth 2 -name manifest.json -print | sort
