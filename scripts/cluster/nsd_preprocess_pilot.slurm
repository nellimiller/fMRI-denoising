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

: "${NSD_RAW:?Set NSD_RAW to the raw NSD root containing ppdata/ and timeseries/}"
: "${NSD_PROCESSED:?Set NSD_PROCESSED to a fresh pilot output directory}"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found. Load the cluster's uv module or add its install directory to PATH." >&2
    exit 1
fi

if [[ ! -d "$NSD_RAW/ppdata" || ! -d "$NSD_RAW/timeseries" ]]; then
    echo "NSD_RAW must contain ppdata/ and timeseries/: $NSD_RAW" >&2
    exit 1
fi

read -r -a SUBJECTS <<< "${PILOT_SUBJECTS:-subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08}"
if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
    echo "PILOT_SUBJECTS must contain at least one subject ID." >&2
    exit 1
fi

existing_manifest="$(find "$NSD_PROCESSED" -name manifest.json -print -quit 2>/dev/null || true)"
if [[ -n "$existing_manifest" ]]; then
    echo "Found an existing manifest under $NSD_PROCESSED; use a fresh pilot directory." >&2
    exit 1
fi

mkdir -p "$NSD_PROCESSED"

echo "Preprocessing subjects: ${SUBJECTS[*]}"
echo "Window length: ${PILOT_TIME_FRAMES:-8}; stride: ${PILOT_STRIDE:-10000}"

for subject in "${SUBJECTS[@]}"; do
    echo "Preprocessing $subject"
    uv run --no-sync python -m src.data_preprocessing.nsd.preprocess \
        --raw-root "$NSD_RAW" \
        --output-root "$NSD_PROCESSED" \
        --subject "$subject" \
        --window-length "${PILOT_TIME_FRAMES:-8}" \
        --stride "${PILOT_STRIDE:-10000}"
done

echo "Pilot preprocessing complete:"
find "$NSD_PROCESSED" -mindepth 2 -maxdepth 2 -name manifest.json -print | sort
