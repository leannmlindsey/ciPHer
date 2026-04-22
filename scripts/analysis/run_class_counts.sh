#!/usr/bin/env bash
#
# SLURM wrapper for scripts/analysis/class_counts.py.
# Reads label_encoders.json from every LA experiment dir and prints K/O
# class counts + deltas. No GPU, no cipher imports — this is pure file I/O,
# wrapped in sbatch only to match the "SLURM only on Delta" convention.
#
# Usage:
#   bash scripts/analysis/run_class_counts.sh
#   bash scripts/analysis/run_class_counts.sh experiments/light_attention/la_seg4_match_sweep
#   DRY_RUN=1 bash scripts/analysis/run_class_counts.sh

set -euo pipefail

ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="${CIPHER_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"

# Tiny job — no GPU needed for file reads. Still queued on ghx4 for simplicity.
GPUS=0
CPUS=2
MEM="4G"
TIME="00:10:00"
JOB_NAME="class_counts"

DRY_RUN="${DRY_RUN:-0}"
ARGS="$@"

JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${CIPHER_DIR}/logs/${JOB_NAME}_%j.log
#SBATCH --error=${CIPHER_DIR}/logs/${JOB_NAME}_%j.log

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

echo \"======================================\"
echo \"Class counts analysis\"
echo \"Started: \$(date)\"
echo \"======================================\"
python3 scripts/analysis/class_counts.py ${ARGS}
echo \"======================================\"
echo \"Done: \$(date)\"
echo \"======================================\"
"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would submit:"
    echo "$JOB_SCRIPT"
else
    mkdir -p "${CIPHER_DIR}/logs"
    JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
    echo "Submitted ${JOB_ID} — ${JOB_NAME}"
    echo "Log: ${CIPHER_DIR}/logs/${JOB_NAME}_${JOB_ID}.log"
fi
