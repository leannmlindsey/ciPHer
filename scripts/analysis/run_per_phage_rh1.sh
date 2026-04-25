#!/usr/bin/env bash
#
# SLURM wrapper for scripts/analysis/per_phage_rh1.py.
# Produces the per-phage rh@1 CSV agent 5 asked for in their
# 2026-04-24 broadcast handoff.
#
# Defaults:
#   --head-mode k_only    (per the 2026-04-23 finding — best PHL mode)
#   --datasets PhageHostLearn PBIP
#   --output results/analysis/per_phage_rh1_phl_pbip.csv
#
# Usage:
#   bash scripts/analysis/run_per_phage_rh1.sh <run_dir>
#   bash scripts/analysis/run_per_phage_rh1.sh <run_dir> --head-mode both
#   DRY_RUN=1 bash scripts/analysis/run_per_phage_rh1.sh <run_dir>

set -euo pipefail

ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="${CIPHER_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"

GPUS=1
CPUS=4
MEM="16G"
TIME="00:30:00"

DRY_RUN="${DRY_RUN:-0}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_dir> [extra args passed to per_phage_rh1.py]" >&2
    exit 1
fi

RUN_DIR="$1"; shift
ABS_RUN_DIR="$(cd "$RUN_DIR" && pwd)"
EXTRA_ARGS="$@"

if [ ! -f "${ABS_RUN_DIR}/experiment.json" ]; then
    echo "ERROR: ${ABS_RUN_DIR} does not contain experiment.json" >&2
    exit 1
fi

JOB_NAME="pphr1_$(basename "$ABS_RUN_DIR")"

JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=${GPUS}
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
echo \"Per-phage rh@1: ${ABS_RUN_DIR}\"
echo \"Started: \$(date)\"
echo \"======================================\"

python3 scripts/analysis/per_phage_rh1.py \\
    \"${ABS_RUN_DIR}\" \\
    --head-mode k_only \\
    --datasets PhageHostLearn PBIP \\
    -o results/analysis/per_phage_rh1_phl_pbip.csv \\
    ${EXTRA_ARGS}

echo \"\"
echo \"======================================\"
echo \"Done: \$(date)\"
echo \"CSV at: ${CIPHER_DIR}/results/analysis/per_phage_rh1_phl_pbip.csv\"
echo \"======================================\"
"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would submit:"
    echo "$JOB_SCRIPT"
else
    mkdir -p "${CIPHER_DIR}/logs" "${CIPHER_DIR}/results/analysis"
    JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
    echo "Submitted ${JOB_ID} — ${JOB_NAME}"
    echo "Log: ${CIPHER_DIR}/logs/${JOB_NAME}_${JOB_ID}.log"
fi
