#!/usr/bin/env bash
#
# SLURM wrapper for scripts/analysis/decision_breakdown.py.
# Submits one GPU job that runs the per-decision breakdown (which head drove
# each max score) on every run_dir passed as an argument. Defaults to all
# la_seg4_* experiments under experiments/light_attention/.
#
# Usage:
#   bash scripts/analysis/run_decision_breakdown.sh
#   bash scripts/analysis/run_decision_breakdown.sh experiments/light_attention/la_seg4_match_sweep
#   DRY_RUN=1 bash scripts/analysis/run_decision_breakdown.sh

set -euo pipefail

ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="${CIPHER_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"

GPUS=1
CPUS=4
MEM="16G"

DRY_RUN="${DRY_RUN:-0}"

if [ $# -gt 0 ]; then
    RUN_DIRS=("$@")
else
    RUN_DIRS=()
    for d in "${CIPHER_DIR}/experiments/light_attention"/la_seg4_*/; do
        [ -d "$d" ] || continue
        RUN_DIRS+=("${d%/}")
    done
fi

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "ERROR: no run_dirs given and no la_seg4_* experiments found under ${CIPHER_DIR}/experiments/light_attention/" >&2
    exit 1
fi

ABS_RUN_DIRS=()
for d in "${RUN_DIRS[@]}"; do
    abs="$(cd "$d" && pwd)"
    if [ ! -f "${abs}/experiment.json" ]; then
        echo "ERROR: ${abs} does not contain experiment.json — not a run dir" >&2
        exit 1
    fi
    ABS_RUN_DIRS+=("$abs")
done

# Decision breakdown does a single (not triple) eval pass per run, so budget
# less time than run_eval_per_head.sh.
N=${#ABS_RUN_DIRS[@]}
MIN_TOTAL=$(( N * 6 + 5 ))
TIME_HH=$(printf '%02d' $(( MIN_TOTAL / 60 )))
TIME_MM=$(printf '%02d' $(( MIN_TOTAL % 60 )))
TIME="${TIME_HH}:${TIME_MM}:00"

FIRST_NAME="$(basename "${ABS_RUN_DIRS[0]}")"
if [ $N -eq 1 ]; then
    JOB_NAME="dbd_${FIRST_NAME}"
else
    JOB_NAME="dbd_${FIRST_NAME}_plus$((N - 1))"
fi

echo "============================================================"
echo "DECISION BREAKDOWN (K-vs-O provenance at rank-1 + per positive)"
echo "  Cipher:    ${CIPHER_DIR}"
echo "  Runs (${N}):"
for d in "${ABS_RUN_DIRS[@]}"; do echo "    ${d}"; done
echo "  SLURM:     ${TIME}, ${GPUS} GPU, ${CPUS} CPU, ${MEM}"
echo "============================================================"
echo ""

RUN_LIST="$(printf '    "%s"\n' "${ABS_RUN_DIRS[@]}")"

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

RUNS=(
${RUN_LIST})

for RUN in \"\${RUNS[@]}\"; do
    echo \"\"
    echo \"======================================\"
    echo \"Decision breakdown: \${RUN}\"
    echo \"Started: \$(date)\"
    echo \"======================================\"
    python3 scripts/analysis/decision_breakdown.py \"\${RUN}\"
done

echo \"\"
echo \"======================================\"
echo \"All decision breakdowns finished: \$(date)\"
echo \"======================================\"
"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would submit the following job:"
    echo "---"
    echo "$JOB_SCRIPT"
    echo "---"
    echo "DRY RUN complete. Set DRY_RUN=0 to submit."
else
    mkdir -p "${CIPHER_DIR}/logs"
    JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
    echo "Submitted ${JOB_ID} — ${JOB_NAME}"
    echo "Monitor: squeue -u \$USER"
    echo "Log:     tail -f ${CIPHER_DIR}/logs/${JOB_NAME}_${JOB_ID}.log"
fi
