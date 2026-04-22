#!/usr/bin/env bash
#
# SLURM wrapper: re-evaluate existing LA experiments with --score-norm raw
# instead of the default --score-norm zscore.
#
# Directly tests the "z-score penalizes O" hypothesis from the O-silence
# diagnostic thread: if O contribution rises significantly under raw scoring,
# z-score normalization is the cause (or a contributor). No retraining —
# cipher-evaluate loads the trained model and re-runs ranking with a different
# aggregation. Writes output to {run_dir}/results/evaluation_raw.json so the
# original zscore evaluation.json is preserved for comparison.
#
# Compare the two afterward with:
#   python scripts/analysis/show_eval_all.py                   # zscore
#   python scripts/analysis/show_eval_all.py --variant raw     # raw
#
# Usage:
#   bash scripts/analysis/run_score_norm_raw.sh
#       # re-eval every la_* experiment sequentially in one job
#
#   bash scripts/analysis/run_score_norm_raw.sh experiments/light_attention/la_seg4_match_sweep
#       # re-eval a single run
#
#   DRY_RUN=1 bash scripts/analysis/run_score_norm_raw.sh

set -euo pipefail

# ============================================================
# Delta-AI configuration
# ============================================================
ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="${CIPHER_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"

GPUS=1
CPUS=4
MEM="16G"

DRY_RUN="${DRY_RUN:-0}"

# ============================================================
# Collect run_dirs (args or default glob)
# ============================================================
if [ $# -gt 0 ]; then
    RUN_DIRS=("$@")
else
    RUN_DIRS=()
    for d in "${CIPHER_DIR}/experiments/light_attention"/la_*/; do
        [ -d "$d" ] || continue
        RUN_DIRS+=("${d%/}")
    done
fi

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "ERROR: no run_dirs given and no la_* experiments found under ${CIPHER_DIR}/experiments/light_attention/" >&2
    exit 1
fi

# Abs paths + validate
ABS_RUN_DIRS=()
for d in "${RUN_DIRS[@]}"; do
    abs="$(cd "$d" && pwd)"
    if [ ! -f "${abs}/experiment.json" ]; then
        echo "ERROR: ${abs} does not contain experiment.json — not a run dir" >&2
        exit 1
    fi
    ABS_RUN_DIRS+=("$abs")
done

# Budget: ~3 min per run + 5 min startup (re-eval is a single pass, much
# lighter than eval_per_head which does 3 passes)
N=${#ABS_RUN_DIRS[@]}
MIN_TOTAL=$(( N * 3 + 5 ))
TIME_HH=$(printf '%02d' $(( MIN_TOTAL / 60 )))
TIME_MM=$(printf '%02d' $(( MIN_TOTAL % 60 )))
TIME="${TIME_HH}:${TIME_MM}:00"

FIRST_NAME="$(basename "${ABS_RUN_DIRS[0]}")"
if [ $N -eq 1 ]; then
    JOB_NAME="snraw_${FIRST_NAME}"
else
    JOB_NAME="snraw_${FIRST_NAME}_plus$((N - 1))"
fi

echo "============================================================"
echo "SCORE-NORM RAW RE-EVALUATION"
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
    echo \"score-norm=raw re-eval: \${RUN}\"
    echo \"Started: \$(date)\"
    echo \"======================================\"
    python3 -m cipher.evaluation.runner \"\${RUN}\" --score-norm raw \\
        -o \"\${RUN}/results/evaluation_raw.json\"
done

echo \"\"
echo \"======================================\"
echo \"All raw re-evals finished: \$(date)\"
echo \"Compare via: python3 scripts/analysis/show_eval_all.py --variant raw\"
echo \"======================================\"
"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would submit:"
    echo "---"
    echo "$JOB_SCRIPT"
    echo "---"
else
    mkdir -p "${CIPHER_DIR}/logs"
    JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
    echo "Submitted ${JOB_ID} — ${JOB_NAME}"
    echo "Log: ${CIPHER_DIR}/logs/${JOB_NAME}_${JOB_ID}.log"
fi
