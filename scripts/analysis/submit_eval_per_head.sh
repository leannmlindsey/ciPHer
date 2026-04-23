#!/usr/bin/env bash
#
# Submit eval_per_head.py as SLURM batch jobs against one or more existing
# experiment directories. No retraining — this just re-runs inference with
# the K head blanked, O head blanked, or both (the default). Output is a
# per-dataset HR@1 table showing dK and dO contributions per agent 2's
# analysis.
#
# Motivation: agent 2 found that on LA + ProtT5-XL seg8 + v1 highconf,
# K-only inference gives PHL rh@1 = 0.235 vs the default z-score combined
# rh@1 = 0.136 — because the O head is miscalibrated (dO = −0.42 on PBIP).
# Same test on the attention_mlp runs may reveal a similar gap.
#
# Usage:
#   # Default: fire at v1 + v2_strict + v2_uat attention_mlp runs
#   bash scripts/analysis/submit_eval_per_head.sh
#
#   # One specific run:
#   RUNS=experiments/attention_mlp/v2_strict_prott5_mean \
#       bash scripts/analysis/submit_eval_per_head.sh
#
#   # Preview:
#   DRY_RUN=1 bash scripts/analysis/submit_eval_per_head.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

# Default run list — the three attention_mlp runs worth re-evaluating
# against agent 2's K-only framing. Override via RUNS env var.
DEFAULT_RUNS=(
    "experiments/attention_mlp/highconf_pipeline_K_prott5_mean"
    "experiments/attention_mlp/v2_strict_prott5_mean"
    "experiments/attention_mlp/v2_uat_prott5_mean"
)
if [ -n "${RUNS:-}" ]; then
    read -ra RUN_LIST <<< "$RUNS"
else
    RUN_LIST=("${DEFAULT_RUNS[@]}")
fi

GPUS=1
CPUS=4
MEM="32G"
TIME="1:00:00"

DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "SUBMIT eval_per_head.py — K/O head ablation on trained runs"
echo "  Targets:"
for r in "${RUN_LIST[@]}"; do
    echo "    ${r}"
done
echo "============================================================"
echo ""

N_SUBMITTED=0
for run in "${RUN_LIST[@]}"; do
    full_run="${CIPHER_DIR}/${run}"
    if [ ! -d "$full_run" ]; then
        echo "  SKIP ${run} — directory not found"
        continue
    fi

    NAME="evh_$(basename "$run")"
    LOG="${CIPHER_DIR}/logs/${NAME}_%j.log"

    JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG}
#SBATCH --error=${LOG}

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

echo \"======================================\"
echo \"eval_per_head: ${run}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

python ${CIPHER_DIR}/scripts/analysis/eval_per_head.py ${full_run}

echo \"======================================\"
echo \"Done: ${NAME} at \$(date)\"
echo \"======================================\"
"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] ${NAME}"
    else
        mkdir -p "${CIPHER_DIR}/logs"
        JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
        echo "  Submitted ${JOB_ID} — ${NAME}"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    fi
done

echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN complete."
else
    echo "Submitted ${N_SUBMITTED} eval-per-head job(s)."
    echo ""
    echo "Each job's log shows a three-column table:"
    echo "  both (default z-score) | K-only | O-only | dK | dO"
    echo "per dataset, for both rank_hosts and rank_phages."
fi
