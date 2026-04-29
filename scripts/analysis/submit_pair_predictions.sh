#!/usr/bin/env bash
#
# Submit export_pair_predictions.py as a SLURM batch job.
#
# Produces under <experiment_dir>/results/analysis/:
#   <dataset>_pair_predictions.csv      one row per (phage, positive_host)
#                                       with rank_of_true_host + k_top5_preds
#   <dataset>_protein_k_probs.npz       per-protein K-prob vectors
#   <dataset>_k_class_index.json        class order for the NPZ
#
# Defaults to the current best-PHL run. Override via env vars.
#
# Usage:
#   bash scripts/analysis/submit_pair_predictions.sh
#   EXPERIMENT=experiments/attention_mlp/other_run bash scripts/...
#   DATASET=PBIP bash scripts/analysis/submit_pair_predictions.sh
#   NO_PROTEIN_PROBS=1 bash scripts/analysis/submit_pair_predictions.sh
#   DRY_RUN=1 bash scripts/analysis/submit_pair_predictions.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

EXPERIMENT="${EXPERIMENT:-experiments/attention_mlp/highconf_pipeline_K_prott5_mean}"
DATASET="${DATASET:-PhageHostLearn}"
NO_PROTEIN_PROBS="${NO_PROTEIN_PROBS:-0}"

GPUS=1
CPUS=4
MEM="16G"
TIME="1:00:00"

EXTRA_FLAGS=""
if [ "$NO_PROTEIN_PROBS" = "1" ]; then
    EXTRA_FLAGS="--no-protein-probs"
fi

NAME="pair_preds_${DATASET}_$(basename "${EXPERIMENT}")"
LOG="${CIPHER_DIR}/logs/${NAME}_%j.log"
DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "SUBMIT export_pair_predictions.py"
echo "  experiment: ${EXPERIMENT}"
echo "  dataset:    ${DATASET}"
echo "  protein probs: $([ "$NO_PROTEIN_PROBS" = "1" ] && echo disabled || echo enabled)"
echo "============================================================"

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
echo \"Pair-predictions export: ${NAME}\"
echo \"  experiment: ${EXPERIMENT}\"
echo \"  dataset:    ${DATASET}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

python ${CIPHER_DIR}/scripts/analysis/export_pair_predictions.py \\
    ${EXPERIMENT} \\
    --dataset ${DATASET} \\
    ${EXTRA_FLAGS}

echo \"======================================\"
echo \"Done: ${NAME} at \$(date)\"
echo \"Outputs:\"
ls -la ${CIPHER_DIR}/${EXPERIMENT}/results/analysis/ 2>/dev/null || true
echo \"======================================\"
"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would submit: ${NAME}"
else
    mkdir -p "${CIPHER_DIR}/logs"
    JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
    echo "Submitted ${JOB_ID} - ${NAME}"
fi
