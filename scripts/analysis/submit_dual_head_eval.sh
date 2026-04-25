#!/usr/bin/env bash
#
# Evaluate a dual-head configuration: take the K head from one trained
# experiment and the O head from another. Each head consumes its own
# (potentially different) embedding family at inference time.
#
# Concrete default use case:
#   K from: experiments/attention_mlp/highconf_pipeline_K_prott5_mean
#           (our strong K head, trained on ProtT5 mean)
#   O from: experiments/attention_mlp/repro_old_v3_full_in_cipher
#           (the old-recipe O head we just retrained, ESM-2 650M mean)
#
# Override via env vars:
#   K_DIR=<path>        K experiment dir
#   O_DIR=<path>        O experiment dir
#   K_VAL_EMB=<path>    K validation embedding NPZ
#   O_VAL_EMB=<path>    O validation embedding NPZ
#   SCORE_NORM={zscore,raw}   default zscore
#   TIE_METHOD={competition,arbitrary}   default competition
#
# Usage:
#   bash scripts/analysis/submit_dual_head_eval.sh
#   DRY_RUN=1 bash scripts/analysis/submit_dual_head_eval.sh
#   SCORE_NORM=raw TIE_METHOD=arbitrary bash scripts/analysis/submit_dual_head_eval.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

K_DIR="${K_DIR:-${CIPHER_DIR}/experiments/attention_mlp/highconf_pipeline_K_prott5_mean}"
O_DIR="${O_DIR:-${CIPHER_DIR}/experiments/attention_mlp/repro_old_v3_full_in_cipher}"

# Default validation embeddings for the default K + O pair above
K_VAL_EMB="${K_VAL_EMB:-/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz}"
O_VAL_EMB="${O_VAL_EMB:-/projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/validation_data/combined/validation_inputs/validation_embeddings_md5.npz}"

SCORE_NORM="${SCORE_NORM:-zscore}"
TIE_METHOD="${TIE_METHOD:-competition}"

VAL_FASTA="${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${CIPHER_DIR}/data/validation_data/HOST_RANGE"

GPUS=1
CPUS=4
MEM="32G"
TIME="1:00:00"

# Tag the run so multiple eval-mode variants don't clobber each other
NAME="dual_$(basename ${K_DIR})_x_$(basename ${O_DIR})_${SCORE_NORM}_${TIE_METHOD}"
LOG="${CIPHER_DIR}/logs/${NAME}_%j.log"

DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "DUAL-HEAD EVAL"
echo "  K from: ${K_DIR}"
echo "  O from: ${O_DIR}"
echo "  K val embedding: ${K_VAL_EMB}"
echo "  O val embedding: ${O_VAL_EMB}"
echo "  score-norm: ${SCORE_NORM}"
echo "  tie-method: ${TIE_METHOD}"
echo "============================================================"
echo ""

JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${NAME:0:32}
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

echo \"=== DUAL-HEAD EVAL: ${NAME} ===\"
echo \"  Started: \$(date)\"

python -m cipher.evaluation.runner \\
    --k-experiment-dir ${K_DIR} \\
    --o-experiment-dir ${O_DIR} \\
    --k-val-embedding-file ${K_VAL_EMB} \\
    --o-val-embedding-file ${O_VAL_EMB} \\
    --val-fasta ${VAL_FASTA} \\
    --val-datasets-dir ${VAL_DATASETS_DIR} \\
    --score-norm ${SCORE_NORM} \\
    --tie-method ${TIE_METHOD}

echo \"\"
echo \"Done: \$(date)\"
"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would submit: ${NAME}"
    echo ""
    echo "Recipe to also try (after the default zscore-competition lands):"
    echo "  SCORE_NORM=raw TIE_METHOD=arbitrary bash scripts/analysis/submit_dual_head_eval.sh"
    echo "  (this matches the old-eval methodology that produced 0.291)"
    exit 0
fi

mkdir -p "${CIPHER_DIR}/logs"
JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
echo "Submitted ${JOB_ID} - ${NAME}"
echo ""
echo "Log: ${CIPHER_DIR}/logs/${NAME}_${JOB_ID}.log"
echo ""
echo "Worth running both eval modes once O is available:"
echo "  bash scripts/analysis/submit_dual_head_eval.sh                                  # zscore (our default)"
echo "  SCORE_NORM=raw TIE_METHOD=arbitrary bash scripts/analysis/submit_dual_head_eval.sh   # old-style"
