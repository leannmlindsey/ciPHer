#!/usr/bin/env bash
#
# Submit a SLURM job on Delta-AI that runs per_head_strict_eval.py
# (the strict-denominator + any-hit metric introduced in main 2026-04-27)
# against a completed light_attention_binary experiment.
#
# This is the eval that produces the headline numbers (any-hit HR@k) used
# in cross-tool comparisons. Use this AFTER the in-job
# cipher-evaluate has finished -- per_head_strict_eval.py emits a
# separate `<experiment>/results/per_head_strict_eval.json` file in the
# new schema; the legacy `evaluation.json` is left untouched.
#
# Usage:
#   bash scripts/strict_eval_light_attention_binary.sh
#   EXP_NAME=lab_esm2_650m_full_highconf_pipeline \
#       bash scripts/strict_eval_light_attention_binary.sh
#   DRY_RUN=1 bash scripts/strict_eval_light_attention_binary.sh

set -euo pipefail

# ============================================================
# Delta-AI configuration
# ============================================================
ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="${CIPHER_DIR:-$(dirname "$SCRIPT_DIR")}"
DATA_DIR="${DATA_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer/data}"

# Strict eval is predictor inference + ranking math; same scale as the
# legacy eval (~1500 val proteins x 5 datasets x 3 head-modes).
GPUS="${GPUS:-1}"
CPUS="${CPUS:-4}"
MEM="${MEM:-64G}"
TIME="${TIME:-1:00:00}"

# ============================================================
# Experiment + val embedding selection
# ============================================================
MODEL="light_attention_binary"
EXP_NAME="${EXP_NAME:-lab_prott5_xl_full_highconf_pipeline}"
EXP_DIR="${CIPHER_DIR}/experiments/${MODEL}/${EXP_NAME}"

if [ -z "${VAL_EMB:-}" ]; then
    case "${EXP_NAME}" in
        *prott5_xl_full*)
            VAL_EMB="/work/hdd/bfzj/llindsey1/val_prott5_xl_full/validation_prott5_full_md5.npz"
            ;;
        *esm2_650m_full*)
            VAL_EMB="/work/hdd/bfzj/llindsey1/val_esm2_650m_full/validation_esm2_full_md5.npz"
            ;;
        *)
            echo "ERROR: cannot infer VAL_EMB for ${EXP_NAME}; set VAL_EMB=..." >&2
            exit 1
            ;;
    esac
fi

DRY_RUN="${DRY_RUN:-0}"
NAME="strict_eval_${EXP_NAME}"

# ============================================================
# Pre-submit checks
# ============================================================
echo "============================================================"
echo "LightAttentionBinary strict-eval job (any-hit HR@k)"
echo "  Cipher dir: ${CIPHER_DIR}"
echo "  Experiment: ${EXP_DIR}"
echo "  Val emb:    ${VAL_EMB}"
echo "============================================================"

if [ ! -d "${EXP_DIR}" ] && [ "${DRY_RUN}" != "1" ]; then
    echo "ERROR: experiment dir does not exist: ${EXP_DIR}" >&2
    exit 1
fi
if [ ! -f "${VAL_EMB}" ] && [ "${DRY_RUN}" != "1" ]; then
    echo "ERROR: validation NPZ does not exist: ${VAL_EMB}" >&2
    exit 1
fi

EVAL_CMD="python scripts/analysis/per_head_strict_eval.py ${EXP_DIR} --val-embedding-file ${VAL_EMB}"

# ============================================================
# Assemble SLURM job
# ============================================================
JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${CIPHER_DIR}/logs/${NAME}_%j.log
#SBATCH --error=${CIPHER_DIR}/logs/${NAME}_%j.log

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

echo \"======================================\"
echo \"Strict eval: ${EXP_NAME}\"
echo \"  Experiment: ${EXP_DIR}\"
echo \"  Val emb:    ${VAL_EMB}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

${EVAL_CMD}

echo \"\"
echo \"=== output ===\"
ls -la ${EXP_DIR}/results/per_head_strict_eval.json

echo \"\"
echo \"======================================\"
echo \"Done: ${NAME}\"
echo \"  Finished: \$(date)\"
echo \"======================================\"
"

# ============================================================
# Submit (or dry-run)
# ============================================================
if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY RUN] job script follows:"
    echo "----------------------------------------"
    echo "${JOB_SCRIPT}"
    echo "----------------------------------------"
else
    mkdir -p "${CIPHER_DIR}/logs"
    JOB_ID=$(echo "${JOB_SCRIPT}" | sbatch | awk '{print $NF}')
    echo "Submitted job ${JOB_ID} -- ${NAME}"
    echo "Log: ${CIPHER_DIR}/logs/${NAME}_${JOB_ID}.log"
fi
