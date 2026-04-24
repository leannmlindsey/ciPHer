#!/usr/bin/env bash
#
# Submit a SLURM job on Delta-AI that runs cipher-evaluate against a
# completed light_attention_binary experiment. Useful when the training
# job's built-in eval step was skipped (e.g. validation embeddings weren't
# ready at train time, or the training script had a wrong val path).
#
# Usage:
#   # Evaluate the default run (lab_prott5_xl_full_highconf_pipeline):
#   bash scripts/evaluate_light_attention_binary.sh
#
#   # Evaluate a specific run:
#   EXP_NAME=lab_esm2_650m_full_highconf_pipeline \
#       bash scripts/evaluate_light_attention_binary.sh
#
#   # Override the val embedding NPZ:
#   EXP_NAME=<run_name> VAL_EMB=/path/to/val.npz \
#       bash scripts/evaluate_light_attention_binary.sh
#
#   # Dry run (print job script, don't submit):
#   DRY_RUN=1 bash scripts/evaluate_light_attention_binary.sh
#

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

# Evaluation is modest: predictor inference over ~1500 val proteins × 5
# datasets. 1 GPU, 64G mem, 1h wall is plenty.
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

# Map run name to the right validation NPZ if not explicitly set.
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
NAME="eval_${EXP_NAME}"

# ============================================================
# Pre-submit sanity checks
# ============================================================
echo "============================================================"
echo "LightAttentionBinary evaluation job"
echo "  Cipher dir:   ${CIPHER_DIR}"
echo "  Experiment:   ${EXP_DIR}"
echo "  Val emb:      ${VAL_EMB}"
echo "============================================================"

if [ ! -d "${EXP_DIR}" ] && [ "${DRY_RUN}" != "1" ]; then
    echo "ERROR: experiment dir does not exist: ${EXP_DIR}" >&2
    exit 1
fi
if [ ! -f "${VAL_EMB}" ] && [ "${DRY_RUN}" != "1" ]; then
    echo "ERROR: validation NPZ does not exist: ${VAL_EMB}" >&2
    exit 1
fi

EVAL_CMD="python -m cipher.evaluation.runner ${EXP_DIR} --val-embedding-file ${VAL_EMB}"

# ============================================================
# Assemble the job script
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

echo \"======================================\"
echo \"Evaluate: ${EXP_NAME}\"
echo \"  Experiment: ${EXP_DIR}\"
echo \"  Val emb:    ${VAL_EMB}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

${EVAL_CMD}

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
    echo "Submitted job ${JOB_ID} — ${NAME}"
    echo "Log: ${CIPHER_DIR}/logs/${NAME}_${JOB_ID}.log"
fi
