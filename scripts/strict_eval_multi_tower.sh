#!/usr/bin/env bash
#
# Strict-eval (any-hit + per-pair HR@k under fixed strict denominator)
# for a trained MultiTowerAttentionMLP experiment. Two-step pipeline:
#
# 1. Build a composite val NPZ that concatenates the three tower
#    val embeddings (A || B || C), keyed by MD5. Cached in
#    ${CIPHER_DIR}/cache/multi_tower_composite_val.npz so this step
#    is one-time; subsequent eval runs skip it.
# 2. Run scripts/analysis/per_head_strict_eval.py with the composite
#    NPZ as --val-embedding-file. The canonical script invokes our
#    MultiTowerPredictor which splits the composite back into 3
#    sub-vectors before the model forward.
#
# Output: <experiment>/results/per_head_strict_eval.json
#
# Usage:
#   bash scripts/strict_eval_multi_tower.sh
#   EXP_NAME=mt_v3uat_prott5xlseg8_esm23b_kmeraa20 \
#       bash scripts/strict_eval_multi_tower.sh
#   DRY_RUN=1 bash scripts/strict_eval_multi_tower.sh

set -euo pipefail

ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="${CIPHER_DIR:-$(dirname "$SCRIPT_DIR")}"
DATA_DIR="${DATA_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer/data}"

# Resources: composite build is CPU-bound (~1500 proteins, just NPZ I/O
# + concat); strict eval is predictor inference over ~1500 val proteins
# x 5 datasets x 3 head-modes. Bumped mem because the kmer block (160k-d)
# stays in memory.
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-128G}"
TIME="${TIME:-2:00:00}"

MODEL="multi_tower_attention_mlp"
EXP_NAME="${EXP_NAME:-mt_v3uat_prott5xlseg8_esm23b_kmeraa20}"
EXP_DIR="${CIPHER_DIR}/experiments/${MODEL}/${EXP_NAME}"

# Three validation NPZs (tower order). Defaults match the training
# script's defaults; override via env if you trained against different
# embeddings.
VAL_EMB_A="${VAL_EMB_A:-/work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments8/validation_prott5_xl_segments8_md5.npz}"
VAL_EMB_B="${VAL_EMB_B:-/work/hdd/bfzj/llindsey1/validation_embeddings_esm2_3b/validation_embeddings_md5.npz}"
VAL_EMB_C="${VAL_EMB_C:-/work/hdd/bfzj/llindsey1/kmer_features/validation_aa20_k4.npz}"

# Composite cache. Shared across runs since the val NPZs don't change
# between model training rounds.
CACHE_DIR="${CIPHER_DIR}/cache"
COMPOSITE="${COMPOSITE:-${CACHE_DIR}/multi_tower_composite_val.npz}"

NAME="strict_eval_${EXP_NAME}"
DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "MultiTowerAttentionMLP strict-eval"
echo "  Cipher dir:   ${CIPHER_DIR}"
echo "  Experiment:   ${EXP_DIR}"
echo "  Val A:        ${VAL_EMB_A}"
echo "  Val B:        ${VAL_EMB_B}"
echo "  Val C:        ${VAL_EMB_C}"
echo "  Composite:    ${COMPOSITE}"
echo "============================================================"

if [ ! -d "${EXP_DIR}" ] && [ "${DRY_RUN}" != "1" ]; then
    echo "ERROR: experiment dir does not exist: ${EXP_DIR}" >&2
    exit 1
fi
for f in "${VAL_EMB_A}" "${VAL_EMB_B}" "${VAL_EMB_C}"; do
    if [ ! -f "${f}" ] && [ "${DRY_RUN}" != "1" ]; then
        echo "ERROR: val NPZ missing: ${f}" >&2
        exit 1
    fi
done

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
echo \"Strict eval (multi-tower): ${EXP_NAME}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

# Step 1: build composite val NPZ if not cached.
if [ -f \"${COMPOSITE}\" ]; then
    echo \"Composite val NPZ already exists, skipping build:\"
    echo \"  ${COMPOSITE}\"
else
    mkdir -p \"${CACHE_DIR}\"
    echo \"=== STEP 1: build composite val NPZ ===\"
    python scripts/build_multi_tower_composite_val.py \\
        --val-emb-a ${VAL_EMB_A} \\
        --val-emb-b ${VAL_EMB_B} \\
        --val-emb-c ${VAL_EMB_C} \\
        --out ${COMPOSITE}
fi

# Step 2: canonical strict-eval against the composite NPZ. Our predict.py's
# MultiTowerPredictor splits the composite vector into 3 tower inputs
# inside predict_protein.
echo \"\"
echo \"=== STEP 2: strict-eval ===\"
python scripts/analysis/per_head_strict_eval.py ${EXP_DIR} \\
    --val-embedding-file ${COMPOSITE}

echo \"\"
echo \"=== output ===\"
ls -la ${EXP_DIR}/results/per_head_strict_eval.json

echo \"\"
echo \"======================================\"
echo \"Done: ${NAME}\"
echo \"  Finished: \$(date)\"
echo \"======================================\"
"

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
