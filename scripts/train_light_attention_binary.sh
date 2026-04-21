#!/usr/bin/env bash
#
# Submit a SLURM job on Delta-AI that trains the LightAttentionBinary model
# (K head + O head, sequentially) on ESM-2 650M per-residue embeddings, then
# runs cipher-evaluate on the resulting experiment.
#
# CIPHER_DIR (code + experiments + logs) is auto-detected from the script's
# location. DATA_DIR (shared training/validation data) defaults to the main
# worktree at /projects/bfzj/llindsey1/PHI_TSP/ciPHer/data so it doesn't
# need to be duplicated per branch.
#
# Usage:
#   bash scripts/train_light_attention_binary.sh                 # submit
#   DRY_RUN=1 bash scripts/train_light_attention_binary.sh       # print only
#   NAME=my_run bash scripts/train_light_attention_binary.sh     # override run name
#   LR=1e-4 BATCH_SIZE=16 bash scripts/train_light_attention_binary.sh
#   DATA_DIR=/some/other/data bash scripts/train_light_attention_binary.sh
#

set -euo pipefail

# ============================================================
# Delta-AI configuration
# ============================================================
ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"

# Auto-detect CIPHER_DIR from this script's location (repo root = dir
# containing scripts/). Override with CIPHER_DIR=... to point elsewhere.
# CIPHER_DIR is where code + experiments + logs live.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="${CIPHER_DIR:-$(dirname "$SCRIPT_DIR")}"

# DATA_DIR is the canonical data home on Delta — shared across all worktrees
# / clones so we don't duplicate 10s of GB of TSVs + validation data. The
# main worktree at /projects/bfzj/llindsey1/PHI_TSP/ciPHer holds the data.
DATA_DIR="${DATA_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer/data}"

GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-0}"          # 0 = all memory on the node
TIME="${TIME:-24:00:00}"

# ============================================================
# Data paths (resolved from DATA_DIR, not CIPHER_DIR)
# ============================================================
ASSOC_MAP="${ASSOC_MAP:-${DATA_DIR}/training_data/metadata/host_phage_protein_map.tsv}"
GLYCAN_BINDERS="${GLYCAN_BINDERS:-${DATA_DIR}/training_data/metadata/glycan_binders_custom.tsv}"
VAL_FASTA="${VAL_FASTA:-${DATA_DIR}/validation_data/metadata/validation_rbps_all.faa}"
VAL_DATASETS_DIR="${VAL_DATASETS_DIR:-${DATA_DIR}/validation_data/HOST_RANGE}"

# Per-residue ESM-2 650M embeddings (variable-length, used by ConvAttn pooler).
# Paths updated 2026-04-21 to the new full-coverage extraction; the old
# /work/hdd/bfzj/llindsey1/embeddings_full/... file only covered ~4.5K MD5s
# and left the model training on 14% of the filtered set.
TRAIN_EMB="${TRAIN_EMB:-/work/hdd/bfzj/llindsey1/embeddings/esm2_650m_full/candidates_esm2_650m_full_md5.npz}"
VAL_EMB="${VAL_EMB:-/work/hdd/bfzj/llindsey1/validation_embeddings/esm2_650m_full/validation_esm2_650m_full_md5.npz}"
EMBEDDING_TYPE="${EMBEDDING_TYPE:-esm2_650m_full}"

# Training-set filter and cluster-stratified downsampling (see
# memory/project_training_filters.md). Default to the advisor's curated
# positive list + 70%-identity cluster-stratified sampling — the combo
# that improved PHL performance in the attention_mlp sweep.
POSITIVE_LIST="${POSITIVE_LIST:-${DATA_DIR}/training_data/metadata/pipeline_positive.list}"
CLUSTER_FILE="${CLUSTER_FILE:-${DATA_DIR}/training_data/metadata/candidates_clusters.tsv}"
CLUSTER_THRESHOLD="${CLUSTER_THRESHOLD:-70}"

# ============================================================
# Model / training config (overridable via env)
# ============================================================
MODEL="light_attention_binary"
LR="${LR:-5e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-30}"
LABEL_STRATEGY="${LABEL_STRATEGY:-multi_label_threshold}"
MIN_CLASS_SAMPLES="${MIN_CLASS_SAMPLES:-25}"
MAX_SAMPLES_K="${MAX_SAMPLES_K:-1000}"
MAX_SAMPLES_O="${MAX_SAMPLES_O:-3000}"
MIN_SOURCES="${MIN_SOURCES:-1}"

NAME="${NAME:-lab_${EMBEDDING_TYPE}_posList_cl${CLUSTER_THRESHOLD}}"
DRY_RUN="${DRY_RUN:-0}"

# ============================================================
# Pre-submit sanity checks
# ============================================================
echo "============================================================"
echo "LightAttentionBinary training job"
echo "  Cipher dir:     ${CIPHER_DIR}"
echo "  Run name:       ${NAME}"
echo "  Embedding:      ${EMBEDDING_TYPE}"
echo "  Train emb:      ${TRAIN_EMB}"
echo "  Val emb:        ${VAL_EMB}"
echo "  Positive list:  ${POSITIVE_LIST}"
echo "  Cluster file:   ${CLUSTER_FILE} (threshold=${CLUSTER_THRESHOLD})"
echo "============================================================"

for f in "${TRAIN_EMB}" "${ASSOC_MAP}" "${GLYCAN_BINDERS}" "${VAL_FASTA}" \
         "${POSITIVE_LIST}" "${CLUSTER_FILE}"; do
    if [ ! -e "${f}" ] && [ "${DRY_RUN}" != "1" ]; then
        echo "ERROR: required path missing: ${f}" >&2
        exit 1
    fi
done

VAL_STATUS="ready"
if [ ! -f "${VAL_EMB}" ]; then
    VAL_STATUS="missing (will train only)"
fi

# ============================================================
# Commands run inside the SLURM job
# ============================================================
TRAIN_CMD="python -m cipher.cli.train_runner \
    --model ${MODEL} \
    --positive_list ${POSITIVE_LIST} \
    --cluster_file ${CLUSTER_FILE} \
    --cluster_threshold ${CLUSTER_THRESHOLD} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --label_strategy ${LABEL_STRATEGY} \
    --min_class_samples ${MIN_CLASS_SAMPLES} \
    --max_samples_per_k ${MAX_SAMPLES_K} \
    --max_samples_per_o ${MAX_SAMPLES_O} \
    --min_sources ${MIN_SOURCES} \
    --embedding_type ${EMBEDDING_TYPE} \
    --embedding_file ${TRAIN_EMB} \
    --association_map ${ASSOC_MAP} \
    --glycan_binders ${GLYCAN_BINDERS} \
    --val_fasta ${VAL_FASTA} \
    --val_datasets_dir ${VAL_DATASETS_DIR} \
    --val_embedding_file ${VAL_EMB} \
    --name ${NAME}"

EXP_DIR="${CIPHER_DIR}/experiments/${MODEL}/${NAME}"
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
echo \"LightAttentionBinary: ${NAME}\"
echo \"  Train embeddings: ${TRAIN_EMB}\"
echo \"  Val embeddings:   ${VAL_EMB}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

if [ ! -f \"${TRAIN_EMB}\" ]; then
    echo \"ERROR: training embedding file not found: ${TRAIN_EMB}\"
    exit 1
fi

VAL_READY=true
if [ ! -f \"${VAL_EMB}\" ]; then
    echo \"WARNING: validation embeddings not found: ${VAL_EMB}\"
    echo \"  Will train but skip evaluation.\"
    VAL_READY=false
fi

echo \"\"
echo \"=== TRAINING ===\"
${TRAIN_CMD}

if [ \"\${VAL_READY}\" = true ]; then
    echo \"\"
    echo \"=== EVALUATING ===\"
    ${EVAL_CMD}
else
    echo \"\"
    echo \"SKIPPING evaluation: validation embeddings not yet available.\"
    echo \"After they are generated, run:\"
    echo \"  ${EVAL_CMD}\"
fi

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
    echo "Val status: ${VAL_STATUS}"
else
    mkdir -p "${CIPHER_DIR}/logs"
    JOB_ID=$(echo "${JOB_SCRIPT}" | sbatch | awk '{print $NF}')
    echo "Submitted job ${JOB_ID} — ${NAME} (val: ${VAL_STATUS})"
    echo "Log: ${CIPHER_DIR}/logs/${NAME}_${JOB_ID}.log"
fi
