#!/usr/bin/env bash
#
# Submit a SLURM job on Delta-AI that trains the LightAttentionBinary model
# (K head + O head, sequentially) on ESM-2 650M per-residue embeddings, then
# runs cipher-evaluate on the resulting experiment.
#
# Usage:
#   bash scripts/train_light_attention_binary.sh                 # submit
#   DRY_RUN=1 bash scripts/train_light_attention_binary.sh       # print only
#   NAME=my_run bash scripts/train_light_attention_binary.sh     # override run name
#   LR=1e-4 BATCH_SIZE=16 bash scripts/train_light_attention_binary.sh
#

set -euo pipefail

# ============================================================
# Delta-AI configuration
# ============================================================
ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/cipher-light-attention-binary}"

GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-0}"          # 0 = all memory on the node
TIME="${TIME:-24:00:00}"

# ============================================================
# Data paths on Delta
# ============================================================
ASSOC_MAP="${ASSOC_MAP:-${CIPHER_DIR}/data/training_data/metadata/host_phage_protein_map.tsv}"
GLYCAN_BINDERS="${GLYCAN_BINDERS:-${CIPHER_DIR}/data/training_data/metadata/glycan_binders_custom.tsv}"
VAL_FASTA="${VAL_FASTA:-${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa}"
VAL_DATASETS_DIR="${VAL_DATASETS_DIR:-${CIPHER_DIR}/data/validation_data/HOST_RANGE}"

# Per-residue ESM-2 650M embeddings (variable-length, used by ConvAttn pooler)
TRAIN_EMB="${TRAIN_EMB:-/work/hdd/bfzj/llindsey1/embeddings_full/candidates_embeddings_full_md5.npz}"
VAL_EMB="${VAL_EMB:-/work/hdd/bfzj/llindsey1/validation_embeddings_full/validation_embeddings_full_md5.npz}"
EMBEDDING_TYPE="${EMBEDDING_TYPE:-esm2_650m_full}"

# ============================================================
# Model / training config (overridable via env)
# ============================================================
MODEL="light_attention_binary"
TOOLS="${TOOLS:-DepoScope,PhageRBPdetect}"
LR="${LR:-5e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-30}"
LABEL_STRATEGY="${LABEL_STRATEGY:-multi_label_threshold}"
MIN_CLASS_SAMPLES="${MIN_CLASS_SAMPLES:-25}"
MAX_SAMPLES_K="${MAX_SAMPLES_K:-1000}"
MAX_SAMPLES_O="${MAX_SAMPLES_O:-3000}"
MIN_SOURCES="${MIN_SOURCES:-1}"

NAME="${NAME:-lab_${EMBEDDING_TYPE}}"
DRY_RUN="${DRY_RUN:-0}"

# ============================================================
# Pre-submit sanity checks
# ============================================================
echo "============================================================"
echo "LightAttentionBinary training job"
echo "  Cipher dir: ${CIPHER_DIR}"
echo "  Run name:   ${NAME}"
echo "  Embedding:  ${EMBEDDING_TYPE}"
echo "  Train emb:  ${TRAIN_EMB}"
echo "  Val emb:    ${VAL_EMB}"
echo "============================================================"

for f in "${TRAIN_EMB}" "${ASSOC_MAP}" "${GLYCAN_BINDERS}" "${VAL_FASTA}"; do
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
    --tools ${TOOLS} \
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
