#!/usr/bin/env bash
#
# Submit a SLURM job on Delta-AI that trains MultiTowerAttentionMLP:
# three identical-architecture AttentionMLP towers fed by ProtT5-XL seg8,
# ESM-2 3B mean, and kmer aa20 k4. Per-head training via v3_uat lists
# (HC_K_UAT_multitop + HC_O_UAT_multitop). Single shared model, K and O
# heads on top of a 960-d joint.
#
# CIPHER_DIR (code + experiments + logs) is auto-detected from the script's
# location. DATA_DIR (shared training/validation data) defaults to the main
# worktree so branches don't duplicate data.
#
# Usage:
#   bash scripts/train_multi_tower_attention_mlp.sh                # submit
#   DRY_RUN=1 bash scripts/train_multi_tower_attention_mlp.sh      # preview

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

GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
# Memory: kmer_aa20_k4 ~160k-d sparse-but-stored-dense; the per-tower SE
# attention bottleneck has ~200M params on the kmer side; combined with
# ProtT5-XL seg8 (8192-d) and ESM-2 3B (2560-d), peak RAM during training
# is large. Agent 1 recommended 192G; bumping a bit to be safe.
MEM="${MEM:-192G}"
TIME="${TIME:-12:00:00}"

# ============================================================
# Data paths
# ============================================================
ASSOC_MAP="${ASSOC_MAP:-${DATA_DIR}/training_data/metadata/host_phage_protein_map.tsv}"
GLYCAN_BINDERS="${GLYCAN_BINDERS:-${DATA_DIR}/training_data/metadata/glycan_binders_custom.tsv}"
VAL_FASTA="${VAL_FASTA:-${DATA_DIR}/validation_data/metadata/validation_rbps_all.faa}"
VAL_DATASETS_DIR="${VAL_DATASETS_DIR:-${DATA_DIR}/validation_data/HOST_RANGE}"

# Per-head training filters (v3_uat = HC_K/O_UAT_multitop). Override only
# if running a different filter recipe.
POSITIVE_LIST_K="${POSITIVE_LIST_K:-${DATA_DIR}/training_data/metadata/highconf_v3_multitop/HC_K_UAT_multitop.list}"
POSITIVE_LIST_O="${POSITIVE_LIST_O:-${DATA_DIR}/training_data/metadata/highconf_v3_multitop/HC_O_UAT_multitop.list}"

# Three training-side embedding NPZs (tower order A -> B -> C).
EMB_A="${EMB_A:-/work/hdd/bfzj/llindsey1/embeddings/prott5_xl_segments8/candidates_prott5_xl_segments8_md5.npz}"
EMB_B="${EMB_B:-/projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_esm2_3b/candidates_embeddings_md5.npz}"
EMB_C="${EMB_C:-/work/hdd/bfzj/llindsey1/kmer_features/candidates_aa20_k4.npz}"
# Three validation-side embedding NPZs (same tower order). Saved into
# config.yaml so the strict-eval job can find them.
VAL_EMB_A="${VAL_EMB_A:-/work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments8/validation_prott5_xl_segments8_md5.npz}"
VAL_EMB_B="${VAL_EMB_B:-/work/hdd/bfzj/llindsey1/validation_embeddings_esm2_3b/validation_embeddings_md5.npz}"
VAL_EMB_C="${VAL_EMB_C:-/work/hdd/bfzj/llindsey1/kmer_features/validation_aa20_k4.npz}"

EMBEDDING_TYPE="${EMBEDDING_TYPE:-multi_tower_prott5xlseg8_esm23b_kmeraa20}"

# ============================================================
# Model / training config (overridable via env)
# ============================================================
MODEL="multi_tower_attention_mlp"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-30}"
LABEL_STRATEGY="${LABEL_STRATEGY:-multi_label_threshold}"
MIN_CLASS_SAMPLES="${MIN_CLASS_SAMPLES:-25}"

NAME="${NAME:-mt_v3uat_prott5xlseg8_esm23b_kmeraa20}"
DRY_RUN="${DRY_RUN:-0}"

# ============================================================
# Pre-submit sanity checks
# ============================================================
echo "============================================================"
echo "MultiTowerAttentionMLP training job"
echo "  Cipher dir:        ${CIPHER_DIR}"
echo "  Run name:          ${NAME}"
echo "  K positive list:   ${POSITIVE_LIST_K}"
echo "  O positive list:   ${POSITIVE_LIST_O}"
echo "  Tower A (emb):     ${EMB_A}"
echo "  Tower B (emb):     ${EMB_B}"
echo "  Tower C (emb):     ${EMB_C}"
echo "  Tower A val:       ${VAL_EMB_A}"
echo "  Tower B val:       ${VAL_EMB_B}"
echo "  Tower C val:       ${VAL_EMB_C}"
echo "============================================================"

REQUIRED_PATHS=("${ASSOC_MAP}" "${GLYCAN_BINDERS}" "${VAL_FASTA}"
                "${POSITIVE_LIST_K}" "${POSITIVE_LIST_O}"
                "${EMB_A}" "${EMB_B}" "${EMB_C}")
for f in "${REQUIRED_PATHS[@]}"; do
    if [ ! -e "${f}" ] && [ "${DRY_RUN}" != "1" ]; then
        echo "ERROR: required path missing: ${f}" >&2
        exit 1
    fi
done

# ============================================================
# Build train command
# ============================================================
TRAIN_CMD="python -m cipher.cli.train_runner \
    --model ${MODEL} \
    --positive_list_k ${POSITIVE_LIST_K} \
    --positive_list_o ${POSITIVE_LIST_O} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --label_strategy ${LABEL_STRATEGY} \
    --min_class_samples ${MIN_CLASS_SAMPLES} \
    --embedding_type ${EMBEDDING_TYPE} \
    --embedding_file_a ${EMB_A} \
    --embedding_file_b ${EMB_B} \
    --embedding_file_c ${EMB_C} \
    --val_embedding_file_a ${VAL_EMB_A} \
    --val_embedding_file_b ${VAL_EMB_B} \
    --val_embedding_file_c ${VAL_EMB_C} \
    --association_map ${ASSOC_MAP} \
    --glycan_binders ${GLYCAN_BINDERS} \
    --val_fasta ${VAL_FASTA} \
    --val_datasets_dir ${VAL_DATASETS_DIR} \
    --name ${NAME}"

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
echo \"MultiTowerAttentionMLP: ${NAME}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

echo \"\"
echo \"=== TRAINING ===\"
${TRAIN_CMD}

echo \"\"
echo \"======================================\"
echo \"Done: ${NAME}\"
echo \"  Finished: \$(date)\"
echo \"======================================\"
echo \"\"
echo \"Strict-eval still pending. Run after this job:\"
echo \"  EXP_NAME=${NAME} bash scripts/strict_eval_multi_tower.sh\"
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
