#!/usr/bin/env bash
#
# Submit a SLURM job on Delta-AI that trains LightAttentionBinary (K head +
# O head, sequentially) on ESM-2 650M per-residue (full) embeddings using
# the highconf_pipeline_positive_K positive list, then runs cipher-evaluate.
#
# Filter rationale (from agent 1's 2026-04-22 notes): the
# highconf_pipeline_positive_K list (12,481 proteins) already encodes the
# cluster-level filter, so --cluster_file and --max_samples_per_* are
# intentionally NOT passed — matches scripts/run_highconf_experiment.sh.
#
# CIPHER_DIR (code + experiments + logs) is auto-detected from the script's
# location. DATA_DIR (shared training/validation data) defaults to the main
# worktree so branches don't duplicate data.
#
# Usage:
#   bash scripts/train_light_attention_binary_esm2.sh                # submit
#   DRY_RUN=1 bash scripts/train_light_attention_binary_esm2.sh      # preview
#   NAME=my_run bash scripts/train_light_attention_binary_esm2.sh
#   LR=1e-4 BATCH_SIZE=16 bash scripts/train_light_attention_binary_esm2.sh
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

# Shared data lives in the main worktree, not this branch's worktree.
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
# Updated 2026-04-25 to point at the per-length-bin DIRECTORY, not a merged
# NPZ. The merge step kept failing under /work/hdd/bfzj quota pressure
# (213 GB output truncated mid-write twice). train.py's
# `load_embeddings_or_bins` handles a directory transparently and saves
# the ~199 GB merged-file footprint.
TRAIN_EMB="${TRAIN_EMB:-/work/hdd/bfzj/llindsey1/embeddings_full/split_embeddings}"
VAL_EMB="${VAL_EMB:-/work/hdd/bfzj/llindsey1/val_esm2_650m_full/validation_esm2_full_md5.npz}"
EMBEDDING_TYPE="${EMBEDDING_TYPE:-esm2_650m_full}"

# Training-set filter. Two recipes supported via env vars:
#
#   (1) v1 highconf  -- single positive list, no cluster, no caps.
#       Default below: POSITIVE_LIST=highconf_pipeline_positive_K.list
#       (12,481 proteins). Per agent 5's 2026-04-25 finding, v1 highconf
#       PHL classifier ~= retrieval; doesn't beat embedding nearest-neighbour.
#
#   (2) posList + cl70 (recommended for PHL-targeted runs)
#       POSITIVE_LIST=pipeline_positive.list (~59K proteins)
#       CLUSTER_FILE=candidates_clusters.tsv  CLUSTER_THRESHOLD=70
#       MAX_SAMPLES_K=1000  MAX_SAMPLES_O=3000
#       This is what every cl70-suffixed leaderboard run uses.
POSITIVE_LIST="${POSITIVE_LIST:-${DATA_DIR}/training_data/metadata/highconf_pipeline_positive_K.list}"
# Per-head positive lists (v2/v3/v4 highconf families). When BOTH are set,
# train.py uses per-head training: K head trains on POSITIVE_LIST_K only,
# O head trains on POSITIVE_LIST_O only. Mutex with POSITIVE_LIST + cluster
# / cap flags (the per-head lists already encode cluster + cap filtering).
# Examples:
#   v3_uat (production baseline):
#     POSITIVE_LIST_K=$DATA_DIR/training_data/metadata/highconf_v3_multitop/HC_K_UAT_multitop.list
#     POSITIVE_LIST_O=$DATA_DIR/training_data/metadata/highconf_v3_multitop/HC_O_UAT_multitop.list
#   v2 strict:
#     POSITIVE_LIST_K=$DATA_DIR/training_data/metadata/highconf_v2/HC_K_cl95.list
#     POSITIVE_LIST_O=$DATA_DIR/training_data/metadata/highconf_v2/HC_O_cl95_full_coverage.list
POSITIVE_LIST_K="${POSITIVE_LIST_K:-}"
POSITIVE_LIST_O="${POSITIVE_LIST_O:-}"
# Cluster-stratified downsampling. Empty / unset = OFF (the v1 highconf
# recipe). For posList+cl70: set CLUSTER_FILE + CLUSTER_THRESHOLD.
CLUSTER_FILE="${CLUSTER_FILE:-}"
CLUSTER_THRESHOLD="${CLUSTER_THRESHOLD:-70}"
# Per-class downsampling caps. Empty / unset = OFF (the highconf recipe).
# For posList+cl70: set MAX_SAMPLES_K=1000 MAX_SAMPLES_O=3000.
MAX_SAMPLES_K="${MAX_SAMPLES_K:-}"
MAX_SAMPLES_O="${MAX_SAMPLES_O:-}"

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
MIN_SOURCES="${MIN_SOURCES:-1}"

# Pooler architecture: conv_attn (default; PPAM-TDM ConvolutionalAttention)
# or transformer (CLS-token TransformerEncoder).
POOLER_TYPE="${POOLER_TYPE:-conv_attn}"
# C-terminal crop: keep only the last N residues per protein before pooling.
# Empty / unset = no cropping. Try 300 to focus on the binding domain.
C_TERMINAL_CROP="${C_TERMINAL_CROP:-}"
# Linear warmup epochs before cosine decay. Empty / unset = no warmup
# (cosine from step 1). Try 5-10 when using POOLER_TYPE=transformer.
WARMUP_EPOCHS="${WARMUP_EPOCHS:-}"

# Default run name encodes (filter recipe + architecture) so A/B
# comparisons sort together in experiments/light_attention_binary/.
FILTER_LABEL="highconf_pipeline"
if [ -n "${POSITIVE_LIST_K}" ] && [ -n "${POSITIVE_LIST_O}" ]; then
    # Derive label from K-list filename, e.g. HC_K_UAT_multitop -> v3_uat
    PL_BASE="$(basename "${POSITIVE_LIST_K}" .list)"
    case "${PL_BASE}" in
        HC_K_UAT_multitop)             FILTER_LABEL="v3_uat" ;;
        HC_K_cl95_multitop)            FILTER_LABEL="v3_strict" ;;
        HC_K_UAT)                      FILTER_LABEL="v2_uat" ;;
        HC_K_cl95)                     FILTER_LABEL="v2_strict" ;;
        HC_K_v4)                       FILTER_LABEL="v4" ;;
        *)                             FILTER_LABEL="${PL_BASE}" ;;
    esac
elif [ -n "${CLUSTER_FILE}" ]; then
    FILTER_LABEL="posList_cl${CLUSTER_THRESHOLD}"
fi
NAME_SUFFIX="${POOLER_TYPE}"
if [ -n "${C_TERMINAL_CROP}" ]; then
    NAME_SUFFIX="${NAME_SUFFIX}_crop${C_TERMINAL_CROP}"
fi
NAME="${NAME:-lab_${EMBEDDING_TYPE}_${FILTER_LABEL}_${NAME_SUFFIX}}"
DRY_RUN="${DRY_RUN:-0}"

# ============================================================
# Pre-submit sanity checks
# ============================================================
echo "============================================================"
echo "LightAttentionBinary training job (ESM-2 650M full)"
echo "  Cipher dir:     ${CIPHER_DIR}"
echo "  Run name:       ${NAME}"
echo "  Embedding:      ${EMBEDDING_TYPE}"
echo "  Train emb:      ${TRAIN_EMB}"
echo "  Val emb:        ${VAL_EMB}"
if [ -n "${POSITIVE_LIST_K}" ] && [ -n "${POSITIVE_LIST_O}" ]; then
echo "  K positive list:${POSITIVE_LIST_K}"
echo "  O positive list:${POSITIVE_LIST_O}"
else
echo "  Positive list:  ${POSITIVE_LIST}"
fi
echo "  Cluster file:   ${CLUSTER_FILE:-none} (threshold=${CLUSTER_THRESHOLD})"
echo "  Per-K cap:      ${MAX_SAMPLES_K:-none}"
echo "  Per-O cap:      ${MAX_SAMPLES_O:-none}"
echo "  Pooler:         ${POOLER_TYPE}"
echo "  C-term crop:    ${C_TERMINAL_CROP:-none}"
echo "============================================================"

REQUIRED_PATHS=("${TRAIN_EMB}" "${ASSOC_MAP}" "${GLYCAN_BINDERS}" "${VAL_FASTA}")
if [ -n "${POSITIVE_LIST_K}" ] && [ -n "${POSITIVE_LIST_O}" ]; then
    REQUIRED_PATHS+=("${POSITIVE_LIST_K}" "${POSITIVE_LIST_O}")
else
    REQUIRED_PATHS+=("${POSITIVE_LIST}")
fi
if [ -n "${CLUSTER_FILE}" ]; then
    REQUIRED_PATHS+=("${CLUSTER_FILE}")
fi
for f in "${REQUIRED_PATHS[@]}"; do
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
EXTRA_FLAGS="--pooler_type ${POOLER_TYPE}"
if [ -n "${C_TERMINAL_CROP}" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --c_terminal_crop ${C_TERMINAL_CROP}"
fi
if [ -n "${WARMUP_EPOCHS}" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --warmup_epochs ${WARMUP_EPOCHS}"
fi
if [ -n "${CLUSTER_FILE}" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --cluster_file ${CLUSTER_FILE} --cluster_threshold ${CLUSTER_THRESHOLD}"
fi
if [ -n "${MAX_SAMPLES_K}" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --max_samples_per_k ${MAX_SAMPLES_K}"
fi
if [ -n "${MAX_SAMPLES_O}" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --max_samples_per_o ${MAX_SAMPLES_O}"
fi

# Filter flags: per-head if both POSITIVE_LIST_K/_O are set, else single-list.
if [ -n "${POSITIVE_LIST_K}" ] && [ -n "${POSITIVE_LIST_O}" ]; then
    FILTER_FLAGS="--positive_list_k ${POSITIVE_LIST_K} --positive_list_o ${POSITIVE_LIST_O}"
else
    FILTER_FLAGS="--positive_list ${POSITIVE_LIST}"
fi

TRAIN_CMD="python -m cipher.cli.train_runner \
    --model ${MODEL} \
    ${FILTER_FLAGS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --label_strategy ${LABEL_STRATEGY} \
    --min_class_samples ${MIN_CLASS_SAMPLES} \
    --min_sources ${MIN_SOURCES} \
    --embedding_type ${EMBEDDING_TYPE} \
    --embedding_file ${TRAIN_EMB} \
    --association_map ${ASSOC_MAP} \
    --glycan_binders ${GLYCAN_BINDERS} \
    --val_fasta ${VAL_FASTA} \
    --val_datasets_dir ${VAL_DATASETS_DIR} \
    --val_embedding_file ${VAL_EMB} \
    ${EXTRA_FLAGS} \
    --name ${NAME}"

EXP_DIR="${CIPHER_DIR}/experiments/${MODEL}/${NAME}"
# Use the new strict-denominator + any-hit eval (post-2026-04-27).
# Legacy `cipher.evaluation.runner` is deprecated and intentionally not
# called -- it produced numbers that were silently incompatible with the
# headline metric and led to confusing duplicate JSONs in results/.
EVAL_CMD="python scripts/analysis/per_head_strict_eval.py ${EXP_DIR} --val-embedding-file ${VAL_EMB}"

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
export PYTHONUNBUFFERED=1   # so train.py prints stream to the SLURM log live

echo \"======================================\"
echo \"LightAttentionBinary (ESM-2 650M full): ${NAME}\"
echo \"  Train embeddings: ${TRAIN_EMB}\"
echo \"  Val embeddings:   ${VAL_EMB}\"
echo \"  Positive list:    ${POSITIVE_LIST}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

if [ ! -e \"${TRAIN_EMB}\" ]; then
    echo \"ERROR: training embedding path not found: ${TRAIN_EMB}\"
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
