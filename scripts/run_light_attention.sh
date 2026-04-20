#!/usr/bin/env bash
#
# Light Attention experiments on Delta-AI (SLURM).
#
# Trains + evaluates the Light Attention model under one of two profiles:
#   la_seg4_reproduce_old — base_config.yaml (single_label, no tool filter,
#                            min_sources=3) — reproduces the old klebsiella
#                            repo's seg4 LA run under the ciPHer framework.
#   la_seg4_match_sweep   — sweep_config.yaml (multi_label_threshold,
#                            DepoScope+PhageRBPdetect, min_class_samples=25,
#                            downsampling) — comparable to the embedding
#                            sweep rows produced by run_embedding_sweep.sh.
#
# Filter-mode and clustering are controlled via env vars, matching the pattern
# in scripts/run_embedding_sweep.sh so you can flip behavior across profiles
# with one submission:
#   FILTER_MODE=""             use whatever the profile's yaml specifies (default)
#   FILTER_MODE=positive_list  override: --positive_list + clear --tools
#   USE_CLUSTERS=0|1           add --cluster_file + --cluster_threshold
#   CLUSTER_THRESHOLD=70       which cluster column to stratify by
#
# Usage:
#   bash scripts/run_light_attention.sh                           # both profiles, yaml defaults
#   bash scripts/run_light_attention.sh la_seg4_match_sweep       # one profile
#   FILTER_MODE=positive_list USE_CLUSTERS=1 bash scripts/run_light_attention.sh
#       # both profiles, overridden to positive-list + cl70 (appends _posList_cl70 to names)
#   DRY_RUN=1 bash scripts/run_light_attention.sh                 # preview

set -euo pipefail

# ============================================================
# Delta-AI configuration
# ============================================================
ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"

# Auto-detect CIPHER_DIR from the script's own location (repo root = dir
# containing this scripts/ folder). Override with CIPHER_DIR=... to point
# elsewhere. CIPHER_DIR is where code + experiments + logs live.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="${CIPHER_DIR:-$(dirname "$SCRIPT_DIR")}"

# DATA_DIR is the canonical data home on Delta — shared across all worktrees
# / clones so we don't duplicate 10s of GB of TSVs + validation data.
# Override with DATA_DIR=... if the data lives elsewhere.
DATA_DIR="${DATA_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer/data}"

# ============================================================
# Data paths (resolved from DATA_DIR, not CIPHER_DIR)
# ============================================================
ASSOC_MAP="${DATA_DIR}/training_data/metadata/host_phage_protein_map.tsv"
GLYCAN_BINDERS="${DATA_DIR}/training_data/metadata/glycan_binders_custom.tsv"
VAL_FASTA="${DATA_DIR}/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${DATA_DIR}/validation_data/HOST_RANGE"

# Advisor's curated TSP/RBP list (used when FILTER_MODE=positive_list).
POSITIVE_LIST="${DATA_DIR}/training_data/metadata/pipeline_positive.list"
# Sequence-cluster TSV with multi-threshold columns (used when USE_CLUSTERS=1);
# CLUSTER_THRESHOLD selects which column to stratify by.
CLUSTER_FILE="${DATA_DIR}/training_data/metadata/candidates_clusters.tsv"

# ============================================================
# Filter-mode env vars (mirror scripts/run_embedding_sweep.sh)
# ============================================================
FILTER_MODE="${FILTER_MODE:-}"   # "" (use yaml as-is) or "positive_list"
USE_CLUSTERS="${USE_CLUSTERS:-0}"
CLUSTER_THRESHOLD="${CLUSTER_THRESHOLD:-70}"

case "$FILTER_MODE" in
    ""|positive_list) ;;
    *) echo "ERROR: FILTER_MODE must be '' or 'positive_list', got '$FILTER_MODE'" >&2
       exit 1 ;;
esac
if [[ "$FILTER_MODE" == "positive_list" && ! -f "$POSITIVE_LIST" ]]; then
    echo "ERROR: POSITIVE_LIST not found: $POSITIVE_LIST" >&2
    exit 1
fi
if [[ "$USE_CLUSTERS" == "1" && ! -f "$CLUSTER_FILE" ]]; then
    echo "ERROR: CLUSTER_FILE not found: $CLUSTER_FILE" >&2
    exit 1
fi

# Segments-4 embeddings (from run_embedding_sweep.sh)
SEG4_TRAIN_EMB="/work/hdd/bfzj/llindsey1/embeddings_segments4/candidates_embeddings_segments4_md5.npz"
SEG4_VAL_EMB="/work/hdd/bfzj/llindsey1/validation_embeddings_segments4/validation_embeddings_segments4_md5.npz"

# SLURM resources
GPUS=1
CPUS=8
TIME="24:00:00"
MEM=0   # 0 = all available

# ============================================================
# Profiles
# Format: "name  config_file_rel_to_cipher_dir  train_emb  val_emb"
# The config file replaces base_config.yaml via --config; CLI flags in
# TRAIN_CMD below override embedding paths and names on top.
# ============================================================
MODEL="light_attention"

PROFILES=(
    "la_seg4_reproduce_old   models/light_attention/base_config.yaml   ${SEG4_TRAIN_EMB}  ${SEG4_VAL_EMB}"
    "la_seg4_match_sweep     models/light_attention/sweep_config.yaml  ${SEG4_TRAIN_EMB}  ${SEG4_VAL_EMB}"
)

# ============================================================
# Main loop
# ============================================================
FILTER="${1:-}"
DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "LIGHT ATTENTION EXPERIMENTS"
echo "  Model:       ${MODEL}"
echo "  Cipher:      ${CIPHER_DIR}"
echo "  Data:        ${DATA_DIR}"
echo "  FilterMode:  ${FILTER_MODE:-<yaml default>}"
echo "  UseClusters: ${USE_CLUSTERS} (threshold=${CLUSTER_THRESHOLD})"
echo "============================================================"
echo ""

N_SUBMITTED=0
N_SKIPPED=0

for entry in "${PROFILES[@]}"; do
    read -r BASE_NAME CONFIG_REL TRAIN_EMB VAL_EMB <<< "$entry"

    if [ -n "$FILTER" ] && [ "$BASE_NAME" != "$FILTER" ]; then
        continue
    fi

    CONFIG_ABS="${CIPHER_DIR}/${CONFIG_REL}"
    EMB_TYPE="esm2_650m_seg4"

    # Apply filter-mode + clustering env vars. Auto-append suffixes to NAME so
    # each variant gets its own experiment dir.
    NAME="$BASE_NAME"
    EXTRA_ARGS=""
    if [[ "$FILTER_MODE" == "positive_list" ]]; then
        NAME="${NAME}_posList"
        # apply_overrides() in train_runner.py automatically clears any
        # tools/exclude_tools from the loaded yaml when --positive_list is
        # set, so the mutex in TrainingConfig.__post_init__ is satisfied.
        EXTRA_ARGS="${EXTRA_ARGS} --positive_list ${POSITIVE_LIST}"
    fi
    if [[ "$USE_CLUSTERS" == "1" ]]; then
        NAME="${NAME}_cl${CLUSTER_THRESHOLD}"
        EXTRA_ARGS="${EXTRA_ARGS} --cluster_file ${CLUSTER_FILE} --cluster_threshold ${CLUSTER_THRESHOLD}"
    fi

    # Pre-submit checks
    if [ ! -f "$TRAIN_EMB" ]; then
        echo "  SKIP ${NAME} — training embeddings not found: ${TRAIN_EMB}"
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi
    if [ ! -f "$CONFIG_ABS" ]; then
        echo "  SKIP ${NAME} — config not found: ${CONFIG_ABS}"
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi
    VAL_STATUS="ready"
    if [ ! -f "$VAL_EMB" ]; then
        VAL_STATUS="missing (will train only)"
    fi

    TRAIN_CMD="python -m cipher.cli.train_runner \
        --model ${MODEL} \
        --config ${CONFIG_ABS} \
        --embedding_type ${EMB_TYPE} \
        --embedding_file ${TRAIN_EMB} \
        --association_map ${ASSOC_MAP} \
        --glycan_binders ${GLYCAN_BINDERS} \
        --val_fasta ${VAL_FASTA} \
        --val_datasets_dir ${VAL_DATASETS_DIR} \
        --val_embedding_file ${VAL_EMB}${EXTRA_ARGS} \
        --name ${NAME}"

    EXP_DIR="${CIPHER_DIR}/experiments/${MODEL}/${NAME}"
    EVAL_CMD="python -m cipher.evaluation.runner ${EXP_DIR} --val-embedding-file ${VAL_EMB}"

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
echo \"Light Attention: ${NAME}\"
echo \"  Config:           ${CONFIG_ABS}\"
echo \"  Train embeddings: ${TRAIN_EMB}\"
echo \"  Val embeddings:   ${VAL_EMB}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

if [ ! -f \"${TRAIN_EMB}\" ]; then
    echo \"ERROR: Training embedding file not found: ${TRAIN_EMB}\"
    exit 1
fi

VAL_READY=true
if [ ! -f \"${VAL_EMB}\" ]; then
    echo \"WARNING: Validation embeddings not found: ${VAL_EMB}\"
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
    echo \"Generate them, then run:\"
    echo \"  ${EVAL_CMD}\"
fi

echo \"\"
echo \"======================================\"
echo \"Done: ${NAME}\"
echo \"  Finished: \$(date)\"
echo \"======================================\"
"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] ${NAME}"
        echo "    Config: ${CONFIG_ABS}"
        echo "    Train:  ${TRAIN_EMB}"
        echo "    Val:    ${VAL_EMB} (${VAL_STATUS})"
        if [ -n "$EXTRA_ARGS" ]; then
            echo "    Extras:${EXTRA_ARGS}"
        fi
        echo ""
    else
        mkdir -p "${CIPHER_DIR}/logs"
        JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
        echo "  Submitted ${JOB_ID} — ${NAME} (val: ${VAL_STATUS})"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    fi
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN complete. Set DRY_RUN=0 to submit."
else
    echo "Submitted ${N_SUBMITTED} jobs."
    echo "Monitor: squeue -u \$USER"
fi
echo "============================================================"
