#!/usr/bin/env bash
#
# Light Attention × segmented-embedding sweep under the highconf positive list.
# Direct companion to scripts/run_highconf_experiment.sh (attention_mlp +
# ProtT5 mean + highconf_pipeline → team's current best PHL rh@1 = 0.188).
# This script swaps the model (LA instead of attention_mlp) and sweeps over
# segmented embeddings (seg4/seg8/seg16 for ESM-2 650M and ProtT5-XL).
#
# Same filter (highconf_pipeline_positive_K.list by default) and same training
# hyperparameters as the team baseline, so results are directly comparable to
# the team's single best number.
#
# Usage:
#   # Submit all 6 variants (default: pipeline list):
#   bash scripts/run_light_attention_highconf_sweep.sh
#
#   # Submit one variant:
#   bash scripts/run_light_attention_highconf_sweep.sh prott5_xl_seg8
#
#   # Use the tsp list instead of pipeline:
#   HIGHCONF_VARIANT=tsp bash scripts/run_light_attention_highconf_sweep.sh
#
#   # Dry run:
#   DRY_RUN=1 bash scripts/run_light_attention_highconf_sweep.sh

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

ASSOC_MAP="${DATA_DIR}/training_data/metadata/host_phage_protein_map.tsv"
GLYCAN_BINDERS="${DATA_DIR}/training_data/metadata/glycan_binders_custom.tsv"
VAL_FASTA="${DATA_DIR}/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${DATA_DIR}/validation_data/HOST_RANGE"

# Highconf positive list — pipeline (12,481 proteins) is the default.
HIGHCONF_VARIANT="${HIGHCONF_VARIANT:-pipeline}"
case "$HIGHCONF_VARIANT" in
    pipeline) HIGHCONF_LIST="${DATA_DIR}/training_data/metadata/highconf_pipeline_positive_K.list" ;;
    tsp)      HIGHCONF_LIST="${DATA_DIR}/training_data/metadata/highconf_tsp_K.list" ;;
    *) echo "ERROR: HIGHCONF_VARIANT must be 'pipeline' or 'tsp', got '$HIGHCONF_VARIANT'" >&2
       exit 1 ;;
esac

CONFIG_ABS="${CIPHER_DIR}/models/light_attention/highconf_config.yaml"

MODEL="light_attention"

# SLURM resources
GPUS=1
CPUS=8
TIME="12:00:00"
DEFAULT_MEM="64G"

# ============================================================
# Embedding variants
# Format: "label  n_segments  train_path  val_path  mem"
# Note: ESM-2 650M seg4 uses the LEGACY flat path — the other segmented
# files all live under {embeddings,validation_embeddings}/<tag>_segmentsN/.
# ============================================================
EMBEDDINGS=(
    "esm2_650m_seg4   4  /work/hdd/bfzj/llindsey1/embeddings_segments4/candidates_embeddings_segments4_md5.npz                          /work/hdd/bfzj/llindsey1/validation_embeddings_segments4/validation_embeddings_segments4_md5.npz                      64G"
    "esm2_650m_seg8   8  /work/hdd/bfzj/llindsey1/embeddings/esm2_650m_segments8/candidates_esm2_650m_segments8_md5.npz                 /work/hdd/bfzj/llindsey1/validation_embeddings/esm2_650m_segments8/validation_esm2_650m_segments8_md5.npz              64G"
    "esm2_650m_seg16 16  /work/hdd/bfzj/llindsey1/embeddings/esm2_650m_segments16/candidates_esm2_650m_segments16_md5.npz               /work/hdd/bfzj/llindsey1/validation_embeddings/esm2_650m_segments16/validation_esm2_650m_segments16_md5.npz            64G"
    "prott5_xl_seg4   4  /work/hdd/bfzj/llindsey1/embeddings/prott5_xl_segments4/candidates_prott5_xl_segments4_md5.npz                 /work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments4/validation_prott5_xl_segments4_md5.npz              64G"
    "prott5_xl_seg8   8  /work/hdd/bfzj/llindsey1/embeddings/prott5_xl_segments8/candidates_prott5_xl_segments8_md5.npz                 /work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments8/validation_prott5_xl_segments8_md5.npz              64G"
    "prott5_xl_seg16 16  /work/hdd/bfzj/llindsey1/embeddings/prott5_xl_segments16/candidates_prott5_xl_segments16_md5.npz               /work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments16/validation_prott5_xl_segments16_md5.npz            64G"
)

FILTER="${1:-}"
DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "LA × SEGMENTED × HIGHCONF SWEEP"
echo "  Model:           ${MODEL}"
echo "  Cipher dir:      ${CIPHER_DIR}"
echo "  Data dir:        ${DATA_DIR}"
echo "  Positive list:   ${HIGHCONF_LIST}"
echo "  Base config:     ${CONFIG_ABS}"
echo "============================================================"
echo ""

if [ ! -f "$HIGHCONF_LIST" ]; then
    echo "ERROR: HIGHCONF_LIST not found: $HIGHCONF_LIST" >&2
    exit 1
fi
if [ ! -f "$CONFIG_ABS" ]; then
    echo "ERROR: config not found: $CONFIG_ABS" >&2
    exit 1
fi

N_SUBMITTED=0
N_SKIPPED=0

for entry in "${EMBEDDINGS[@]}"; do
    read -r LABEL N_SEG TRAIN_EMB VAL_EMB MEM <<< "$entry"

    if [ -n "$FILTER" ] && [ "$LABEL" != "$FILTER" ]; then
        continue
    fi

    NAME="la_highconf_${HIGHCONF_VARIANT}_${LABEL}"

    # Pre-submit checks
    if [ ! -f "$TRAIN_EMB" ]; then
        echo "  SKIP ${LABEL} — training embedding not found: ${TRAIN_EMB}"
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
        --positive_list ${HIGHCONF_LIST} \
        --embedding_type ${LABEL} \
        --embedding_file ${TRAIN_EMB} \
        --n_segments ${N_SEG} \
        --association_map ${ASSOC_MAP} \
        --glycan_binders ${GLYCAN_BINDERS} \
        --val_fasta ${VAL_FASTA} \
        --val_datasets_dir ${VAL_DATASETS_DIR} \
        --val_embedding_file ${VAL_EMB} \
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
echo \"LA highconf sweep: ${NAME}\"
echo \"  embedding:     ${LABEL}  (n_segments=${N_SEG})\"
echo \"  train emb:     ${TRAIN_EMB}\"
echo \"  val emb:       ${VAL_EMB}\"
echo \"  positive_list: ${HIGHCONF_LIST}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

if [ ! -f \"${TRAIN_EMB}\" ]; then
    echo \"ERROR: Training embedding file not found: ${TRAIN_EMB}\"
    exit 1
fi

VAL_READY=true
if [ ! -f \"${VAL_EMB}\" ]; then
    echo \"WARNING: Validation embeddings not found: ${VAL_EMB}\"
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
fi

echo \"\"
echo \"======================================\"
echo \"Done: ${NAME} at \$(date)\"
echo \"======================================\"
"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] ${NAME}"
        echo "    embedding: ${LABEL}  n_segments=${N_SEG}"
        echo "    train:     ${TRAIN_EMB}"
        echo "    val:       ${VAL_EMB} (${VAL_STATUS})"
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
    echo "Submitted ${N_SUBMITTED} jobs (${N_SKIPPED} skipped)."
    echo "Monitor: squeue -u \$USER"
fi
echo "============================================================"
