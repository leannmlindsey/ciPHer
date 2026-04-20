#!/usr/bin/env bash
#
# Kmer + pLM concatenation sweep. Trains + evaluates attention_mlp on the
# concatenation of two embeddings per MD5. Each pair runs as one SLURM job.
#
# Pairs live in the EMBEDDING_PAIRS array below — edit them to match the
# winners of the single-embedding sweeps once those results are in.
#
# Usage:
#   # Submit all pairs (tool filter, random sampling — default):
#   bash scripts/run_concat_sweep.sh
#
#   # Single pair by label:
#   bash scripts/run_concat_sweep.sh esm2_3b_mean+kmer_aa20_k4
#
#   # Dry run:
#   DRY_RUN=1 bash scripts/run_concat_sweep.sh
#
#   # Positive-list filter or cluster sampling — same env vars as
#   # run_embedding_sweep.sh; names get 'posList_'/'cl70' decorators.
#   FILTER_MODE=positive_list bash scripts/run_concat_sweep.sh
#   USE_CLUSTERS=1 bash scripts/run_concat_sweep.sh
#   FILTER_MODE=positive_list USE_CLUSTERS=1 bash scripts/run_concat_sweep.sh

set -euo pipefail

# ============================================================
# Delta-AI configuration
# ============================================================
ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="/projects/bfzj/llindsey1/PHI_TSP/ciPHer"

# ============================================================
# Data paths (same as run_embedding_sweep.sh)
# ============================================================
ASSOC_MAP="${CIPHER_DIR}/data/training_data/metadata/host_phage_protein_map.tsv"
GLYCAN_BINDERS="${CIPHER_DIR}/data/training_data/metadata/glycan_binders_custom.tsv"
POSITIVE_LIST="${CIPHER_DIR}/data/training_data/metadata/pipeline_positive.list"
CLUSTER_FILE="${CIPHER_DIR}/data/training_data/metadata/candidates_clusters.tsv"
CLUSTER_THRESHOLD="${CLUSTER_THRESHOLD:-70}"
VAL_FASTA="${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${CIPHER_DIR}/data/validation_data/HOST_RANGE"

# SLURM resources (MEM is per-pair, set in the EMBEDDING_PAIRS array)
GPUS=1
CPUS=8
TIME="12:00:00"

# ============================================================
# Fixed model configuration (held constant; matches run_embedding_sweep.sh)
# ============================================================
MODEL="attention_mlp"
TOOLS="DepoScope,PhageRBPdetect"
LR="1e-05"
BATCH_SIZE=512
EPOCHS=1000
PATIENCE=30
LABEL_STRATEGY="multi_label_threshold"
MIN_CLASS_SAMPLES=25
MAX_SAMPLES_K=1000
MAX_SAMPLES_O=3000
MIN_SOURCES=1

FILTER_MODE="${FILTER_MODE:-tools}"
if [[ "$FILTER_MODE" != "tools" && "$FILTER_MODE" != "positive_list" ]]; then
    echo "ERROR: FILTER_MODE must be 'tools' or 'positive_list', got '$FILTER_MODE'" >&2
    exit 1
fi
if [[ "$FILTER_MODE" == "positive_list" && ! -f "$POSITIVE_LIST" ]]; then
    echo "ERROR: POSITIVE_LIST not found: $POSITIVE_LIST" >&2
    exit 1
fi
USE_CLUSTERS="${USE_CLUSTERS:-0}"
if [[ "$USE_CLUSTERS" == "1" && ! -f "$CLUSTER_FILE" ]]; then
    echo "ERROR: CLUSTER_FILE not found: $CLUSTER_FILE" >&2
    exit 1
fi

# ============================================================
# Embedding pair configurations
# Format:
#   "plm_label  kmer_label  plm_train  plm_val  kmer_train  kmer_val  mem"
#
# The run label is "<plm>+<kmer>". MEM should cover the larger of the two
# embeddings' memory footprint plus training overhead (dominated by kmer
# for the big ones — mirror the MEM value from run_embedding_sweep.sh).
#
# Starting placeholders: top 2 pLMs (esm2_3b_mean, prott5_mean) × top 2
# k-mers (kmer_aa20_k4, kmer_murphy8_k5) from the initial sweep. Edit
# once sweeps 2/3/4 results are in.
# ============================================================
TRAIN_EMB_ROOT="/projects/bfzj/llindsey1/RBP_Structural_Similarity/output"
KMER_ROOT="/work/hdd/bfzj/llindsey1/kmer_features"
VAL_EMB_ROOT="/projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/validation_data/combined/validation_inputs"

EMBEDDING_PAIRS=(
    # Pairs picked from sweeps 1–4 analysis (2026-04-20):
    #   pLM side:  esm2_3b_mean, esm2_650m_seg4, prott5_mean (best PHL performers)
    #   kmer side: kmer_li10_k5 (highest PHL rh@1), kmer_murphy8_k5 (top combined kmer)
    "esm2_3b_mean      kmer_li10_k5     ${TRAIN_EMB_ROOT}/embeddings_esm2_3b/candidates_embeddings_md5.npz                                    /work/hdd/bfzj/llindsey1/validation_embeddings_esm2_3b/validation_embeddings_md5.npz                  ${KMER_ROOT}/candidates_li10_k5.npz     ${KMER_ROOT}/validation_li10_k5.npz     192G"
    "esm2_3b_mean      kmer_murphy8_k5  ${TRAIN_EMB_ROOT}/embeddings_esm2_3b/candidates_embeddings_md5.npz                                    /work/hdd/bfzj/llindsey1/validation_embeddings_esm2_3b/validation_embeddings_md5.npz                  ${KMER_ROOT}/candidates_murphy8_k5.npz  ${KMER_ROOT}/validation_murphy8_k5.npz  128G"
    "esm2_650m_seg4    kmer_li10_k5     /work/hdd/bfzj/llindsey1/embeddings_segments4/candidates_embeddings_segments4_md5.npz                 /work/hdd/bfzj/llindsey1/validation_embeddings_segments4/validation_embeddings_segments4_md5.npz      ${KMER_ROOT}/candidates_li10_k5.npz     ${KMER_ROOT}/validation_li10_k5.npz     192G"
    "prott5_mean       kmer_li10_k5     ${TRAIN_EMB_ROOT}/embeddings_prott5/candidates_embeddings_md5.npz                                     /work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz                   ${KMER_ROOT}/candidates_li10_k5.npz     ${KMER_ROOT}/validation_li10_k5.npz     192G"
)

# ============================================================
# Main loop
# ============================================================
FILTER="${1:-}"   # optional: filter to one combined label
DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "CONCAT SWEEP (pLM + k-mer)"
echo "  Model:       ${MODEL}"
echo "  Filter mode: ${FILTER_MODE}"
echo "  Clusters:    ${USE_CLUSTERS} (threshold ${CLUSTER_THRESHOLD})"
echo "  Cipher:      ${CIPHER_DIR}"
echo "============================================================"
echo ""

N_SUBMITTED=0
N_SKIPPED=0

for entry in "${EMBEDDING_PAIRS[@]}"; do
    read -r PLM_LABEL KMER_LABEL PLM_TRAIN PLM_VAL KMER_TRAIN KMER_VAL MEM <<< "$entry"
    COMBINED_LABEL="${PLM_LABEL}+${KMER_LABEL}"

    if [ -n "$FILTER" ] && [ "$COMBINED_LABEL" != "$FILTER" ]; then
        continue
    fi

    # Compose filter + cluster decorations into the run name so variants
    # coexist in experiments/attention_mlp/ alongside single-embedding runs.
    if [[ "$FILTER_MODE" == "positive_list" ]]; then
        NAME="concat_posList_${COMBINED_LABEL}"
        FILTER_ARG="--positive_list ${POSITIVE_LIST}"
    else
        NAME="concat_${COMBINED_LABEL}"
        FILTER_ARG="--tools ${TOOLS}"
    fi
    if [[ "$USE_CLUSTERS" == "1" ]]; then
        NAME="${NAME}_cl${CLUSTER_THRESHOLD}"
        CLUSTER_ARG="--cluster_file ${CLUSTER_FILE} --cluster_threshold ${CLUSTER_THRESHOLD}"
    else
        CLUSTER_ARG=""
    fi

    # Pre-submit checks
    MISSING=""
    for f in "$PLM_TRAIN" "$KMER_TRAIN"; do
        [ -f "$f" ] || MISSING="$MISSING $f"
    done
    if [ -n "$MISSING" ]; then
        echo "  SKIP ${COMBINED_LABEL} — training embedding(s) not found:${MISSING}"
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi
    VAL_STATUS="ready"
    for f in "$PLM_VAL" "$KMER_VAL"; do
        if [ ! -f "$f" ]; then
            VAL_STATUS="missing (will train only)"
        fi
    done

    TRAIN_CMD="python -m cipher.cli.train_runner \
        --model ${MODEL} \
        ${FILTER_ARG} \
        ${CLUSTER_ARG} \
        --lr ${LR} \
        --batch_size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --patience ${PATIENCE} \
        --label_strategy ${LABEL_STRATEGY} \
        --min_class_samples ${MIN_CLASS_SAMPLES} \
        --max_samples_per_k ${MAX_SAMPLES_K} \
        --max_samples_per_o ${MAX_SAMPLES_O} \
        --min_sources ${MIN_SOURCES} \
        --embedding_type ${PLM_LABEL} \
        --embedding_file ${PLM_TRAIN} \
        --embedding_type_2 ${KMER_LABEL} \
        --embedding_file_2 ${KMER_TRAIN} \
        --association_map ${ASSOC_MAP} \
        --glycan_binders ${GLYCAN_BINDERS} \
        --val_fasta ${VAL_FASTA} \
        --val_datasets_dir ${VAL_DATASETS_DIR} \
        --val_embedding_file ${PLM_VAL} \
        --val_embedding_file_2 ${KMER_VAL} \
        --name ${NAME}"

    EXP_DIR="${CIPHER_DIR}/experiments/${MODEL}/${NAME}"
    EVAL_CMD="python -m cipher.evaluation.runner ${EXP_DIR} \
        --val-embedding-file ${PLM_VAL} \
        --val-embedding-file-2 ${KMER_VAL}"

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
echo \"Concat sweep: ${COMBINED_LABEL}\"
echo \"  pLM train:   ${PLM_TRAIN}\"
echo \"  kmer train:  ${KMER_TRAIN}\"
echo \"  pLM val:     ${PLM_VAL}\"
echo \"  kmer val:    ${KMER_VAL}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

VAL_READY=true
for f in \"${PLM_VAL}\" \"${KMER_VAL}\"; do
    if [ ! -f \"\$f\" ]; then
        echo \"WARNING: Validation embedding missing: \$f\"
        VAL_READY=false
    fi
done

echo \"\"
echo \"=== TRAINING ===\"
${TRAIN_CMD}

if [ \"\${VAL_READY}\" = true ]; then
    echo \"\"
    echo \"=== EVALUATING ===\"
    ${EVAL_CMD}
else
    echo \"\"
    echo \"SKIPPING evaluation: validation embeddings not available.\"
    echo \"Rerun evaluation manually once they exist:\"
    echo \"  ${EVAL_CMD}\"
fi

echo \"\"
echo \"======================================\"
echo \"Done: ${COMBINED_LABEL}\"
echo \"  Finished: \$(date)\"
echo \"======================================"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] ${NAME}  (mem=${MEM}, time=${TIME})"
        echo "    pLM:  ${PLM_TRAIN}"
        echo "    kmer: ${KMER_TRAIN}"
        echo "    val:  (${VAL_STATUS})"
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
