#!/usr/bin/env bash
#
# Embedding sweep: train + evaluate the same model config with different embeddings.
# Designed for Delta-AI (SLURM). Each embedding gets its own job.
#
# Usage:
#   # Submit all jobs (default: tool-filtered training set):
#   bash scripts/run_embedding_sweep.sh
#
#   # Submit a single embedding:
#   bash scripts/run_embedding_sweep.sh esm2_650m_mean
#
#   # Dry run (print commands without submitting):
#   DRY_RUN=1 bash scripts/run_embedding_sweep.sh
#
#   # Use the pipeline-positive list as the training filter (instead of
#   # --tools). Run names get a 'posList' prefix so results coexist with
#   # the tool-filter sweep in experiments/.
#   FILTER_MODE=positive_list bash scripts/run_embedding_sweep.sh
#
#   # Cluster-stratified downsampling (round-robin across clusters instead
#   # of random sampling). Run names get a 'cl70' suffix.
#   USE_CLUSTERS=1 bash scripts/run_embedding_sweep.sh
#
#   # Combine both toggles:
#   FILTER_MODE=positive_list USE_CLUSTERS=1 bash scripts/run_embedding_sweep.sh

set -euo pipefail

# ============================================================
# Delta-AI configuration
# ============================================================
ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="/projects/bfzj/llindsey1/PHI_TSP/ciPHer"

# ============================================================
# Data paths on Delta (no symlinks needed)
# ============================================================
ASSOC_MAP="${CIPHER_DIR}/data/training_data/metadata/host_phage_protein_map.tsv"
GLYCAN_BINDERS="${CIPHER_DIR}/data/training_data/metadata/glycan_binders_custom.tsv"
POSITIVE_LIST="${CIPHER_DIR}/data/training_data/metadata/pipeline_positive.list"
CLUSTER_FILE="${CIPHER_DIR}/data/training_data/metadata/candidates_clusters.tsv"
CLUSTER_THRESHOLD="${CLUSTER_THRESHOLD:-70}"
VAL_FASTA="${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${CIPHER_DIR}/data/validation_data/HOST_RANGE"

# SLURM resources (per-embedding MEM is set in the EMBEDDINGS array below)
GPUS=1
CPUS=8
TIME="12:00:00"

# ============================================================
# Fixed model configuration (held constant across sweep)
# ============================================================
MODEL="attention_mlp"
TOOLS="DepoScope,PhageRBPdetect"
LR="1e-05"

# Training-set filter mode. Controls which flag is passed to cipher-train.
#   tools         -> --tools ${TOOLS}     (default, matches original sweep)
#   positive_list -> --positive_list ${POSITIVE_LIST}
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
BATCH_SIZE=512
EPOCHS=1000
PATIENCE=30
LABEL_STRATEGY="multi_label_threshold"
MIN_CLASS_SAMPLES=25
MAX_SAMPLES_K=1000
MAX_SAMPLES_O=3000
MIN_SOURCES=1

# ============================================================
# Embedding configurations
# Format: "label  train_embedding_file  val_embedding_file  mem"
# Set mem per-entry so small jobs don't block on full-node reservations.
# ============================================================
TRAIN_EMB_ROOT="/projects/bfzj/llindsey1/RBP_Structural_Similarity/output"
KMER_ROOT="/work/hdd/bfzj/llindsey1/kmer_features"
VAL_EMB_ROOT="/projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/validation_data/combined/validation_inputs"

EMBEDDINGS=(
    # PLM embeddings
    "esm2_650m_mean     ${TRAIN_EMB_ROOT}/embeddings_binned/candidates_embeddings_md5.npz                                   ${VAL_EMB_ROOT}/validation_embeddings_md5.npz                                                    64G"
    "esm2_650m_seg4     /work/hdd/bfzj/llindsey1/embeddings_segments4/candidates_embeddings_segments4_md5.npz               /work/hdd/bfzj/llindsey1/validation_embeddings_segments4/validation_embeddings_segments4_md5.npz  64G"
    "esm2_3b_mean       ${TRAIN_EMB_ROOT}/embeddings_esm2_3b/candidates_embeddings_md5.npz                                  /work/hdd/bfzj/llindsey1/validation_embeddings_esm2_3b/validation_embeddings_md5.npz            128G"
    "esm2_150m_mean     ${TRAIN_EMB_ROOT}/embeddings_esm2_150m/candidates_embeddings_md5.npz                                /work/hdd/bfzj/llindsey1/validation_embeddings_esm2_150m/validation_embeddings_md5.npz           64G"
    "prott5_mean        ${TRAIN_EMB_ROOT}/embeddings_prott5/candidates_embeddings_md5.npz                                   /work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz              64G"

    # K-mer features (best separation results only)
    "kmer_murphy8_k5    ${KMER_ROOT}/candidates_murphy8_k5.npz      ${KMER_ROOT}/validation_murphy8_k5.npz   64G"
    "kmer_murphy10_k5   ${KMER_ROOT}/candidates_murphy10_k5.npz     ${KMER_ROOT}/validation_murphy10_k5.npz  192G"
    "kmer_li10_k5       ${KMER_ROOT}/candidates_li10_k5.npz         ${KMER_ROOT}/validation_li10_k5.npz      64G"
    "kmer_aa20_k3       ${KMER_ROOT}/candidates_aa20_k3.npz         ${KMER_ROOT}/validation_aa20_k3.npz      64G"
    "kmer_aa20_k4       ${KMER_ROOT}/candidates_aa20_k4.npz         ${KMER_ROOT}/validation_aa20_k4.npz      256G"
)

# ============================================================
# Main loop
# ============================================================
FILTER="${1:-}"  # optional: only run this label
DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "EMBEDDING SWEEP"
echo "  Model:    ${MODEL}"
echo "  Tools:    ${TOOLS}"
echo "  Strategy: ${LABEL_STRATEGY}"
echo "  Cipher:   ${CIPHER_DIR}"
echo "============================================================"
echo ""

N_SUBMITTED=0
N_SKIPPED=0

for entry in "${EMBEDDINGS[@]}"; do
    read -r LABEL TRAIN_EMB VAL_EMB MEM <<< "$entry"

    # Filter to single embedding if requested
    if [ -n "$FILTER" ] && [ "$LABEL" != "$FILTER" ]; then
        continue
    fi

    if [[ "$FILTER_MODE" == "positive_list" ]]; then
        NAME="sweep_posList_${LABEL}"
        FILTER_ARG="--positive_list ${POSITIVE_LIST}"
    else
        NAME="sweep_${LABEL}"
        FILTER_ARG="--tools ${TOOLS}"
    fi
    if [[ "$USE_CLUSTERS" == "1" ]]; then
        NAME="${NAME}_cl${CLUSTER_THRESHOLD}"
        CLUSTER_ARG="--cluster_file ${CLUSTER_FILE} --cluster_threshold ${CLUSTER_THRESHOLD}"
    else
        CLUSTER_ARG=""
    fi

    # Pre-submit checks
    if [ ! -f "$TRAIN_EMB" ]; then
        echo "  SKIP ${LABEL} — training embeddings not found: ${TRAIN_EMB}"
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi
    VAL_STATUS="ready"
    if [ ! -f "$VAL_EMB" ]; then
        VAL_STATUS="missing (will train only)"
    fi

    # Build train command
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
        --embedding_type ${LABEL} \
        --embedding_file ${TRAIN_EMB} \
        --association_map ${ASSOC_MAP} \
        --glycan_binders ${GLYCAN_BINDERS} \
        --val_fasta ${VAL_FASTA} \
        --val_datasets_dir ${VAL_DATASETS_DIR} \
        --val_embedding_file ${VAL_EMB} \
        --name ${NAME}"

    # Build evaluate command (reads paths from saved config.yaml)
    EXP_DIR="${CIPHER_DIR}/experiments/${MODEL}/${NAME}"
    EVAL_CMD="python -m cipher.evaluation.runner ${EXP_DIR} --val-embedding-file ${VAL_EMB}"

    # Full job script
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
echo \"Embedding sweep: ${LABEL}\"
echo \"  Train embeddings: ${TRAIN_EMB}\"
echo \"  Val embeddings:   ${VAL_EMB}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

# Check training embeddings exist
if [ ! -f \"${TRAIN_EMB}\" ]; then
    echo \"ERROR: Training embedding file not found: ${TRAIN_EMB}\"
    exit 1
fi

# Check validation files exist before starting
VAL_READY=true
if [ ! -f \"${VAL_EMB}\" ]; then
    echo \"WARNING: Validation embeddings not found: ${VAL_EMB}\"
    echo \"  Will train but skip evaluation.\"
    VAL_READY=false
fi

# Train
echo \"\"
echo \"=== TRAINING ===\"
${TRAIN_CMD}

# Evaluate
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
echo \"Done: ${LABEL}\"
echo \"  Finished: \$(date)\"
echo \"======================================"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] ${LABEL}  (mem=${MEM}, time=${TIME})"
        echo "    Train: ${TRAIN_EMB}"
        echo "    Val:   ${VAL_EMB} (${VAL_STATUS})"
        echo ""
    else
        # Create logs directory
        mkdir -p "${CIPHER_DIR}/logs"

        # Submit
        JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
        echo "  Submitted ${JOB_ID} — ${LABEL} (val: ${VAL_STATUS})"
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
    echo ""
    echo "After completion, compare results:"
    echo "  python scripts/analysis/compare_experiments.py"
fi
echo "============================================================"
