#!/usr/bin/env bash
#
# Embedding sweep: train + evaluate the same model config with different embeddings.
# Designed for Delta-AI (SLURM). Each embedding gets its own job.
#
# Usage:
#   # Submit all jobs:
#   bash scripts/run_embedding_sweep.sh
#
#   # Submit a single embedding:
#   bash scripts/run_embedding_sweep.sh esm2_650m_mean
#
#   # Dry run (print commands without submitting):
#   DRY_RUN=1 bash scripts/run_embedding_sweep.sh

set -euo pipefail

# ============================================================
# Delta-AI configuration
# ============================================================
ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="/u/llindsey1/llindsey/PHI_TSP/cipher"

# SLURM resources
GPUS=1
CPUS=8
TIME="24:00:00"
MEM=0   # 0 = all available

# ============================================================
# Fixed model configuration (held constant across sweep)
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

# ============================================================
# Embedding configurations
# Format: "label  train_embedding_file  val_embedding_file"
# ============================================================
TRAIN_EMB_ROOT="/projects/bfzj/llindsey1/RBP_Structural_Similarity/output"
KMER_ROOT="/work/hdd/bfzj/llindsey1/kmer_features"
VAL_EMB_ROOT="/projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/validation_data/combined/validation_inputs"

EMBEDDINGS=(
    # PLM embeddings
    "esm2_650m_mean     ${TRAIN_EMB_ROOT}/embeddings_binned/candidates_embeddings_md5.npz                   ${VAL_EMB_ROOT}/validation_embeddings_md5.npz"
    "esm2_650m_seg4     /work/hdd/bfzj/llindsey1/embeddings_segments4/candidates_embeddings_segments4_md5.npz  /work/hdd/bfzj/llindsey1/validation_embeddings_segments4/validation_embeddings_segments4_md5.npz"
    "esm2_3b_mean       ${TRAIN_EMB_ROOT}/embeddings_esm2_3b/candidates_embeddings_md5.npz                  MISSING"
    "esm2_150m_mean     ${TRAIN_EMB_ROOT}/embeddings_esm2_150m/candidates_embeddings_md5.npz                MISSING"
    "prott5_mean        ${TRAIN_EMB_ROOT}/embeddings_prott5/candidates_embeddings_md5.npz                   MISSING"

    # K-mer features (various alphabets and k values)
    "kmer_murphy8_k5    ${KMER_ROOT}/candidates_murphy8_k5.npz      MISSING"
    "kmer_murphy8_k6    ${KMER_ROOT}/candidates_murphy8_k6.npz      MISSING"
    "kmer_murphy8_k456  ${KMER_ROOT}/candidates_murphy8_k456.npz    MISSING"
    "kmer_murphy10_k5   ${KMER_ROOT}/candidates_murphy10_k5.npz     MISSING"
    "kmer_murphy10_k45  ${KMER_ROOT}/candidates_murphy10_k45.npz    MISSING"
    "kmer_li10_k5       ${KMER_ROOT}/candidates_li10_k5.npz         MISSING"
    "kmer_li10_k45      ${KMER_ROOT}/candidates_li10_k45.npz        MISSING"
    "kmer_aa20_k3       ${KMER_ROOT}/candidates_aa20_k3.npz         MISSING"
    "kmer_aa20_k4       ${KMER_ROOT}/candidates_aa20_k4.npz         MISSING"
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
    read -r LABEL TRAIN_EMB VAL_EMB <<< "$entry"

    # Filter to single embedding if requested
    if [ -n "$FILTER" ] && [ "$LABEL" != "$FILTER" ]; then
        continue
    fi

    NAME="sweep_${LABEL}"

    # Build train command
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
        --embedding_type ${LABEL} \
        --embedding_file ${TRAIN_EMB} \
        --name ${NAME}"

    # Build evaluate command
    EXP_DIR="${CIPHER_DIR}/experiments/${MODEL}/${NAME}"
    EVAL_CMD="python -m cipher.evaluation.runner ${EXP_DIR}"
    if [ "$VAL_EMB" != "MISSING" ]; then
        EVAL_CMD="${EVAL_CMD} --val-embedding-file ${VAL_EMB}"
    fi

    # Full job script
    JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${CIPHER_DIR}/logs/${NAME}_%j.out
#SBATCH --error=${CIPHER_DIR}/logs/${NAME}_%j.err

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}

echo \"======================================\"
echo \"Embedding sweep: ${LABEL}\"
echo \"  Train embeddings: ${TRAIN_EMB}\"
echo \"  Val embeddings:   ${VAL_EMB}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

# Check files exist
if [ ! -f \"${TRAIN_EMB}\" ]; then
    echo \"ERROR: Training embedding file not found: ${TRAIN_EMB}\"
    exit 1
fi

# Train
echo \"\"
echo \"=== TRAINING ===\"
${TRAIN_CMD}

# Evaluate (only if val embeddings exist)
if [ \"${VAL_EMB}\" != \"MISSING\" ] && [ -f \"${VAL_EMB}\" ]; then
    echo \"\"
    echo \"=== EVALUATING ===\"
    ${EVAL_CMD}
else
    echo \"\"
    echo \"SKIPPING evaluation: validation embeddings not available (${VAL_EMB})\"
    echo \"Run evaluation later with:\"
    echo \"  ${EVAL_CMD}\"
fi

echo \"\"
echo \"======================================\"
echo \"Done: ${LABEL}\"
echo \"  Finished: \$(date)\"
echo \"======================================"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] ${LABEL}"
        echo "    Train: ${TRAIN_EMB}"
        echo "    Val:   ${VAL_EMB}"
        echo ""
    else
        # Create logs directory
        mkdir -p "${CIPHER_DIR}/logs"

        # Submit
        JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
        echo "  Submitted ${JOB_ID} — ${LABEL}"
        echo "    Train: ${TRAIN_EMB}"
        echo "    Val:   ${VAL_EMB}"
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
    echo "  python scripts/compare_experiments.py"
fi
echo "============================================================"
