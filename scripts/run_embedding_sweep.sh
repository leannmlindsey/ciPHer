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
CIPHER_DIR="/projects/bfzj/llindsey1/PHI_TSP/ciPHer"

# ============================================================
# Data paths on Delta (no symlinks needed)
# ============================================================
ASSOC_MAP="/projects/bfzj/llindsey1/PHI_TSP/ciPHer/data/training_data/metadata/host_phage_protein_map.tsv"
GLYCAN_BINDERS="/projects/bfzj/llindsey1/RBP_Structural_Similarity/input/glycan_binders_custom.tsv"
VAL_FASTA="/projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/validation_data/combined/validation_inputs/validation_rbps_all.faa"
VAL_DATASETS_DIR="/projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/validation_data/combined/HOST_RANGE"

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
    "esm2_3b_mean       ${TRAIN_EMB_ROOT}/embeddings_esm2_3b/candidates_embeddings_md5.npz                  /work/hdd/bfzj/llindsey1/validation_embeddings_esm2_3b/validation_embeddings_md5.npz"
    "esm2_150m_mean     ${TRAIN_EMB_ROOT}/embeddings_esm2_150m/candidates_embeddings_md5.npz                /work/hdd/bfzj/llindsey1/validation_embeddings_esm2_150m/validation_embeddings_md5.npz"
    "prott5_mean        ${TRAIN_EMB_ROOT}/embeddings_prott5/candidates_embeddings_md5.npz                   /work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz"

    # K-mer features (best separation results only)
    "kmer_murphy8_k5    ${KMER_ROOT}/candidates_murphy8_k5.npz      ${KMER_ROOT}/validation_murphy8_k5.npz"
    "kmer_murphy10_k5   ${KMER_ROOT}/candidates_murphy10_k5.npz     ${KMER_ROOT}/validation_murphy10_k5.npz"
    "kmer_li10_k5       ${KMER_ROOT}/candidates_li10_k5.npz         ${KMER_ROOT}/validation_li10_k5.npz"
    "kmer_aa20_k3       ${KMER_ROOT}/candidates_aa20_k3.npz         ${KMER_ROOT}/validation_aa20_k3.npz"
    "kmer_aa20_k4       ${KMER_ROOT}/candidates_aa20_k4.npz         ${KMER_ROOT}/validation_aa20_k4.npz"
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
        echo "  [DRY RUN] ${LABEL}"
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
    echo "  python scripts/compare_experiments.py"
fi
echo "============================================================"
