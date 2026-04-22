#!/usr/bin/env bash
#
# Train attention_mlp on ProtT5 mean embeddings using one of the two
# high-confidence cluster-filtered positive lists produced by the
# sibling CLAUDE_PHI_DATA_ANALYSIS analysis.
#
# Two variants, one SLURM job each (train + evaluate):
#   tsp           — highconf_tsp_K.list              (7,438 proteins)
#   pipeline      — highconf_pipeline_positive_K.list (12,481 proteins)
#
# No cluster-stratified downsampling and no per-K cap — the positive
# list itself already encodes the cluster-level filter. Everything else
# matches the ProtT5 mean baseline for direct comparison.
#
# Usage:
#   # Submit both:
#   bash scripts/run_highconf_experiment.sh
#
#   # Submit one:
#   bash scripts/run_highconf_experiment.sh tsp
#   bash scripts/run_highconf_experiment.sh pipeline
#
#   # Dry run:
#   DRY_RUN=1 bash scripts/run_highconf_experiment.sh

set -euo pipefail

# ============================================================
# Delta-AI configuration
# ============================================================
ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="/projects/bfzj/llindsey1/PHI_TSP/ciPHer"

ASSOC_MAP="${CIPHER_DIR}/data/training_data/metadata/host_phage_protein_map.tsv"
GLYCAN_BINDERS="${CIPHER_DIR}/data/training_data/metadata/glycan_binders_custom.tsv"
VAL_FASTA="${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${CIPHER_DIR}/data/validation_data/HOST_RANGE"

# ProtT5 mean embeddings (re-using sweep paths)
TRAIN_EMB="/projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_prott5/candidates_embeddings_md5.npz"
VAL_EMB="/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz"

# Training hyperparameters — mirror the ProtT5 mean baseline
MODEL="attention_mlp"
LR="1e-05"
BATCH_SIZE=512
EPOCHS=1000
PATIENCE=30
LABEL_STRATEGY="multi_label_threshold"
MIN_CLASS_SAMPLES=25
MIN_SOURCES=1

# SLURM resources
GPUS=1
CPUS=8
MEM="64G"
TIME="12:00:00"

# ============================================================
# Variant table:
#   "label  positive_list_path"
# ============================================================
VARIANTS=(
    "tsp       ${CIPHER_DIR}/data/training_data/metadata/highconf_tsp_K.list"
    "pipeline  ${CIPHER_DIR}/data/training_data/metadata/highconf_pipeline_positive_K.list"
)

FILTER="${1:-}"
DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "HIGH-CONFIDENCE CLUSTER EXPERIMENT"
echo "  Model:      ${MODEL} on prott5_mean embeddings"
echo "  Cipher dir: ${CIPHER_DIR}"
echo "============================================================"
echo ""

N_SUBMITTED=0
for entry in "${VARIANTS[@]}"; do
    read -r LABEL POSLIST <<< "$entry"
    if [ -n "$FILTER" ] && [ "$LABEL" != "$FILTER" ]; then
        continue
    fi

    NAME="highconf_${LABEL}_K_prott5_mean"

    if [ ! -f "$POSLIST" ]; then
        echo "  SKIP ${LABEL} — positive list not found on Delta: ${POSLIST}"
        continue
    fi

    TRAIN_CMD="python -m cipher.cli.train_runner \
        --model ${MODEL} \
        --positive_list ${POSLIST} \
        --lr ${LR} \
        --batch_size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --patience ${PATIENCE} \
        --label_strategy ${LABEL_STRATEGY} \
        --min_class_samples ${MIN_CLASS_SAMPLES} \
        --min_sources ${MIN_SOURCES} \
        --embedding_type prott5_mean \
        --embedding_file ${TRAIN_EMB} \
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
echo \"High-conf experiment: ${NAME}\"
echo \"  positive_list: ${POSLIST}\"
echo \"  train emb:     ${TRAIN_EMB}\"
echo \"  val emb:       ${VAL_EMB}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

echo \"=== TRAINING ===\"
${TRAIN_CMD}

echo \"=== EVALUATING ===\"
${EVAL_CMD}

echo \"======================================\"
echo \"Done: ${NAME} at \$(date)\"
echo \"======================================"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] ${NAME}"
        echo "    positive_list: ${POSLIST}"
        echo ""
    else
        mkdir -p "${CIPHER_DIR}/logs"
        JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
        echo "  Submitted ${JOB_ID} — ${NAME}"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    fi
done

echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN complete."
else
    echo "Submitted ${N_SUBMITTED} jobs."
fi
