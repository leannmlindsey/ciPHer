#!/usr/bin/env bash
#
# Attribution control for the highconf_pipeline result.
#
# Goal: isolate whether the jump from PHL rh@1 0.17 -> 0.19 came from the
# K-type-cluster-purity filter or just from training on a smaller dataset.
#
# Method:
#   1. Build a deterministic random subset of pipeline_positive.list
#      matched in size to highconf_pipeline (~12,481 proteins).
#   2. Train attention_mlp on ProtT5 mean with that subset as --positive_list.
#   3. Compare PHL rh@1 against the 0.188 highconf_pipeline result.
#
# Interpretation:
#   - Random-subset PHL rh@1 ~0.19  -> highconf filter did nothing beyond
#                                       shrinking data; 0.188 is a size effect.
#   - Random-subset PHL rh@1 <=0.13 -> highconf's K-purity filter is the real
#                                       cause; we should build highconf_v2.
#
# Usage:
#   bash scripts/run_control_random_subset.sh
#   DRY_RUN=1 bash scripts/run_control_random_subset.sh

set -euo pipefail

# ============================================================
# Delta-AI configuration (env-overridable)
# ============================================================
ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

ASSOC_MAP="${CIPHER_DIR}/data/training_data/metadata/host_phage_protein_map.tsv"
GLYCAN_BINDERS="${CIPHER_DIR}/data/training_data/metadata/glycan_binders_custom.tsv"
VAL_FASTA="${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${CIPHER_DIR}/data/validation_data/HOST_RANGE"

# ProtT5 mean embeddings
TRAIN_EMB="${TRAIN_EMB:-/projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_prott5/candidates_embeddings_md5.npz}"
VAL_EMB="${VAL_EMB:-/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz}"
EMBEDDING_TYPE="prott5_mean"

POSLIST="${CIPHER_DIR}/data/training_data/metadata/pipeline_positive.list"
RANDOM_LIST="${CIPHER_DIR}/data/training_data/metadata/random12481_pipeline_positive_K.list"
SUBSET_N="${SUBSET_N:-12481}"
SUBSET_SEED="${SUBSET_SEED:-42}"

# Match the highconf_pipeline attention_mlp hyperparameters exactly
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
TIME="6:00:00"

NAME="control_random${SUBSET_N}_prott5_mean"
LOG="${CIPHER_DIR}/logs/${NAME}_%j.log"

DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "ATTRIBUTION CONTROL - random subset vs highconf_pipeline"
echo "  Model:        ${MODEL} on ${EMBEDDING_TYPE}"
echo "  Random subset: ${SUBSET_N} proteins from pipeline_positive.list (seed=${SUBSET_SEED})"
echo "  Cipher dir:    ${CIPHER_DIR}"
echo "============================================================"
echo ""

TRAIN_CMD="python -m cipher.cli.train_runner \
    --model ${MODEL} \
    --positive_list ${RANDOM_LIST} \
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
#SBATCH --output=${LOG}
#SBATCH --error=${LOG}

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

echo \"======================================\"
echo \"Attribution control: ${NAME}\"
echo \"  random list: ${RANDOM_LIST}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

echo \"=== STEP 1: build random-${SUBSET_N} subset (idempotent) ===\"
if [ -s \"${RANDOM_LIST}\" ]; then
    echo \"  Subset list already exists: ${RANDOM_LIST}\"
else
    python ${CIPHER_DIR}/scripts/utils/build_random_subset_list.py \\
        --in ${POSLIST} \\
        --out ${RANDOM_LIST} \\
        --n ${SUBSET_N} \\
        --seed ${SUBSET_SEED}
fi

echo \"=== STEP 2: TRAINING ===\"
${TRAIN_CMD}

echo \"=== STEP 3: EVALUATING ===\"
${EVAL_CMD}

echo \"======================================\"
echo \"Done: ${NAME} at \$(date)\"
echo \"======================================\"
"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would submit: ${NAME}"
    echo "  Random list: ${RANDOM_LIST}"
    echo "  Experiment:  ${EXP_DIR}"
else
    mkdir -p "${CIPHER_DIR}/logs"
    JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
    echo "Submitted ${JOB_ID} - ${NAME}"
fi
