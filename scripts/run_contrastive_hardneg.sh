#!/usr/bin/env bash
#
# Hard-negative-mining variant of the contrastive encoder.
#
# Mirrors the K-only m=0.5 s=30 sweep variant exactly (same filter, same
# embedding, same hyperparameters) EXCEPT hard-negative mining is enabled
# in the PK sampler. Direct A/B test vs the corresponding sweep run:
#
#   vanilla (from run_contrastive_sweep.sh):
#     contrastive_Konly_m05_s30_highconf_prott5_mean
#   hard-neg (this script):
#     contrastive_Konly_m05_s30_hardneg_highconf_prott5_mean
#
# What hard-negative mining does here (see models/contrastive_encoder/train.py):
#   - every HN_UPDATE_EVERY epochs after HN_START_EPOCH, recompute per-class
#     prototypes (mean encoder output on train split) and pairwise cosine
#     similarity between them
#   - each class's "hardness score" = max cosine to another prototype
#   - PK sampler draws the P classes per batch proportional to softmaxed
#     hardness (with temperature HN_TEMPERATURE), so the encoder sees the
#     currently-confused class-pairs more often
#
# Usage:
#   bash scripts/run_contrastive_hardneg.sh
#   DRY_RUN=1 bash scripts/run_contrastive_hardneg.sh
#   HN_TEMPERATURE=0.5 HN_START_EPOCH=5 bash scripts/run_contrastive_hardneg.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

ASSOC_MAP="${CIPHER_DIR}/data/training_data/metadata/host_phage_protein_map.tsv"
GLYCAN_BINDERS="${CIPHER_DIR}/data/training_data/metadata/glycan_binders_custom.tsv"
VAL_FASTA="${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${CIPHER_DIR}/data/validation_data/HOST_RANGE"

TRAIN_EMB="${TRAIN_EMB:-/projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_prott5/candidates_embeddings_md5.npz}"
VAL_EMB="${VAL_EMB:-/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz}"
EMBEDDING_TYPE="prott5_mean"

HIGHCONF_LIST="${CIPHER_DIR}/data/training_data/metadata/highconf_pipeline_positive_K.list"

# Shared hyperparameters (must match the vanilla counterpart in the sweep)
ARCFACE_MARGIN="${ARCFACE_MARGIN:-0.5}"
ARCFACE_SCALE="${ARCFACE_SCALE:-30}"

# Hard-negative-specific (env-overridable)
HN_START_EPOCH="${HN_START_EPOCH:-10}"
HN_UPDATE_EVERY="${HN_UPDATE_EVERY:-5}"
HN_TEMPERATURE="${HN_TEMPERATURE:-1.0}"

# Downstream attention_mlp hyperparameters
DOWN_LR="1e-05"
DOWN_BATCH=512
DOWN_EPOCHS=1000
DOWN_PATIENCE=30
DOWN_LABEL_STRATEGY="multi_label_threshold"
DOWN_MIN_CLASS_SAMPLES=25
DOWN_MIN_SOURCES=1

GPUS=1
CPUS=8
MEM="96G"
TIME="6:00:00"

DRY_RUN="${DRY_RUN:-0}"

ENC_NAME="contrastive_Konly_m${ARCFACE_MARGIN//./}_s${ARCFACE_SCALE%.*}_hardneg_highconf_prott5_mean"
DOWN_NAME="downstream_${ENC_NAME}"
ENC_DIR="${CIPHER_DIR}/experiments/contrastive_encoder/${ENC_NAME}"
DOWN_DIR="${CIPHER_DIR}/experiments/attention_mlp/${DOWN_NAME}"
CT_TRAIN_NPZ="${ENC_DIR}/contrastive_train_md5.npz"
CT_VAL_NPZ="${ENC_DIR}/contrastive_val_md5.npz"
LOG="${CIPHER_DIR}/logs/${ENC_NAME}_%j.log"

echo "============================================================"
echo "CONTRASTIVE ENCODER - HARD-NEGATIVE MINING"
echo "  ArcFace:     margin=${ARCFACE_MARGIN}  scale=${ARCFACE_SCALE}  (K-only, lambda_o=0)"
echo "  HN params:   start_epoch=${HN_START_EPOCH}  update_every=${HN_UPDATE_EVERY}  temperature=${HN_TEMPERATURE}"
echo "  Backbone:    ${EMBEDDING_TYPE}"
echo "  Filter:      highconf_pipeline"
echo "============================================================"
echo ""

ENC_CMD="python -m cipher.cli.train_runner \
    --model contrastive_encoder \
    --positive_list ${HIGHCONF_LIST} \
    --embedding_type ${EMBEDDING_TYPE} \
    --embedding_file ${TRAIN_EMB} \
    --association_map ${ASSOC_MAP} \
    --glycan_binders ${GLYCAN_BINDERS} \
    --val_fasta ${VAL_FASTA} \
    --val_datasets_dir ${VAL_DATASETS_DIR} \
    --val_embedding_file ${VAL_EMB} \
    --lambda_k 1.0 \
    --lambda_o 0.0 \
    --arcface_margin ${ARCFACE_MARGIN} \
    --arcface_scale ${ARCFACE_SCALE} \
    --sampler_hard_negative_mining true \
    --sampler_hard_negative_start_epoch ${HN_START_EPOCH} \
    --name ${ENC_NAME}"

GATE_CMD="python ${CIPHER_DIR}/scripts/analysis/phl_neighbor_labels.py \
    --train-emb ${CT_TRAIN_NPZ} \
    --val-emb ${CT_VAL_NPZ} \
    --out-dir ${ENC_DIR}/analysis \
    --restrict-to-labeled"

DOWN_CMD="python -m cipher.cli.train_runner \
    --model attention_mlp \
    --positive_list ${HIGHCONF_LIST} \
    --lr ${DOWN_LR} \
    --batch_size ${DOWN_BATCH} \
    --epochs ${DOWN_EPOCHS} \
    --patience ${DOWN_PATIENCE} \
    --label_strategy ${DOWN_LABEL_STRATEGY} \
    --min_class_samples ${DOWN_MIN_CLASS_SAMPLES} \
    --min_sources ${DOWN_MIN_SOURCES} \
    --embedding_type ${EMBEDDING_TYPE} \
    --embedding_file ${CT_TRAIN_NPZ} \
    --association_map ${ASSOC_MAP} \
    --glycan_binders ${GLYCAN_BINDERS} \
    --val_fasta ${VAL_FASTA} \
    --val_datasets_dir ${VAL_DATASETS_DIR} \
    --val_embedding_file ${CT_VAL_NPZ} \
    --name ${DOWN_NAME}"

EVAL_CMD="python -m cipher.evaluation.runner ${DOWN_DIR} --val-embedding-file ${CT_VAL_NPZ}"

JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${ENC_NAME}
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
echo \"Contrastive hard-neg: ${ENC_NAME}\"
echo \"  margin=${ARCFACE_MARGIN}  scale=${ARCFACE_SCALE}  K-only\"
echo \"  HN: start=${HN_START_EPOCH} every=${HN_UPDATE_EVERY} temp=${HN_TEMPERATURE}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

echo \"=== STEP 1: TRAIN ENCODER (hard-negative) ===\"
${ENC_CMD}

echo \"=== STEP 2: QUALITY GATE (PHL K-match) ===\"
${GATE_CMD} || echo \"WARNING: quality gate failed; continuing\"

echo \"=== STEP 3: DOWNSTREAM attention_mlp ===\"
${DOWN_CMD}

echo \"=== STEP 4: EVALUATE ===\"
${EVAL_CMD}

echo \"======================================\"
echo \"Done: ${ENC_NAME} at \$(date)\"
echo \"======================================\"
"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] ${ENC_NAME}"
    echo "  enc_dir:  ${ENC_DIR}"
    echo "  down_dir: ${DOWN_DIR}"
else
    mkdir -p "${CIPHER_DIR}/logs"
    JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
    echo "Submitted ${JOB_ID} - ${ENC_NAME}"
fi
