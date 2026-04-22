#!/usr/bin/env bash
#
# Contrastive encoder K-only hyperparameter sweep.
#
# All three variants share:
#   - input embedding: ProtT5 mean
#   - training filter: highconf_pipeline_positive_K.list
#   - lambda_k = 1.0, lambda_o = 0.0  (K-head only — drop O to avoid
#     pulling the encoder toward O-discriminative features)
#
# The three variants differ only in ArcFace (margin, scale):
#   m03_s30   margin=0.3, scale=30   softer angular margin
#   m05_s30   margin=0.5, scale=30   K-only version of the original baseline
#   m05_s50   margin=0.5, scale=50   sharper softmax
#
# Each job: train encoder -> quality gate (PHL K-match) -> train
# attention_mlp on the learned NPZs -> evaluate.
#
# Usage:
#   bash scripts/run_contrastive_sweep.sh                 # all 3
#   bash scripts/run_contrastive_sweep.sh m03_s30         # one
#   DRY_RUN=1 bash scripts/run_contrastive_sweep.sh       # print only

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

# Downstream attention_mlp hyperparameters (mirror the highconf baseline)
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

# Variant table:
#   "label  margin  scale"
VARIANTS=(
    "m03_s30  0.3  30"
    "m05_s30  0.5  30"
    "m05_s50  0.5  50"
)

FILTER="${1:-}"
DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "CONTRASTIVE ENCODER K-ONLY HP SWEEP"
echo "  Backbone:     ${EMBEDDING_TYPE}"
echo "  Filter:       highconf_pipeline"
echo "  K-only:       lambda_o=0"
echo "  Cipher dir:   ${CIPHER_DIR}"
echo "============================================================"
echo ""

N_SUBMITTED=0
for entry in "${VARIANTS[@]}"; do
    read -r LABEL MARGIN SCALE <<< "$entry"
    if [ -n "$FILTER" ] && [ "$LABEL" != "$FILTER" ]; then
        continue
    fi

    ENC_NAME="contrastive_Konly_${LABEL}_highconf_prott5_mean"
    DOWN_NAME="downstream_${ENC_NAME}"
    ENC_DIR="${CIPHER_DIR}/experiments/contrastive_encoder/${ENC_NAME}"
    DOWN_DIR="${CIPHER_DIR}/experiments/attention_mlp/${DOWN_NAME}"
    CT_TRAIN_NPZ="${ENC_DIR}/contrastive_train_md5.npz"
    CT_VAL_NPZ="${ENC_DIR}/contrastive_val_md5.npz"
    LOG="${CIPHER_DIR}/logs/${ENC_NAME}_%j.log"

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
        --arcface_margin ${MARGIN} \
        --arcface_scale ${SCALE} \
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
echo \"Contrastive K-only sweep: ${ENC_NAME}\"
echo \"  margin=${MARGIN}  scale=${SCALE}  lambda_o=0\"
echo \"  Started: \$(date)\"
echo \"======================================\"

echo \"=== STEP 1: TRAIN ENCODER ===\"
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
        echo "  [DRY RUN] ${ENC_NAME}  (margin=${MARGIN} scale=${SCALE})"
    else
        mkdir -p "${CIPHER_DIR}/logs"
        JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
        echo "  Submitted ${JOB_ID} - ${ENC_NAME}  (margin=${MARGIN} scale=${SCALE})"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    fi
done

echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN complete."
else
    echo "Submitted ${N_SUBMITTED} K-only sweep job(s)."
fi
