#!/usr/bin/env bash
#
# First end-to-end test of the contrastive_encoder model.
#
# Two experiments, each a single SLURM job that chains:
#   1. Train the contrastive encoder (ArcFace + PK-cluster sampler) ->
#      produces contrastive_train_md5.npz + contrastive_val_md5.npz
#   2. Quality gate: PHL neighbour-label audit on the new NPZs
#      (target: top-1 K-match > 20% vs 11.3% baseline on raw ESM-2)
#   3. Train attention_mlp on the contrastive NPZs
#   4. Evaluate on all 5 validation datasets
#
# Variants
# --------
#   highconf_pipeline : matches today's best attention_mlp run (PHL rh@1=0.188)
#                       Direct head-to-head: same filter, same input embedding.
#   poslist_cl70      : larger training set (~59K MD5s) with cluster-70
#                       stratified sampling. Tests whether contrastive scales
#                       with more (noisier) data.
#
# Input backbone for both = ProtT5 mean (best K-type separation of our
# available pLMs). K + O heads both active (lambda_k=lambda_o=1 from the
# base_config).
#
# Usage:
#   bash scripts/run_contrastive_experiments.sh                    # both
#   bash scripts/run_contrastive_experiments.sh highconf_pipeline  # one
#   bash scripts/run_contrastive_experiments.sh poslist_cl70
#   DRY_RUN=1 bash scripts/run_contrastive_experiments.sh          # print only

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
CLUSTER_FILE="${CIPHER_DIR}/data/training_data/metadata/candidates_clusters.tsv"
VAL_FASTA="${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${CIPHER_DIR}/data/validation_data/HOST_RANGE"

# ProtT5 mean embeddings (same files as the highconf + sweep runs)
TRAIN_EMB="${TRAIN_EMB:-/projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_prott5/candidates_embeddings_md5.npz}"
VAL_EMB="${VAL_EMB:-/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz}"
EMBEDDING_TYPE="prott5_mean"

# Positive lists
HIGHCONF_LIST="${CIPHER_DIR}/data/training_data/metadata/highconf_pipeline_positive_K.list"
POSLIST="${CIPHER_DIR}/data/training_data/metadata/pipeline_positive.list"

# Downstream attention_mlp hyperparameters - mirror the highconf baseline so
# the comparison is one variable at a time (the embedding layer).
DOWN_LR="1e-05"
DOWN_BATCH=512
DOWN_EPOCHS=1000
DOWN_PATIENCE=30
DOWN_LABEL_STRATEGY="multi_label_threshold"
DOWN_MIN_CLASS_SAMPLES=25
DOWN_MIN_SOURCES=1

# SLURM resources per job
GPUS=1
CPUS=8
MEM="96G"
TIME="6:00:00"

# ============================================================
# Variant table:
#   "label  filter_flags_for_train_runner"
# filter_flags are shell-safe; they get pasted verbatim into cipher-train.
# ============================================================
VARIANTS=(
    "highconf_pipeline  --positive_list ${HIGHCONF_LIST}"
    "poslist_cl70       --positive_list ${POSLIST} --cluster_file ${CLUSTER_FILE} --cluster_threshold 70"
)

FILTER="${1:-}"
DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "CONTRASTIVE ENCODER - FIRST END-TO-END TEST"
echo "  Input embedding: ${EMBEDDING_TYPE}"
echo "  Cipher dir:      ${CIPHER_DIR}"
echo "  Variants:        ${#VARIANTS[@]}"
echo "============================================================"
echo ""

N_SUBMITTED=0
for entry in "${VARIANTS[@]}"; do
    LABEL="${entry%% *}"
    FILTER_FLAGS="${entry#* }"

    if [ -n "$FILTER" ] && [ "$LABEL" != "$FILTER" ]; then
        continue
    fi

    ENC_NAME="contrastive_${LABEL}_${EMBEDDING_TYPE}"
    DOWN_NAME="downstream_${ENC_NAME}"
    ENC_DIR="${CIPHER_DIR}/experiments/contrastive_encoder/${ENC_NAME}"
    DOWN_DIR="${CIPHER_DIR}/experiments/attention_mlp/${DOWN_NAME}"
    CONTRASTIVE_TRAIN_NPZ="${ENC_DIR}/contrastive_train_md5.npz"
    CONTRASTIVE_VAL_NPZ="${ENC_DIR}/contrastive_val_md5.npz"

    JOB_NAME="${ENC_NAME}"
    LOG="${CIPHER_DIR}/logs/${JOB_NAME}_%j.log"

    # ----------------------------------------------------------
    # Commands for the SBATCH script (heredoc expanded below)
    # ----------------------------------------------------------
    ENC_CMD="python -m cipher.cli.train_runner \
        --model contrastive_encoder \
        ${FILTER_FLAGS} \
        --embedding_type ${EMBEDDING_TYPE} \
        --embedding_file ${TRAIN_EMB} \
        --association_map ${ASSOC_MAP} \
        --glycan_binders ${GLYCAN_BINDERS} \
        --val_fasta ${VAL_FASTA} \
        --val_datasets_dir ${VAL_DATASETS_DIR} \
        --val_embedding_file ${VAL_EMB} \
        --name ${ENC_NAME}"

    GATE_CMD="python ${CIPHER_DIR}/scripts/analysis/phl_neighbor_labels.py \
        --train-emb ${CONTRASTIVE_TRAIN_NPZ} \
        --val-emb ${CONTRASTIVE_VAL_NPZ} \
        --out-dir ${ENC_DIR}/analysis \
        --restrict-to-labeled"

    DOWN_CMD="python -m cipher.cli.train_runner \
        --model attention_mlp \
        ${FILTER_FLAGS} \
        --lr ${DOWN_LR} \
        --batch_size ${DOWN_BATCH} \
        --epochs ${DOWN_EPOCHS} \
        --patience ${DOWN_PATIENCE} \
        --label_strategy ${DOWN_LABEL_STRATEGY} \
        --min_class_samples ${DOWN_MIN_CLASS_SAMPLES} \
        --min_sources ${DOWN_MIN_SOURCES} \
        --embedding_type ${EMBEDDING_TYPE} \
        --embedding_file ${CONTRASTIVE_TRAIN_NPZ} \
        --association_map ${ASSOC_MAP} \
        --glycan_binders ${GLYCAN_BINDERS} \
        --val_fasta ${VAL_FASTA} \
        --val_datasets_dir ${VAL_DATASETS_DIR} \
        --val_embedding_file ${CONTRASTIVE_VAL_NPZ} \
        --name ${DOWN_NAME}"

    EVAL_CMD="python -m cipher.evaluation.runner ${DOWN_DIR} --val-embedding-file ${CONTRASTIVE_VAL_NPZ}"

    JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
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
echo \"Contrastive experiment: ${ENC_NAME}\"
echo \"  input embedding: ${EMBEDDING_TYPE}\"
echo \"  filter flags:    ${FILTER_FLAGS}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

echo \"=== STEP 1: TRAIN CONTRASTIVE ENCODER ===\"
${ENC_CMD}

echo \"=== STEP 2: QUALITY GATE (PHL neighbour K-match) ===\"
echo \"Target: top-1 K-match > 20% (raw ESM-2 650M baseline was 11.3%)\"
${GATE_CMD} || echo \"WARNING: quality gate script failed; continuing to downstream anyway\"

echo \"=== STEP 3: DOWNSTREAM attention_mlp ===\"
${DOWN_CMD}

echo \"=== STEP 4: EVALUATE ===\"
${EVAL_CMD}

echo \"======================================\"
echo \"Done: ${ENC_NAME} at \$(date)\"
echo \"======================================\"
"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] ${ENC_NAME}"
        echo "    enc_dir:   ${ENC_DIR}"
        echo "    down_dir:  ${DOWN_DIR}"
        echo "    filter:    ${FILTER_FLAGS}"
        echo ""
    else
        mkdir -p "${CIPHER_DIR}/logs"
        JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
        echo "  Submitted ${JOB_ID} - ${ENC_NAME}"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    fi
done

echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN complete."
else
    echo "Submitted ${N_SUBMITTED} contrastive-encoder job(s)."
    echo ""
    echo "Per-job log format: ${CIPHER_DIR}/logs/contrastive_<label>_<embedding>_<jobid>.log"
fi
