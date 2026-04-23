#!/usr/bin/env bash
#
# Train attention_mlp on ProtT5 mean embeddings using the new v2 per-head
# highconf training lists (from agent 4's data-analysis workstream). This
# is the first use of the `--positive_list_k` / `--positive_list_o`
# flags added by the 2026-04-22 loss-masking refactor.
#
# Two variants, one SLURM job each (train + evaluate):
#   strict  — HC_K_cl95.list + HC_O_cl95_full_coverage.list (23,299 / 14,677)
#   uat     — HC_K_UAT.list  + HC_O_UAT.list                (25,924 / 15,568)
#
# Everything else (embedding, attention_mlp hyperparameters, val data)
# mirrors the v1 highconf_pipeline run so the comparison against the
# current 0.188 PHL rh@1 baseline is apples-to-apples.
#
# Usage:
#   bash scripts/run_v2_experiment.sh                # both variants
#   bash scripts/run_v2_experiment.sh strict         # cl95 strict only
#   bash scripts/run_v2_experiment.sh uat            # UAT only
#   DRY_RUN=1 bash scripts/run_v2_experiment.sh      # print only

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

# ProtT5 mean embeddings (same files as the v1 highconf baseline)
TRAIN_EMB="${TRAIN_EMB:-/projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_prott5/candidates_embeddings_md5.npz}"
VAL_EMB="${VAL_EMB:-/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz}"

HC_V2_DIR="${CIPHER_DIR}/data/training_data/metadata/highconf_v2"

# attention_mlp hyperparameters — match the v1 highconf baseline exactly
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
#   "label  K_list_path  O_list_path"
# ============================================================
VARIANTS=(
    "strict  ${HC_V2_DIR}/HC_K_cl95.list  ${HC_V2_DIR}/HC_O_cl95_full_coverage.list"
    "uat     ${HC_V2_DIR}/HC_K_UAT.list   ${HC_V2_DIR}/HC_O_UAT.list"
)

FILTER="${1:-}"
DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "V2 PER-HEAD HIGHCONF EXPERIMENT"
echo "  Model:      ${MODEL} on prott5_mean embeddings"
echo "  Cipher dir: ${CIPHER_DIR}"
echo "  v2 lists:   ${HC_V2_DIR}/"
echo "============================================================"
echo ""

N_SUBMITTED=0
for entry in "${VARIANTS[@]}"; do
    read -r LABEL K_LIST O_LIST <<< "$entry"
    if [ -n "$FILTER" ] && [ "$LABEL" != "$FILTER" ]; then
        continue
    fi

    NAME="v2_${LABEL}_prott5_mean"

    if [ ! -f "$K_LIST" ]; then
        echo "  SKIP ${LABEL} — K list not found: ${K_LIST}"
        continue
    fi
    if [ ! -f "$O_LIST" ]; then
        echo "  SKIP ${LABEL} — O list not found: ${O_LIST}"
        continue
    fi

    TRAIN_CMD="python -m cipher.cli.train_runner \
        --model ${MODEL} \
        --positive_list_k ${K_LIST} \
        --positive_list_o ${O_LIST} \
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
echo \"v2 per-head experiment: ${NAME}\"
echo \"  K list: ${K_LIST}\"
echo \"  O list: ${O_LIST}\"
echo \"  train emb: ${TRAIN_EMB}\"
echo \"  val emb:   ${VAL_EMB}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

echo \"=== TRAINING ===\"
${TRAIN_CMD}

echo \"=== EVALUATING ===\"
${EVAL_CMD}

echo \"======================================\"
echo \"Done: ${NAME} at \$(date)\"
echo \"======================================\"
"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] ${NAME}"
        echo "    K list: ${K_LIST}"
        echo "    O list: ${O_LIST}"
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
    echo "Submitted ${N_SUBMITTED} v2 experiment job(s)."
    echo ""
    echo "Baselines to compare against (v1 highconf_pipeline = PHL rh@1 = 0.188):"
    echo "  experiments/${MODEL}/highconf_pipeline_K_prott5_mean/"
fi
