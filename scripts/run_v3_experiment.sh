#!/usr/bin/env bash
#
# Train attention_mlp on ProtT5 mean embeddings using v3_multitop
# highconf lists (agent 4, 2026-04-23). v3 relaxes v2's top-1-must-be-≥90%
# rule to "top-3-together-must-be-≥90%", which restores broad-range
# proteins that v2 was dropping at ~10× the rate of specific ones.
# Motivated by analysis A15; full rationale in
# data/training_data/metadata/highconf_v3_multitop/README.md.
#
# Three variants, one SLURM job each (train + evaluate):
#   strict    — HC_K_cl95_multitop + HC_O_cl95_multitop_full_coverage  (35,249 / 35,429)
#   uat       — HC_K_UAT_multitop  + HC_O_UAT_multitop                 (37,158 / 37,178)
#   k_v2o     — HC_K_cl95_multitop (v3)  + HC_O_cl95_full_coverage (v2)
#               Direct test of agent 4's specific question: "is v3 O
#               actually helping, or absorbing host-background noise?"
#
# Everything else (embedding, attention_mlp hyperparameters, val data)
# mirrors v2 run for apples-to-apples comparison.
#
# Baselines to beat:
#   v1 highconf:     PHL rh@1 = 0.188 (combined) / 0.232 (K-only)
#   v2_strict:       PHL rh@1 = 0.144 (combined) / 0.161 (K-only)
#   v2_uat:          PHL rh@1 = 0.132 (combined) / 0.144 (K-only)
#
# Usage:
#   bash scripts/run_v3_experiment.sh                # all three variants
#   bash scripts/run_v3_experiment.sh strict         # one variant only
#   bash scripts/run_v3_experiment.sh k_v2o          # cross variant only
#   DRY_RUN=1 bash scripts/run_v3_experiment.sh      # print only

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

# ProtT5 mean embeddings (same files as v1/v2 baselines)
TRAIN_EMB="${TRAIN_EMB:-/projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_prott5/candidates_embeddings_md5.npz}"
VAL_EMB="${VAL_EMB:-/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz}"

HC_V2_DIR="${CIPHER_DIR}/data/training_data/metadata/highconf_v2"
HC_V3_DIR="${CIPHER_DIR}/data/training_data/metadata/highconf_v3_multitop"

# attention_mlp hyperparameters — match v1/v2 baselines exactly
MODEL="attention_mlp"
LR="1e-05"
BATCH_SIZE=512
EPOCHS=1000
PATIENCE=30
LABEL_STRATEGY="multi_label_threshold"
MIN_CLASS_SAMPLES=25
MIN_SOURCES=1

# SLURM resources — v3 is ~2.3× v2 scale (35k vs 15k proteins). v2
# needed 128G (64G OOMed on the O head). Bumping v3 to 160G for
# headroom since the O head specifically sees ~35k samples vs v2's 10k.
GPUS=1
CPUS=8
MEM="160G"
TIME="24:00:00"

# ============================================================
# Variant table:
#   "label          K_list_path                      O_list_path"
# ============================================================
VARIANTS=(
    "strict  ${HC_V3_DIR}/HC_K_cl95_multitop.list  ${HC_V3_DIR}/HC_O_cl95_multitop_full_coverage.list"
    "uat     ${HC_V3_DIR}/HC_K_UAT_multitop.list   ${HC_V3_DIR}/HC_O_UAT_multitop.list"
    "k_v2o   ${HC_V3_DIR}/HC_K_cl95_multitop.list  ${HC_V2_DIR}/HC_O_cl95_full_coverage.list"
)

FILTER="${1:-}"
DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "V3 MULTI-TOP HIGHCONF EXPERIMENT"
echo "  Model:      ${MODEL} on prott5_mean embeddings"
echo "  Cipher dir: ${CIPHER_DIR}"
echo "  v3 lists:   ${HC_V3_DIR}/"
echo "============================================================"
echo ""

N_SUBMITTED=0
for entry in "${VARIANTS[@]}"; do
    read -r LABEL K_LIST O_LIST <<< "$entry"
    if [ -n "$FILTER" ] && [ "$LABEL" != "$FILTER" ]; then
        continue
    fi

    NAME="v3_${LABEL}_prott5_mean"

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
echo \"v3 multitop experiment: ${NAME}\"
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
    echo "Submitted ${N_SUBMITTED} v3 experiment job(s)."
    echo ""
    echo "Baselines to compare against:"
    echo "  v1 highconf:  PHL rh@1 = 0.188 combined / 0.232 K-only"
    echo "  v2_strict:    PHL rh@1 = 0.144 combined / 0.161 K-only"
    echo "  v2_uat:       PHL rh@1 = 0.132 combined / 0.144 K-only"
fi
