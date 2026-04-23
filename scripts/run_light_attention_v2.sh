#!/usr/bin/env bash
#
# LA + ProtT5-XL seg8 + v2 per-head highconf. Direct companion to
# scripts/run_v2_experiment.sh (agent 1's attention_mlp + ProtT5 mean + v2
# run) but with LA segmented pooling.
#
# Stacks three partial wins we have as of 2026-04-22:
#   * LA + ProtT5 seg8 architecture (PHL K-only rh@1 = 0.235 under v1 highconf)
#   * v2 per-head highconf training data (agent 1's A/B showed v2 recovered
#     GORODNICHIV, doubled CHEN, quadrupled UCSD on attention_mlp — v1 PHL
#     0.188 regressed to 0.144 at default eval but rh@5 held, so under
#     K-only inference the regression may flip)
#   * K-only inference at eval time
#
# Recommended eval flow after training:
#   bash scripts/analysis/run_eval_per_head.sh experiments/light_attention/la_v2_strict_prott5_xl_seg8
#   bash scripts/analysis/run_score_norm_raw.sh experiments/light_attention/la_v2_strict_prott5_xl_seg8
#   python3 scripts/analysis/show_eval_all.py experiments/light_attention/la_v2_strict_prott5_xl_seg8
#
# Usage:
#   bash scripts/run_light_attention_v2.sh
#   DRY_RUN=1 bash scripts/run_light_attention_v2.sh

set -euo pipefail

# ============================================================
# Delta-AI configuration (env-overridable)
# ============================================================
ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="${CIPHER_DIR:-$(dirname "$SCRIPT_DIR")}"
DATA_DIR="${DATA_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer/data}"

ASSOC_MAP="${DATA_DIR}/training_data/metadata/host_phage_protein_map.tsv"
GLYCAN_BINDERS="${DATA_DIR}/training_data/metadata/glycan_binders_custom.tsv"
VAL_FASTA="${DATA_DIR}/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${DATA_DIR}/validation_data/HOST_RANGE"

# v2 per-head highconf lists (strict cl95 variant — agent 1's A/B winner)
HC_V2_DIR="${DATA_DIR}/training_data/metadata/highconf_v2"
K_LIST="${HC_V2_DIR}/HC_K_cl95.list"
O_LIST="${HC_V2_DIR}/HC_O_cl95_full_coverage.list"

# ProtT5-XL seg8 paths (same as yesterday's successful la_highconf_pipeline_prott5_xl_seg8)
TRAIN_EMB="/work/hdd/bfzj/llindsey1/embeddings/prott5_xl_segments8/candidates_prott5_xl_segments8_md5.npz"
VAL_EMB="/work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments8/validation_prott5_xl_segments8_md5.npz"

CONFIG_ABS="${CIPHER_DIR}/models/light_attention/highconf_config.yaml"

MODEL="light_attention"
EMB_TYPE="prott5_xl_seg8"
N_SEGMENTS=8

# SLURM — bumped per agent 1's warning (v2 ~2× v1 scale; attention_mlp OOMed at 64G)
GPUS=1
CPUS=8
MEM="128G"
TIME="12:00:00"

NAME="la_v2_strict_prott5_xl_seg8"

DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "LA + PROT T5 seg8 + v2 PER-HEAD HIGHCONF (strict)"
echo "  Model:       ${MODEL}"
echo "  Embedding:   ${EMB_TYPE}  (n_segments=${N_SEGMENTS})"
echo "  K list:      ${K_LIST}"
echo "  O list:      ${O_LIST}"
echo "  Cipher dir:  ${CIPHER_DIR}"
echo "  SLURM:       ${TIME}, ${GPUS} GPU, ${CPUS} CPU, ${MEM}"
echo "============================================================"
echo ""

# Pre-submit checks
if [ ! -f "$K_LIST" ]; then
    echo "ERROR: K list not found: $K_LIST" >&2; exit 1
fi
if [ ! -f "$O_LIST" ]; then
    echo "ERROR: O list not found: $O_LIST" >&2; exit 1
fi
if [ ! -f "$CONFIG_ABS" ]; then
    echo "ERROR: config not found: $CONFIG_ABS" >&2; exit 1
fi
if [ ! -f "$TRAIN_EMB" ]; then
    echo "  SKIP ${NAME} — training embedding not found: ${TRAIN_EMB}"
    exit 0
fi
VAL_STATUS="ready"
if [ ! -f "$VAL_EMB" ]; then
    VAL_STATUS="missing (will train only)"
fi

TRAIN_CMD="python -m cipher.cli.train_runner \
    --model ${MODEL} \
    --config ${CONFIG_ABS} \
    --positive_list_k ${K_LIST} \
    --positive_list_o ${O_LIST} \
    --embedding_type ${EMB_TYPE} \
    --embedding_file ${TRAIN_EMB} \
    --n_segments ${N_SEGMENTS} \
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
echo \"LA v2 strict: ${NAME}\"
echo \"  K list:    ${K_LIST}\"
echo \"  O list:    ${O_LIST}\"
echo \"  train emb: ${TRAIN_EMB}\"
echo \"  val emb:   ${VAL_EMB}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

VAL_READY=true
if [ ! -f \"${VAL_EMB}\" ]; then
    echo \"WARNING: Validation embeddings not found: ${VAL_EMB}\"
    VAL_READY=false
fi

echo \"\"
echo \"=== TRAINING ===\"
${TRAIN_CMD}

if [ \"\${VAL_READY}\" = true ]; then
    echo \"\"
    echo \"=== EVALUATING ===\"
    ${EVAL_CMD}
fi

echo \"\"
echo \"======================================\"
echo \"Done: ${NAME} at \$(date)\"
echo \"Next: bash scripts/analysis/run_eval_per_head.sh \${EXP_DIR}\"
echo \"      bash scripts/analysis/run_score_norm_raw.sh \${EXP_DIR}\"
echo \"======================================\"
"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] ${NAME}"
    echo "  val: ${VAL_STATUS}"
    echo "---"
    echo "$JOB_SCRIPT"
else
    mkdir -p "${CIPHER_DIR}/logs"
    JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
    echo "Submitted ${JOB_ID} — ${NAME} (val: ${VAL_STATUS})"
    echo ""
    echo "Baselines to beat (PHL rh@1 / PBIP rh@1):"
    echo "  LA v1 highconf prott5_xl_seg8, K-only: 0.235 / 0.890  (our previous best)"
    echo "  attention_mlp v2_strict, default eval: 0.144 / 0.740  (agent 1 last night)"
    echo "  target for 'win':                       ≥ 0.30 / —    (K-only PHL)"
fi
