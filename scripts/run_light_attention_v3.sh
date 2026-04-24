#!/usr/bin/env bash
#
# LA + ProtT5-XL seg8 + v3 multi-top highconf. Direct companion to
# scripts/run_v3_experiment.sh (agent 1's attention_mlp + ProtT5 mean + v3
# run) but with LA segmented pooling on our best architecture.
#
# v3 multi-top restores broad-range proteins that v2 dropped at ~10× the
# rate of specific ones (audit A15). Full rationale in
# data/training_data/metadata/highconf_v3_multitop/README.md.
#
# Three variants:
#   strict   — HC_K_cl95_multitop + HC_O_cl95_multitop_full_coverage
#              (35,249 / 35,429 proteins). Direct replacement for v2_strict.
#   uat      — HC_K_UAT_multitop + HC_O_UAT_multitop
#              (37,158 / 37,178 proteins). Maximal coverage.
#   k_v2o    — HC_K_cl95_multitop (v3) + HC_O_cl95_full_coverage (v2).
#              Cross variant testing whether v3 O is genuinely helping or
#              absorbing host-background noise (agent 4's specific question
#              from the v3 README).
#
# Everything else (embedding, LA hyperparameters, val data) mirrors
# la_v2_strict_prott5_xl_seg8 for apples-to-apples comparison against v2.
#
# Baselines (LA + ProtT5 seg8, host HR@1 — K-only inference):
#   la_highconf_pipeline (v1):  see v2 handoff for the numbers we care about
#   la_v2_strict:               see lab_notebook_agent2.txt today's entry
#
# Usage:
#   bash scripts/run_light_attention_v3.sh                 # all three
#   bash scripts/run_light_attention_v3.sh strict          # one variant
#   bash scripts/run_light_attention_v3.sh k_v2o
#   DRY_RUN=1 bash scripts/run_light_attention_v3.sh       # preview

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

HC_V2_DIR="${DATA_DIR}/training_data/metadata/highconf_v2"
HC_V3_DIR="${DATA_DIR}/training_data/metadata/highconf_v3_multitop"

# ProtT5-XL seg8 paths (same as v2; our best LA architecture)
TRAIN_EMB="/work/hdd/bfzj/llindsey1/embeddings/prott5_xl_segments8/candidates_prott5_xl_segments8_md5.npz"
VAL_EMB="/work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments8/validation_prott5_xl_segments8_md5.npz"

CONFIG_ABS="${CIPHER_DIR}/models/light_attention/highconf_config.yaml"

MODEL="light_attention"
EMB_TYPE="prott5_xl_seg8"
N_SEGMENTS=8

# SLURM resources — v3 is ~2.3× v2 scale (35k vs 15k proteins). Mirror
# agent 1's v3 allocation: 160G / 24h.
GPUS=1
CPUS=8
MEM="160G"
TIME="24:00:00"

# ============================================================
# Variant table:
#   "label   K_list_path   O_list_path"
# ============================================================
VARIANTS=(
    "strict  ${HC_V3_DIR}/HC_K_cl95_multitop.list  ${HC_V3_DIR}/HC_O_cl95_multitop_full_coverage.list"
    "uat     ${HC_V3_DIR}/HC_K_UAT_multitop.list   ${HC_V3_DIR}/HC_O_UAT_multitop.list"
    "k_v2o   ${HC_V3_DIR}/HC_K_cl95_multitop.list  ${HC_V2_DIR}/HC_O_cl95_full_coverage.list"
)

FILTER="${1:-}"
DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "LA + PROT T5 seg8 + v3 MULTI-TOP HIGHCONF"
echo "  Model:      ${MODEL}"
echo "  Embedding:  ${EMB_TYPE}  (n_segments=${N_SEGMENTS})"
echo "  Config:     ${CONFIG_ABS}"
echo "  Cipher dir: ${CIPHER_DIR}"
echo "  v3 lists:   ${HC_V3_DIR}/"
echo "  SLURM:      ${TIME}, ${GPUS} GPU, ${CPUS} CPU, ${MEM}"
echo "============================================================"
echo ""

if [ ! -f "$CONFIG_ABS" ]; then
    echo "ERROR: config not found: $CONFIG_ABS" >&2; exit 1
fi

N_SUBMITTED=0
N_SKIPPED=0

for entry in "${VARIANTS[@]}"; do
    read -r LABEL K_LIST O_LIST <<< "$entry"
    if [ -n "$FILTER" ] && [ "$LABEL" != "$FILTER" ]; then
        continue
    fi

    NAME="la_v3_${LABEL}_prott5_xl_seg8"

    if [ ! -f "$K_LIST" ]; then
        echo "  SKIP ${LABEL} — K list not found: ${K_LIST}"
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi
    if [ ! -f "$O_LIST" ]; then
        echo "  SKIP ${LABEL} — O list not found: ${O_LIST}"
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi
    if [ ! -f "$TRAIN_EMB" ]; then
        echo "  SKIP ${LABEL} — training embedding not found: ${TRAIN_EMB}"
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
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
    EVAL_K_CMD="python -m cipher.evaluation.runner ${EXP_DIR} --val-embedding-file ${VAL_EMB} --head-mode k_only -o ${EXP_DIR}/results/evaluation_k_only.json"
    EVAL_O_CMD="python -m cipher.evaluation.runner ${EXP_DIR} --val-embedding-file ${VAL_EMB} --head-mode o_only -o ${EXP_DIR}/results/evaluation_o_only.json"
    EVAL_RAW_CMD="python -m cipher.evaluation.runner ${EXP_DIR} --val-embedding-file ${VAL_EMB} --score-norm raw -o ${EXP_DIR}/results/evaluation_raw.json"

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
echo \"LA v3 ${LABEL}: ${NAME}\"
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
    echo \"=== EVALUATING (default: zscore combined) ===\"
    ${EVAL_CMD}

    echo \"\"
    echo \"=== EVALUATING (K-only via --head-mode k_only) ===\"
    ${EVAL_K_CMD}

    echo \"\"
    echo \"=== EVALUATING (O-only via --head-mode o_only) ===\"
    ${EVAL_O_CMD}

    echo \"\"
    echo \"=== EVALUATING (raw combined via --score-norm raw) ===\"
    ${EVAL_RAW_CMD}
fi

echo \"\"
echo \"======================================\"
echo \"Done: ${NAME} at \$(date)\"
echo \"Eval JSONs saved to \${EXP_DIR}/results/:\"
echo \"  evaluation.json           — default (zscore combined)\"
echo \"  evaluation_k_only.json    — --head-mode k_only\"
echo \"  evaluation_o_only.json    — --head-mode o_only\"
echo \"  evaluation_raw.json       — --score-norm raw\"
echo \"Compare across runs: python3 scripts/analysis/show_eval_all.py [--variant k_only|o_only|raw]\"
echo \"======================================\"
"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] ${NAME}"
        echo "    K list: ${K_LIST}"
        echo "    O list: ${O_LIST}"
        echo "    val: ${VAL_STATUS}"
        echo ""
    else
        mkdir -p "${CIPHER_DIR}/logs"
        JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
        echo "  Submitted ${JOB_ID} — ${NAME} (val: ${VAL_STATUS})"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    fi
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN complete. Set DRY_RUN=0 to submit."
else
    echo "Submitted ${N_SUBMITTED} job(s) (${N_SKIPPED} skipped)."
    echo "Monitor: squeue -u \$USER"
fi
echo "============================================================"
