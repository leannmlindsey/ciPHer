#!/usr/bin/env bash
#
# LA cap sweep on the v3_uat baseline. Trains la_v3_uat_prott5_xl_seg8 with
# `--max_samples_per_k ∈ {300, 500, 750}` (current production cap is 1000).
# Each variant runs the standard 4-way eval (default / k_only / o_only / raw)
# AND the per_head_strict_eval that populates harvest any-hit columns.
#
# Why: per leann's 2026-04-28-2320 ask. Earlier same day the team confirmed
# `nocap` is worse than cap=1000 on multiple architectures (LA included —
# see harvest row `v3uat_nocap_light_attention_prott5_xl_seg8`). This sweep
# tests whether cap < 1000 helps further; the attention_mlp side already
# produced `sweep_prott5_mean_cl70_cap300` and
# `concat_prott5_mean+kmer_li10_k5_cap300` which are competitive with their
# uncapped baselines on PHL_OR.
#
# All inputs mirror la_v3_uat_prott5_xl_seg8 exactly except --max_samples_per_k.
# Output runs land at `experiments/light_attention/la_v3_uat_prott5_xl_seg8_cap${CAP}`.
#
# Usage:
#   bash scripts/run_light_attention_cap_sweep.sh                     # all 3 (300, 500, 750)
#   bash scripts/run_light_attention_cap_sweep.sh 300                 # one cap
#   CAPS="300 500" bash scripts/run_light_attention_cap_sweep.sh      # custom subset
#   DRY_RUN=1 bash scripts/run_light_attention_cap_sweep.sh           # render but don't submit

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

HC_V3_DIR="${DATA_DIR}/training_data/metadata/highconf_v3_multitop"
K_LIST="${HC_V3_DIR}/HC_K_UAT_multitop.list"
O_LIST="${HC_V3_DIR}/HC_O_UAT_multitop.list"

# Match v3_uat exactly: ProtT5-XL seg8.
TRAIN_EMB="/work/hdd/bfzj/llindsey1/embeddings/prott5_xl_segments8/candidates_prott5_xl_segments8_md5.npz"
VAL_EMB="/work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments8/validation_prott5_xl_segments8_md5.npz"

CONFIG_ABS="${CIPHER_DIR}/models/light_attention/highconf_config.yaml"

MODEL="light_attention"
EMB_TYPE="prott5_xl_seg8"
N_SEGMENTS=8
BASE_NAME="la_v3_uat_prott5_xl_seg8"

# Match v3_uat resource allocation: 24h walltime, 1 GPU, 8 CPUs, 160G.
GPUS=1
CPUS=8
MEM="160G"
TIME="24:00:00"

# ============================================================
# Cap values to sweep
# ============================================================
CAPS_DEFAULT="300 500 750"
if [ -n "${1:-}" ]; then
    CAPS="$1"
elif [ -n "${CAPS:-}" ]; then
    CAPS="${CAPS}"
else
    CAPS="${CAPS_DEFAULT}"
fi

DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "LA v3_uat CAP SWEEP"
echo "  Base name:  ${BASE_NAME}"
echo "  Caps:       ${CAPS}"
echo "  Embedding:  ${EMB_TYPE}  (n_segments=${N_SEGMENTS})"
echo "  Config:     ${CONFIG_ABS}"
echo "  Cipher dir: ${CIPHER_DIR}"
echo "  Data dir:   ${DATA_DIR}"
echo "  K list:     ${K_LIST}"
echo "  O list:     ${O_LIST}"
echo "  SLURM:      ${TIME}, ${GPUS} GPU, ${CPUS} CPU, ${MEM}"
echo "============================================================"
echo ""

if [ ! -f "$CONFIG_ABS" ]; then
    echo "ERROR: config not found: $CONFIG_ABS" >&2; exit 1
fi
if [ ! -f "$K_LIST" ] || [ ! -f "$O_LIST" ]; then
    echo "ERROR: K/O list not found:" >&2
    echo "  K: $K_LIST"  >&2
    echo "  O: $O_LIST"  >&2
    exit 1
fi

mkdir -p "${CIPHER_DIR}/logs"

N_SUBMITTED=0

for CAP in $CAPS; do
    NAME="${BASE_NAME}_cap${CAP}"

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
        --max_samples_per_k ${CAP} \
        --name ${NAME}"

    EXP_DIR="${CIPHER_DIR}/experiments/${MODEL}/${NAME}"
    EVAL_CMD="python -m cipher.evaluation.runner ${EXP_DIR} --val-embedding-file ${VAL_EMB}"
    EVAL_K_CMD="python -m cipher.evaluation.runner ${EXP_DIR} --val-embedding-file ${VAL_EMB} --head-mode k_only -o ${EXP_DIR}/results/evaluation_k_only.json"
    EVAL_O_CMD="python -m cipher.evaluation.runner ${EXP_DIR} --val-embedding-file ${VAL_EMB} --head-mode o_only -o ${EXP_DIR}/results/evaluation_o_only.json"
    EVAL_RAW_CMD="python -m cipher.evaluation.runner ${EXP_DIR} --val-embedding-file ${VAL_EMB} --score-norm raw -o ${EXP_DIR}/results/evaluation_raw.json"
    STRICT_EVAL_CMD="python ${CIPHER_DIR}/scripts/analysis/per_head_strict_eval.py ${EXP_DIR} \
        --val-embedding-file ${VAL_EMB} \
        --val-fasta ${VAL_FASTA} \
        --val-datasets-dir ${VAL_DATASETS_DIR}"

    JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${NAME:0:32}
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
echo \"LA cap sweep: ${NAME}  (max_samples_per_k=${CAP})\"
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
    echo \"=== EVAL: default (zscore combined) ===\"
    ${EVAL_CMD}

    echo \"\"
    echo \"=== EVAL: K-only ===\"
    ${EVAL_K_CMD}

    echo \"\"
    echo \"=== EVAL: O-only ===\"
    ${EVAL_O_CMD}

    echo \"\"
    echo \"=== EVAL: raw combined ===\"
    ${EVAL_RAW_CMD}

    echo \"\"
    echo \"=== EVAL: per_head_strict_eval (any-hit harvest columns) ===\"
    ${STRICT_EVAL_CMD}
fi

echo \"\"
echo \"======================================\"
echo \"Done: ${NAME} at \$(date)\"
echo \"Eval JSONs saved to \${EXP_DIR}/results/:\"
echo \"  evaluation.json              — default (zscore combined)\"
echo \"  evaluation_k_only.json       — --head-mode k_only\"
echo \"  evaluation_o_only.json       — --head-mode o_only\"
echo \"  evaluation_raw.json          — --score-norm raw\"
echo \"  per_head_strict_eval.json    — any-hit + per-pair, fixed denom\"
echo \"======================================\"
"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] ${NAME} (cap=${CAP})"
        echo ""
    else
        JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
        echo "  Submitted ${JOB_ID} — ${NAME} (cap=${CAP})"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    fi
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN complete. Set DRY_RUN=0 to submit."
else
    echo "Submitted ${N_SUBMITTED} cap-sweep job(s)."
    echo "Monitor: squeue -u \$USER"
    echo "Once they finish, refresh harvest in main ciPHer worktree:"
    echo "  python scripts/analysis/harvest_results.py --experiments-dirs \\"
    echo "    experiments \\"
    echo "    ../cipher-light-attention/experiments \\"
    echo "    ../cipher-light-attention-binary/experiments"
fi
echo "============================================================"
