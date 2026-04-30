#!/usr/bin/env bash
#
# Submit one Delta SLURM job per run to emit the per-phage TSV
# (k_only_rank, o_only_rank, ...) for hybrid-OR analysis. Uses
# scripts/analysis/per_head_strict_eval.py with --per-phage-out.
#
# Defaults to the top K-head and top O-head candidates by PHL/overall
# leaderboard (2026-04-30 harvest refresh). Edit RUNS to extend.
#
# Usage:
#   bash scripts/run_per_phage_extract.sh                    # all defaults
#   bash scripts/run_per_phage_extract.sh sweep_kmer_aa20_k4 # one run
#   DRY_RUN=1 bash scripts/run_per_phage_extract.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"
DATA_DIR="${DATA_DIR:-${CIPHER_DIR}/data}"
GPUS=1; CPUS=4; MEM="32G"; TIME="00:30:00"
LOG_DIR="${CIPHER_DIR}/scripts/_logs/per_phage_extract"
mkdir -p "$LOG_DIR"

VAL_DATASETS_DIR="${DATA_DIR}/validation_data/HOST_RANGE"
OUT_DIR="${CIPHER_DIR}/results/analysis/per_phage"

# Default candidates: top 3 K-heads + top 3 O-heads from PHL leaderboard.
# Edit to add/remove. Skip any that already have a per-phage TSV.
RUNS_DEFAULT=(
    # Top K-head sources (PHL K-only HR@1)
    "attention_mlp/sweep_posList_esm2_650m_seg4_cl70"
    "attention_mlp/sweep_posList_esm2_3b_mean_cl70"
    "attention_mlp/sweep_prott5_mean_cl70_dropout0.2"
    # Top O-head sources (PHL O-only HR@1)
    "attention_mlp/sweep_kmer_aa20_k4"
    "attention_mlp/concat_prott5_mean+kmer_li10_k5_cap300"
)

if [[ $# -gt 0 ]]; then
    RUNS=()
    for arg in "$@"; do
        # Allow either "arch/run" or just "run" (default arch=attention_mlp)
        if [[ "$arg" == */* ]]; then RUNS+=("$arg")
        else RUNS+=("attention_mlp/$arg"); fi
    done
else
    RUNS=("${RUNS_DEFAULT[@]}")
fi

for ARCH_RUN in "${RUNS[@]}"; do
    ARCH="${ARCH_RUN%%/*}"
    RUN_NAME="${ARCH_RUN##*/}"
    EXP_DIR="${CIPHER_DIR}/experiments/${ARCH}/${RUN_NAME}"
    OUT_TSV="${OUT_DIR}/per_phage_${RUN_NAME}.tsv"

    if [[ -f "$OUT_TSV" ]]; then
        echo "skip ${RUN_NAME}: ${OUT_TSV} already exists"
        continue
    fi
    if [[ ! -d "$EXP_DIR" ]]; then
        echo "skip ${RUN_NAME}: experiment dir not found at ${EXP_DIR}"
        continue
    fi

    JOB_NAME="ppx_${ARCH}_${RUN_NAME//[\/+]/_}"
    CMD="
        source \$(conda info --base)/etc/profile.d/conda.sh
        conda activate ${CONDA_ENV}
        cd ${CIPHER_DIR}
        export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}
        mkdir -p ${OUT_DIR}
        python scripts/analysis/per_head_strict_eval.py \
            ${EXP_DIR} \
            --val-datasets-dir ${VAL_DATASETS_DIR} \
            --per-phage-out ${OUT_TSV}
    "

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY: sbatch --job-name=${JOB_NAME} (writes ${OUT_TSV})"
    else
        sbatch \
            --account="$ACCOUNT" --partition="$PARTITION" \
            --gpus-per-node=$GPUS --cpus-per-task=$CPUS \
            --mem="$MEM" --time="$TIME" \
            --job-name="$JOB_NAME" \
            --output="${LOG_DIR}/${JOB_NAME}.%j.log" \
            --wrap="$CMD"
    fi
done
