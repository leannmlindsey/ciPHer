#!/usr/bin/env bash
#
# Per-phage top-1 K-class predictions for 3 headline cipher models.
# Generates the per_phage_top1_<run>.tsv files agent 4 asked for in
# 2026-05-04, which the K-label-disagreement table consumes (need
# the model's actual top-1 K to compare against the closest training
# neighbour's K-label).
#
# Models:
#   1. sweep_kmer_aa20_k4              — best single
#   2. sweep_posList_esm2_3b_mean_cl70 — best hybrid K-source
#   3. concat_prott5_mean+kmer_li10_k5 — concat baseline
#
# Output land at:
#   results/analysis/per_phage_top1/per_phage_top1_<run>.tsv
#
# Usage:
#   bash scripts/2026-05-05_dump_top1_3headline_models.sh
#   MODELS="sweep_kmer_aa20_k4" bash scripts/2026-05-05_dump_top1_3headline_models.sh
#   DRY_RUN=1 bash scripts/2026-05-05_dump_top1_3headline_models.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

MODELS="${MODELS:-sweep_kmer_aa20_k4 sweep_posList_esm2_3b_mean_cl70 concat_prott5_mean+kmer_li10_k5}"
OUT_DIR="${CIPHER_DIR}/results/analysis/per_phage_top1"

LOG_DIR="${CIPHER_DIR}/scripts/_logs/per_phage_top1"
mkdir -p "$LOG_DIR" "$OUT_DIR"
DRY_RUN="${DRY_RUN:-0}"

submit_one() {
    local EXP_NAME="$1"
    local EXP_DIR="${CIPHER_DIR}/experiments/attention_mlp/${EXP_NAME}"
    if [[ ! -d "$EXP_DIR" ]]; then
        echo "WARNING: experiment dir missing: $EXP_DIR — skipping" >&2
        return
    fi

    local JOB_NAME="dump_top1_${EXP_NAME}"
    local CMD="
        source \$(conda info --base)/etc/profile.d/conda.sh
        conda activate ${CONDA_ENV}
        cd ${CIPHER_DIR}
        export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}
        python ${CIPHER_DIR}/scripts/analysis/generate_per_phage_top1.py \
            ${EXP_DIR} \
            --out-dir ${OUT_DIR}
    "

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "DRY [$EXP_NAME]: would write ${OUT_DIR}/per_phage_top1_${EXP_NAME}.tsv"
        return
    fi

    sbatch \
        --account="$ACCOUNT" --partition="$PARTITION" \
        --gpus-per-node=1 --cpus-per-task=4 --mem=48G --time=01:00:00 \
        --job-name="$JOB_NAME" \
        --output="${LOG_DIR}/${JOB_NAME}_%j.log" \
        --wrap="$CMD"
}

for m in $MODELS; do
    submit_one "$m"
done
