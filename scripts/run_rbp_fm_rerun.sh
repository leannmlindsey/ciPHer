#!/usr/bin/env bash
#
# Re-run per_head_strict_eval on the top attention_mlp models with the
# eval RBP filter MATCHED to each model's own training filter — leann's
# 2026-05-04 2330 broadcast Ask 1.
#
# Outputs use the `_rbp_fm` suffix (rbp_filter_matched) so they coexist
# with the existing strict-eval JSONs:
#   <exp>/results/per_head_strict_eval_rbp_fm.json
#   results/analysis/per_phage/per_phage_<exp>_rbp_fm.tsv
#
# Models covered = top 3 single MLP runs + every K-source contributing to
# the current top-3 hybrids:
#   1. sweep_kmer_aa20_k4                    (best single, top hybrid O-source)
#   2. sweep_posList_esm2_3b_mean_cl70       (top hybrid #1 K-source)
#   3. sweep_posList_esm2_650m_seg4_cl70     (top hybrid #2 K-source)
#   4. sweep_prott5_mean_cl70                (top hybrid #3 K-source)
#
# Usage:
#   bash scripts/run_rbp_fm_rerun.sh                    # all 4 models
#   MODELS="sweep_kmer_aa20_k4" bash scripts/run_rbp_fm_rerun.sh
#   DRY_RUN=1 bash scripts/run_rbp_fm_rerun.sh
#

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

GLYCAN_BINDERS="${GLYCAN_BINDERS:-/projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/validation_data/combined/validation_inputs/glycan_binders_custom.tsv}"

MODELS="${MODELS:-sweep_kmer_aa20_k4 sweep_posList_esm2_3b_mean_cl70 sweep_posList_esm2_650m_seg4_cl70 sweep_prott5_mean_cl70}"

LOG_DIR="${CIPHER_DIR}/scripts/_logs/rbp_fm"
mkdir -p "$LOG_DIR"

DRY_RUN="${DRY_RUN:-0}"

# Pre-flight check
if [[ ! -f "$GLYCAN_BINDERS" ]]; then
    echo "ERROR: glycan_binders TSV not found at $GLYCAN_BINDERS" >&2
    exit 1
fi
echo "Validation glycan_binders: $GLYCAN_BINDERS ($(wc -l < $GLYCAN_BINDERS) lines)"

submit_one() {
    local EXP_NAME="$1"
    local EXP_DIR="${CIPHER_DIR}/experiments/attention_mlp/${EXP_NAME}"
    if [[ ! -d "$EXP_DIR" ]]; then
        echo "WARNING: experiment dir missing: $EXP_DIR — skipping $EXP_NAME" >&2
        return
    fi

    local OUT_JSON="${EXP_DIR}/results/per_head_strict_eval_rbp_fm.json"
    local PER_PHAGE="${CIPHER_DIR}/results/analysis/per_phage/per_phage_${EXP_NAME}_rbp_fm.tsv"
    local JOB_NAME="rbp_fm_${EXP_NAME}"
    mkdir -p "$(dirname "$PER_PHAGE")"

    local CMD="
        source \$(conda info --base)/etc/profile.d/conda.sh
        conda activate ${CONDA_ENV}
        cd ${CIPHER_DIR}
        export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}
        python ${CIPHER_DIR}/scripts/analysis/per_head_strict_eval.py ${EXP_DIR} \
            --glycan-binders ${GLYCAN_BINDERS} \
            --out-json ${OUT_JSON} \
            --per-phage-out ${PER_PHAGE}
    "

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "DRY [$EXP_NAME]:"
        echo "  exp:    $EXP_DIR"
        echo "  out:    $OUT_JSON"
        echo "  perph:  $PER_PHAGE"
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
