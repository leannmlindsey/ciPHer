#!/usr/bin/env bash
#
# Hybrid OR pipeline — single SLURM job that:
#   1. Runs per-phage strict eval for la_v3_uat_prott5_xl_seg8
#      (cipher-light-attention worktree) — emits per-phage TSV
#   2. Runs per-phage strict eval for sweep_prott5_mean_cl70 (cipher
#      main worktree) — emits per-phage TSV
#   3. Runs cross_model_or_union.py to compute the hybrid OR curves
#      (K from LA + O from MLP) per dataset, k=1..20
#
# Output:
#   results/analysis/per_phage/per_phage_la_v3_uat_prott5_xl_seg8.tsv
#   results/analysis/per_phage/per_phage_sweep_prott5_mean_cl70.tsv
#   results/analysis/hybrid_or_la_K_sweep_O.tsv
#   results/analysis/hybrid_or_la_K_sweep_O_curves.json
#
# Env overrides:
#   ACCOUNT, PARTITION, CONDA_ENV, CIPHER_DIR
#   DRY_RUN=1   render sbatch but do not submit
#
# Usage:
#   bash scripts/run_hybrid_or_pipeline.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"
DRY_RUN="${DRY_RUN:-0}"

LA_EXP="${CIPHER_DIR}/../cipher-light-attention/experiments/light_attention/la_v3_uat_prott5_xl_seg8"
LA_VAL_NPZ="/work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments8/validation_prott5_xl_segments8_md5.npz"

SWEEP_EXP="${CIPHER_DIR}/experiments/attention_mlp/sweep_prott5_mean_cl70"
SWEEP_VAL_NPZ="/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz"

# Third top model — concat (MLP + ProtT5 mean + k-mer features). Needs
# its strict-eval JSON refreshed too (so the host→phage OR ceiling
# field is populated for the 4-panel figure).
CONCAT_EXP="${CIPHER_DIR}/experiments/attention_mlp/concat_prott5_mean+kmer_li10_k5"
CONCAT_VAL_NPZ_1="/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz"
CONCAT_VAL_NPZ_2="/work/hdd/bfzj/llindsey1/kmer_features/validation_li10_k5.npz"

OUT_DIR="${CIPHER_DIR}/results/analysis"
PER_PHAGE_DIR="${OUT_DIR}/per_phage"

mkdir -p "${CIPHER_DIR}/logs"

NAME="hybrid_or_pipeline_$(date +%Y%m%d_%H%M%S)"
SBATCH_FILE="${CIPHER_DIR}/logs/${NAME}.sbatch"
LOG="${CIPHER_DIR}/logs/${NAME}_%j.log"

cat > "$SBATCH_FILE" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=hybrid_or
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=${LOG}
#SBATCH --error=${LOG}

set -euo pipefail
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

mkdir -p ${PER_PHAGE_DIR}

echo "============================================================"
echo "STEP 1 — per-phage strict eval: la_v3_uat_prott5_xl_seg8 (LA, K source)"
echo "============================================================"
python scripts/analysis/per_head_strict_eval.py \\
    "${LA_EXP}" \\
    --val-embedding-file "${LA_VAL_NPZ}" \\
    --per-phage-out "${PER_PHAGE_DIR}/per_phage_la_v3_uat_prott5_xl_seg8.tsv"

echo ""
echo "============================================================"
echo "STEP 2 — per-phage strict eval: sweep_prott5_mean_cl70 (MLP, O source)"
echo "============================================================"
python scripts/analysis/per_head_strict_eval.py \\
    "${SWEEP_EXP}" \\
    --val-embedding-file "${SWEEP_VAL_NPZ}" \\
    --per-phage-out "${PER_PHAGE_DIR}/per_phage_sweep_prott5_mean_cl70.tsv"

echo ""
echo "============================================================"
echo "STEP 2.5 — refresh per_head_strict_eval JSON for concat (top PHL_OR)"
echo "============================================================"
python scripts/analysis/per_head_strict_eval.py \\
    "${CONCAT_EXP}" \\
    --val-embedding-file "${CONCAT_VAL_NPZ_1}" \\
    --val-embedding-file-2 "${CONCAT_VAL_NPZ_2}" \\
    --per-phage-out "${PER_PHAGE_DIR}/per_phage_concat_prott5_mean+kmer_li10_k5.tsv"

echo ""
echo "============================================================"
echo "STEP 3 — cross-model OR union (K from LA + O from MLP)"
echo "============================================================"
python scripts/analysis/cross_model_or_union.py \\
    --k-tsv  "${PER_PHAGE_DIR}/per_phage_la_v3_uat_prott5_xl_seg8.tsv" \\
    --o-tsv  "${PER_PHAGE_DIR}/per_phage_sweep_prott5_mean_cl70.tsv" \\
    --k-label la_v3_uat_prott5_xl_seg8 \\
    --o-label sweep_prott5_mean_cl70 \\
    --out-tsv "${OUT_DIR}/hybrid_or_la_K_sweep_O.tsv" \\
    --out-curves-json "${OUT_DIR}/hybrid_or_la_K_sweep_O_curves.json"

echo ""
echo "============================================================"
echo "Done: \$(date)"
echo "Outputs:"
echo "  ${PER_PHAGE_DIR}/per_phage_la_v3_uat_prott5_xl_seg8.tsv"
echo "  ${PER_PHAGE_DIR}/per_phage_sweep_prott5_mean_cl70.tsv"
echo "  ${OUT_DIR}/hybrid_or_la_K_sweep_O.tsv"
echo "  ${OUT_DIR}/hybrid_or_la_K_sweep_O_curves.json"
echo "============================================================"
SBATCH_EOF

echo "============================================================"
echo "Hybrid OR pipeline"
echo "  Script: ${SBATCH_FILE}"
echo "  Log:    ${LOG}"
echo "============================================================"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] sbatch ${SBATCH_FILE}"
    exit 0
fi

JOB_ID=$(sbatch "${SBATCH_FILE}" | awk '{print $NF}')
echo "Submitted ${JOB_ID}"
echo ""
echo "When done, push the outputs:"
echo "  # cipher main: per-phage TSVs, hybrid curves, refreshed JSONs"
echo "  git add results/analysis/per_phage/ results/analysis/hybrid_or_*"
echo "  git add -f experiments/attention_mlp/sweep_prott5_mean_cl70/results/per_head_strict_eval.json"
echo "  git add -f experiments/attention_mlp/concat_prott5_mean+kmer_li10_k5/results/per_head_strict_eval.json"
echo "  git commit -m 'per-phage data + hybrid OR curves + refreshed strict-eval JSONs'"
echo "  git push"
echo ""
echo "  # LA worktree: refreshed la_v3_uat JSON"
echo "  cd ../cipher-light-attention"
echo "  git add -f experiments/light_attention/la_v3_uat_prott5_xl_seg8/results/per_head_strict_eval.json"
echo "  git commit -m 'refresh strict-eval JSON with host→phage OR field'"
echo "  git push"
