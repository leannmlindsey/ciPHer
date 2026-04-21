#!/usr/bin/env bash
#
# Submit within-class vs between-class cosine analyses for every
# embedding we have. Each job reads one NPZ, computes K-type and O-type
# cosine stats, and appends a row to
#   results/analysis/within_between_summary.tsv
#
# Output is intended for the paper appendix — one comprehensive table
# covering every pLM size / pooling variant / k-mer encoding we've
# extracted.
#
# Usage:
#   bash scripts/analysis/submit_within_between_analyses.sh
#   bash scripts/analysis/submit_within_between_analyses.sh prott5_mean
#   DRY_RUN=1 bash scripts/analysis/submit_within_between_analyses.sh
#
# Missing NPZs (extractions still in progress) are skipped with a warning
# at submit time, not at job start — safe to re-run once more extractions
# land.

set -euo pipefail

ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"   # only needs numpy + cipher package
CIPHER_DIR="/projects/bfzj/llindsey1/PHI_TSP/ciPHer"

# Format: "label  train_npz  mem  time"
# Mem sized to fit the NPZ plus scratch — bigger dim → more mem.
ANALYSES=(
    # ESM-2 family, mean and segmented pooling
    "esm2_150m_mean         /projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_esm2_150m/candidates_embeddings_md5.npz                       32G   1:00:00"
    "esm2_650m_mean         /projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_binned/candidates_embeddings_md5.npz                          32G   1:00:00"
    "esm2_650m_seg4         /work/hdd/bfzj/llindsey1/embeddings_segments4/candidates_embeddings_segments4_md5.npz                                              64G   1:00:00"
    "esm2_650m_seg8         /work/hdd/bfzj/llindsey1/embeddings/esm2_650m_segments8/candidates_esm2_650m_segments8_md5.npz                                     96G   1:00:00"
    "esm2_650m_seg16        /work/hdd/bfzj/llindsey1/embeddings/esm2_650m_segments16/candidates_esm2_650m_segments16_md5.npz                                  128G   2:00:00"
    "esm2_3b_mean           /projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_esm2_3b/candidates_embeddings_md5.npz                         64G   1:00:00"

    # ProtT5 family
    "prott5_mean            /projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_prott5/candidates_embeddings_md5.npz                          32G   1:00:00"
    "prott5_xl_segments4    /work/hdd/bfzj/llindsey1/embeddings/prott5_xl_segments4/candidates_prott5_xl_segments4_md5.npz                                     64G   1:00:00"
    "prott5_xl_segments8    /work/hdd/bfzj/llindsey1/embeddings/prott5_xl_segments8/candidates_prott5_xl_segments8_md5.npz                                     96G   1:00:00"
    "prott5_xl_segments16   /work/hdd/bfzj/llindsey1/embeddings/prott5_xl_segments16/candidates_prott5_xl_segments16_md5.npz                                  128G   2:00:00"
    "prott5_xxl_mean        /work/hdd/bfzj/llindsey1/embeddings/prott5_xxl_mean/candidates_prott5_xxl_mean_md5.npz                                             64G   1:00:00"

    # K-mer features
    "kmer_aa20_k3           /work/hdd/bfzj/llindsey1/kmer_features/candidates_aa20_k3.npz                                                                      32G   1:00:00"
    "kmer_aa20_k4           /work/hdd/bfzj/llindsey1/kmer_features/candidates_aa20_k4.npz                                                                    192G   2:00:00"
    "kmer_murphy8_k5        /work/hdd/bfzj/llindsey1/kmer_features/candidates_murphy8_k5.npz                                                                   64G   1:00:00"
    "kmer_murphy10_k5       /work/hdd/bfzj/llindsey1/kmer_features/candidates_murphy10_k5.npz                                                                 128G   2:00:00"
    "kmer_li10_k5           /work/hdd/bfzj/llindsey1/kmer_features/candidates_li10_k5.npz                                                                    128G   2:00:00"
)

FILTER="${1:-}"
DRY_RUN="${DRY_RUN:-0}"

N=0
N_SKIPPED=0
for entry in "${ANALYSES[@]}"; do
    read -r LABEL TRAIN MEM TIME <<< "$entry"
    [ -n "$FILTER" ] && [ "$LABEL" != "$FILTER" ] && continue

    # Skip if NPZ isn't there yet (ongoing extraction). Safe to re-run.
    if [ ! -f "$TRAIN" ]; then
        echo "  SKIP ${LABEL} — NPZ not found: ${TRAIN}"
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi

    NAME="within_between_${LABEL}"

    JOB="#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${CIPHER_DIR}/logs/${NAME}_%j.log
#SBATCH --error=${CIPHER_DIR}/logs/${NAME}_%j.log

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

echo '======================================'
echo 'within-between cosine: ${LABEL}'
echo '  train: ${TRAIN}'
echo '  Started:' \$(date)
echo '======================================'

python scripts/analysis/within_between_class_cosine.py \\
    --train-emb ${TRAIN} \\
    --label ${LABEL} \\
    --label-type both

echo '======================================'
echo 'Done:' \$(date)
echo '======================================'
"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] ${NAME}  (mem=${MEM}, time=${TIME})"
        N=$((N + 1))
    else
        mkdir -p "${CIPHER_DIR}/logs"
        JOB_ID=$(echo "$JOB" | sbatch | awk '{print $NF}')
        echo "  Submitted ${JOB_ID} — ${NAME}  (mem=${MEM})"
        N=$((N + 1))
    fi
done

echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN: ${N} would submit, ${N_SKIPPED} skipped (NPZ missing)."
else
    echo "Submitted ${N}, skipped ${N_SKIPPED} (NPZ missing)."
fi
