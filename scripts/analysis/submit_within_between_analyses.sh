#!/usr/bin/env bash
#
# Submit within-class vs between-class cosine analyses for the Delta-resident
# embeddings (ProtT5 mean, ESM-2 650M seg4, kmer li10_k5). Complements the
# esm2_650m_mean run that was done locally.
#
# Each job loads one NPZ, computes cosine stats for same-K-type vs
# different-K-type pairs, and appends a row to
# results/analysis/within_between_summary.tsv.
#
# Usage:
#   bash scripts/analysis/submit_within_between_analyses.sh
#   bash scripts/analysis/submit_within_between_analyses.sh prott5_mean
#   DRY_RUN=1 bash scripts/analysis/submit_within_between_analyses.sh

set -euo pipefail

ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"   # only needs numpy + cipher package
CIPHER_DIR="/projects/bfzj/llindsey1/PHI_TSP/ciPHer"

# Format: "label  train_npz  mem  time"
ANALYSES=(
    "prott5_mean      /projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_prott5/candidates_embeddings_md5.npz                                   32G   1:00:00"
    "esm2_650m_seg4   /work/hdd/bfzj/llindsey1/embeddings_segments4/candidates_embeddings_segments4_md5.npz                                                       64G   1:00:00"
    "kmer_li10_k5     /work/hdd/bfzj/llindsey1/kmer_features/candidates_li10_k5.npz                                                                              128G   2:00:00"
)

FILTER="${1:-}"
DRY_RUN="${DRY_RUN:-0}"

for entry in "${ANALYSES[@]}"; do
    read -r LABEL TRAIN MEM TIME <<< "$entry"
    [ -n "$FILTER" ] && [ "$LABEL" != "$FILTER" ] && continue

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
    --label ${LABEL}

echo '======================================'
echo 'Done:' \$(date)
echo '======================================'
"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY RUN] ${NAME}  (mem=${MEM}, time=${TIME})"
    else
        mkdir -p "${CIPHER_DIR}/logs"
        JOB_ID=$(echo "$JOB" | sbatch | awk '{print $NF}')
        echo "Submitted ${JOB_ID} — ${NAME}  (mem=${MEM})"
    fi
done
