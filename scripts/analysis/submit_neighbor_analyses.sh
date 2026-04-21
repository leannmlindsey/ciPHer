#!/usr/bin/env bash
#
# Submit PHL nearest-neighbour K-type-agreement analyses on Delta-AI.
# One SLURM job per embedding. Reuses scripts/analysis/phl_neighbor_labels.py.
#
# Outputs land in results/analysis/<label>/ for each embedding.
#
# Usage:
#   bash scripts/analysis/submit_neighbor_analyses.sh            # all rows
#   bash scripts/analysis/submit_neighbor_analyses.sh prott5_mean
#   DRY_RUN=1 bash scripts/analysis/submit_neighbor_analyses.sh
#
# Why SLURM: large kmer embeddings (~22 GB RAM when loaded) don't fit on
# login nodes. CPU-only job; no GPU needed.

set -euo pipefail

ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="/projects/bfzj/llindsey1/PHI_TSP/ciPHer"

# Format: "label  train_npz  val_npz  mem  time"
# Paths match those used by run_embedding_sweep.sh for consistency.
ANALYSES=(
    "prott5_mean      /projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_prott5/candidates_embeddings_md5.npz               /work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz              32G  1:00:00"
    "esm2_650m_seg4   /work/hdd/bfzj/llindsey1/embeddings_segments4/candidates_embeddings_segments4_md5.npz                                   /work/hdd/bfzj/llindsey1/validation_embeddings_segments4/validation_embeddings_segments4_md5.npz  64G  1:00:00"
    "kmer_li10_k5     /work/hdd/bfzj/llindsey1/kmer_features/candidates_li10_k5.npz                                                            /work/hdd/bfzj/llindsey1/kmer_features/validation_li10_k5.npz                                    128G  2:00:00"
)

FILTER="${1:-}"
DRY_RUN="${DRY_RUN:-0}"

for entry in "${ANALYSES[@]}"; do
    read -r LABEL TRAIN VAL MEM TIME <<< "$entry"
    [ -n "$FILTER" ] && [ "$LABEL" != "$FILTER" ] && continue

    NAME="neighbor_${LABEL}"
    OUT="${CIPHER_DIR}/results/analysis/${LABEL}"

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
echo 'PHL neighbour analysis: ${LABEL}'
echo '  train: ${TRAIN}'
echo '  val:   ${VAL}'
echo '  Started:' \$(date)
echo '======================================'

mkdir -p ${OUT}

python scripts/analysis/phl_neighbor_labels.py \\
    --train-emb ${TRAIN} \\
    --val-emb ${VAL} \\
    --out-dir ${OUT} \\
    --restrict-to-labeled

echo '======================================'
echo 'Done:' \$(date)
echo '======================================'
"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY RUN] ${NAME}  (mem=${MEM})"
        echo "   out: ${OUT}"
    else
        mkdir -p "${CIPHER_DIR}/logs"
        JOB_ID=$(echo "$JOB" | sbatch | awk '{print $NF}')
        echo "Submitted ${JOB_ID} — ${NAME}  (mem=${MEM})"
    fi
done
