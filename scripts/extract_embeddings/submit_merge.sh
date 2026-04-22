#!/usr/bin/env bash
#
# Submit a streaming NPZ merge as a SLURM batch job.
#
# Use this when a merge is too big or slow for an interactive session.
# The merge step itself is CPU + I/O bound (no GPU compute), but Delta-AI
# requires every job to request at least one GPU, so we ask for 1.
#
# Defaults target the ProtT5-XL full per-residue merge; override the
# INPUT / OUTPUT env vars for any other merge.
#
# Usage:
#   bash scripts/extract_embeddings/submit_merge.sh
#   DRY_RUN=1 bash scripts/extract_embeddings/submit_merge.sh
#   COMPRESS=1 bash scripts/extract_embeddings/submit_merge.sh   # zlib-deflate output (slow)
#
#   INPUT=/work/hdd/bfzj/llindsey1/embeddings_full/split_embeddings \
#   OUTPUT=/work/hdd/bfzj/llindsey1/embeddings_full/candidates_embeddings_full_md5.npz \
#       bash scripts/extract_embeddings/submit_merge.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

INPUT="${INPUT:-/work/hdd/bfzj/llindsey1/prott5_xl_full/split_embeddings}"
OUTPUT="${OUTPUT:-/work/hdd/bfzj/llindsey1/prott5_xl_full/candidates_prott5_xl_full_md5.npz}"

COMPRESS="${COMPRESS:-0}"
NO_COMPRESS_FLAG=""
if [ "$COMPRESS" = "0" ]; then
    NO_COMPRESS_FLAG="--no-compress"
fi

GPUS=1
CPUS=4
MEM="16G"
TIME="2:00:00"

NAME="merge_$(basename "${OUTPUT%.npz}")"
LOG="${CIPHER_DIR}/logs/${NAME}_%j.log"

DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "SUBMIT STREAMING MERGE AS SLURM BATCH JOB"
echo "  input dir:  ${INPUT}"
echo "  output:     ${OUTPUT}"
echo "  compress:   ${COMPRESS}  (1=zlib, 0=uncompressed / faster)"
echo "============================================================"
echo ""

JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG}
#SBATCH --error=${LOG}

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

echo \"======================================\"
echo \"Streaming merge: ${NAME}\"
echo \"  input:  ${INPUT}\"
echo \"  output: ${OUTPUT}\"
echo \"  compress: ${COMPRESS}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

# Remove any partial output from a prior interrupted run — zip files can't
# be resumed, only restarted from scratch.
if [ -e \"${OUTPUT}\" ]; then
    echo \"Removing prior (possibly partial) output: ${OUTPUT}\"
    rm -f \"${OUTPUT}\"
fi

python ${CIPHER_DIR}/scripts/extract_embeddings/merge_split_embeddings.py \\
    ${NO_COMPRESS_FLAG} \\
    -i ${INPUT} \\
    -o ${OUTPUT}

echo \"======================================\"
echo \"Done: ${NAME} at \$(date)\"
echo \"Output size: \$(du -h ${OUTPUT} | cut -f1)\"
echo \"======================================\"
"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would submit: ${NAME}"
    echo "  command:"
    echo "    python scripts/extract_embeddings/merge_split_embeddings.py ${NO_COMPRESS_FLAG} \\"
    echo "      -i ${INPUT} -o ${OUTPUT}"
else
    mkdir -p "${CIPHER_DIR}/logs"
    JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
    echo "Submitted ${JOB_ID} - ${NAME}"
fi
