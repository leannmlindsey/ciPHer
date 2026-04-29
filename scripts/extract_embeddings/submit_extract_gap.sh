#!/usr/bin/env bash
#
# Submit a targeted ESM-2 extraction for the "gap" proteins in the
# maxlen1024 bin — i.e., the ~893 proteins that were processed in the
# checkpoint run but never made it into the final NPZ due to the disk
# quota crash.
#
# Prerequisite: you've already run
#     python scripts/utils/find_missing_md5s.py \
#         --fasta <split_fasta>/candidates_maxlen1024.faa \
#         --npz   <split_embeddings>/candidates_maxlen1024_embeddings.npz \
#         --out   <split_fasta>/candidates_maxlen1024_gap.faa
# so that `candidates_maxlen1024_gap.faa` exists and contains only the
# missing sequences.
#
# This job extracts embeddings for those sequences into a SMALL NPZ in
# the same split_embeddings directory. The naming is important: it
# matches the pattern the merge script globs on, so running
# `submit_esm2_full_train_merge.sh` afterwards will merge all 7 bins
# + this gap NPZ together into the final output.
#
# Usage:
#   bash scripts/extract_embeddings/submit_extract_gap.sh
#   DRY_RUN=1 bash scripts/extract_embeddings/submit_extract_gap.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

ROOT="${ROOT:-/work/hdd/bfzj/llindsey1/embeddings_full}"
SPLIT_FASTA_DIR="${ROOT}/split_fasta"
SPLIT_EMB_DIR="${ROOT}/split_embeddings"
GAP_FASTA="${GAP_FASTA:-${SPLIT_FASTA_DIR}/candidates_maxlen1024_gap.faa}"
GAP_NPZ="${GAP_NPZ:-${SPLIT_EMB_DIR}/candidates_maxlen1024_gap_embeddings.npz}"

# Must match the extraction settings used for the original maxlen1024 bin.
MODEL="${MODEL:-esm2_t33_650M_UR50D}"
LAYER="${LAYER:-33}"
POOLING="${POOLING:-full}"
MAX_LEN=1024
# ESM tokenizer reserves cls/eos, so pad by +2 (same convention used
# by resubmit_missing_bins.sh).
TOK_MAX=$((MAX_LEN + 2))

GPUS=1
CPUS=8
MEM="32G"
TIME="2:00:00"

DRY_RUN="${DRY_RUN:-0}"

NAME="esm2_gap_maxlen1024_full"
LOG="${CIPHER_DIR}/logs/${NAME}_%j.log"

echo "============================================================"
echo "ESM-2 maxlen1024 GAP EXTRACTION"
echo "  model:     ${MODEL}  layer=${LAYER}  pooling=${POOLING}"
echo "  gap fasta: ${GAP_FASTA}"
echo "  gap npz:   ${GAP_NPZ}"
echo "============================================================"

if [ ! -f "$GAP_FASTA" ]; then
    echo "ERROR: gap FASTA not found: ${GAP_FASTA}"
    echo "  Produce it first with:"
    echo "    python scripts/utils/find_missing_md5s.py \\"
    echo "        --fasta ${SPLIT_FASTA_DIR}/candidates_maxlen1024.faa \\"
    echo "        --npz   ${SPLIT_EMB_DIR}/candidates_maxlen1024_embeddings.npz \\"
    echo "        --out   ${GAP_FASTA}"
    exit 1
fi

N_SEQS=$(grep -c '^>' "$GAP_FASTA" || true)
echo "  gap n_seqs: ${N_SEQS}"
echo ""

if [ -s "$GAP_NPZ" ]; then
    echo "  Gap NPZ already exists at ${GAP_NPZ}"
    echo "  Remove it if you want to re-extract."
    exit 0
fi

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
export TORCH_HOME=/projects/bfzj/llindsey1/RBP_Structural_Similarity/models
export HF_HOME=\${TORCH_HOME}

echo \"======================================\"
echo \"Gap extraction: ${NAME}\"
echo \"  fasta:   ${GAP_FASTA}\"
echo \"  output:  ${GAP_NPZ}\"
echo \"  seqs:    ${N_SEQS}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

python ${CIPHER_DIR}/scripts/extract_embeddings/esm2_extract.py \\
    ${GAP_FASTA} ${GAP_NPZ} \\
    --model ${MODEL} --layer ${LAYER} \\
    --max_length ${TOK_MAX} \\
    --pooling ${POOLING} \\
    --key_by_md5

echo \"======================================\"
echo \"Done: ${NAME} at \$(date)\"
echo \"Output size: \$(du -h ${GAP_NPZ} | cut -f1)\"
echo \"\"
echo \"Next step — run the delete-as-you-go merge:\"
echo \"  SKIP_RECOMPRESS=1 bash scripts/extract_embeddings/submit_esm2_full_train_merge.sh\"
echo \"======================================\"
"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would submit: ${NAME}"
    echo "  gap fasta: ${GAP_FASTA}  (${N_SEQS} seqs)"
    echo "  gap npz:   ${GAP_NPZ}"
else
    mkdir -p "${CIPHER_DIR}/logs"
    JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
    echo "Submitted ${JOB_ID} - ${NAME}"
fi
