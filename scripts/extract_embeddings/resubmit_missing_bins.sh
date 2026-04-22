#!/usr/bin/env bash
#
# Submit ESM-2 per-residue extraction jobs ONLY for length bins whose
# NPZ is missing from <root>/split_embeddings/. Input FASTAs are read
# from <root>/split_fasta/. Intended for recovering from a partial
# length-binned extraction without re-running bins that already completed.
#
# Defaults target the existing ESM-2 650M full extraction at
# /work/hdd/bfzj/llindsey1/embeddings_full/. Override --root to point
# elsewhere.
#
# Usage:
#   bash scripts/extract_embeddings/resubmit_missing_bins.sh
#   DRY_RUN=1 bash scripts/extract_embeddings/resubmit_missing_bins.sh
#
# After all submitted jobs finish, re-merge with:
#   python scripts/extract_embeddings/merge_split_embeddings.py \
#       -i <root>/split_embeddings \
#       -o <root>/candidates_embeddings_full_md5.npz

set -euo pipefail

ROOT="${ROOT:-/work/hdd/bfzj/llindsey1/embeddings_full}"
MODEL="${MODEL:-esm2_t33_650M_UR50D}"
LAYER="${LAYER:-33}"
POOLING="${POOLING:-full}"

ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="/projects/bfzj/llindsey1/PHI_TSP/ciPHer"

SPLIT_FASTA="${ROOT}/split_fasta"
SPLIT_EMB="${ROOT}/split_embeddings"
EXTRACT_PY="${CIPHER_DIR}/scripts/extract_embeddings/esm2_extract.py"

DRY_RUN="${DRY_RUN:-0}"

if [ ! -d "$SPLIT_FASTA" ]; then
    echo "ERROR: ${SPLIT_FASTA} does not exist" >&2
    exit 1
fi
mkdir -p "$SPLIT_EMB"

echo "============================================================"
echo "RESUBMIT MISSING LENGTH BINS"
echo "  root:       ${ROOT}"
echo "  model:      ${MODEL}  layer=${LAYER}  pooling=${POOLING}"
echo "  split_fasta: ${SPLIT_FASTA}"
echo "  split_embeddings: ${SPLIT_EMB}"
echo "============================================================"
echo ""

N_SUBMITTED=0
for fasta in "${SPLIT_FASTA}"/*.faa; do
    [ -f "$fasta" ] || continue
    basename=$(basename "$fasta" .faa)
    npz="${SPLIT_EMB}/${basename}_embeddings.npz"

    # Skip if NPZ already exists and is non-trivial
    if [ -s "$npz" ]; then
        echo "  SKIP ${basename} — NPZ already present ($(du -h "$npz" | cut -f1))"
        continue
    fi

    # Extract max_length from filename (candidates_maxlen1024 -> 1024)
    MAX_LEN=$(echo "$basename" | grep -o 'maxlen[0-9]*' | grep -o '[0-9]*' || true)
    if [ -z "$MAX_LEN" ]; then
        echo "  WARNING: cannot parse maxlen from ${basename}; skipping"
        continue
    fi

    N_SEQS=$(grep -c "^>" "$fasta" || true)

    # Memory / time scale with bin size; larger bins need more
    if [ "$MAX_LEN" -le 512 ]; then
        MEM="64G"; TIME="8:00:00"
    elif [ "$MAX_LEN" -le 1024 ]; then
        MEM="128G"; TIME="18:00:00"
    elif [ "$MAX_LEN" -le 2048 ]; then
        MEM="192G"; TIME="24:00:00"
    else
        MEM="192G"; TIME="24:00:00"
    fi

    # ESM tokenizer reserves CLS/EOS, so pad --max_length by +2.
    TOK_MAX=$((MAX_LEN + 2))

    JOB_NAME="esm2_${MAX_LEN}_${POOLING}"
    LOG="${CIPHER_DIR}/logs/${JOB_NAME}_%j.log"

    JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
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
echo \"Resubmit bin: ${basename}\"
echo \"  FASTA:  ${fasta}\"
echo \"  Output: ${npz}\"
echo \"  Model:  ${MODEL}  layer=${LAYER}  pooling=${POOLING}\"
echo \"  Seqs:   ${N_SEQS}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

python ${EXTRACT_PY} \\
    ${fasta} ${npz} \\
    --model ${MODEL} --layer ${LAYER} \\
    --max_length ${TOK_MAX} \\
    --pooling ${POOLING} \\
    --key_by_md5

echo \"======================================\"
echo \"Done: ${basename} at \$(date)\"
echo \"======================================\"
"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] ${basename}  (seqs=${N_SEQS}, mem=${MEM}, time=${TIME})"
    else
        mkdir -p "${CIPHER_DIR}/logs"
        JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
        echo "  Submitted ${JOB_ID} — ${basename}  (seqs=${N_SEQS}, mem=${MEM}, time=${TIME})"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    fi
done

echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN complete."
else
    echo "Submitted ${N_SUBMITTED} missing-bin extraction jobs."
    echo ""
    echo "After all complete, re-merge with:"
    echo "  python scripts/extract_embeddings/merge_split_embeddings.py \\"
    echo "      -i ${SPLIT_EMB} \\"
    echo "      -o ${ROOT}/candidates_embeddings_full_md5.npz"
fi
