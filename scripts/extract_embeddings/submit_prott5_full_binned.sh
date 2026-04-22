#!/usr/bin/env bash
#
# End-to-end ProtT5 per-residue extraction pipeline, length-binned.
#
# Pipeline
# --------
#   1. Filter an input FASTA to a protein-ID list (default: pipeline_positive.list)
#   2. Split the filtered FASTA into length bins (128, 256, 512, 1024, 2048, 4096, 8192)
#   3. Submit one SLURM job per bin to prott5_extract.py with
#      --pooling full, --half_precision, and appropriate --max_length
#   4. Print the merge command to run after all jobs finish
#
# The filter + split steps are skipped if the expected outputs already
# exist under $ROOT, so the script is safe to rerun after a partial
# failure (resumes by submitting only bins whose NPZ is missing).
#
# Usage:
#   bash scripts/extract_embeddings/submit_prott5_full_binned.sh
#   DRY_RUN=1 bash scripts/extract_embeddings/submit_prott5_full_binned.sh
#   MODEL=Rostlab/prot_t5_xxl_uniref50 \
#       bash scripts/extract_embeddings/submit_prott5_full_binned.sh

set -euo pipefail

# ============================================================
# Configuration (env-overridable)
# ============================================================
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"
ROOT="${ROOT:-/work/hdd/bfzj/llindsey1/prott5_xl_full}"
INPUT_FASTA="${INPUT_FASTA:-${CIPHER_DIR}/data/training_data/metadata/candidates.faa}"
POSITIVE_LIST="${POSITIVE_LIST:-${CIPHER_DIR}/data/training_data/metadata/pipeline_positive.list}"
MODEL="${MODEL:-Rostlab/prot_t5_xl_uniref50}"
POOLING="${POOLING:-full}"

ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-prott5}"

FILTERED_FASTA="${ROOT}/candidates_positive.faa"
SPLIT_FASTA_DIR="${ROOT}/split_fasta"
SPLIT_EMB_DIR="${ROOT}/split_embeddings"
EXTRACT_PY="${CIPHER_DIR}/scripts/extract_embeddings/prott5_extract.py"
FILTER_PY="${CIPHER_DIR}/scripts/extract_embeddings/filter_fasta.py"
SPLIT_PY="${CIPHER_DIR}/scripts/extract_embeddings/split_fasta_by_length.py"

DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${ROOT}" "${SPLIT_EMB_DIR}"

echo "============================================================"
echo "ProtT5 FULL PER-RESIDUE EXTRACTION (length-binned)"
echo "  model:          ${MODEL}"
echo "  pooling:        ${POOLING}"
echo "  output root:    ${ROOT}"
echo "  input FASTA:    ${INPUT_FASTA}"
echo "  positive list:  ${POSITIVE_LIST}"
echo "============================================================"
echo ""

# ------------------------------------------------------------
# Step 1: filter FASTA to positive list (idempotent)
# ------------------------------------------------------------
if [ ! -s "${FILTERED_FASTA}" ]; then
    echo "Step 1: filtering ${INPUT_FASTA} to ${POSITIVE_LIST}"
    python "${FILTER_PY}" \
        --in "${INPUT_FASTA}" \
        --list "${POSITIVE_LIST}" \
        --out "${FILTERED_FASTA}"
    echo ""
else
    echo "Step 1: filtered FASTA already exists at ${FILTERED_FASTA} — skipping."
    echo ""
fi

# ------------------------------------------------------------
# Step 2: split by length (idempotent)
# ------------------------------------------------------------
if ! ls "${SPLIT_FASTA_DIR}"/*_maxlen*.faa >/dev/null 2>&1; then
    echo "Step 2: splitting ${FILTERED_FASTA} into length bins under ${SPLIT_FASTA_DIR}"
    mkdir -p "${SPLIT_FASTA_DIR}"
    python "${SPLIT_PY}" "${FILTERED_FASTA}" "${SPLIT_FASTA_DIR}"
    echo ""
else
    echo "Step 2: split FASTAs already present under ${SPLIT_FASTA_DIR} — skipping."
    echo ""
fi

# ------------------------------------------------------------
# Step 3: submit one SLURM job per missing bin
# ------------------------------------------------------------
echo "Step 3: submitting extraction jobs for missing bins"
echo ""

N_SUBMITTED=0
for fasta in "${SPLIT_FASTA_DIR}"/*.faa; do
    [ -f "$fasta" ] || continue
    basename=$(basename "$fasta" .faa)
    npz="${SPLIT_EMB_DIR}/${basename}_embeddings.npz"

    if [ -s "$npz" ]; then
        echo "  SKIP ${basename} — NPZ already present ($(du -h "$npz" | cut -f1))"
        continue
    fi

    MAX_LEN=$(echo "$basename" | grep -o 'maxlen[0-9]*' | grep -o '[0-9]*' || true)
    if [ -z "$MAX_LEN" ]; then
        echo "  WARNING: cannot parse maxlen from ${basename}; skipping"
        continue
    fi
    N_SEQS=$(grep -c "^>" "$fasta" || true)

    # Per-bin SLURM resources. ProtT5 at fp16: attention O(L^2), so the
    # longer bins need substantially more VRAM and wall time.
    if [ "$MAX_LEN" -le 512 ]; then
        MEM="64G";  TIME="8:00:00"
    elif [ "$MAX_LEN" -le 1024 ]; then
        MEM="96G";  TIME="18:00:00"
    elif [ "$MAX_LEN" -le 2048 ]; then
        MEM="128G"; TIME="24:00:00"
    elif [ "$MAX_LEN" -le 4096 ]; then
        MEM="192G"; TIME="30:00:00"
    else
        MEM="256G"; TIME="36:00:00"
    fi

    JOB_NAME="prott5_${MAX_LEN}_${POOLING}"
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
export TRANSFORMERS_CACHE=\${TORCH_HOME}

echo \"======================================\"
echo \"ProtT5 full per-residue: ${basename}\"
echo \"  FASTA:  ${fasta}\"
echo \"  Output: ${npz}\"
echo \"  Model:  ${MODEL}  pooling=${POOLING}\"
echo \"  Seqs:   ${N_SEQS}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

python ${EXTRACT_PY} \\
    ${fasta} ${npz} \\
    --model_name ${MODEL} \\
    --pooling ${POOLING} \\
    --half_precision \\
    --max_length ${MAX_LEN} \\
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
    echo "Submitted ${N_SUBMITTED} extraction job(s)."
    echo ""
    echo "After all complete, re-merge with:"
    echo "  python scripts/extract_embeddings/merge_split_embeddings.py \\"
    echo "      -i ${SPLIT_EMB_DIR} \\"
    echo "      -o ${ROOT}/candidates_prott5_xl_full_md5.npz"
fi
