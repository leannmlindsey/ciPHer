#!/usr/bin/env bash
#
# Submit length-binned full per-residue extractions on the VALIDATION
# FASTA (validation_rbps_all.faa) for ESM-2 650M and/or ProtT5-XL.
#
# Mirrors the training-side submit_prott5_full_binned.sh but:
#   - No filter step: validation isn't filtered to a positive list.
#   - Handles both model families in one script (conda env, extractor,
#     and CLI args differ).
#   - Smaller per-bin resources (validation has ~1.5k proteins, not 59k).
#
# After every per-bin job finishes, run the streaming merge job
# (submit_merge.sh) to produce a single validation NPZ, then verify
# with check_full_npz_coverage.py --expected-dim <D>.
#
# Usage:
#   bash scripts/extract_embeddings/submit_full_validation.sh          # both
#   bash scripts/extract_embeddings/submit_full_validation.sh esm2     # one
#   bash scripts/extract_embeddings/submit_full_validation.sh prott5   # the other
#   DRY_RUN=1 bash scripts/extract_embeddings/submit_full_validation.sh
#
# Env overrides: CIPHER_DIR, VAL_FASTA, ROOT_ESM2, ROOT_PROTT5,
# CONDA_ESM2, CONDA_PROTT5, ACCOUNT, PARTITION.

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"
VAL_FASTA="${VAL_FASTA:-${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa}"
SPLIT_PY="${CIPHER_DIR}/scripts/extract_embeddings/split_fasta_by_length.py"

WHICH="${1:-both}"
DRY_RUN="${DRY_RUN:-0}"

# ------------------------------------------------------------
# Per-bin resource table. Validation bins are <=650 sequences
# (tail bins <100), so requests are much smaller than the
# training-side binned extraction.
# ------------------------------------------------------------
resources_for() {
    # echoes "MEM TIME" for a given max_length
    local L="$1"
    if [ "$L" -le 512 ]; then
        echo "32G 2:00:00"
    elif [ "$L" -le 1024 ]; then
        echo "48G 4:00:00"
    elif [ "$L" -le 2048 ]; then
        echo "64G 6:00:00"
    elif [ "$L" -le 4096 ]; then
        echo "96G 8:00:00"
    else
        echo "128G 12:00:00"
    fi
}

submit_for_model() {
    local MODEL_KEY="$1"  # esm2 | prott5
    local ROOT CONDA_ENV EXTRACTOR MODEL_NAME EXPECTED_DIM

    if [ "$MODEL_KEY" = "esm2" ]; then
        ROOT="${ROOT_ESM2:-/work/hdd/bfzj/llindsey1/val_esm2_650m_full}"
        CONDA_ENV="${CONDA_ESM2:-esmfold2}"
        EXTRACTOR="${CIPHER_DIR}/scripts/extract_embeddings/esm2_extract.py"
        MODEL_NAME="esm2_t33_650M_UR50D"
        EXPECTED_DIM=1280
    elif [ "$MODEL_KEY" = "prott5" ]; then
        ROOT="${ROOT_PROTT5:-/work/hdd/bfzj/llindsey1/val_prott5_xl_full}"
        CONDA_ENV="${CONDA_PROTT5:-prott5}"
        EXTRACTOR="${CIPHER_DIR}/scripts/extract_embeddings/prott5_extract.py"
        MODEL_NAME="Rostlab/prot_t5_xl_uniref50"
        EXPECTED_DIM=1024
    else
        echo "ERROR: unknown model key: ${MODEL_KEY}" >&2
        return 1
    fi

    local SPLIT_FASTA_DIR="${ROOT}/split_fasta"
    local SPLIT_EMB_DIR="${ROOT}/split_embeddings"
    local FINAL_NPZ="${ROOT}/validation_${MODEL_KEY}_full_md5.npz"

    echo ""
    echo "----------------------------------------"
    echo "MODEL: ${MODEL_KEY}"
    echo "  root:           ${ROOT}"
    echo "  conda env:      ${CONDA_ENV}"
    echo "  model name:     ${MODEL_NAME}  dim=${EXPECTED_DIM}"
    echo "  val fasta:      ${VAL_FASTA}"
    echo "  merge target:   ${FINAL_NPZ}"
    echo "----------------------------------------"

    if [ -s "${FINAL_NPZ}" ]; then
        echo "  Final merged NPZ already exists; nothing to do."
        echo "  Remove it to force re-extraction: rm ${FINAL_NPZ}"
        return 0
    fi

    mkdir -p "${ROOT}" "${SPLIT_EMB_DIR}"

    # Step 1: split the validation FASTA (idempotent per model root)
    if ! ls "${SPLIT_FASTA_DIR}"/*_maxlen*.faa >/dev/null 2>&1; then
        echo "  Splitting ${VAL_FASTA} into length bins under ${SPLIT_FASTA_DIR}"
        mkdir -p "${SPLIT_FASTA_DIR}"
        if [ "$DRY_RUN" = "1" ]; then
            echo "  [DRY RUN] would run: python ${SPLIT_PY} ${VAL_FASTA} ${SPLIT_FASTA_DIR}"
        else
            python "${SPLIT_PY}" "${VAL_FASTA}" "${SPLIT_FASTA_DIR}"
        fi
    else
        echo "  Split FASTAs already present - skipping split."
    fi

    # Step 2: submit per-bin jobs (skip bins whose NPZ already exists)
    local N_SUBMITTED=0
    for fasta in "${SPLIT_FASTA_DIR}"/*.faa; do
        [ -f "$fasta" ] || continue
        local bname=$(basename "$fasta" .faa)
        local npz="${SPLIT_EMB_DIR}/${bname}_embeddings.npz"

        if [ -s "$npz" ]; then
            echo "  SKIP ${bname} - NPZ already present ($(du -h "$npz" | cut -f1))"
            continue
        fi

        local MAX_LEN=$(echo "$bname" | grep -o 'maxlen[0-9]*' | grep -o '[0-9]*' || true)
        if [ -z "$MAX_LEN" ]; then
            echo "  WARNING: cannot parse maxlen from ${bname}; skipping"
            continue
        fi
        local N_SEQS=$(grep -c "^>" "$fasta" || true)

        local RES=($(resources_for "$MAX_LEN"))
        local MEM="${RES[0]}"
        local TIME="${RES[1]}"

        local JOB_NAME="val_${MODEL_KEY}_${MAX_LEN}_full"
        local LOG="${CIPHER_DIR}/logs/${JOB_NAME}_%j.log"

        # Build model-specific extractor command
        local EXTRACT_CMD
        if [ "$MODEL_KEY" = "esm2" ]; then
            # ESM tokenizer reserves cls/eos -> pad max_length by +2.
            local TOK_MAX=$((MAX_LEN + 2))
            EXTRACT_CMD="python ${EXTRACTOR} ${fasta} ${npz} \\
    --model ${MODEL_NAME} --layer 33 \\
    --pooling full --key_by_md5 \\
    --max_length ${TOK_MAX}"
        else  # prott5
            EXTRACT_CMD="python ${EXTRACTOR} ${fasta} ${npz} \\
    --model_name ${MODEL_NAME} \\
    --pooling full --half_precision --key_by_md5 \\
    --max_length ${MAX_LEN}"
        fi

        local JOB_SCRIPT="#!/bin/bash
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
export TRANSFORMERS_CACHE=\${TORCH_HOME}

echo \"======================================\"
echo \"Val ${MODEL_KEY} full: ${bname}\"
echo \"  fasta:  ${fasta}\"
echo \"  output: ${npz}\"
echo \"  seqs:   ${N_SEQS}  max_length=${MAX_LEN}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

${EXTRACT_CMD}

echo \"======================================\"
echo \"Done: ${bname} at \$(date)\"
echo \"======================================\"
"

        if [ "$DRY_RUN" = "1" ]; then
            echo "  [DRY RUN] ${JOB_NAME}  (seqs=${N_SEQS}, mem=${MEM}, time=${TIME})"
        else
            mkdir -p "${CIPHER_DIR}/logs"
            local JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
            echo "  Submitted ${JOB_ID} - ${JOB_NAME}  (seqs=${N_SEQS}, mem=${MEM}, time=${TIME})"
            N_SUBMITTED=$((N_SUBMITTED + 1))
        fi
    done

    echo ""
    if [ "$DRY_RUN" != "1" ]; then
        echo "  Submitted ${N_SUBMITTED} ${MODEL_KEY} bin job(s)."
    fi
    echo "  After all bins complete, merge with:"
    echo "    INPUT=${SPLIT_EMB_DIR} OUTPUT=${FINAL_NPZ} \\"
    echo "        bash scripts/extract_embeddings/submit_merge.sh"
    echo "  Then verify with:"
    echo "    python scripts/utils/check_full_npz_coverage.py \\"
    echo "        --npz ${FINAL_NPZ} \\"
    echo "        --fasta ${VAL_FASTA} \\"
    echo "        --positive-list /nonexistent --tsp-K-list /nonexistent --pos-K-list /nonexistent \\"
    echo "        --expected-dim ${EXPECTED_DIM}"
}

echo "============================================================"
echo "VALIDATION FULL EXTRACTIONS (length-binned)"
echo "  validation fasta: ${VAL_FASTA}"
echo "  which model(s):   ${WHICH}"
echo "============================================================"

case "$WHICH" in
    esm2)    submit_for_model esm2 ;;
    prott5)  submit_for_model prott5 ;;
    both)    submit_for_model esm2; submit_for_model prott5 ;;
    *)       echo "ERROR: first arg must be 'esm2' | 'prott5' | 'both'" >&2; exit 1 ;;
esac

echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN complete."
fi
