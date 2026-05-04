#!/usr/bin/env bash
#
# Submit Delta SLURM jobs for the H5 (broader-PHL) eval — extract three
# embedding variants on agent 4's broader (~8,223-protein) PHL FASTA so
# we can re-run cipher's eval against the broader set instead of the
# strict 8/8 tail.
#
# Embeddings extracted (matches agent 4's spec in 2026-05-03-0507):
#   1. ProtT5-XL segments8
#   2. ESM-2 3B mean
#   3. kmer aa20_k4
#
# Output NPZs live alongside the existing 589-key validation NPZs with
# a `_broad_phl` suffix so the strict-set NPZs are not overwritten.
#
# Usage:
#   bash scripts/run_h5_broader_phl_extractions.sh                     # all three
#   WHICH="prott5 esm2"     bash scripts/run_h5_broader_phl_extractions.sh
#   WHICH=kmer DRY_RUN=1    bash scripts/run_h5_broader_phl_extractions.sh
#
# Pre-flight (caller's responsibility — the script will fail early if
# these are missing):
#   - /projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/PI_INFO/
#         phagehostlearn_phold_aa.fasta            (8,223 records)
#   - /projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/PI_INFO/
#         phagehostlearn_phold_per_cds_predictions.tsv
#

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

BROAD_FASTA="${BROAD_FASTA:-/projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/PI_INFO/phagehostlearn_phold_aa.fasta}"

KLEB_DIR="/projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella"
KMER_EXTRACTOR="${KLEB_DIR}/scripts/data_prep/compute_kmer_features.py"

OUT_PROTT5="/work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments8/validation_embeddings_broad_phl_md5.npz"
OUT_ESM2_3B="/work/hdd/bfzj/llindsey1/validation_embeddings_esm2_3b/validation_embeddings_broad_phl_md5.npz"
OUT_KMER="/work/hdd/bfzj/llindsey1/kmer_features/validation_aa20_k4_broad_phl.npz"

CONDA_PROTT5="${CONDA_PROTT5:-prott5}"
CONDA_ESM2="${CONDA_ESM2:-esmfold2}"
CONDA_KMER="${CONDA_KMER:-esmfold2}"

LOG_DIR="${CIPHER_DIR}/scripts/_logs/h5_extractions"
mkdir -p "$LOG_DIR"

WHICH="${WHICH:-prott5 esm2 kmer}"
DRY_RUN="${DRY_RUN:-0}"

# Pre-flight check
if [[ ! -f "$BROAD_FASTA" ]]; then
    echo "ERROR: broader-PHL FASTA not found at $BROAD_FASTA" >&2
    echo "  scp it from your laptop first." >&2
    exit 1
fi
N_SEQS=$(grep -c '^>' "$BROAD_FASTA")
echo "broader PHL FASTA: $BROAD_FASTA  ($N_SEQS sequences)"

submit_prott5() {
    local JOB_NAME="h5_prott5_xl_seg8"
    local OUT_DIR=$(dirname "$OUT_PROTT5")
    mkdir -p "$OUT_DIR"

    local CMD="
        source \$(conda info --base)/etc/profile.d/conda.sh
        conda activate ${CONDA_PROTT5}
        cd ${CIPHER_DIR}
        export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}
        export TORCH_HOME=/projects/bfzj/llindsey1/RBP_Structural_Similarity/models
        export HF_HOME=\${TORCH_HOME}
        export TRANSFORMERS_CACHE=\${TORCH_HOME}
        python ${CIPHER_DIR}/scripts/extract_embeddings/prott5_extract.py \
            ${BROAD_FASTA} ${OUT_PROTT5} \
            --model_name Rostlab/prot_t5_xl_uniref50 \
            --pooling segments8 --half_precision --key_by_md5
    "
    submit_one "$JOB_NAME" "48G" "04:00:00" "$CMD"
}

submit_esm2_3b() {
    local JOB_NAME="h5_esm2_3b_mean"
    local OUT_DIR=$(dirname "$OUT_ESM2_3B")
    mkdir -p "$OUT_DIR"

    local CMD="
        source \$(conda info --base)/etc/profile.d/conda.sh
        conda activate ${CONDA_ESM2}
        cd ${CIPHER_DIR}
        export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}
        export TORCH_HOME=/projects/bfzj/llindsey1/RBP_Structural_Similarity/models
        export HF_HOME=\${TORCH_HOME}
        export TRANSFORMERS_CACHE=\${TORCH_HOME}
        python ${CIPHER_DIR}/scripts/extract_embeddings/esm2_extract.py \
            ${BROAD_FASTA} ${OUT_ESM2_3B} \
            --model esm2_t36_3B_UR50D --layer 36 \
            --pooling mean --key_by_md5
    "
    submit_one "$JOB_NAME" "64G" "06:00:00" "$CMD"
}

submit_kmer() {
    local JOB_NAME="h5_kmer_aa20_k4"
    local OUT_DIR=$(dirname "$OUT_KMER")
    mkdir -p "$OUT_DIR"

    local CMD="
        source \$(conda info --base)/etc/profile.d/conda.sh
        conda activate ${CONDA_KMER}
        cd ${KLEB_DIR}
        python ${KMER_EXTRACTOR} \
            --fasta ${BROAD_FASTA} \
            --output ${OUT_KMER} \
            --k 4
    "
    submit_one "$JOB_NAME" "24G" "00:30:00" "$CMD"
}

submit_one() {
    local JOB_NAME="$1"
    local MEM="$2"
    local TIME="$3"
    local CMD="$4"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "DRY [$JOB_NAME mem=$MEM time=$TIME]:"
        echo "$CMD"
        return
    fi

    sbatch \
        --account="$ACCOUNT" --partition="$PARTITION" \
        --gpus-per-node=1 --cpus-per-task=8 --mem="$MEM" --time="$TIME" \
        --job-name="$JOB_NAME" \
        --output="${LOG_DIR}/${JOB_NAME}_%j.log" \
        --wrap="$CMD"
}

for w in $WHICH; do
    case "$w" in
        prott5)  submit_prott5 ;;
        esm2)    submit_esm2_3b ;;
        kmer)    submit_kmer ;;
        *)       echo "WARNING: unknown model key '$w' (expected: prott5|esm2|kmer)" >&2 ;;
    esac
done
