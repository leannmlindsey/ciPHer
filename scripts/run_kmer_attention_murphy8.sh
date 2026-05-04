#!/usr/bin/env bash
#
# Submit Delta SLURM jobs that extract SE-attention weights from
# sweep_kmer_murphy8_k5's K-head — for agent 5's K1-specific motif vs
# control delta analysis (analysis 35-style).
#
# Per agent 5's 2026-05-04 pivot: Murphy8-k5 is the native motif
# alphabet; AA20-k4 splits attention across multiple AA20 instances of
# the same chemistry-class motif. K-head accuracy parity verified
# (Murphy8 PHL=0.098 vs AA20 PHL=0.107; PBIP 0.800 vs 0.767).
#
# After the first round, K30 (intended control) and K3 (parallel test)
# both failed because the Murphy8 K-head correctly predicts neither
# class on any val phage (--require-correct dropped them all). Two
# fallbacks now baked in:
#
#   - "anyhost" mode (--include-all-target-phages): average attention
#     over every K=class-host phage regardless of whether the K-head
#     predicted that class. Output gets `_anyhost` suffix.
#   - K47 as alternate strict control (better-populated K class than
#     K30 in HC_K_cl95 — has more chance of any-correct phages).
#
# JOB FORMAT: each entry is "TARGET:MODE" where MODE = strict | anyhost
#
# Usage:
#   bash scripts/run_kmer_attention_murphy8.sh                   # default JOBS list
#   JOBS="K47:strict K30:anyhost" bash scripts/run_kmer_attention_murphy8.sh
#   DRY_RUN=1 bash scripts/run_kmer_attention_murphy8.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

# Default: K47 strict control + K30/K3 anyhost fallbacks. K1 not re-run
# (already landed on first batch).
JOBS="${JOBS:-K47:strict K30:anyhost K3:anyhost}"

EXP_DIR="${CIPHER_DIR}/experiments/attention_mlp/sweep_kmer_murphy8_k5"
VAL_FASTA="${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa"
VAL_EMB="/work/hdd/bfzj/llindsey1/kmer_features/validation_murphy8_k5.npz"
VAL_DS="${CIPHER_DIR}/data/validation_data/HOST_RANGE"

CC_K1K30="${CIPHER_DIR}/scripts/analysis/crosscheck_murphy8_k5_k1k30_motifs.json"
CC_K3="${CIPHER_DIR}/scripts/analysis/crosscheck_murphy8_k5_k3.json"

LOG_DIR="${CIPHER_DIR}/scripts/_logs/kmer_attention"
mkdir -p "$LOG_DIR"

for JOB in $JOBS; do
    TARGET="${JOB%%:*}"
    MODE="${JOB##*:}"

    case "$TARGET" in
        K3)   CC_JSON="$CC_K3" ;;
        *)    CC_JSON="$CC_K1K30" ;;
    esac

    case "$MODE" in
        strict)
            MODE_FLAG=""
            SUFFIX=""
            ;;
        anyhost)
            MODE_FLAG="--include-all-target-phages"
            SUFFIX="_anyhost"
            ;;
        *)
            echo "WARNING: unknown mode '$MODE' for $TARGET (expected: strict|anyhost) — skipping" >&2
            continue
            ;;
    esac

    OUT_TSV="${CIPHER_DIR}/results/analysis/kmer_attention_${TARGET}${SUFFIX}_murphy8.tsv"
    JOB_NAME="kmer_attn_murphy8_${TARGET}${SUFFIX}"

    CMD="
        source \$(conda info --base)/etc/profile.d/conda.sh
        conda activate ${CONDA_ENV}
        cd ${CIPHER_DIR}
        export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}
        python scripts/analysis/extract_kmer_attention_for_class.py \
            --experiment-dir ${EXP_DIR} \
            --val-fasta ${VAL_FASTA} \
            --val-emb ${VAL_EMB} \
            --val-datasets-dir ${VAL_DS} \
            --target-class ${TARGET} \
            --alphabet murphy8 --k 5 \
            --crosscheck-json ${CC_JSON} \
            ${MODE_FLAG} \
            --out-tsv ${OUT_TSV}
    "

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY [${TARGET}/${MODE}]: ${CMD}"
    else
        sbatch \
            --account="$ACCOUNT" --partition="$PARTITION" \
            --gpus-per-node=1 --cpus-per-task=4 --mem=32G --time=00:30:00 \
            --job-name="$JOB_NAME" \
            --output="${LOG_DIR}/${JOB_NAME}_%j.log" \
            --wrap="$CMD"
    fi
done
