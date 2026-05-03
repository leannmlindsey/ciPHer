#!/usr/bin/env bash
#
# Submit Delta SLURM jobs that extract SE-attention weights from
# sweep_kmer_murphy8_k5's K-head conditioned on K1, K30 (control), and
# K3 (parallel test) — for agent 5's K1-specific motif vs control delta
# analysis (analysis 35-style).
#
# Per agent 5's 2026-05-04 pivot: Murphy8-k5 is the native motif
# alphabet; AA20-k4 splits attention across multiple AA20 instances of
# the same chemistry-class motif. K-head accuracy parity verified
# (Murphy8 PHL=0.098 vs AA20 PHL=0.107; PBIP 0.800 vs 0.767).
#
# Usage:
#   bash scripts/run_kmer_attention_murphy8.sh                # default = K1, K30, K3
#   TARGETS="K1" bash scripts/run_kmer_attention_murphy8.sh   # subset
#   TARGETS="K47 K3" bash scripts/run_kmer_attention_murphy8.sh
#
# Heads-up: K3 has lower training support than K1 in HC_K_cl95;
# n_phages_contributed could be very small. The script emits an empty
# TSV and exits with a clear message in that case.

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

TARGETS="${TARGETS:-K1 K30 K3}"
EXP_DIR="${CIPHER_DIR}/experiments/attention_mlp/sweep_kmer_murphy8_k5"
VAL_FASTA="${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa"
VAL_EMB="/work/hdd/bfzj/llindsey1/kmer_features/validation_murphy8_k5.npz"
VAL_DS="${CIPHER_DIR}/data/validation_data/HOST_RANGE"

CC_K1K30="${CIPHER_DIR}/scripts/analysis/crosscheck_murphy8_k5_k1k30_motifs.json"
CC_K3="${CIPHER_DIR}/scripts/analysis/crosscheck_murphy8_k5_k3.json"

LOG_DIR="${CIPHER_DIR}/scripts/_logs/kmer_attention"
mkdir -p "$LOG_DIR"

for TARGET in $TARGETS; do
    case "$TARGET" in
        K3)   CC_JSON="$CC_K3" ;;
        *)    CC_JSON="$CC_K1K30" ;;
    esac

    OUT_TSV="${CIPHER_DIR}/results/analysis/kmer_attention_${TARGET}_murphy8.tsv"
    JOB_NAME="kmer_attn_murphy8_${TARGET}"

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
            --out-tsv ${OUT_TSV}
    "

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY [$TARGET]: ${CMD}"
    else
        sbatch \
            --account="$ACCOUNT" --partition="$PARTITION" \
            --gpus-per-node=1 --cpus-per-task=4 --mem=32G --time=00:30:00 \
            --job-name="$JOB_NAME" \
            --output="${LOG_DIR}/${JOB_NAME}_%j.log" \
            --wrap="$CMD"
    fi
done
