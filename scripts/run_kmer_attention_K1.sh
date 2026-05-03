#!/usr/bin/env bash
#
# Submit a Delta SLURM job that extracts SE-attention weights from
# sweep_kmer_aa20_k4's K-head conditioned on K1 prediction. Output is a
# TSV agent 5 will use for the analysis-32 vs analysis-33 motif comparison.
#
# Usage:
#   bash scripts/run_kmer_attention_K1.sh                    # default = K1
#   TARGET=K47 bash scripts/run_kmer_attention_K1.sh         # different class

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

TARGET="${TARGET:-K1}"
EXP_DIR="${CIPHER_DIR}/experiments/attention_mlp/sweep_kmer_aa20_k4"
VAL_FASTA="${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa"
VAL_EMB="/work/hdd/bfzj/llindsey1/kmer_features/validation_aa20_k4.npz"
VAL_DS="${CIPHER_DIR}/data/validation_data/HOST_RANGE"
OUT_TSV="${CIPHER_DIR}/results/analysis/kmer_attention_${TARGET}.tsv"

JOB_NAME="kmer_attn_${TARGET}"
LOG_DIR="${CIPHER_DIR}/scripts/_logs/kmer_attention"
mkdir -p "$LOG_DIR"

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
        --out-tsv ${OUT_TSV}
"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY: ${CMD}"
else
    sbatch \
        --account="$ACCOUNT" --partition="$PARTITION" \
        --gpus-per-node=1 --cpus-per-task=4 --mem=64G --time=00:30:00 \
        --job-name="$JOB_NAME" \
        --output="${LOG_DIR}/${JOB_NAME}_%j.log" \
        --wrap="$CMD"
fi
