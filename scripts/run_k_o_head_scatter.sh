#!/usr/bin/env bash
#
# Submit a Delta-AI SLURM job that computes the K-head/O-head per-pair
# rank scatter for one cipher experiment + one validation dataset.
#
# Defaults: concat_prott5_mean+kmer_li10_k5 on PhageHostLearn.
#
# Outputs (under $CIPHER_DIR after the job finishes):
#   results/analysis/k_o_head_ranks_<run>_<dataset>.tsv
#   results/figures/k_o_head_scatter_<run>_<dataset>.{svg,png}
#
# Usage:
#   bash scripts/run_k_o_head_scatter.sh                    # default
#   bash scripts/run_k_o_head_scatter.sh sweep_prott5_mean_cl70 PhageHostLearn
#   DRY_RUN=1 bash scripts/run_k_o_head_scatter.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"
DATA_DIR="${DATA_DIR:-${CIPHER_DIR}/data}"

GPUS=1
CPUS=4
MEM="32G"
TIME="00:30:00"

RUN_NAME="${1:-concat_prott5_mean+kmer_li10_k5}"
DATASET="${2:-PhageHostLearn}"

# Default embedding pairs per run (extend if you add more concat runs).
case "$RUN_NAME" in
    concat_prott5_mean+kmer_li10_k5)
        EMB1="/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz"
        EMB2="/work/hdd/bfzj/llindsey1/kmer_features/validation_li10_k5.npz"
        ARCH="attention_mlp"
        ;;
    sweep_prott5_mean_cl70|sweep_prott5_mean)
        EMB1="/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz"
        EMB2=""
        ARCH="attention_mlp"
        ;;
    highconf_pipeline_K_prott5_mean)
        EMB1="/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz"
        EMB2=""
        ARCH="attention_mlp"
        ;;
    *)
        # Fall back to reading val_embedding_file from config.yaml
        ARCH="attention_mlp"
        EMB1=""
        EMB2=""
        ;;
esac

EXP_DIR="${CIPHER_DIR}/experiments/${ARCH}/${RUN_NAME}"
VAL_FASTA="${DATA_DIR}/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${DATA_DIR}/validation_data/HOST_RANGE"

JOB_NAME="koscat_${RUN_NAME//[\/+]/_}_${DATASET}"
LOG_DIR="${CIPHER_DIR}/scripts/_logs/k_o_head_scatter"
mkdir -p "$LOG_DIR"

EMB_ARGS="--val-embedding-file ${EMB1}"
if [[ -n "$EMB2" ]]; then EMB_ARGS+=" --val-embedding-file-2 ${EMB2}"; fi

CMD="
    source \$(conda info --base)/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV}
    cd ${CIPHER_DIR}
    export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}
    python scripts/analysis/plot_k_o_head_scatter.py \
        --experiment-dir ${EXP_DIR} \
        --val-fasta ${VAL_FASTA} \
        ${EMB_ARGS} \
        --val-datasets-dir ${VAL_DATASETS_DIR} \
        --dataset ${DATASET}
"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY: sbatch --job-name=$JOB_NAME --account=$ACCOUNT --partition=$PARTITION \\"
    echo "       --gpus-per-node=$GPUS --cpus-per-task=$CPUS --mem=$MEM --time=$TIME \\"
    echo "       --output=${LOG_DIR}/${JOB_NAME}.%j.log --wrap='${CMD}'"
else
    sbatch \
        --account="$ACCOUNT" --partition="$PARTITION" \
        --gpus-per-node=$GPUS --cpus-per-task=$CPUS --mem="$MEM" --time="$TIME" \
        --job-name="$JOB_NAME" \
        --output="${LOG_DIR}/${JOB_NAME}.%j.log" \
        --wrap="$CMD"
fi
