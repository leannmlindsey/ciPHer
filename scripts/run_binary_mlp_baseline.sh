#!/usr/bin/env bash
#
# binary_mlp baseline run on kmer_aa20_k4 — direct A/B vs the current cipher
# headline single model (`sweep_kmer_aa20_k4`, attention_mlp + multi_label_
# threshold + kmer_aa20_k4 + pipeline_positive + cl70, PHL_OR=0.560).
#
# What this is testing:
#   The structural change is the model architecture: shared trunk + per-class
#   binary heads (each class gets a 2-layer MLP head over the trunk's last
#   activation, instead of a single Linear(h, n_classes) projection).
#   Training data, features, label strategy, and cluster filter all match
#   `sweep_kmer_aa20_k4` so the only varied axis is the model.
#
# After training, runs:
#   - default eval (cipher-evaluate, zscore K+O)
#   - K-only and O-only head-mode evals
#   - per_head_strict_eval (any-hit + per-pair, fixed denominator) — the
#     output that lands in the harvest's any-hit headline columns.
#
# Env overrides:
#   CIPHER_DIR    binary-onevsrest worktree on Delta
#                 (default: /projects/bfzj/llindsey1/PHI_TSP/cipher-binary-onevsrest)
#   DATA_DIR      main ciPHer data dir (default: /projects/bfzj/llindsey1/PHI_TSP/ciPHer/data)
#   POSITIVE_LIST default: pipeline_positive.list (matches sweep_kmer_aa20_k4)
#   CLUSTER_FILE  default: candidates_clusters.tsv (matches sweep_kmer_aa20_k4)
#   CLUSTER_TH    default: 70
#   MAX_K_CAP     optional --max_samples_per_k cap (default: unset = no cap)
#   ACCOUNT, PARTITION, CONDA_ENV  (Delta defaults)
#   DRY_RUN=1     render but don't submit
#
# Usage:
#   bash scripts/run_binary_mlp_baseline.sh
#   MAX_K_CAP=300 bash scripts/run_binary_mlp_baseline.sh   # try cap=300 (LA's sweet spot)
#   DRY_RUN=1 bash scripts/run_binary_mlp_baseline.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="${CIPHER_DIR:-$(dirname "$SCRIPT_DIR")}"
DATA_DIR="${DATA_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer/data}"

ASSOC_MAP="${DATA_DIR}/training_data/metadata/host_phage_protein_map.tsv"
GLYCAN_BINDERS="${DATA_DIR}/training_data/metadata/glycan_binders_custom.tsv"
VAL_FASTA="${DATA_DIR}/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${DATA_DIR}/validation_data/HOST_RANGE"

POSITIVE_LIST="${POSITIVE_LIST:-${DATA_DIR}/training_data/metadata/pipeline_positive.list}"
CLUSTER_FILE="${CLUSTER_FILE:-${DATA_DIR}/training_data/metadata/candidates_clusters.tsv}"
CLUSTER_TH="${CLUSTER_TH:-70}"

# kmer_aa20_k4 features (160000-d) — training + validation NPZs, already
# on Delta from the existing kmer pipeline.
TRAIN_EMB="/work/hdd/bfzj/llindsey1/kmer_features/candidates_aa20_k4.npz"
VAL_EMB="/work/hdd/bfzj/llindsey1/kmer_features/validation_aa20_k4.npz"

MODEL="binary_mlp"
EMB_TYPE="kmer_aa20_k4"

# Optional cap. attention_mlp's cap=300 helped LA's PHL_OR (+0.045);
# unset for the first pass to match sweep_kmer_aa20_k4 baseline exactly.
MAX_K_CAP="${MAX_K_CAP:-}"
CAP_TAG=""
CAP_FLAG=""
if [ -n "$MAX_K_CAP" ]; then
    CAP_TAG="_cap${MAX_K_CAP}"
    CAP_FLAG="--max_samples_per_k ${MAX_K_CAP}"
fi

NAME="binary_mlp_kmer_aa20_k4_cl${CLUSTER_TH}${CAP_TAG}"

# 167M params on a kmer_aa20 input is slightly larger than attention_mlp;
# keep the same SLURM allocation as the sweep_kmer_aa20_k4 baseline.
GPUS=1
CPUS=8
MEM="160G"
TIME="24:00:00"

EXP_DIR="${CIPHER_DIR}/experiments/${MODEL}/${NAME}"

TRAIN_CMD="python -u -m cipher.cli.train_runner \
    --model ${MODEL} \
    --positive_list ${POSITIVE_LIST} \
    --cluster_file ${CLUSTER_FILE} \
    --cluster_threshold ${CLUSTER_TH} \
    --embedding_type ${EMB_TYPE} \
    --embedding_file ${TRAIN_EMB} \
    --association_map ${ASSOC_MAP} \
    --glycan_binders ${GLYCAN_BINDERS} \
    --val_fasta ${VAL_FASTA} \
    --val_datasets_dir ${VAL_DATASETS_DIR} \
    --val_embedding_file ${VAL_EMB} \
    --label_strategy multi_label_threshold \
    --min_class_samples 25 \
    ${CAP_FLAG} \
    --name ${NAME}"

# NOTE: cipher.cli.train_runner now auto-runs per_head_strict_eval after
# training finishes (writes <exp>/results/per_head_strict_eval.json — the
# JSON the harvest CSV reads for headline any-hit + per-pair columns). No
# need for the launcher to chain a separate eval step. The legacy
# `cipher.evaluation.runner` (per-pair zscore-combined evaluation.json)
# is still callable manually if needed but is no longer the headline
# metric — drop it from launchers.

mkdir -p "${CIPHER_DIR}/logs"

JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${NAME:0:32}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${CIPHER_DIR}/logs/${NAME}_%j.log
#SBATCH --error=${CIPHER_DIR}/logs/${NAME}_%j.log

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

echo \"======================================\"
echo \"binary_mlp baseline: ${NAME}\"
echo \"  positive list: ${POSITIVE_LIST}\"
echo \"  cluster filter: ${CLUSTER_FILE} @ cl${CLUSTER_TH}\"
echo \"  train emb: ${TRAIN_EMB}\"
echo \"  val emb:   ${VAL_EMB}\"
echo \"  cap:       ${MAX_K_CAP:-none}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

if [ ! -f \"${VAL_EMB}\" ]; then
    echo \"WARNING: Validation embeddings not found: ${VAL_EMB}\"
    echo \"   train_runner will skip the auto strict-eval step.\"
fi

echo \"\"
echo \"=== TRAIN + AUTO STRICT-EVAL ===\"
${TRAIN_CMD}

echo \"\"
echo \"======================================\"
echo \"Done: ${NAME} at \$(date)\"
echo \"Result JSON: ${EXP_DIR}/results/per_head_strict_eval.json\"
echo \"  (any-hit + per-pair, fixed denominator — the harvest CSV reads this)\"
echo \"======================================\"
"

echo "============================================================"
echo "binary_mlp baseline"
echo "  Name:       ${NAME}"
echo "  Cipher dir: ${CIPHER_DIR}"
echo "  Embedding:  ${EMB_TYPE}"
echo "  Positive:   ${POSITIVE_LIST}"
echo "  Cluster:    ${CLUSTER_FILE} @ cl${CLUSTER_TH}"
echo "  Cap:        ${MAX_K_CAP:-none}"
echo "  SLURM:      ${TIME}, ${GPUS} GPU, ${CPUS} CPU, ${MEM}"
echo "============================================================"
echo ""

DRY_RUN="${DRY_RUN:-0}"
if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] sbatch script not submitted."
    echo "Submit with:"
    echo "$JOB_SCRIPT" | head -10
    exit 0
fi

JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
echo "Submitted ${JOB_ID} — ${NAME}"
echo "Monitor: squeue -j ${JOB_ID}"
