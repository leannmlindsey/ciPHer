#!/usr/bin/env bash
#
# ciphermil baseline run — EvoMIL-style attention-based MIL on phage→host
# K and O serotype prediction. ProtT5-XL mean (1024-d) per leann's
# 2026-05-05 design call.
#
# What this is testing:
#   Whether attention-based MIL — a bag-level architecture that pools
#   over a phage's proteins via Ilse-2018 attention — beats per-protein
#   classifier architectures (attention_mlp / binary_mlp / LA) on
#   PhageHostLearn HR@1 (any-hit). EvoMIL reported AUC > 0.95 on
#   prokaryotic virus-host prediction with a comparable training-set
#   size to ours.
#
# Holds OTHER axes constant:
#   - architecture: ciphermil (Ilse-2018 attention MIL, multi-class softmax)
#   - features: ProtT5-XL mean (1024-d)
#   - cluster_threshold: 70 (cl70, our usual baseline)
#   - min_class_samples: 25 (cipher-wide convention)
#   - label_strategy: single_label_softmax (different from existing models'
#     multi_label_threshold; baked into ciphermil's training loop)
#   - SLURM walltime: 24h training (per generous-timeout rule)
#
# After training, train_runner's mandatory auto-strict-eval produces
# `per_head_strict_eval.json` (filter-matched if --glycan-binders
# resolves; for this baseline we don't pass a filter, so eval uses
# every protein in phage_protein_mapping.csv — same as our existing
# baselines that don't pass --glycan-binders either).
#
# Naming: <arch>_<emb>[_<extras>]:
#   ciphermil_prott5_xl_cl70
#
# Env overrides:
#   CIPHER_DIR    cipher-mil worktree on Delta
#                 (default: /projects/bfzj/llindsey1/PHI_TSP/cipher-mil)
#   DATA_DIR      ciPHer data dir (default: /projects/bfzj/llindsey1/PHI_TSP/ciPHer/data)
#   POSITIVE_LIST default: pipeline_positive.list
#   CLUSTER_FILE  default: candidates_clusters.tsv
#   CLUSTER_TH    default: 70
#   ACCOUNT, PARTITION, CONDA_ENV  (Delta defaults)
#   DRY_RUN=1     render but don't submit
#
# Usage:
#   bash scripts/run_ciphermil_baseline.sh
#   DRY_RUN=1 bash scripts/run_ciphermil_baseline.sh

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

# ProtT5-XL mean embeddings (1024-d). Same paths as the existing prott5
# attention_mlp baselines so we can compare fairly.
TRAIN_EMB="/projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_prott5/candidates_embeddings_md5.npz"
VAL_EMB="/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz"

MODEL="ciphermil"
EMB_TYPE="prott5_xl"

NAME="ciphermil_prott5_xl_cl${CLUSTER_TH}"

# Generous SLURM allocation per the no-tight-timeouts rule. ciphermil
# trains ONE bag at a time (no batching across bags), so per-epoch is
# slower than a per-protein MLP. 24h covers the K head + O head safely.
GPUS=1
CPUS=8
MEM="64G"     # smaller than our usual 160G — bag-by-bag training has small footprint
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
    --label_strategy single_label \
    --min_class_samples 25 \
    --name ${NAME}"

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
echo \"ciphermil baseline: ${NAME}\"
echo \"  positive list: ${POSITIVE_LIST}\"
echo \"  cluster filter: ${CLUSTER_FILE} @ cl${CLUSTER_TH}\"
echo \"  train emb: ${TRAIN_EMB}\"
echo \"  val emb:   ${VAL_EMB}\"
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
echo "ciphermil baseline"
echo "  Name:       ${NAME}"
echo "  Cipher dir: ${CIPHER_DIR}"
echo "  Embedding:  ${EMB_TYPE} (1024-d ProtT5-XL mean)"
echo "  Positive:   ${POSITIVE_LIST}"
echo "  Cluster:    ${CLUSTER_FILE} @ cl${CLUSTER_TH}"
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
