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
#   ciphermil_esm2_3b_mean_cl70
#   ciphermil_kmer_aa20_k4_cl70
#
# Env overrides:
#   EMB_TYPE      embedding key. Recognized presets auto-fill TRAIN_EMB/VAL_EMB:
#                   prott5_xl       — ProtT5-XL mean (1024-d, default)
#                   esm2_3b_mean    — ESM-2 3B mean (2560-d)
#                   esm2_650m_mean  — ESM-2 650M mean (1280-d)
#                   kmer_aa20_k4    — kmer aa20 k=4 (160000-d sparse)
#                 Anything else: must also set TRAIN_EMB and VAL_EMB.
#   TRAIN_EMB     training NPZ (auto-set if EMB_TYPE is a preset)
#   VAL_EMB       validation NPZ (auto-set if EMB_TYPE is a preset)
#   NAME          run name (auto-generated as ciphermil_<EMB_TYPE>_cl<CLUSTER_TH>)
#   MEM           SLURM memory (auto-bumped to 256G for kmer_aa20_k4)
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
#   bash scripts/run_ciphermil_baseline.sh                              # ProtT5 default
#   EMB_TYPE=esm2_3b_mean bash scripts/run_ciphermil_baseline.sh        # ESM-2 3B
#   EMB_TYPE=kmer_aa20_k4 bash scripts/run_ciphermil_baseline.sh        # kmer
#   DRY_RUN=1 bash scripts/run_ciphermil_baseline.sh                    # inspect only

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

MODEL="ciphermil"
EMB_TYPE="${EMB_TYPE:-prott5_xl}"

# Embedding presets — recognized EMB_TYPE values map to canonical NPZ
# paths on Delta. Override TRAIN_EMB / VAL_EMB explicitly to use a
# custom variant. Same paths as the corresponding attention_mlp
# baselines so per-architecture comparisons are apples-to-apples.
case "$EMB_TYPE" in
    prott5_xl)
        DEFAULT_TRAIN_EMB="/projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_prott5/candidates_embeddings_md5.npz"
        DEFAULT_VAL_EMB="/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz"
        DEFAULT_MEM="64G"
        ;;
    esm2_3b_mean|esm2_3b)
        DEFAULT_TRAIN_EMB="/projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_esm2_3b/candidates_embeddings_md5.npz"
        DEFAULT_VAL_EMB="/work/hdd/bfzj/llindsey1/validation_embeddings_esm2_3b/validation_embeddings_md5.npz"
        DEFAULT_MEM="96G"   # 2560-d input → ~7M params on the trunk Linear; bag forward still small
        ;;
    esm2_650m_mean|esm2_650m)
        DEFAULT_TRAIN_EMB="/projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_binned/candidates_embeddings_md5.npz"
        DEFAULT_VAL_EMB="/projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/validation_data/combined/validation_inputs/validation_embeddings_md5.npz"
        DEFAULT_MEM="64G"
        ;;
    kmer_aa20_k4)
        DEFAULT_TRAIN_EMB="/work/hdd/bfzj/llindsey1/kmer_features/candidates_aa20_k4.npz"
        DEFAULT_VAL_EMB="/work/hdd/bfzj/llindsey1/kmer_features/validation_aa20_k4.npz"
        # 160000-d input + the trunk Linear(160000, 800) = 128M params for
        # the trunk alone. Plus ~70k MD5s × 160000-d float32 NPZ load.
        # Bump memory to be safe.
        DEFAULT_MEM="256G"
        ;;
    *)
        DEFAULT_TRAIN_EMB=""
        DEFAULT_VAL_EMB=""
        DEFAULT_MEM="64G"
        ;;
esac

TRAIN_EMB="${TRAIN_EMB:-${DEFAULT_TRAIN_EMB}}"
VAL_EMB="${VAL_EMB:-${DEFAULT_VAL_EMB}}"

if [ -z "$TRAIN_EMB" ] || [ -z "$VAL_EMB" ]; then
    echo "ERROR: EMB_TYPE='${EMB_TYPE}' is not a known preset and no" >&2
    echo "       TRAIN_EMB/VAL_EMB env vars were set." >&2
    echo "       Recognized presets: prott5_xl, esm2_3b_mean, esm2_650m_mean, kmer_aa20_k4" >&2
    exit 1
fi

NAME="${NAME:-ciphermil_${EMB_TYPE}_cl${CLUSTER_TH}}"

# Generous SLURM allocation per the no-tight-timeouts rule. ciphermil
# trains ONE bag at a time (no batching across bags), so per-epoch is
# slower than a per-protein MLP. 24h covers the K head + O head safely.
GPUS=1
CPUS=8
MEM="${MEM:-${DEFAULT_MEM}}"
TIME="${TIME:-24:00:00}"

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
echo "  Embedding:  ${EMB_TYPE}"
echo "  Train NPZ:  ${TRAIN_EMB}"
echo "  Val NPZ:    ${VAL_EMB}"
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
