#!/usr/bin/env bash
#
# Train an attention_mlp run that replicates the OLD-repo
# `host_prediction_local_test_v3` recipe end-to-end inside the new
# ciPHer codebase.
#
# Old recipe (from klebsiella/config_local_v3.yaml):
#   embedding:        ESM-2 650M mean (layer 33, 1280-d)
#   protein_set:      all_glycan_binders   (no tool filter)
#   positive_list:    pipeline_positive.list
#   min_sources:      3
#   max_k_types:      3
#   max_o_types:      3
#   label_strategy:   single_label  (majority vote → CrossEntropy)
#   keep_null_classes: true   (null/N/A treated as valid predictable classes
#                              — our pipeline does this by default since we
#                              don't drop null classes during label-vector
#                              construction; just don't set min_class_samples
#                              high enough to cull "null" as a class)
#   model arch:       SE-AttentionMLP, hidden_dims=[1280,640,320,160], se_dim=640
#   batch_size: 64, lr: 1e-5, epochs: 200, patience: 30, seed: 42
#
# Old result on PHL: rh@1 = 0.291 (raw merge, arbitrary ties).
# This run will reproduce the trained model. To match the old
# eval methodology too, we also re-evaluate with --score-norm raw and
# --tie-method arbitrary.
#
# Usage:
#   bash scripts/run_repro_old_v3_in_cipher.sh
#   DRY_RUN=1 bash scripts/run_repro_old_v3_in_cipher.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

ASSOC_MAP="${CIPHER_DIR}/data/training_data/metadata/host_phage_protein_map.tsv"
GLYCAN_BINDERS="${CIPHER_DIR}/data/training_data/metadata/glycan_binders_custom.tsv"
POSITIVE_LIST="${CIPHER_DIR}/data/training_data/metadata/pipeline_positive.list"
VAL_FASTA="${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${CIPHER_DIR}/data/validation_data/HOST_RANGE"

# ESM-2 650M mean training and validation embeddings (match old config)
TRAIN_EMB="${TRAIN_EMB:-/projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_binned/candidates_embeddings_md5.npz}"
VAL_EMB="${VAL_EMB:-/projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/validation_data/combined/validation_inputs/validation_embeddings_md5.npz}"

# Match old hyperparameters exactly
NAME="repro_old_v3_full_in_cipher"
MODEL="attention_mlp"
LR="1e-05"
BATCH_SIZE=64
EPOCHS=200
PATIENCE=30
LABEL_STRATEGY="single_label"
MIN_SOURCES=3
MAX_K_TYPES=3
MAX_O_TYPES=3
# Critically: do NOT set min_class_samples (defaults to None → no class drop,
# preserving "null" / "N/A" as classes — matches old keep_null_classes=true)

GPUS=1
CPUS=8
MEM="64G"
TIME="6:00:00"

EXP_DIR="${CIPHER_DIR}/experiments/${MODEL}/${NAME}"
LOG="${CIPHER_DIR}/logs/${NAME}_%j.log"
DRY_RUN="${DRY_RUN:-0}"

TRAIN_CMD="python -m cipher.cli.train_runner \
    --model ${MODEL} \
    --positive_list ${POSITIVE_LIST} \
    --min_sources ${MIN_SOURCES} \
    --max_k_types ${MAX_K_TYPES} \
    --max_o_types ${MAX_O_TYPES} \
    --label_strategy ${LABEL_STRATEGY} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --embedding_type esm2_650m_mean \
    --embedding_file ${TRAIN_EMB} \
    --association_map ${ASSOC_MAP} \
    --glycan_binders ${GLYCAN_BINDERS} \
    --val_fasta ${VAL_FASTA} \
    --val_datasets_dir ${VAL_DATASETS_DIR} \
    --val_embedding_file ${VAL_EMB} \
    --name ${NAME}"

EVAL_DEFAULT="python -m cipher.evaluation.runner ${EXP_DIR} \
    --val-embedding-file ${VAL_EMB}"

EVAL_RAW="python -m cipher.evaluation.runner ${EXP_DIR} \
    --val-embedding-file ${VAL_EMB} \
    --score-norm raw \
    -o ${EXP_DIR}/results_raw/evaluation.json"

EVAL_RAW_ARB="python -m cipher.evaluation.runner ${EXP_DIR} \
    --val-embedding-file ${VAL_EMB} \
    --score-norm raw \
    --tie-method arbitrary \
    -o ${EXP_DIR}/results_raw_arbitrary/evaluation.json"

EVAL_PER_HEAD="python ${CIPHER_DIR}/scripts/analysis/eval_per_head.py ${EXP_DIR}"

echo "============================================================"
echo "REPRODUCE OLD v3 RECIPE INSIDE THE NEW CIPHER CODEBASE"
echo "  Name:        ${NAME}"
echo "  Model:       ${MODEL} (SE-AttentionMLP)"
echo "  Embedding:   esm2_650m_mean (matches old)"
echo "  Filter:      pipeline_positive + min_sources=3 + max_{k,o}_types=3"
echo "  Strategy:    single_label (majority vote → CrossEntropy)"
echo "  null/N/A:    kept as classes (default; min_class_samples not set)"
echo "  Target:      PHL rh@1 ≈ 0.291 (raw merge, arbitrary ties)"
echo "============================================================"
echo ""

JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG}
#SBATCH --error=${LOG}

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

echo \"=== TRAIN (old v3 recipe in new codebase) ===\"
echo \"  Started: \$(date)\"
${TRAIN_CMD}

echo \"\"
echo \"=== EVAL — default (zscore + competition) ===\"
${EVAL_DEFAULT}

echo \"\"
echo \"=== EVAL — raw merge, default ties ===\"
mkdir -p ${EXP_DIR}/results_raw
${EVAL_RAW}

echo \"\"
echo \"=== EVAL — raw merge, arbitrary ties (full old-style) ===\"
mkdir -p ${EXP_DIR}/results_raw_arbitrary
${EVAL_RAW_ARB}

echo \"\"
echo \"=== EVAL — per-head (K-only / O-only / combined) ===\"
${EVAL_PER_HEAD}

echo \"\"
echo \"=== PHL HR@1 across all eval modes ===\"
python -c \"
import json
for variant in ['results', 'results_raw', 'results_raw_arbitrary']:
    p = '${EXP_DIR}/' + variant + '/evaluation.json'
    try:
        d = json.load(open(p))
        rh = d['PhageHostLearn']['rank_hosts']['hr_at_k']['1']
        print(f'  {variant:<25} PHL rh@1 = {rh:.4f}')
    except FileNotFoundError:
        print(f'  {variant:<25} (not present)')
\"

echo \"\"
echo \"=== Saved K/O classes (look for null/N/A) ===\"
python -c \"
import json
e = json.load(open('${EXP_DIR}/label_encoders.json'))
k = e['k_classes']
o = e['o_classes']
print(f'  K classes ({len(k)}): null/N/A present?', any(c in {'null','N/A','Unknown',''} for c in k))
print(f'  O classes ({len(o)}): null/N/A present?', any(c in {'null','N/A','Unknown',''} for c in o))
\"

echo \"\"
echo \"Done: \$(date)\"
"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would submit: ${NAME}"
    exit 0
fi

mkdir -p "${CIPHER_DIR}/logs"
JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
echo "Submitted ${JOB_ID} - ${NAME}"
echo ""
echo "Log: ${CIPHER_DIR}/logs/${NAME}_${JOB_ID}.log"
echo ""
echo "If PHL rh@1 in results_raw_arbitrary ≈ 0.291, the new codebase"
echo "fully reproduces the old recipe. If still below 0.29, the gap is"
echo "either keep_null_classes-explicit-handling or some downstream"
echo "filter difference."
