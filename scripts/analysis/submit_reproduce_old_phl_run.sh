#!/usr/bin/env bash
#
# Investigate why old-repo `host_prediction_local_test_v3` reached
# PHL rh@1 = 0.291 while our current best (`highconf_pipeline_K_prott5_mean`)
# is at 0.188. Submits TWO SLURM jobs unconditionally so a fire-and-forget
# invocation answers both questions by the time you come back.
#
# STAGE 1 — eval-methodology probe (fast, ~15 min).
#   Re-evaluates the existing best-PHL model with:
#     (a) --tie-method arbitrary          (old convention, can inflate HR@1)
#     (b) eval_per_head.py                (K-only / O-only / combined table)
#   If STAGE 1 bumps PHL rh@1 to ~0.29, the eval methodology alone
#   explains the old-vs-new gap and we haven't actually regressed —
#   we've just been measuring more strictly.
#
# STAGE 2 — training-recipe probe (slow, ~3–5 h).
#   Trains a new attention_mlp with a best-effort reproduction of old
#   `config_local_v3.yaml`:
#     - ESM-2 650M mean embeddings (not ProtT5)
#     - no tool filter + --positive_list pipeline_positive.list
#     - --min_sources 3
#     - --label_strategy single_label
#     - --max_k_types 3 / --max_o_types 3
#   Then evaluates the new model BOTH old-style (arbitrary ties, O-only
#   via eval_per_head) and new-style (competition, combined).
#
#   Gap still unclosed after STAGE 2 → it was about
#   `keep_null_classes=true` (not yet supported in our pipeline). That
#   would be a third stage, not landed yet.
#
# Usage:
#   bash scripts/analysis/submit_reproduce_old_phl_run.sh
#   DRY_RUN=1 bash scripts/analysis/submit_reproduce_old_phl_run.sh
#
# Outputs:
#   experiments/attention_mlp/highconf_pipeline_K_prott5_mean/results_arbitrary/evaluation.json
#   logs/repro_stage1_evh_highconf_pipeline_<jobid>.log
#   logs/repro_stage1_arbitrary_<jobid>.log
#   experiments/attention_mlp/repro_old_v3_esm2_650m_mean/
#   logs/repro_stage2_train_<jobid>.log
#   logs/repro_stage2_evh_<jobid>.log

set -euo pipefail

# ============================================================
# Delta-AI configuration
# ============================================================
ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

ASSOC_MAP="${CIPHER_DIR}/data/training_data/metadata/host_phage_protein_map.tsv"
GLYCAN_BINDERS="${CIPHER_DIR}/data/training_data/metadata/glycan_binders_custom.tsv"
POSITIVE_LIST="${CIPHER_DIR}/data/training_data/metadata/pipeline_positive.list"
VAL_FASTA="${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${CIPHER_DIR}/data/validation_data/HOST_RANGE"

# ESM-2 650M mean embeddings (match the old config's embedding choice)
ESM2_TRAIN_EMB="${ESM2_TRAIN_EMB:-/projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_binned/candidates_embeddings_md5.npz}"
ESM2_VAL_EMB="${ESM2_VAL_EMB:-/projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/validation_data/combined/validation_inputs/validation_embeddings_md5.npz}"

# Existing best-PHL run to re-evaluate in STAGE 1
STAGE1_RUN="experiments/attention_mlp/highconf_pipeline_K_prott5_mean"

# STAGE 2: new run name + its attention_mlp hyperparameters (match old v3 config)
STAGE2_NAME="repro_old_v3_esm2_650m_mean"
STAGE2_LR="1e-05"
STAGE2_BATCH=64
STAGE2_EPOCHS=200
STAGE2_PATIENCE=30
STAGE2_MAX_K_TYPES=3
STAGE2_MAX_O_TYPES=3
STAGE2_MIN_SOURCES=3

DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "REPRODUCE OLD-REPO PHL RESULT — STAGE 1 + STAGE 2"
echo "  Old run to match: host_prediction_local_test_v3  (PHL rh@1 = 0.291)"
echo "  Current best:     highconf_pipeline_K_prott5_mean (PHL rh@1 = 0.188)"
echo "  Cipher dir:       ${CIPHER_DIR}"
echo "============================================================"
echo ""

# ============================================================
# STAGE 1 — eval-methodology probe
# ============================================================
# 1a. Arbitrary-tie re-eval of the existing model.
S1A_NAME="repro_stage1_arbitrary"
S1A_LOG="${CIPHER_DIR}/logs/${S1A_NAME}_%j.log"
S1A_SCRIPT="#!/bin/bash
#SBATCH --job-name=${S1A_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=${S1A_LOG}
#SBATCH --error=${S1A_LOG}

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

echo \"=== STAGE 1a: ARBITRARY-TIE RE-EVAL OF EXISTING BEST ===\"
echo \"  Run: ${STAGE1_RUN}\"
echo \"  Started: \$(date)\"

mkdir -p ${STAGE1_RUN}/results_arbitrary
python -m cipher.evaluation.runner ${STAGE1_RUN} \\
    --tie-method arbitrary \\
    -o ${STAGE1_RUN}/results_arbitrary/evaluation.json

echo \"=== done stage 1a ===\"
"

# 1b. eval_per_head on the existing model — K-only / O-only / combined breakdown.
S1B_NAME="repro_stage1_evh_highconf_pipeline"
S1B_LOG="${CIPHER_DIR}/logs/${S1B_NAME}_%j.log"
S1B_SCRIPT="#!/bin/bash
#SBATCH --job-name=${S1B_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=${S1B_LOG}
#SBATCH --error=${S1B_LOG}

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

echo \"=== STAGE 1b: eval_per_head (K / O / combined) ON EXISTING BEST ===\"
echo \"  Run: ${STAGE1_RUN}\"
echo \"  Started: \$(date)\"

python ${CIPHER_DIR}/scripts/analysis/eval_per_head.py ${STAGE1_RUN}

echo \"=== done stage 1b ===\"
"

# ============================================================
# STAGE 2 — training-recipe probe
# ============================================================
# Train a new attention_mlp with old config approximations, then evaluate
# with both old-style (arbitrary + per-head) and new-style (default).
S2_LOG="${CIPHER_DIR}/logs/repro_stage2_%j.log"

STAGE2_TRAIN_CMD="python -m cipher.cli.train_runner \
    --model attention_mlp \
    --positive_list ${POSITIVE_LIST} \
    --min_sources ${STAGE2_MIN_SOURCES} \
    --max_k_types ${STAGE2_MAX_K_TYPES} \
    --max_o_types ${STAGE2_MAX_O_TYPES} \
    --label_strategy single_label \
    --lr ${STAGE2_LR} \
    --batch_size ${STAGE2_BATCH} \
    --epochs ${STAGE2_EPOCHS} \
    --patience ${STAGE2_PATIENCE} \
    --embedding_type esm2_650m_mean \
    --embedding_file ${ESM2_TRAIN_EMB} \
    --association_map ${ASSOC_MAP} \
    --glycan_binders ${GLYCAN_BINDERS} \
    --val_fasta ${VAL_FASTA} \
    --val_datasets_dir ${VAL_DATASETS_DIR} \
    --val_embedding_file ${ESM2_VAL_EMB} \
    --name ${STAGE2_NAME}"

S2_EXP_DIR="${CIPHER_DIR}/experiments/attention_mlp/${STAGE2_NAME}"

S2_SCRIPT="#!/bin/bash
#SBATCH --job-name=repro_stage2
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=${S2_LOG}
#SBATCH --error=${S2_LOG}

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

echo \"=== STAGE 2: TRAIN ATTENTION_MLP WITH OLD-v3-CONFIG APPROXIMATION ===\"
echo \"  Name:   ${STAGE2_NAME}\"
echo \"  Started: \$(date)\"
echo \"  Known limitation: does NOT yet support keep_null_classes=true.\"
echo \"  If PHL rh@1 still trails 0.291 after this, null-class handling\"
echo \"  may be the remaining gap.\"

echo \"=== TRAIN ===\"
${STAGE2_TRAIN_CMD}

echo \"=== EVAL (default: competition ties, combined z-score) ===\"
python -m cipher.evaluation.runner ${S2_EXP_DIR} \\
    --val-embedding-file ${ESM2_VAL_EMB}

echo \"=== EVAL (old-style: arbitrary ties, combined) ===\"
mkdir -p ${S2_EXP_DIR}/results_arbitrary
python -m cipher.evaluation.runner ${S2_EXP_DIR} \\
    --tie-method arbitrary \\
    --val-embedding-file ${ESM2_VAL_EMB} \\
    -o ${S2_EXP_DIR}/results_arbitrary/evaluation.json

echo \"=== EVAL (per-head: K-only / O-only / combined) ===\"
python ${CIPHER_DIR}/scripts/analysis/eval_per_head.py ${S2_EXP_DIR}

echo \"=== done stage 2 at \$(date) ===\"
"

# ============================================================
# Submit both stages
# ============================================================
if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would submit:"
    echo "  STAGE 1a — ${S1A_NAME}    (arbitrary-tie re-eval)"
    echo "  STAGE 1b — ${S1B_NAME}    (eval_per_head on existing model)"
    echo "  STAGE 2  — repro_stage2   (retrain + 3 evals)"
    exit 0
fi

mkdir -p "${CIPHER_DIR}/logs"

S1A_ID=$(echo "$S1A_SCRIPT" | sbatch | awk '{print $NF}')
S1B_ID=$(echo "$S1B_SCRIPT" | sbatch | awk '{print $NF}')
S2_ID=$(echo "$S2_SCRIPT" | sbatch | awk '{print $NF}')

echo "Submitted ${S1A_ID} — ${S1A_NAME}  (STAGE 1a, arbitrary-tie re-eval)"
echo "Submitted ${S1B_ID} — ${S1B_NAME}  (STAGE 1b, eval_per_head)"
echo "Submitted ${S2_ID}  — repro_stage2  (STAGE 2, retrain + 3-way eval)"
echo ""
echo "Logs will land at:"
echo "  ${CIPHER_DIR}/logs/${S1A_NAME}_${S1A_ID}.log"
echo "  ${CIPHER_DIR}/logs/${S1B_NAME}_${S1B_ID}.log"
echo "  ${CIPHER_DIR}/logs/repro_stage2_${S2_ID}.log"
echo ""
echo "When you're back, compare PHL rh@1 across:"
echo "  (stage 1a) ${STAGE1_RUN}/results_arbitrary/evaluation.json"
echo "  (stage 1b) stdout of the eval_per_head log"
echo "  (stage 2)  experiments/attention_mlp/${STAGE2_NAME}/results{,_arbitrary}/evaluation.json"
echo "  + eval_per_head stdout in stage 2 log"
echo ""
echo "Target to match (old repo host_prediction_local_test_v3): PHL rh@1 = 0.291"
