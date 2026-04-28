#!/usr/bin/env bash
#
# Submit the full DpoTropi-data training pipeline on Delta-AI.
#
# Pipeline:
#   1. Embed DpoTropi training proteins with ProtT5 mean (1024-dim)
#   2. Embed DpoTropi training proteins with prott5_xl_seg8 (8192-dim)
#   3. Train attention_mlp K-only on DpoTropi data + ProtT5 mean
#   4. Train light_attention K-only on DpoTropi data + prott5_xl_seg8
#
# Each step is a separate sbatch job. Steps 3/4 depend on the
# corresponding embedding job finishing — uses --dependency=afterok.
#
# Why these architectures + embeddings:
#   - attention_mlp + ProtT5 mean: the cipher leaderboard top under any-hit
#     (sweep_prott5_mean_cl70 was #1)
#   - light_attention + prott5_xl_seg8: the best-on-PHL LA experiment
#     (la_v3_uat_prott5_xl_seg8 had PHL_rh1 = 0.178)
#
# Why K-only: DpoTropi training data has no O labels. Cipher's
# train_runner accepts --heads k to train K head only.
#
# Usage (on Delta):
#   bash scripts/run_dpotropi_data_pipeline.sh
#   DRY_RUN=1 bash scripts/run_dpotropi_data_pipeline.sh
#   STEP=1 bash scripts/run_dpotropi_data_pipeline.sh   # only embedding step 1

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

DPOTROPI_DIR="${CIPHER_DIR}/data/training_data/dpotropi"
DPOTROPI_FAA="${DPOTROPI_DIR}/dpotropi_train_proteins.faa"
ASSOC_MAP="${DPOTROPI_DIR}/dpotropi_phage_protein_map.tsv"
GLYCAN_TSV="${DPOTROPI_DIR}/dpotropi_glycan_binders.tsv"
POSITIVE_LIST="${DPOTROPI_DIR}/dpotropi_positive.list"

EMB_DIR="${EMB_DIR:-/work/hdd/bfzj/llindsey1/dpotropi_embeddings}"
PROTT5_MEAN_NPZ="${EMB_DIR}/dpotropi_prott5_mean.npz"
PROTT5_SEG8_NPZ="${EMB_DIR}/dpotropi_prott5_xl_seg8.npz"

# Validation paths — same as existing cipher experiments
VAL_FASTA="${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${CIPHER_DIR}/data/validation_data/HOST_RANGE"
VAL_PROTT5_MEAN_NPZ="/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz"
VAL_PROTT5_SEG8_NPZ="/work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments8/validation_prott5_xl_segments8_md5.npz"

mkdir -p "${CIPHER_DIR}/logs" "${EMB_DIR}"
DRY_RUN="${DRY_RUN:-0}"
STEP="${STEP:-all}"

submit() {
    local job_script="$1"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY RUN] would submit: ${job_script}"
        cat "$job_script"
        echo "---"
        echo "0"  # fake job id
        return
    fi
    sbatch "$@" | awk '{print $NF}'
}

# ────────────────────────────────────────────────────────────────────
# Step 1: ProtT5 mean embedding of DpoTropi proteins
# ────────────────────────────────────────────────────────────────────

if [ "$STEP" = "all" ] || [ "$STEP" = "1" ]; then
JOB1_SBATCH="${CIPHER_DIR}/logs/dpotropi_emb_prott5_mean_$(date +%Y%m%d_%H%M%S).sbatch"
cat > "$JOB1_SBATCH" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=dpotropi_emb_pt5_mean
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=${CIPHER_DIR}/logs/dpotropi_emb_prott5_mean_%j.log
#SBATCH --error=${CIPHER_DIR}/logs/dpotropi_emb_prott5_mean_%j.log

set -euo pipefail
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate prott5
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

echo "Embedding ${DPOTROPI_FAA} → ${PROTT5_MEAN_NPZ}"
python scripts/extract_embeddings/prott5_extract.py \\
    "${DPOTROPI_FAA}" "${PROTT5_MEAN_NPZ}" \\
    --key_by_md5
echo "Done: \$(date)"
SBATCH_EOF
echo "Step 1 script: ${JOB1_SBATCH}"
JOB1_ID=$(submit "$JOB1_SBATCH")
echo "Step 1 (ProtT5 mean embed): JOB ${JOB1_ID}"
fi

# ────────────────────────────────────────────────────────────────────
# Step 2: ProtT5 XL seg8 embedding (parallel to step 1, both GPU)
# ────────────────────────────────────────────────────────────────────

if [ "$STEP" = "all" ] || [ "$STEP" = "2" ]; then
JOB2_SBATCH="${CIPHER_DIR}/logs/dpotropi_emb_prott5_seg8_$(date +%Y%m%d_%H%M%S).sbatch"
cat > "$JOB2_SBATCH" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=dpotropi_emb_pt5_seg8
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=${CIPHER_DIR}/logs/dpotropi_emb_prott5_seg8_%j.log
#SBATCH --error=${CIPHER_DIR}/logs/dpotropi_emb_prott5_seg8_%j.log

set -euo pipefail
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate prott5
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

echo "Embedding ${DPOTROPI_FAA} → ${PROTT5_SEG8_NPZ} (segments8)"
python scripts/extract_embeddings/prott5_extract.py \\
    "${DPOTROPI_FAA}" "${PROTT5_SEG8_NPZ}" \\
    --key_by_md5 \\
    --pooling segments8
echo "Done: \$(date)"
SBATCH_EOF
echo "Step 2 script: ${JOB2_SBATCH}"
JOB2_ID=$(submit "$JOB2_SBATCH")
echo "Step 2 (ProtT5 seg8 embed): JOB ${JOB2_ID}"
fi

# ────────────────────────────────────────────────────────────────────
# Step 3: Train attention_mlp K-only on DpoTropi + ProtT5 mean
#   (depends on step 1)
# ────────────────────────────────────────────────────────────────────

if [ "$STEP" = "all" ] || [ "$STEP" = "3" ]; then
DEP="${JOB1_ID:-}"
DEP_FLAG=""
if [ -n "$DEP" ] && [ "$DRY_RUN" != "1" ] && [ "${NO_DEP:-0}" != "1" ]; then
    DEP_FLAG="--dependency=afterok:${DEP}"
fi

JOB3_SBATCH="${CIPHER_DIR}/logs/dpotropi_train_mlp_prott5_mean_$(date +%Y%m%d_%H%M%S).sbatch"
cat > "$JOB3_SBATCH" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=dpotropi_mlp_pt5
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=${CIPHER_DIR}/logs/dpotropi_train_mlp_prott5_mean_%j.log
#SBATCH --error=${CIPHER_DIR}/logs/dpotropi_train_mlp_prott5_mean_%j.log

set -euo pipefail
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate esmfold2
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

NAME="dpotropi_attention_mlp_prott5_mean_K_only"
echo "Training: \${NAME}"
echo "  Training data: ${DPOTROPI_DIR}"
echo "  Embedding:     ${PROTT5_MEAN_NPZ}"

python -m cipher.cli.train_runner \\
    --model attention_mlp \\
    --heads k \\
    --positive_list "${POSITIVE_LIST}" \\
    --label_strategy single_label \\
    --split-style canonical \\
    --lr 1e-05 \\
    --batch_size 64 \\
    --epochs 200 \\
    --patience 30 \\
    --embedding_type prott5_mean \\
    --embedding_file "${PROTT5_MEAN_NPZ}" \\
    --association_map "${ASSOC_MAP}" \\
    --glycan_binders "${GLYCAN_TSV}" \\
    --val_fasta "${VAL_FASTA}" \\
    --val_datasets_dir "${VAL_DATASETS_DIR}" \\
    --val_embedding_file "${VAL_PROTT5_MEAN_NPZ}" \\
    --name "\${NAME}"

EXP_DIR="${CIPHER_DIR}/experiments/attention_mlp/\${NAME}"
echo ""
echo "Eval: default (z-score, competition)"
python -m cipher.evaluation.runner "\${EXP_DIR}" \\
    --val-embedding-file "${VAL_PROTT5_MEAN_NPZ}"

echo ""
echo "Eval: per-head strict (any-hit + per-pair)"
python ${CIPHER_DIR}/scripts/analysis/per_head_strict_eval.py "\${EXP_DIR}" \\
    --val-embedding-file "${VAL_PROTT5_MEAN_NPZ}" \\
    --val-fasta "${VAL_FASTA}" \\
    --val-datasets-dir "${VAL_DATASETS_DIR}"

echo "Done: \$(date)"
SBATCH_EOF
echo "Step 3 script: ${JOB3_SBATCH}"
if [ "$DRY_RUN" != "1" ]; then
    JOB3_ID=$(sbatch ${DEP_FLAG} "$JOB3_SBATCH" | awk '{print $NF}')
    echo "Step 3 (MLP K-only train, ProtT5 mean): JOB ${JOB3_ID}  (depends on ${DEP})"
else
    echo "[DRY RUN] would sbatch ${DEP_FLAG} ${JOB3_SBATCH}"
fi
fi

# ────────────────────────────────────────────────────────────────────
# Step 4: Train light_attention K-only on DpoTropi + ProtT5 seg8
# ────────────────────────────────────────────────────────────────────

if [ "$STEP" = "all" ] || [ "$STEP" = "4" ]; then
DEP="${JOB2_ID:-}"
DEP_FLAG=""
if [ -n "$DEP" ] && [ "$DRY_RUN" != "1" ] && [ "${NO_DEP:-0}" != "1" ]; then
    DEP_FLAG="--dependency=afterok:${DEP}"
fi

JOB4_SBATCH="${CIPHER_DIR}/logs/dpotropi_train_la_prott5_seg8_$(date +%Y%m%d_%H%M%S).sbatch"
cat > "$JOB4_SBATCH" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=dpotropi_la_pt5seg8
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=${CIPHER_DIR}/logs/dpotropi_train_la_prott5_seg8_%j.log
#SBATCH --error=${CIPHER_DIR}/logs/dpotropi_train_la_prott5_seg8_%j.log

set -euo pipefail
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate esmfold2
LA_DIR="\$(cd ${CIPHER_DIR}/../cipher-light-attention && pwd)"
cd "\${LA_DIR}"
export PYTHONPATH="\${LA_DIR}/src:${CIPHER_DIR}/src:\${PYTHONPATH:-}"

NAME="dpotropi_light_attention_prott5_xl_seg8_K_only"
echo "Training: \${NAME}"
echo "  Training data: ${DPOTROPI_DIR}"
echo "  Embedding:     ${PROTT5_SEG8_NPZ}"

python -m cipher.cli.train_runner \\
    --model light_attention \\
    --heads k \\
    --positive_list "${POSITIVE_LIST}" \\
    --label_strategy single_label \\
    --lr 1e-05 \\
    --batch_size 64 \\
    --epochs 200 \\
    --patience 30 \\
    --embedding_type prott5_xl_seg8 \\
    --embedding_file "${PROTT5_SEG8_NPZ}" \\
    --association_map "${ASSOC_MAP}" \\
    --glycan_binders "${GLYCAN_TSV}" \\
    --val_fasta "${VAL_FASTA}" \\
    --val_datasets_dir "${VAL_DATASETS_DIR}" \\
    --val_embedding_file "${VAL_PROTT5_SEG8_NPZ}" \\
    --name "\${NAME}"

EXP_DIR="\${LA_DIR}/experiments/light_attention/\${NAME}"
echo ""
echo "Eval: default"
python -m cipher.evaluation.runner "\${EXP_DIR}" \\
    --val-embedding-file "${VAL_PROTT5_SEG8_NPZ}"

echo ""
echo "Eval: per-head strict (any-hit + per-pair)"
python ${CIPHER_DIR}/scripts/analysis/per_head_strict_eval.py "\${EXP_DIR}" \\
    --val-embedding-file "${VAL_PROTT5_SEG8_NPZ}" \\
    --val-fasta "${VAL_FASTA}" \\
    --val-datasets-dir "${VAL_DATASETS_DIR}"

echo "Done: \$(date)"
SBATCH_EOF
echo "Step 4 script: ${JOB4_SBATCH}"
if [ "$DRY_RUN" != "1" ]; then
    JOB4_ID=$(sbatch ${DEP_FLAG} "$JOB4_SBATCH" | awk '{print $NF}')
    echo "Step 4 (LA K-only train, ProtT5 seg8): JOB ${JOB4_ID}  (depends on ${DEP})"
else
    echo "[DRY RUN] would sbatch ${DEP_FLAG} ${JOB4_SBATCH}"
fi
fi

echo ""
echo "============================================================"
echo "Pipeline scheduled. Tail logs in ${CIPHER_DIR}/logs/"
echo "Re-run individual steps with STEP=N (1, 2, 3, or 4)."
echo "============================================================"
