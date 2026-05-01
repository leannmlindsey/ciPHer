#!/usr/bin/env bash
#
# v3_UAT no-cap ablation — train cipher's two production architectures
# (attention_mlp + light_attention) on the v3_UAT positive lists with
# the per-K and per-O sample caps DROPPED. Tests whether
# `max_samples_per_k=1000` + cluster-stratified round-robin is
# currently throwing away signal we need on common K-types.
#
# Recipe = sweep_prott5_mean_cl70 / la_v3_uat_prott5_xl_seg8 minus:
#   --max_samples_per_k 1000
#   --max_samples_per_o 3000
#   --cluster_file
#   --cluster_threshold 70
#
# All other hyperparameters identical to the v3_UAT production run.
#
# Compare against:
#   sweep_prott5_mean_cl70             (MLP, ProtT5 mean, current cap)
#   la_v3_uat_prott5_xl_seg8           (LA,  ProtT5 xl seg8, current cap)
#
# Two SLURM jobs (no embedding step — uses existing cipher candidate
# embeddings):
#   STEP 1 — attention_mlp v3_UAT no-cap, ProtT5 mean
#   STEP 2 — light_attention v3_UAT no-cap, ProtT5 xl seg8 (parallel)
#
# Env overrides:
#   ACCOUNT, PARTITION, CONDA_ENV, CIPHER_DIR
#   STEP={1,2,all} (default all)
#   DRY_RUN=1      render sbatch but do not submit
#
# Usage:
#   bash scripts/run_v3uat_nocap_pipeline.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"
STEP="${STEP:-all}"
DRY_RUN="${DRY_RUN:-0}"

# v3_UAT positive lists
V3_DIR="${CIPHER_DIR}/data/training_data/metadata/highconf_v3_multitop"
POSITIVE_LIST_K="${V3_DIR}/HC_K_UAT_multitop.list"
POSITIVE_LIST_O="${V3_DIR}/HC_O_UAT_multitop.list"

# Production v3 association map + glycan flags (unchanged)
ASSOC_MAP="${CIPHER_DIR}/data/training_data/metadata/host_phage_protein_map.tsv"
GLYCAN_TSV="${CIPHER_DIR}/data/training_data/metadata/glycan_binders_custom.tsv"

# Existing cipher candidate embeddings (no merge needed — v3_UAT IDs
# are all in the existing pool)
CAND_PROTT5_MEAN_NPZ="/projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_prott5/candidates_embeddings_md5.npz"
CAND_PROTT5_SEG8_NPZ="/work/hdd/bfzj/llindsey1/embeddings/prott5_xl_segments8/candidates_prott5_xl_segments8_md5.npz"

# Validation NPZs
VAL_FASTA="${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${CIPHER_DIR}/data/validation_data/HOST_RANGE"
VAL_PROTT5_MEAN_NPZ="/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz"
VAL_PROTT5_SEG8_NPZ="/work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments8/validation_prott5_xl_segments8_md5.npz"

mkdir -p "${CIPHER_DIR}/logs"

submit() {
    local script="$1"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY] sbatch ${script}"
        echo "DRY_${RANDOM}"
    else
        sbatch "${script}" | awk '{print $NF}'
    fi
}

# ────────────────────────────────────────────────────────────────────
# STEP 1 — attention_mlp v3_UAT no-cap, ProtT5 mean
# ────────────────────────────────────────────────────────────────────
if [ "$STEP" = "all" ] || [ "$STEP" = "1" ]; then
JOB1_SBATCH="${CIPHER_DIR}/logs/v3uat_nocap_mlp_$(date +%Y%m%d_%H%M%S).sbatch"
cat > "$JOB1_SBATCH" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=v3uat_nocap_mlp
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=${CIPHER_DIR}/logs/v3uat_nocap_mlp_%j.log
#SBATCH --error=${CIPHER_DIR}/logs/v3uat_nocap_mlp_%j.log

set -euo pipefail
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

NAME="v3uat_nocap_attention_mlp_prott5_mean"
echo "Training: \${NAME}"
echo "  K positive list: ${POSITIVE_LIST_K}"
echo "  O positive list: ${POSITIVE_LIST_O}"
echo "  Embedding:       ${CAND_PROTT5_MEAN_NPZ}"
echo "  Caps:            DISABLED (no max_samples_per_k/o, no cluster_file)"

python -m cipher.cli.train_runner \\
    --model attention_mlp \\
    --positive_list_k "${POSITIVE_LIST_K}" \\
    --positive_list_o "${POSITIVE_LIST_O}" \\
    --label_strategy multi_label_threshold \\
    --max_k_types 3 \\
    --max_o_types 3 \\
    --min_class_samples 25 \\
    --min_sources 1 \\
    --lr 1e-05 \\
    --batch_size 512 \\
    --epochs 1000 \\
    --patience 30 \\
    --embedding_type prott5_mean \\
    --embedding_file "${CAND_PROTT5_MEAN_NPZ}" \\
    --association_map "${ASSOC_MAP}" \\
    --glycan_binders "${GLYCAN_TSV}" \\
    --val_fasta "${VAL_FASTA}" \\
    --val_datasets_dir "${VAL_DATASETS_DIR}" \\
    --val_embedding_file "${VAL_PROTT5_MEAN_NPZ}" \\
    --name "\${NAME}"

EXP_DIR="${CIPHER_DIR}/experiments/attention_mlp/\${NAME}"
echo ""
echo "Eval: default"
python -m cipher.evaluation.runner "\${EXP_DIR}" \\
    --val-embedding-file "${VAL_PROTT5_MEAN_NPZ}"

echo ""
echo "Eval: per-head strict (any-hit + per-pair, all 5 datasets)"
python ${CIPHER_DIR}/scripts/analysis/per_head_strict_eval.py "\${EXP_DIR}" \\
    --val-embedding-file "${VAL_PROTT5_MEAN_NPZ}"
SBATCH_EOF
echo "STEP 1 script: ${JOB1_SBATCH}"
JOB1_ID=$(submit "$JOB1_SBATCH")
echo "STEP 1 (v3_UAT no-cap MLP, ProtT5 mean): JOB ${JOB1_ID}"
fi

# ────────────────────────────────────────────────────────────────────
# STEP 2 — light_attention v3_UAT no-cap, ProtT5 xl seg8
# ────────────────────────────────────────────────────────────────────
if [ "$STEP" = "all" ] || [ "$STEP" = "2" ]; then
JOB2_SBATCH="${CIPHER_DIR}/logs/v3uat_nocap_la_$(date +%Y%m%d_%H%M%S).sbatch"
cat > "$JOB2_SBATCH" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=v3uat_nocap_la
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=${CIPHER_DIR}/logs/v3uat_nocap_la_%j.log
#SBATCH --error=${CIPHER_DIR}/logs/v3uat_nocap_la_%j.log

set -euo pipefail
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
LA_DIR="\$(cd ${CIPHER_DIR}/../cipher-light-attention && pwd)"
cd "\${LA_DIR}"
export PYTHONPATH="\${LA_DIR}/src:${CIPHER_DIR}/src:\${PYTHONPATH:-}"

NAME="v3uat_nocap_light_attention_prott5_xl_seg8"
echo "Training: \${NAME}"
echo "  K positive list: ${POSITIVE_LIST_K}"
echo "  O positive list: ${POSITIVE_LIST_O}"
echo "  Embedding:       ${CAND_PROTT5_SEG8_NPZ}"
echo "  Caps:            DISABLED (no max_samples_per_k/o, no cluster_file)"

python -m cipher.cli.train_runner \\
    --model light_attention \\
    --positive_list_k "${POSITIVE_LIST_K}" \\
    --positive_list_o "${POSITIVE_LIST_O}" \\
    --label_strategy multi_label_threshold \\
    --max_k_types 3 \\
    --max_o_types 3 \\
    --min_class_samples 25 \\
    --lr 1e-05 \\
    --batch_size 64 \\
    --epochs 200 \\
    --patience 30 \\
    --embedding_type prott5_xl_seg8 \\
    --embedding_file "${CAND_PROTT5_SEG8_NPZ}" \\
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
echo "Eval: per-head strict (any-hit + per-pair, all 5 datasets)"
python ${CIPHER_DIR}/scripts/analysis/per_head_strict_eval.py "\${EXP_DIR}" \\
    --val-embedding-file "${VAL_PROTT5_SEG8_NPZ}"
SBATCH_EOF
echo "STEP 2 script: ${JOB2_SBATCH}"
JOB2_ID=$(submit "$JOB2_SBATCH")
echo "STEP 2 (v3_UAT no-cap LA, ProtT5 xl seg8): JOB ${JOB2_ID}"
fi

echo ""
echo "============================================================"
echo "v3_UAT no-cap ablation submitted. Tail logs in ${CIPHER_DIR}/logs/"
echo "Compare against baselines:"
echo "  sweep_prott5_mean_cl70       (MLP, current cap)"
echo "  la_v3_uat_prott5_xl_seg8     (LA, current cap)"
echo "============================================================"
