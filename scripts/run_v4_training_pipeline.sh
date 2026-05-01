#!/usr/bin/env bash
#
# v4 training pipeline — A/B against v3_UAT.
#
# Trains cipher's two production architectures (attention_mlp +
# light_attention) on the v4 K-list (38,162 IDs) paired with v3's O-list
# (HC_O_UAT_multitop.list — v4 is K-side-only per agent 4's hand-off).
#
# Four SLURM jobs:
#   1. ProtT5 mean embed of v4 additions (1,067 new seqs from A25+A28)
#   2. ProtT5 xl seg8 embed of v4 additions
#   3. Merge embeddings + attention_mlp train (depends on 1)
#   4. Merge embeddings + light_attention train  (depends on 2)
#
# Run scope: same hyperparameters as sweep_prott5_mean_cl70 (MLP) and
# la_v3_uat_prott5_xl_seg8 (LA) — only the K positive list changes.
#
# Env overrides:
#   ACCOUNT, PARTITION, CONDA_ENV, CIPHER_DIR
#   STEP={1,2,3,4,all}  (default all)
#   DRY_RUN=1           render sbatch but do not submit
#   NO_DEP=1            don't add --dependency=afterok (use existing
#                       merged NPZ; useful when re-running just step 3 or 4)
#   POSITIVE_LIST_K     override v4 list (default pipeline_positive_v4.list)
#                       options: pipeline_positive_v4_no_a23.list,
#                                pipeline_positive_v4_clean_only.list

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"
STEP="${STEP:-all}"
DRY_RUN="${DRY_RUN:-0}"
NO_DEP="${NO_DEP:-0}"

V4_DIR="${CIPHER_DIR}/data/training_data/metadata/highconf_v4"
V3_O_LIST="${CIPHER_DIR}/data/training_data/metadata/highconf_v3_multitop/HC_O_UAT_multitop.list"
POSITIVE_LIST_K="${POSITIVE_LIST_K:-${V4_DIR}/pipeline_positive_v4.list}"
ASSOC_MAP_V4="${V4_DIR}/host_phage_protein_map_v4.tsv"
# v4 companion files (built by scripts/data_prep/build_v4_companion_files.py)
GLYCAN_TSV="${V4_DIR}/glycan_binders_v4.tsv"
CLUSTER_FILE_V4="${V4_DIR}/candidates_clusters_v4.tsv"
V4_ADDITIONS_FAA="${V4_DIR}/candidates_v4_additions.faa"

# Existing cipher candidate embeddings (we'll merge v4 additions on top)
EXISTING_PROTT5_MEAN_NPZ="/projects/bfzj/llindsey1/RBP_Structural_Similarity/output/embeddings_prott5/candidates_embeddings_md5.npz"
EXISTING_PROTT5_SEG8_NPZ="/work/hdd/bfzj/llindsey1/embeddings/prott5_xl_segments8/candidates_prott5_xl_segments8_md5.npz"

# v4 additions embedding outputs (just the 1,067 new seqs)
V4_EMB_DIR="/work/hdd/bfzj/llindsey1/v4_embeddings"
V4_PROTT5_MEAN_NPZ="${V4_EMB_DIR}/v4_additions_prott5_mean.npz"
V4_PROTT5_SEG8_NPZ="${V4_EMB_DIR}/v4_additions_prott5_xl_seg8.npz"

# Merged training NPZs (existing ∪ v4 additions, md5-keyed)
MERGED_PROTT5_MEAN_NPZ="${V4_EMB_DIR}/v4_merged_prott5_mean.npz"
MERGED_PROTT5_SEG8_NPZ="${V4_EMB_DIR}/v4_merged_prott5_xl_seg8.npz"

# Validation NPZs (unchanged from v3)
VAL_FASTA="${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${CIPHER_DIR}/data/validation_data/HOST_RANGE"
VAL_PROTT5_MEAN_NPZ="/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz"
VAL_PROTT5_SEG8_NPZ="/work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments8/validation_prott5_xl_segments8_md5.npz"

mkdir -p "${CIPHER_DIR}/logs"

submit() {
    local script="$1"
    local extra_args="${2:-}"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY] sbatch ${extra_args} ${script}"
        echo "DRY_${RANDOM}"
    else
        sbatch ${extra_args} "${script}" | awk '{print $NF}'
    fi
}

# ────────────────────────────────────────────────────────────────────
# Step 1: ProtT5 mean embed of v4 additions
# ────────────────────────────────────────────────────────────────────
JOB1_ID=""
if [ "$STEP" = "all" ] || [ "$STEP" = "1" ]; then
JOB1_SBATCH="${CIPHER_DIR}/logs/v4_emb_prott5_mean_$(date +%Y%m%d_%H%M%S).sbatch"
cat > "$JOB1_SBATCH" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=v4_emb_pt5_mean
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=${CIPHER_DIR}/logs/v4_emb_prott5_mean_%j.log
#SBATCH --error=${CIPHER_DIR}/logs/v4_emb_prott5_mean_%j.log

set -euo pipefail
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate prott5
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}
mkdir -p ${V4_EMB_DIR}

echo "Embedding ${V4_ADDITIONS_FAA} → ${V4_PROTT5_MEAN_NPZ}"
python scripts/extract_embeddings/prott5_extract.py \\
    "${V4_ADDITIONS_FAA}" "${V4_PROTT5_MEAN_NPZ}" \\
    --key_by_md5
echo "Done: \$(date)"
SBATCH_EOF
echo "Step 1 script: ${JOB1_SBATCH}"
JOB1_ID=$(submit "$JOB1_SBATCH")
echo "Step 1 (v4 ProtT5 mean embed): JOB ${JOB1_ID}"
fi

# ────────────────────────────────────────────────────────────────────
# Step 2: ProtT5 xl seg8 embed of v4 additions (parallel to step 1)
# ────────────────────────────────────────────────────────────────────
JOB2_ID=""
if [ "$STEP" = "all" ] || [ "$STEP" = "2" ]; then
JOB2_SBATCH="${CIPHER_DIR}/logs/v4_emb_prott5_seg8_$(date +%Y%m%d_%H%M%S).sbatch"
cat > "$JOB2_SBATCH" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=v4_emb_pt5_seg8
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=${CIPHER_DIR}/logs/v4_emb_prott5_seg8_%j.log
#SBATCH --error=${CIPHER_DIR}/logs/v4_emb_prott5_seg8_%j.log

set -euo pipefail
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate prott5
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}
mkdir -p ${V4_EMB_DIR}

echo "Embedding ${V4_ADDITIONS_FAA} → ${V4_PROTT5_SEG8_NPZ} (segments8)"
python scripts/extract_embeddings/prott5_extract.py \\
    "${V4_ADDITIONS_FAA}" "${V4_PROTT5_SEG8_NPZ}" \\
    --key_by_md5 \\
    --pooling segments8
echo "Done: \$(date)"
SBATCH_EOF
echo "Step 2 script: ${JOB2_SBATCH}"
JOB2_ID=$(submit "$JOB2_SBATCH")
echo "Step 2 (v4 ProtT5 seg8 embed): JOB ${JOB2_ID}"
fi

# ────────────────────────────────────────────────────────────────────
# Step 3: Merge embeddings + train attention_mlp on v4 (depends on 1)
# ────────────────────────────────────────────────────────────────────
if [ "$STEP" = "all" ] || [ "$STEP" = "3" ]; then
JOB3_SBATCH="${CIPHER_DIR}/logs/v4_train_mlp_prott5_mean_$(date +%Y%m%d_%H%M%S).sbatch"
cat > "$JOB3_SBATCH" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=v4_train_mlp
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=${CIPHER_DIR}/logs/v4_train_mlp_prott5_mean_%j.log
#SBATCH --error=${CIPHER_DIR}/logs/v4_train_mlp_prott5_mean_%j.log

set -euo pipefail
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

# 3a. Merge existing candidate ProtT5 mean embeddings + v4 additions.
# python script that loads two NPZs and saves union (md5-keyed).
python -c "
import numpy as np, os
existing = np.load('${EXISTING_PROTT5_MEAN_NPZ}')
v4 = np.load('${V4_PROTT5_MEAN_NPZ}')
merged = {k: existing[k] for k in existing.files}
n_added = 0
for k in v4.files:
    if k not in merged:
        merged[k] = v4[k]
        n_added += 1
np.savez('${MERGED_PROTT5_MEAN_NPZ}', **merged)
print(f'Merged: {len(existing.files)} existing + {n_added} new = {len(merged)} keys')
"

NAME="v4_attention_mlp_prott5_mean"
echo "Training: \${NAME}"
echo "  K positive list: ${POSITIVE_LIST_K}"
echo "  O positive list: ${V3_O_LIST}"
echo "  Embedding:       ${MERGED_PROTT5_MEAN_NPZ}"

python -m cipher.cli.train_runner \\
    --model attention_mlp \\
    --positive_list_k "${POSITIVE_LIST_K}" \\
    --positive_list_o "${V3_O_LIST}" \\
    --label_strategy multi_label_threshold \\
    --max_k_types 3 \\
    --max_o_types 3 \\
    --max_samples_per_k 1000 \\
    --max_samples_per_o 3000 \\
    --min_class_samples 25 \\
    --min_sources 1 \\
    --cluster_file "${CLUSTER_FILE_V4}" \\
    --cluster_threshold 70 \\
    --lr 1e-05 \\
    --batch_size 512 \\
    --epochs 1000 \\
    --patience 30 \\
    --embedding_type prott5_mean \\
    --embedding_file "${MERGED_PROTT5_MEAN_NPZ}" \\
    --association_map "${ASSOC_MAP_V4}" \\
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
echo "Step 3 script: ${JOB3_SBATCH}"
DEP=""
if [ "$NO_DEP" != "1" ] && [ -n "$JOB1_ID" ] && [ "$DRY_RUN" != "1" ]; then
    DEP="--dependency=afterok:${JOB1_ID}"
fi
JOB3_ID=$(submit "$JOB3_SBATCH" "${DEP}")
echo "Step 3 (v4 MLP train + eval, ProtT5 mean): JOB ${JOB3_ID}${DEP:+  (depends on ${JOB1_ID})}"
fi

# ────────────────────────────────────────────────────────────────────
# Step 4: Merge embeddings + train light_attention on v4 (depends on 2)
# ────────────────────────────────────────────────────────────────────
if [ "$STEP" = "all" ] || [ "$STEP" = "4" ]; then
JOB4_SBATCH="${CIPHER_DIR}/logs/v4_train_la_prott5_seg8_$(date +%Y%m%d_%H%M%S).sbatch"
cat > "$JOB4_SBATCH" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=v4_train_la
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=${CIPHER_DIR}/logs/v4_train_la_prott5_seg8_%j.log
#SBATCH --error=${CIPHER_DIR}/logs/v4_train_la_prott5_seg8_%j.log

set -euo pipefail
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
LA_DIR="\$(cd ${CIPHER_DIR}/../cipher-light-attention && pwd)"
cd "\${LA_DIR}"
export PYTHONPATH="\${LA_DIR}/src:${CIPHER_DIR}/src:\${PYTHONPATH:-}"

# Merge existing candidate ProtT5 xl seg8 embeddings + v4 additions
python -c "
import numpy as np, os
existing = np.load('${EXISTING_PROTT5_SEG8_NPZ}')
v4 = np.load('${V4_PROTT5_SEG8_NPZ}')
merged = {k: existing[k] for k in existing.files}
n_added = 0
for k in v4.files:
    if k not in merged:
        merged[k] = v4[k]
        n_added += 1
np.savez('${MERGED_PROTT5_SEG8_NPZ}', **merged)
print(f'Merged: {len(existing.files)} existing + {n_added} new = {len(merged)} keys')
"

NAME="v4_light_attention_prott5_xl_seg8"
echo "Training: \${NAME}"
echo "  K positive list: ${POSITIVE_LIST_K}"
echo "  O positive list: ${V3_O_LIST}"
echo "  Embedding:       ${MERGED_PROTT5_SEG8_NPZ}"

python -m cipher.cli.train_runner \\
    --model light_attention \\
    --positive_list_k "${POSITIVE_LIST_K}" \\
    --positive_list_o "${V3_O_LIST}" \\
    --label_strategy multi_label_threshold \\
    --max_k_types 3 \\
    --max_o_types 3 \\
    --min_class_samples 25 \\
    --cluster_file "${CLUSTER_FILE_V4}" \\
    --cluster_threshold 70 \\
    --lr 1e-05 \\
    --batch_size 64 \\
    --epochs 200 \\
    --patience 30 \\
    --embedding_type prott5_xl_seg8 \\
    --embedding_file "${MERGED_PROTT5_SEG8_NPZ}" \\
    --association_map "${ASSOC_MAP_V4}" \\
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
echo "Step 4 script: ${JOB4_SBATCH}"
DEP=""
if [ "$NO_DEP" != "1" ] && [ -n "$JOB2_ID" ] && [ "$DRY_RUN" != "1" ]; then
    DEP="--dependency=afterok:${JOB2_ID}"
fi
JOB4_ID=$(submit "$JOB4_SBATCH" "${DEP}")
echo "Step 4 (v4 LA train + eval, ProtT5 xl seg8): JOB ${JOB4_ID}${DEP:+  (depends on ${JOB2_ID})}"
fi

echo ""
echo "============================================================"
echo "v4 pipeline scheduled. Tail logs in ${CIPHER_DIR}/logs/"
echo "K positive list: ${POSITIVE_LIST_K}"
echo "Re-run individual steps with STEP=N (1, 2, 3, or 4)."
echo "============================================================"
