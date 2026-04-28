#!/usr/bin/env bash
#
# Submit a SLURM job that runs scripts/analysis/dump_per_pair_phl_predictions.py
# for a Light Attention experiment, appending per-pair PHL ranks to
# results/analysis/per_pair_phl_predictions.csv (the file agent 5 buckets
# by homology-tier).
#
# Default: la_v3_uat_prott5_xl_seg8 (the v3_uat run agent 5 specifically
# asked for in inbox/agent2/2026-04-27-1356-from-agent5-need-v3uat-per-pair-ranks.md).
#
# Per-pair ranks are not affected by the fixed-denominator semantics
# change — unscorable pairs are emitted as blank rank, which is the
# right per-pair representation. So this can run independently of the
# new any-hit aggregation work.
#
# Env overrides:
#   CIPHER_DIR  light-attention worktree on Delta
#               (default: /projects/bfzj/llindsey1/PHI_TSP/cipher-light-attention)
#   EXP_DIR     experiment dir (default: ${CIPHER_DIR}/experiments/light_attention/la_v3_uat_prott5_xl_seg8)
#   MODEL_ID    short label for model_id column (default: la_v3_uat_prott5_xl_seg8_k_only)
#   VAL_EMB     validation NPZ matching the experiment's embedding type
#               (default: ProtT5-XL seg8 validation NPZ)
#   OUT_CSV     output CSV (default: ${CIPHER_DIR}/results/analysis/per_pair_phl_predictions.csv)
#   ACCOUNT, PARTITION, CONDA_ENV  (Delta defaults)
#   DRY_RUN=1   render but don't submit
#
# Usage:
#   bash scripts/analysis/run_dump_per_pair_phl_la.sh                      # v3_uat default
#   EXP_DIR=...la_v3_strict... MODEL_ID=la_v3_strict_prott5_xl_seg8_k_only \
#       bash scripts/analysis/run_dump_per_pair_phl_la.sh                  # any LA run

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer-light-attention}"
DATA_DIR="${DATA_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer/data}"

EXP_DIR="${EXP_DIR:-${CIPHER_DIR}/experiments/light_attention/la_v3_uat_prott5_xl_seg8}"
MODEL_ID="${MODEL_ID:-la_v3_uat_prott5_xl_seg8_k_only}"
VAL_EMB="${VAL_EMB:-/work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments8/validation_prott5_xl_segments8_md5.npz}"

VAL_FASTA="${VAL_FASTA:-${DATA_DIR}/validation_data/metadata/validation_rbps_all.faa}"
VAL_DATASETS_DIR="${VAL_DATASETS_DIR:-${DATA_DIR}/validation_data/HOST_RANGE}"
OUT_CSV="${OUT_CSV:-${CIPHER_DIR}/results/analysis/per_pair_phl_predictions.csv}"

GPUS=1
CPUS=4
MEM="32G"
TIME="1:00:00"

NAME="dump_pp_phl_${MODEL_ID:0:24}"
LOG="${CIPHER_DIR}/logs/${NAME}_%j.log"

mkdir -p "${CIPHER_DIR}/logs" "$(dirname "${OUT_CSV}")"

JOB_SCRIPT="${CIPHER_DIR}/logs/${NAME}_$(date +%Y%m%d_%H%M%S).sbatch"
cat > "$JOB_SCRIPT" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${NAME:0:32}
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

echo "=== DUMP PER-PAIR PHL (LA): ${NAME} ==="
echo "  cipher_dir: ${CIPHER_DIR}"
echo "  exp:        ${EXP_DIR}"
echo "  model_id:   ${MODEL_ID}"
echo "  val NPZ:    ${VAL_EMB}"
echo "  val FASTA:  ${VAL_FASTA}"
echo "  out CSV:    ${OUT_CSV}"
echo "Started: \$(date)"
echo ""

if [ ! -d "${EXP_DIR}" ]; then
    echo "ERROR: experiment dir not found: ${EXP_DIR}"
    ls "\$(dirname ${EXP_DIR})" 2>&1 || true
    exit 1
fi
if [ ! -f "${VAL_EMB}" ]; then
    echo "ERROR: validation NPZ not found: ${VAL_EMB}"
    exit 1
fi

python3 ${CIPHER_DIR}/scripts/analysis/dump_per_pair_phl_predictions.py \\
    "${EXP_DIR}" \\
    --model-id "${MODEL_ID}" \\
    --val-embedding-file "${VAL_EMB}" \\
    --val-fasta "${VAL_FASTA}" \\
    --val-datasets-dir "${VAL_DATASETS_DIR}" \\
    --out-csv "${OUT_CSV}"

echo ""
echo "Done: \$(date)"
echo "Appended rows for model_id=${MODEL_ID} to ${OUT_CSV}"
SBATCH_EOF

echo "============================================================"
echo "DUMP PER-PAIR PHL (LA) — ${MODEL_ID}"
echo "  Job script:    ${JOB_SCRIPT}"
echo "  Log:           ${LOG}"
echo "  Will append to ${OUT_CSV}"
echo "============================================================"

DRY_RUN="${DRY_RUN:-0}"
if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] sbatch script written but not submitted."
    echo "Submit with: sbatch ${JOB_SCRIPT}"
    exit 0
fi

JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $NF}')
echo "Submitted ${JOB_ID}"
