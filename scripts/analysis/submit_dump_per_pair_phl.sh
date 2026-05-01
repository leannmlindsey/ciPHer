#!/usr/bin/env bash
#
# Submit a SLURM job that runs scripts/analysis/dump_per_pair_phl_predictions.py
# for the seg4_cl70 best-K Delta experiment (or any experiment whose
# val NPZ lives only on Delta), appending to results/analysis/per_pair_phl_predictions.csv.
#
# Built for agent 5's homology-bucket analysis. Laptop already wrote the
# laptop_repro and highconf_pipeline_K_prott5_mean rows; this fills in
# the seg4_cl70 row.
#
# Env overrides:
#   EXP_DIR    cipher experiment dir (default: sweep_posList_esm2_650m_seg4_cl70)
#   MODEL_ID   short label for model_id column (default: seg4_cl70)
#   VAL_EMB    validation NPZ matching the experiment's embedding type
#              (default: validation_embeddings_segments4)
#   ACCOUNT, PARTITION, CONDA_ENV, CIPHER_DIR  (standard Delta defaults)
#   DRY_RUN=1  render but don't submit
#
# Usage:
#   bash scripts/analysis/submit_dump_per_pair_phl.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

EXP_DIR="${EXP_DIR:-${CIPHER_DIR}/experiments/attention_mlp/sweep_posList_esm2_650m_seg4_cl70}"
MODEL_ID="${MODEL_ID:-seg4_cl70}"
VAL_EMB="${VAL_EMB:-/work/hdd/bfzj/llindsey1/validation_embeddings_segments4/validation_embeddings_segments4_md5.npz}"

VAL_FASTA="${VAL_FASTA:-${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa}"
VAL_DATASETS_DIR="${VAL_DATASETS_DIR:-${CIPHER_DIR}/data/validation_data/HOST_RANGE}"
OUT_CSV="${OUT_CSV:-${CIPHER_DIR}/results/analysis/per_pair_phl_predictions.csv}"

GPUS=1   # Delta requires a GPU even for CPU-bound jobs
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

echo "=== DUMP PER-PAIR PHL: ${NAME} ==="
echo "  exp:      ${EXP_DIR}"
echo "  model_id: ${MODEL_ID}"
echo "  val NPZ:  ${VAL_EMB}"
echo "  out CSV:  ${OUT_CSV}"
echo "Started: \$(date)"
echo ""

python3 ${CIPHER_DIR}/scripts/analysis/dump_per_pair_phl_predictions.py \\
    "${EXP_DIR}" \\
    --model-id "${MODEL_ID}" \\
    --val-embedding-file "${VAL_EMB}" \\
    --val-fasta "${VAL_FASTA}" \\
    --val-datasets-dir "${VAL_DATASETS_DIR}" \\
    --out-csv "${OUT_CSV}"

echo ""
echo "Done: \$(date)"
SBATCH_EOF

echo "============================================================"
echo "DUMP PER-PAIR PHL — ${MODEL_ID}"
echo "  Job script:  ${JOB_SCRIPT}"
echo "  Log:         ${LOG}"
echo "  Will append to: ${OUT_CSV}"
echo "============================================================"

DRY_RUN="${DRY_RUN:-0}"
if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] sbatch script written but not submitted."
    echo "Submit with: sbatch ${JOB_SCRIPT}"
    exit 0
fi

JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $NF}')
echo "Submitted ${JOB_ID}"
