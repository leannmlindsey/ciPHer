#!/usr/bin/env bash
#
# Submit a SLURM job that runs OLD-style class-ranking eval on a
# dual-head combination — K head from one experiment, O head from
# another, each with its own validation embedding NPZ.
#
# Companion to scripts/analysis/submit_old_style_eval.sh, which scores
# single experiments. Use this when you've identified strong K and O
# heads in different experiments and want to test the mix.
#
# Output: results/dual_head_old_style/<K_run>_x_<O_run>/old_style_eval.json
# Pickup by scripts/analysis/old_style_eval_summary.py for tabulation.
#
# Defaults: best Delta K (sweep_posList_esm2_650m_seg4_cl70) ×
#           strongest known O on PHL (LAPTOP repro O). Override the
#           paths with env vars below.
#
# Env overrides:
#   K_DIR           K experiment dir (must contain model_k/best_model.pt)
#   O_DIR           O experiment dir (must contain model_o/best_model.pt)
#   K_VAL_EMB       validation NPZ for K head (must match K embedding type)
#   O_VAL_EMB       validation NPZ for O head (must match O embedding type)
#   ACCOUNT, PARTITION, CONDA_ENV, CIPHER_DIR  (standard Delta defaults)
#   DRY_RUN=1       render the sbatch but do not submit
#
# Usage:
#   bash scripts/analysis/submit_dual_head_old_style_eval.sh
#   K_DIR=... O_DIR=... K_VAL_EMB=... O_VAL_EMB=... \
#     bash scripts/analysis/submit_dual_head_old_style_eval.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

K_DIR="${K_DIR:-${CIPHER_DIR}/experiments/attention_mlp/sweep_posList_esm2_650m_seg4_cl70}"
# Delta-side equivalent of the LAPTOP repro (same bit-exact OLD recipe).
# Adjust if the actual Delta path differs — verify by checking that
# experiments/attention_mlp/repro_old_v3_full_in_cipher/model_o/best_model.pt exists.
O_DIR="${O_DIR:-${CIPHER_DIR}/experiments/attention_mlp/repro_old_v3_full_in_cipher}"

K_VAL_EMB="${K_VAL_EMB:-/work/hdd/bfzj/llindsey1/validation_embeddings_segments4/validation_embeddings_segments4_md5.npz}"
O_VAL_EMB="${O_VAL_EMB:-/projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/validation_data/combined/validation_inputs/validation_embeddings_md5.npz}"

VAL_FASTA="${VAL_FASTA:-${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa}"
VAL_DATASETS_DIR="${VAL_DATASETS_DIR:-${CIPHER_DIR}/data/validation_data/HOST_RANGE}"

GPUS=1   # Delta requires a GPU even for CPU-bound jobs
CPUS=4
MEM="32G"
TIME="1:00:00"

K_RUN=$(basename "${K_DIR}")
O_RUN=$(basename "${O_DIR}")
NAME="dual_old_eval_${K_RUN:0:24}_x_${O_RUN:0:16}"
LOG="${CIPHER_DIR}/logs/${NAME}_%j.log"

mkdir -p "${CIPHER_DIR}/logs"

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

echo "=== DUAL-HEAD OLD-STYLE EVAL: ${NAME} ==="
echo "  K: ${K_DIR}"
echo "  O: ${O_DIR}"
echo "  K NPZ: ${K_VAL_EMB}"
echo "  O NPZ: ${O_VAL_EMB}"
echo "Started: \$(date)"
echo ""

python3 ${CIPHER_DIR}/scripts/analysis/old_style_eval.py \\
    --k-experiment-dir "${K_DIR}" \\
    --o-experiment-dir "${O_DIR}" \\
    --k-val-embedding-file "${K_VAL_EMB}" \\
    --o-val-embedding-file "${O_VAL_EMB}" \\
    --val-fasta "${VAL_FASTA}" \\
    --val-datasets-dir "${VAL_DATASETS_DIR}"

echo ""
echo "Done: \$(date)"
SBATCH_EOF

echo "============================================================"
echo "DUAL-HEAD OLD-STYLE EVAL"
echo "  K source: ${K_DIR}"
echo "  O source: ${O_DIR}"
echo "  Job script: ${JOB_SCRIPT}"
echo "  Log:        ${LOG}"
echo "============================================================"

DRY_RUN="${DRY_RUN:-0}"
if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] sbatch script written but not submitted."
    echo "Submit with: sbatch ${JOB_SCRIPT}"
    exit 0
fi

JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $NF}')
echo "Submitted ${JOB_ID}"
