#!/usr/bin/env bash
#
# Submit a SLURM job that runs scripts/analysis/per_head_strict_eval.py
# on an existing experiment dir. Use when:
#   - The training launcher's auto-eval failed mid-stream (e.g. an
#     unknown CLI flag tripped `set -e` in an older launcher)
#   - You manually need to re-run strict-eval on an existing experiment
#     (different val NPZ, recovered training, etc.)
#
# Memory: kmer_aa20_k4 (160k-d) × ~600 validation proteins is large
# enough to OOM the head node; that's the failure mode this script
# exists to fix. SLURM-allocated 32G + 1 GPU is comfortable for any
# single-experiment strict-eval.
#
# Env overrides:
#   EXP_DIR     experiment directory (REQUIRED unless --pos arg given)
#   VAL_EMB     validation NPZ matching the experiment's embedding type (REQUIRED)
#   VAL_FASTA   default: ${DATA_DIR}/validation_data/metadata/validation_rbps_all.faa
#   VAL_DS      default: ${DATA_DIR}/validation_data/HOST_RANGE
#   CIPHER_DIR  default: parent of this script's dir (the worktree root)
#   DATA_DIR    default: /projects/bfzj/llindsey1/PHI_TSP/ciPHer/data
#   ACCOUNT, PARTITION, CONDA_ENV  (Delta defaults)
#   DRY_RUN=1   render but don't submit
#
# Usage:
#   EXP_DIR=experiments/binary_mlp/binary_mlp_kmer_aa20_k4_cl70 \
#   VAL_EMB=/work/hdd/bfzj/llindsey1/kmer_features/validation_aa20_k4.npz \
#       bash scripts/analysis/run_strict_eval_sbatch.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="${CIPHER_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
DATA_DIR="${DATA_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer/data}"

VAL_FASTA="${VAL_FASTA:-${DATA_DIR}/validation_data/metadata/validation_rbps_all.faa}"
VAL_DS="${VAL_DS:-${DATA_DIR}/validation_data/HOST_RANGE}"

EXP_DIR="${EXP_DIR:-${1:-}}"
VAL_EMB="${VAL_EMB:-}"

if [ -z "$EXP_DIR" ] || [ -z "$VAL_EMB" ]; then
    echo "ERROR: EXP_DIR and VAL_EMB are required." >&2
    echo "Usage: EXP_DIR=<dir> VAL_EMB=<npz> bash $0" >&2
    exit 1
fi

# Resolve EXP_DIR to absolute (so SLURM compute node finds it)
case "$EXP_DIR" in
    /*) ;;
    *) EXP_DIR="${CIPHER_DIR}/${EXP_DIR}" ;;
esac

if [ ! -d "$EXP_DIR" ]; then
    echo "ERROR: experiment dir not found: $EXP_DIR" >&2
    exit 1
fi
if [ ! -f "$VAL_EMB" ]; then
    echo "ERROR: validation NPZ not found: $VAL_EMB" >&2
    exit 1
fi

NAME="strict_eval_$(basename "$EXP_DIR" | head -c 24)_$(date +%Y%m%d_%H%M%S)"

GPUS=1
CPUS=4
MEM="32G"
# Generous default — per-class architectures (binary_mlp's 156 K-heads,
# 22 O-heads) run inference sequentially per class, so PHL with
# n_candidates×n_proteins×n_classes forward passes can take >30 min.
# Better to over-allocate than time-out partway through. Override with
# TIME=NN:MM:SS env var if a particular run is known fast.
TIME="${TIME:-4:00:00}"

mkdir -p "${CIPHER_DIR}/logs"
LOG="${CIPHER_DIR}/logs/${NAME}_%j.log"

JOB_SCRIPT="${CIPHER_DIR}/logs/${NAME}.sbatch"
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

echo "=== strict-eval rescue: ${NAME} ==="
echo "  exp:       ${EXP_DIR}"
echo "  val NPZ:   ${VAL_EMB}"
echo "  val FASTA: ${VAL_FASTA}"
echo "  val DS:    ${VAL_DS}"
echo "Started: \$(date)"
echo ""

python -u ${CIPHER_DIR}/scripts/analysis/per_head_strict_eval.py "${EXP_DIR}" \\
    --val-embedding-file "${VAL_EMB}" \\
    --val-fasta "${VAL_FASTA}" \\
    --val-datasets-dir "${VAL_DS}"

echo ""
echo "Done: \$(date)"
echo "Result: ${EXP_DIR}/results/per_head_strict_eval.json"
SBATCH_EOF

echo "============================================================"
echo "STRICT-EVAL RESCUE — ${NAME}"
echo "  Exp:        ${EXP_DIR}"
echo "  Val NPZ:    ${VAL_EMB}"
echo "  Job script: ${JOB_SCRIPT}"
echo "  Log:        ${LOG}"
echo "  SLURM:      ${TIME}, ${GPUS} GPU, ${CPUS} CPU, ${MEM}"
echo "============================================================"

DRY_RUN="${DRY_RUN:-0}"
if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] sbatch script written; not submitted."
    echo "Submit with: sbatch ${JOB_SCRIPT}"
    exit 0
fi

JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $NF}')
echo "Submitted ${JOB_ID}"
