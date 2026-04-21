#!/usr/bin/env bash
#
# SLURM wrapper for scripts/analysis/eval_per_head.py.
# Submits one GPU job that runs the per-head (K / O / both) evaluation on
# every run_dir passed as an argument. If no args are given, defaults to all
# la_seg4_* directories under experiments/light_attention/.
#
# Usage:
#   bash scripts/analysis/run_eval_per_head.sh
#       # evaluate every la_seg4_* experiment sequentially in one job
#
#   bash scripts/analysis/run_eval_per_head.sh experiments/light_attention/la_seg4_match_sweep
#       # evaluate a single run
#
#   DRY_RUN=1 bash scripts/analysis/run_eval_per_head.sh
#       # preview the sbatch submission without submitting

set -euo pipefail

# ============================================================
# Delta-AI configuration
# ============================================================
ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="${CIPHER_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"

# SLURM resources — per-head eval is ~6 min per run_dir on seg4; allocate
# 15 min per run to give headroom (cl jobs may be a touch slower).
GPUS=1
CPUS=4
MEM="16G"

DRY_RUN="${DRY_RUN:-0}"

# ============================================================
# Collect run_dirs (args or default glob)
# ============================================================
if [ $# -gt 0 ]; then
    RUN_DIRS=("$@")
else
    # Default: every la_seg4_* run that has results/evaluation.json
    RUN_DIRS=()
    for d in "${CIPHER_DIR}/experiments/light_attention"/la_seg4_*/; do
        [ -d "$d" ] || continue
        RUN_DIRS+=("${d%/}")
    done
fi

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "ERROR: no run_dirs given and no la_seg4_* experiments found under ${CIPHER_DIR}/experiments/light_attention/" >&2
    exit 1
fi

# Convert to absolute paths; verify each looks like a real run_dir
ABS_RUN_DIRS=()
for d in "${RUN_DIRS[@]}"; do
    abs="$(cd "$d" && pwd)"
    if [ ! -f "${abs}/experiment.json" ]; then
        echo "ERROR: ${abs} does not contain experiment.json — not a run dir" >&2
        exit 1
    fi
    ABS_RUN_DIRS+=("$abs")
done

# Budget: 15 min per run_dir + 5 min startup
N=${#ABS_RUN_DIRS[@]}
MIN_TOTAL=$(( N * 15 + 5 ))
TIME_HH=$(printf '%02d' $(( MIN_TOTAL / 60 )))
TIME_MM=$(printf '%02d' $(( MIN_TOTAL % 60 )))
TIME="${TIME_HH}:${TIME_MM}:00"

# Job name: eval_per_head_<first-run-basename>[_plusN]
FIRST_NAME="$(basename "${ABS_RUN_DIRS[0]}")"
if [ $N -eq 1 ]; then
    JOB_NAME="evh_${FIRST_NAME}"
else
    JOB_NAME="evh_${FIRST_NAME}_plus$((N - 1))"
fi

echo "============================================================"
echo "EVAL PER HEAD (K / O / both)"
echo "  Cipher:    ${CIPHER_DIR}"
echo "  Runs (${N}):"
for d in "${ABS_RUN_DIRS[@]}"; do echo "    ${d}"; done
echo "  SLURM:     ${TIME}, ${GPUS} GPU, ${CPUS} CPU, ${MEM}"
echo "============================================================"
echo ""

# ============================================================
# Build the job script heredoc
# ============================================================
RUN_LIST="$(printf '    "%s"\n' "${ABS_RUN_DIRS[@]}")"

JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${CIPHER_DIR}/logs/${JOB_NAME}_%j.log
#SBATCH --error=${CIPHER_DIR}/logs/${JOB_NAME}_%j.log

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

RUNS=(
${RUN_LIST})

for RUN in \"\${RUNS[@]}\"; do
    echo \"\"
    echo \"======================================\"
    echo \"Per-head eval: \${RUN}\"
    echo \"Started: \$(date)\"
    echo \"======================================\"
    python3 scripts/analysis/eval_per_head.py \"\${RUN}\"
done

echo \"\"
echo \"======================================\"
echo \"All per-head evals finished: \$(date)\"
echo \"======================================\"
"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would submit the following job:"
    echo "---"
    echo "$JOB_SCRIPT"
    echo "---"
    echo "DRY RUN complete. Set DRY_RUN=0 to submit."
else
    mkdir -p "${CIPHER_DIR}/logs"
    JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
    echo "Submitted ${JOB_ID} — ${JOB_NAME}"
    echo "Monitor: squeue -u \$USER"
    echo "Log:     tail -f ${CIPHER_DIR}/logs/${JOB_NAME}_${JOB_ID}.log"
fi
