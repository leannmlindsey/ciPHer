#!/usr/bin/env bash
#
# Backfill the three non-default eval JSONs (k_only / o_only / raw) on
# experiments that finished before the launcher auto-ran them. Uses the
# first-class --head-mode and --score-norm flags (not the monkey-patch
# eval_per_head.py).
#
# Writes to {run_dir}/results/:
#   evaluation_k_only.json   — --head-mode k_only
#   evaluation_o_only.json   — --head-mode o_only
#   evaluation_raw.json      — --score-norm raw
#
# Skips any JSON that already exists (idempotent — safe to re-run).
#
# Usage:
#   bash scripts/analysis/run_extra_eval_modes.sh experiments/light_attention/la_v3_strict_prott5_xl_seg8 [...]
#   DRY_RUN=1 bash scripts/analysis/run_extra_eval_modes.sh <run_dir> [...]

set -euo pipefail

ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="${CIPHER_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"

GPUS=1
CPUS=4
MEM="16G"

DRY_RUN="${DRY_RUN:-0}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_dir> [run_dir ...]" >&2
    exit 1
fi

ABS_RUN_DIRS=()
for d in "$@"; do
    abs="$(cd "$d" && pwd)"
    if [ ! -f "${abs}/experiment.json" ]; then
        echo "ERROR: ${abs} does not contain experiment.json" >&2
        exit 1
    fi
    ABS_RUN_DIRS+=("$abs")
done

N=${#ABS_RUN_DIRS[@]}
# ~3 min per eval × 3 modes × N runs + 5 min startup
MIN_TOTAL=$(( N * 3 * 3 + 5 ))
TIME_HH=$(printf '%02d' $(( MIN_TOTAL / 60 )))
TIME_MM=$(printf '%02d' $(( MIN_TOTAL % 60 )))
TIME="${TIME_HH}:${TIME_MM}:00"

FIRST_NAME="$(basename "${ABS_RUN_DIRS[0]}")"
if [ $N -eq 1 ]; then
    JOB_NAME="evmodes_${FIRST_NAME}"
else
    JOB_NAME="evmodes_${FIRST_NAME}_plus$((N - 1))"
fi

echo "============================================================"
echo "BACKFILL EXTRA EVAL MODES (k_only / o_only / raw)"
echo "  Cipher:  ${CIPHER_DIR}"
echo "  Runs:    ${N}"
for d in "${ABS_RUN_DIRS[@]}"; do echo "    ${d}"; done
echo "  SLURM:   ${TIME}, ${GPUS} GPU, ${CPUS} CPU, ${MEM}"
echo "============================================================"

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
    echo \"Extra eval modes: \${RUN}\"
    echo \"Started: \$(date)\"
    echo \"======================================\"

    K_JSON=\"\${RUN}/results/evaluation_k_only.json\"
    O_JSON=\"\${RUN}/results/evaluation_o_only.json\"
    RAW_JSON=\"\${RUN}/results/evaluation_raw.json\"

    if [ -f \"\${K_JSON}\" ]; then
        echo \"SKIP k_only (already exists): \${K_JSON}\"
    else
        echo \"--- --head-mode k_only ---\"
        python3 -m cipher.evaluation.runner \"\${RUN}\" --head-mode k_only -o \"\${K_JSON}\"
    fi

    if [ -f \"\${O_JSON}\" ]; then
        echo \"SKIP o_only (already exists): \${O_JSON}\"
    else
        echo \"--- --head-mode o_only ---\"
        python3 -m cipher.evaluation.runner \"\${RUN}\" --head-mode o_only -o \"\${O_JSON}\"
    fi

    if [ -f \"\${RAW_JSON}\" ]; then
        echo \"SKIP raw (already exists): \${RAW_JSON}\"
    else
        echo \"--- --score-norm raw ---\"
        python3 -m cipher.evaluation.runner \"\${RUN}\" --score-norm raw -o \"\${RAW_JSON}\"
    fi
done

echo \"\"
echo \"======================================\"
echo \"All extra eval modes finished: \$(date)\"
echo \"Compare: python3 scripts/analysis/show_eval_all.py --variant {k_only|o_only|raw|default}\"
echo \"======================================\"
"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would submit:"
    echo "$JOB_SCRIPT"
else
    mkdir -p "${CIPHER_DIR}/logs"
    JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
    echo "Submitted ${JOB_ID} — ${JOB_NAME}"
    echo "Log: ${CIPHER_DIR}/logs/${JOB_NAME}_${JOB_ID}.log"
fi
