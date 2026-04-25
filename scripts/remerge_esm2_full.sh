#!/usr/bin/env bash
#
# Re-merge the ESM-2 650M per-residue split embeddings into a single NPZ.
# Necessary because the 2026-04-24 merge produced a 213 GB file with a
# truncated central-directory record (np.load raises BadZipFile),
# probably due to a quota hit during the final write.
#
# split_embeddings/ itself is intact, so we don't need to re-extract --
# only re-merge.
#
# Usage:
#   bash scripts/remerge_esm2_full.sh
#   DRY_RUN=1 bash scripts/remerge_esm2_full.sh

set -euo pipefail

ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
CONDA_ENV="${CONDA_ENV:-esmfold2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="${CIPHER_DIR:-$(dirname "$SCRIPT_DIR")}"
# Use the main worktree's checkout of the merge script — same script lives
# in this branch via the merge, but DATA_DIR convention applies.
MAIN_CIPHER_DIR="${MAIN_CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

INPUT_DIR="${INPUT_DIR:-/work/hdd/bfzj/llindsey1/embeddings_full/split_embeddings}"
OUTPUT_NPZ="${OUTPUT_NPZ:-/work/hdd/bfzj/llindsey1/embeddings_full/candidates_embeddings_full_md5.npz}"

# Resources: per-residue merge holds all (L, 1280) arrays in a dict before
# np.savez_compressed. ~250-300 GB peak RAM expected. mem=0 takes whole node.
# Compression dominates wall time (the original failed merge took ~7 h);
# 12 h budget is comfortable. The job uses CPU only, but per the
# delta-requires-GPU rule it still requests one.
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-0}"
TIME="${TIME:-12:00:00}"

# Optional: pass --no-compress to merge_split_embeddings.py for speed,
# accepting larger on-disk size.
NO_COMPRESS="${NO_COMPRESS:-0}"
EXTRA_ARGS=""
if [ "${NO_COMPRESS}" = "1" ]; then
    EXTRA_ARGS="--no-compress"
fi

NAME="${NAME:-remerge_esm2_650m_full}"
DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "Re-merge ESM-2 650M full split embeddings"
echo "  Cipher dir:    ${CIPHER_DIR}"
echo "  Input dir:     ${INPUT_DIR}"
echo "  Output NPZ:    ${OUTPUT_NPZ}"
echo "  Compression:   $([ "${NO_COMPRESS}" = "1" ] && echo "off" || echo "on")"
echo "============================================================"

if [ ! -d "${INPUT_DIR}" ] && [ "${DRY_RUN}" != "1" ]; then
    echo "ERROR: split_embeddings dir not found: ${INPUT_DIR}" >&2
    exit 1
fi

JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${CIPHER_DIR}/logs/${NAME}_%j.log
#SBATCH --error=${CIPHER_DIR}/logs/${NAME}_%j.log

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

echo \"======================================\"
echo \"Re-merge ${NAME}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

# Remove the corrupt prior output so the merge writes fresh. (The merge
# script doesn't atomically replace — failed writes can leave junk.)
if [ -f \"${OUTPUT_NPZ}\" ]; then
    echo \"Removing existing (corrupt) ${OUTPUT_NPZ}\"
    rm -f \"${OUTPUT_NPZ}\"
fi

python ${MAIN_CIPHER_DIR}/scripts/extract_embeddings/merge_split_embeddings.py \\
    -i ${INPUT_DIR} \\
    -o ${OUTPUT_NPZ} \\
    ${EXTRA_ARGS}

echo \"\"
echo \"=== verification ===\"
ls -lh ${OUTPUT_NPZ}
python -c \"import numpy as np; d = np.load('${OUTPUT_NPZ}'); keys = list(d.files); print(f'Loaded {len(keys)} keys; first key shape = {d[keys[0]].shape}')\"

echo \"\"
echo \"======================================\"
echo \"Done: ${NAME}\"
echo \"  Finished: \$(date)\"
echo \"======================================\"
"

if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY RUN] job script follows:"
    echo "----------------------------------------"
    echo "${JOB_SCRIPT}"
    echo "----------------------------------------"
else
    mkdir -p "${CIPHER_DIR}/logs"
    JOB_ID=$(echo "${JOB_SCRIPT}" | sbatch | awk '{print $NF}')
    echo "Submitted job ${JOB_ID} — ${NAME}"
    echo "Log: ${CIPHER_DIR}/logs/${NAME}_${JOB_ID}.log"
fi
