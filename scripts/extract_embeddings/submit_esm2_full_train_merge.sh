#!/usr/bin/env bash
#
# Submit the ESM-2 650M full per-residue TRAINING merge as one SLURM job.
#
# Two steps chained:
#   1. Recompress candidates_maxlen1024_embeddings.npz in place
#      (223 GB uncompressed → ~100 GB compressed, saves ~123 GB and
#       brings us back under quota before the merge writes anything).
#   2. Delete-as-you-go merge of all 7 length bins into a single
#      candidates_embeddings_full_md5.npz. Per-split commit + unlink,
#      so peak disk stays near current.
#
# Why this script exists (vs submit_merge.sh):
#   - submit_merge.sh has a 2h TIME limit, not enough for a ~400 GB
#     compressed write of per-residue arrays. This one gets 8h.
#   - Combines the prerequisite recompress step in the same job so you
#     don't have to babysit two.
#   - Default COMPRESS=1 (normal compression) because uncompressed
#     output would blow the hard quota.
#   - Default DELETE_INPUTS=1 because that's the whole point.
#
# Usage:
#   bash scripts/extract_embeddings/submit_esm2_full_train_merge.sh
#   DRY_RUN=1 bash scripts/extract_embeddings/submit_esm2_full_train_merge.sh
#
# After the job finishes, verify with:
#   python scripts/utils/check_full_npz_coverage.py \
#       --npz /work/hdd/bfzj/llindsey1/embeddings_full/candidates_embeddings_full_md5.npz \
#       --expected-dim 1280

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

ROOT="${ROOT:-/work/hdd/bfzj/llindsey1/embeddings_full}"
SPLIT_DIR="${ROOT}/split_embeddings"
OUTPUT="${OUTPUT:-${ROOT}/candidates_embeddings_full_md5.npz}"
MAXLEN1024_NPZ="${SPLIT_DIR}/candidates_maxlen1024_embeddings.npz"

# Set SKIP_RECOMPRESS=1 if you've already recompressed maxlen1024
# interactively (user did this 2026-04-23 pm). Saves the auto-check
# inside the job.
SKIP_RECOMPRESS="${SKIP_RECOMPRESS:-0}"

GPUS=1
CPUS=4
MEM="32G"
TIME="8:00:00"

DRY_RUN="${DRY_RUN:-0}"

NAME="esm2_full_train_merge"
LOG="${CIPHER_DIR}/logs/${NAME}_%j.log"

echo "============================================================"
echo "ESM-2 FULL TRAIN MERGE (recompress + delete-as-you-go)"
echo "  root:       ${ROOT}"
echo "  split dir:  ${SPLIT_DIR}"
echo "  output:     ${OUTPUT}"
echo "  maxlen1024: ${MAXLEN1024_NPZ}  (will be recompressed in-place)"
echo "============================================================"
echo ""

if [ ! -d "$SPLIT_DIR" ]; then
    echo "ERROR: split directory not found: ${SPLIT_DIR}"
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
#SBATCH --output=${LOG}
#SBATCH --error=${LOG}

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

echo \"======================================\"
echo \"ESM-2 full train merge: ${NAME}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

echo \"--- disk state at start ---\"
du -h ${SPLIT_DIR} 2>/dev/null || true
quota 2>/dev/null || true
echo \"\"

if [ \"${SKIP_RECOMPRESS}\" = \"1\" ]; then
    echo \"=== STEP 1: skipped (SKIP_RECOMPRESS=1) ===\"
else
    echo \"=== STEP 1: recompress maxlen1024 in place ===\"
    if [ -e \"${MAXLEN1024_NPZ}\" ]; then
        python ${CIPHER_DIR}/scripts/utils/recompress_npz.py ${MAXLEN1024_NPZ}
    else
        echo \"WARNING: ${MAXLEN1024_NPZ} not present; skipping recompress step.\"
    fi
    echo \"\"
    echo \"--- disk state after recompress ---\"
    du -h ${SPLIT_DIR} 2>/dev/null || true
    echo \"\"
fi

echo \"=== STEP 2: streaming merge with --delete-inputs ===\"
python ${CIPHER_DIR}/scripts/extract_embeddings/merge_split_embeddings.py \\
    --delete-inputs \\
    -i ${SPLIT_DIR} \\
    -o ${OUTPUT}

echo \"\"
echo \"--- disk state at end ---\"
ls -lah ${OUTPUT}
du -h ${SPLIT_DIR} 2>/dev/null || true
quota 2>/dev/null || true

echo \"\"
echo \"======================================\"
echo \"Done: ${NAME} at \$(date)\"
echo \"\"
echo \"Next step: verify with\"
echo \"  python scripts/utils/check_full_npz_coverage.py \\\\
echo \"      --npz ${OUTPUT} --expected-dim 1280\"
echo \"======================================\"
"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would submit: ${NAME}"
    echo "  step 1: recompress ${MAXLEN1024_NPZ}"
    echo "  step 2: merge --delete-inputs -i ${SPLIT_DIR} -o ${OUTPUT}"
else
    mkdir -p "${CIPHER_DIR}/logs"
    JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
    echo "Submitted ${JOB_ID} - ${NAME}"
fi
