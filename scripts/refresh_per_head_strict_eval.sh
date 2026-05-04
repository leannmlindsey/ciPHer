#!/usr/bin/env bash
#
# Refresh per_head_strict_eval.json across every cipher experiment that
# already has a results/per_head_strict_eval.json. Required after the
# 2026-04-30 fix to scripts/analysis/per_head_strict_eval.py removed the
# `if not annotated_md5s: continue` skip — phages with zero annotated
# RBPs are now KEPT in the strict denominator (they count as miss in
# every HR@k), restoring the project-policy fixed denominator (PHL=127,
# PBIP=104, UCSD=11, CHEN=3, GORODNICHIV=3).
#
# Designed for Delta-AI (SLURM). One job per experiment_dir. After this
# completes, regenerate the harvest CSV with:
#   python scripts/analysis/harvest_results.py --experiments-dirs \
#       experiments ../cipher-*/experiments
#
# Usage:
#   bash scripts/refresh_per_head_strict_eval.sh                 # all worktrees
#   bash scripts/refresh_per_head_strict_eval.sh main-only       # just $CIPHER_DIR/experiments
#   DRY_RUN=1 bash scripts/refresh_per_head_strict_eval.sh       # preview
#
# Override paths if not running with the canonical layout:
#   CIPHER_DIR=/path/to/cipher DATA_DIR=/path/to/data \
#       bash scripts/refresh_per_head_strict_eval.sh

set -euo pipefail

# ============================================================
# Delta-AI configuration
# ============================================================
ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"
DATA_DIR="${DATA_DIR:-${CIPHER_DIR}/data}"

GPUS=1
CPUS=4
MEM="32G"
TIME="01:00:00"

VAL_FASTA="${DATA_DIR}/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${DATA_DIR}/validation_data/HOST_RANGE"

# ============================================================
# Find experiment dirs to refresh
# ============================================================
SCOPE="${1:-all}"
ROOTS=()
case "$SCOPE" in
    main-only)
        ROOTS+=("${CIPHER_DIR}/experiments")
        ;;
    all)
        # Main + every sibling worktree (matches harvest_results convention)
        for d in "${CIPHER_DIR}/experiments" \
                 "$(dirname "$CIPHER_DIR")"/cipher-*/experiments; do
            [[ -d "$d" ]] && ROOTS+=("$d")
        done
        ;;
    *)
        echo "ERROR: unknown scope '$SCOPE'. Use 'main-only' or 'all'." >&2
        exit 1
        ;;
esac

echo "Scanning roots:"
for r in "${ROOTS[@]}"; do echo "  $r"; done

# Collect all experiment_dirs that already have a per_head_strict_eval.json.
# (We refresh exactly the runs that have been evaluated before — not the
# whole experiments/ tree, which includes in-progress / abandoned runs.)
EXP_DIRS=()
for root in "${ROOTS[@]}"; do
    while IFS= read -r f; do
        EXP_DIRS+=("$(dirname "$(dirname "$f")")")
    done < <(find "$root" -maxdepth 4 -name 'per_head_strict_eval.json' 2>/dev/null)
done
echo "Found ${#EXP_DIRS[@]} experiment_dirs to refresh."

DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${CIPHER_DIR}/scripts/_logs/refresh_per_head"
mkdir -p "$LOG_DIR"

# ============================================================
# Submit one SLURM job per experiment_dir
# ============================================================
for EXP_DIR in "${EXP_DIRS[@]}"; do
    RUN_NAME="$(basename "$EXP_DIR")"
    ARCH="$(basename "$(dirname "$EXP_DIR")")"
    JOB_NAME="phs_${ARCH}_${RUN_NAME}"
    LOG="${LOG_DIR}/${JOB_NAME}.%j.log"

    # The script auto-detects val_fasta / val_embedding_file from the
    # run's config.yaml; we only override val_datasets_dir to the
    # canonical DATA_DIR location.
    SBATCH_CMD=(sbatch
        --account="$ACCOUNT"
        --partition="$PARTITION"
        --gpus-per-node="$GPUS"
        --cpus-per-task="$CPUS"
        --mem="$MEM"
        --time="$TIME"
        --job-name="$JOB_NAME"
        --output="$LOG"
        --wrap="
            source \$(conda info --base)/etc/profile.d/conda.sh
            conda activate ${CONDA_ENV}
            cd ${CIPHER_DIR}
            export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}
            python scripts/analysis/per_head_strict_eval.py \
                ${EXP_DIR} \
                --val-datasets-dir ${VAL_DATASETS_DIR}
        ")

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "DRY: ${SBATCH_CMD[*]}"
    else
        "${SBATCH_CMD[@]}"
    fi
done

echo
echo "Done. After all jobs complete, refresh the harvest CSV with:"
echo "  cd ${CIPHER_DIR} && python scripts/analysis/harvest_results.py \\"
echo "      --experiments-dirs experiments \$(ls -d ../cipher-*/experiments 2>/dev/null)"
