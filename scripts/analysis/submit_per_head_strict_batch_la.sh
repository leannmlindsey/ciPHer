#!/usr/bin/env bash
#
# Submit a SLURM job that runs scripts/analysis/per_head_strict_eval.py for
# every Light Attention experiment in this worktree. Writes
# <exp>/results/per_head_strict_eval.json under the new any-hit + per-pair
# schema (per agent 1's 2026-04-27-1600 broadcast).
#
# Discovers experiments by globbing CIPHER_DIR/experiments/light_attention/*/
# and keeping any directory that has experiment.json (training completed).
# Override with EXPS=<newline-separated absolute paths> to evaluate a
# specific list instead.
#
# After this lands, the LA-relevant any-hit columns will appear in any
# harvest CSV that pulls these JSONs. Agent 4's per-K HR@1 quality
# comparison is gated on this.
#
# Env overrides:
#   CIPHER_DIR  light-attention worktree on Delta
#               (default: /projects/bfzj/llindsey1/PHI_TSP/cipher-light-attention)
#   DATA_DIR    main ciPHer data dir (default: /projects/bfzj/llindsey1/PHI_TSP/ciPHer/data)
#   EXPS        newline-separated absolute experiment dirs (overrides discovery)
#   ACCOUNT, PARTITION, CONDA_ENV (Delta defaults)
#   DRY_RUN=1   render the sbatch script but do not submit
#
# Usage:
#   bash scripts/analysis/submit_per_head_strict_batch_la.sh
#   DRY_RUN=1 bash scripts/analysis/submit_per_head_strict_batch_la.sh
#   EXPS="/path/to/exp1
#   /path/to/exp2" bash scripts/analysis/submit_per_head_strict_batch_la.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/cipher-light-attention}"
DATA_DIR="${DATA_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer/data}"

VAL_FASTA="${VAL_FASTA:-${DATA_DIR}/validation_data/metadata/validation_rbps_all.faa}"
VAL_DATASETS_DIR="${VAL_DATASETS_DIR:-${DATA_DIR}/validation_data/HOST_RANGE}"

GPUS=1
CPUS=4
MEM="32G"
TIME="3:00:00"

NAME="per_head_strict_la_$(date +%Y%m%d_%H%M%S)"
LOG="${CIPHER_DIR}/logs/${NAME}_%j.log"

# Discover LA experiment dirs (those with experiment.json) unless EXPS given.
if [ -z "${EXPS:-}" ]; then
    EXPS=$(find "${CIPHER_DIR}/experiments/light_attention" -mindepth 2 -maxdepth 2 \
        -name experiment.json -printf '%h\n' 2>/dev/null | sort)
fi

if [ -z "${EXPS}" ]; then
    echo "ERROR: no LA experiments with experiment.json under ${CIPHER_DIR}/experiments/light_attention/" >&2
    exit 1
fi

mkdir -p "${CIPHER_DIR}/logs"

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

echo "=== Per-head strict eval batch (LA) ==="
echo "  cipher_dir: ${CIPHER_DIR}"
echo "  data_dir:   ${DATA_DIR}"
echo "Started: \$(date)"
echo ""

# embedding_type -> Delta validation NPZ (mirrors agent 1's batch wrapper)
declare -A EMB_TO_NPZ=(
  [esm2_650m_mean]=/projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/validation_data/combined/validation_inputs/validation_embeddings_md5.npz
  [esm2_650m]=/projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/validation_data/combined/validation_inputs/validation_embeddings_md5.npz
  [esm2_650m_full]=/work/hdd/bfzj/llindsey1/validation_embeddings_full/validation_embeddings_full_md5.npz
  [esm2_650m_seg4]=/work/hdd/bfzj/llindsey1/validation_embeddings_segments4/validation_embeddings_segments4_md5.npz
  [esm2_650m_seg8]=/work/hdd/bfzj/llindsey1/validation_embeddings/esm2_650m_segments8/validation_esm2_650m_segments8_md5.npz
  [esm2_650m_seg16]=/work/hdd/bfzj/llindsey1/validation_embeddings/esm2_650m_segments16/validation_esm2_650m_segments16_md5.npz
  [esm2_3b_mean]=/work/hdd/bfzj/llindsey1/validation_embeddings_esm2_3b/validation_embeddings_md5.npz
  [esm2_150m_mean]=/work/hdd/bfzj/llindsey1/validation_embeddings_esm2_150m/validation_embeddings_md5.npz
  [prott5_mean]=/work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz
  [prott5_xl_full]=/work/hdd/bfzj/llindsey1/validation_prott5_xl_full/validation_prott5_xl_full_md5.npz
  [prott5_xl_seg4]=/work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments4/validation_prott5_xl_segments4_md5.npz
  [prott5_xl_seg8]=/work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments8/validation_prott5_xl_segments8_md5.npz
  [prott5_xl_seg16]=/work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments16/validation_prott5_xl_segments16_md5.npz
)

EXPS_TMP=\$(mktemp)
cat > "\$EXPS_TMP" <<'EXPS_EOF'
${EXPS}
EXPS_EOF

n_ok=0
n_skip=0
n_fail=0
while IFS= read -r EXP; do
    [ -z "\$EXP" ] && continue
    name=\$(basename "\$EXP")
    echo ""
    echo "------------------------------------------------------------"
    echo ">>> \$name"

    if [ ! -d "\$EXP" ]; then
        echo "    SKIP: dir not found"
        n_skip=\$((n_skip+1)); continue
    fi
    EMB_TYPE=\$(python3 -c "
import json, sys
try:
    d = json.load(open('\$EXP/experiment.json'))
    print(d.get('config', {}).get('data', {}).get('embedding_type', '') or '')
except Exception:
    print('')
")
    if [ -z "\$EMB_TYPE" ]; then
        echo "    SKIP: no embedding_type"; n_skip=\$((n_skip+1)); continue
    fi
    NPZ="\${EMB_TO_NPZ[\$EMB_TYPE]:-}"
    if [ -z "\$NPZ" ] || [ ! -f "\$NPZ" ]; then
        echo "    SKIP: no NPZ for embedding_type=\$EMB_TYPE"
        n_skip=\$((n_skip+1)); continue
    fi
    echo "    embedding_type: \$EMB_TYPE"
    echo "    val NPZ:        \$NPZ"
    if python3 ${CIPHER_DIR}/scripts/analysis/per_head_strict_eval.py "\$EXP" \\
        --val-embedding-file "\$NPZ" \\
        --val-fasta "${VAL_FASTA}" \\
        --val-datasets-dir "${VAL_DATASETS_DIR}" 2>&1 | tail -10
    then
        n_ok=\$((n_ok+1))
    else
        echo "    FAIL"
        n_fail=\$((n_fail+1))
    fi
done < "\$EXPS_TMP"
rm -f "\$EXPS_TMP"

echo ""
echo "============================================================"
echo "BATCH SUMMARY: ok=\$n_ok skip=\$n_skip fail=\$n_fail"
echo "Finished: \$(date)"
echo ""
echo "Each successful run wrote <exp>/results/per_head_strict_eval.json."
echo "Next: run harvest_results.py if you want a CSV across LA runs."
SBATCH_EOF

N_EXPS=$(printf '%s\n' "${EXPS}" | wc -l)

echo "============================================================"
echo "PER-HEAD STRICT EVAL BATCH (LA)"
echo "  N experiments: ${N_EXPS}"
echo "  Job script:    ${JOB_SCRIPT}"
echo "  Log:           ${LOG}"
echo "============================================================"
echo ""
echo "Experiments to evaluate:"
printf '%s\n' "${EXPS}" | sed 's|^|  |'
echo ""

DRY_RUN="${DRY_RUN:-0}"
if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Job script written, not submitted."
    echo "Submit with: sbatch ${JOB_SCRIPT}"
    exit 0
fi

JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $NF}')
echo "Submitted ${JOB_ID}"
