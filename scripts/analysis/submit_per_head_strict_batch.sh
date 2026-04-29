#!/usr/bin/env bash
#
# Submit a SLURM job that runs scripts/analysis/per_head_strict_eval.py
# for the top-N experiments by current PHL+PBIP combined HR@1 (or any
# explicit list via EXPS). Saves per-experiment
# results/per_head_strict_eval.json on each run.
#
# After this lands, re-run harvest_results.py (login node, no SLURM
# needed for that since it just reads JSONs) to get the strict-
# denominator best-head columns + new sort.
#
# Env overrides:
#   N_TOP             how many top experiments to process (default 30)
#   EXPS              newline-separated absolute experiment dirs
#   ACCOUNT, PARTITION, CONDA_ENV, CIPHER_DIR  (Delta defaults)
#   DRY_RUN=1         render the sbatch but do not submit
#
# Usage:
#   bash scripts/analysis/submit_per_head_strict_batch.sh
#   N_TOP=50 bash scripts/analysis/submit_per_head_strict_batch.sh
#   DRY_RUN=1 bash scripts/analysis/submit_per_head_strict_batch.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"
N_TOP="${N_TOP:-30}"

VAL_FASTA="${VAL_FASTA:-${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa}"
VAL_DATASETS_DIR="${VAL_DATASETS_DIR:-${CIPHER_DIR}/data/validation_data/HOST_RANGE}"

GPUS=1
CPUS=4
MEM="32G"
TIME="3:00:00"

NAME="per_head_strict_top${N_TOP}_$(date +%Y%m%d_%H%M%S)"
LOG="${CIPHER_DIR}/logs/${NAME}_%j.log"

if [ -z "${EXPS:-}" ]; then
    EXPS=$(python3 - <<PYEOF
import csv, os
csv_path = os.path.join("${CIPHER_DIR}", "results", "experiment_log.csv")
def f(v):
    try: return float(v)
    except: return None
with open(csv_path) as fh:
    all_rows = list(csv.DictReader(fh))
# Prefer phage-weighted overall; fall back to legacy combined only on a
# cold-start (no per_head_strict_eval has been run for any exp yet).
have_overall = any(f(r.get("overall_anyhit_HR1")) is not None for r in all_rows)
field = "overall_anyhit_HR1" if have_overall else "phl_pbip_combined_hr1"
rows = [r for r in all_rows if f(r.get(field)) is not None]
rows.sort(key=lambda r: -f(r[field]))
for r in rows[:${N_TOP}]:
    print(r["exp_dir"])
PYEOF
)
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

echo "=== Per-head strict eval batch (${N_TOP} experiments) ==="
echo "Started: \$(date)"
echo ""

# embedding_type -> Delta validation NPZ (mirrors submit_old_style_eval.sh)
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
  [kmer_aa20_k3]=/work/hdd/bfzj/llindsey1/kmer_features/validation_aa20_k3.npz
  [kmer_aa20_k4]=/work/hdd/bfzj/llindsey1/kmer_features/validation_aa20_k4.npz
  [kmer_li10_k5]=/work/hdd/bfzj/llindsey1/kmer_features/validation_li10_k5.npz
  [kmer_murphy8_k5]=/work/hdd/bfzj/llindsey1/kmer_features/validation_murphy8_k5.npz
  [kmer_murphy10_k5]=/work/hdd/bfzj/llindsey1/kmer_features/validation_murphy10_k5.npz
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
echo "Next: run harvest_results.py on the login node to refresh CSV."
SBATCH_EOF

echo "============================================================"
echo "PER-HEAD STRICT EVAL BATCH"
echo "  N experiments: $(printf '%s\n' "${EXPS}" | wc -l)"
echo "  Job script:    ${JOB_SCRIPT}"
echo "  Log:           ${LOG}"
echo "============================================================"
echo ""
echo "First 10 to score:"
printf '%s\n' "${EXPS}" | head -10 | sed 's/^/  /'
echo ""

DRY_RUN="${DRY_RUN:-0}"
if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Job script written, not submitted."
    echo "Submit with: sbatch ${JOB_SCRIPT}"
    exit 0
fi

JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $NF}')
echo "Submitted ${JOB_ID}"
