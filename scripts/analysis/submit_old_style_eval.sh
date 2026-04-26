#!/usr/bin/env bash
#
# Submit a SLURM job that scores N cipher experiments under OLD-style
# class-ranking eval (rank K + O serotypes; hit if top-1 matches host's
# K or O). Lets us compare any cipher experiment to OLD klebsiella's
# 0.291 PHL HR@1 baseline on equal footing.
#
# What it does
# ------------
# For each experiment dir in EXPS:
#   1. Reads embedding_type from <exp>/experiment.json
#   2. Resolves the matching validation NPZ on Delta
#   3. Runs scripts/analysis/old_style_eval.py
#   4. Writes <exp>/results_old_style/old_style_eval.json
#
# After the job lands, all per-experiment results are summarised by
# scripts/analysis/old_style_eval_summary.py (run separately on the
# login node — small, fast, no compute).
#
# Defaults: scores the top-15 experiments by NEW eval PHL+PBIP combined
# HR@1, as ranked in results/experiment_log.csv. Override with EXPS to
# pick a custom set.
#
# Env overrides:
#   EXPS              newline-separated absolute experiment dirs (default: top-15)
#   N_TOP             how many top experiments to take from harvest (default 15)
#   ACCOUNT           SLURM account (default bfzj-dtai-gh)
#   PARTITION         SLURM partition (default ghx4)
#   CONDA_ENV         conda env (default esmfold2)
#   CIPHER_DIR        cipher root on Delta (default /projects/bfzj/llindsey1/PHI_TSP/ciPHer)
#   DRY_RUN=1         print the rendered script, do not submit
#
# Usage:
#   bash scripts/analysis/submit_old_style_eval.sh
#   N_TOP=30 bash scripts/analysis/submit_old_style_eval.sh
#   DRY_RUN=1 bash scripts/analysis/submit_old_style_eval.sh
#   EXPS="$(printf '%s\n' /full/path/exp1 /full/path/exp2)" \
#     bash scripts/analysis/submit_old_style_eval.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"
N_TOP="${N_TOP:-15}"

VAL_FASTA="${VAL_FASTA:-${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa}"
VAL_DATASETS_DIR="${VAL_DATASETS_DIR:-${CIPHER_DIR}/data/validation_data/HOST_RANGE}"

GPUS=1   # Delta requires a GPU even for CPU-bound jobs
CPUS=4
MEM="32G"
TIME="2:00:00"

NAME="old_style_eval_top${N_TOP}_$(date +%Y%m%d_%H%M%S)"
LOG="${CIPHER_DIR}/logs/${NAME}_%j.log"

# Resolve EXPS: if not provided, take top N_TOP from the harvest CSV
if [ -z "${EXPS:-}" ]; then
    EXPS=$(python3 - <<PYEOF
import csv, os
csv_path = os.path.join("${CIPHER_DIR}", "results", "experiment_log.csv")
def f(v):
    try: return float(v)
    except: return None
with open(csv_path) as fh:
    rows = [r for r in csv.DictReader(fh) if f(r.get("phl_pbip_combined_hr1")) is not None]
rows.sort(key=lambda r: -f(r["phl_pbip_combined_hr1"]))
for r in rows[:${N_TOP}]:
    print(r["exp_dir"])
PYEOF
)
fi

mkdir -p "${CIPHER_DIR}/logs"

# Render the SLURM body to a file (we can re-submit / inspect easily)
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

echo "=== OLD-style eval batch (${N_TOP} experiments) ==="
echo "Started: \$(date)"
echo ""

# embedding_type -> validation NPZ on Delta
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

EXPS_LIST=$(printf '%q\n' ${EXPS})
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
        echo "    SKIP: experiment dir not found"
        n_skip=\$((n_skip+1))
        continue
    fi
    if [ ! -f "\$EXP/model_k/best_model.pt" ] && [ ! -f "\$EXP/model_o/best_model.pt" ]; then
        # Some models (light_attention*, etc) may use a different layout — try anyway
        if [ ! -f "\$EXP/best_model.pt" ]; then
            echo "    SKIP: no model_k/best_model.pt or model_o/best_model.pt"
            n_skip=\$((n_skip+1))
            continue
        fi
    fi

    # Discover embedding_type via experiment.json
    EMB_TYPE=\$(python3 -c "
import json, sys
try:
    d = json.load(open('\$EXP/experiment.json'))
    et = d.get('config', {}).get('data', {}).get('embedding_type', '')
    print(et)
except Exception as e:
    print('', file=sys.stderr)
")
    if [ -z "\$EMB_TYPE" ]; then
        echo "    SKIP: cannot determine embedding_type"
        n_skip=\$((n_skip+1))
        continue
    fi
    NPZ="\${EMB_TO_NPZ[\$EMB_TYPE]:-}"
    if [ -z "\$NPZ" ] || [ ! -f "\$NPZ" ]; then
        echo "    SKIP: no NPZ mapped for embedding_type='\$EMB_TYPE' (need to add to EMB_TO_NPZ)"
        n_skip=\$((n_skip+1))
        continue
    fi
    echo "    embedding_type: \$EMB_TYPE"
    echo "    val NPZ:        \$NPZ"

    if python3 ${CIPHER_DIR}/scripts/analysis/old_style_eval.py "\$EXP" \\
        --val-embedding-file "\$NPZ" \\
        --val-fasta "${VAL_FASTA}" \\
        --val-datasets-dir "${VAL_DATASETS_DIR}" 2>&1 | tail -10
    then
        n_ok=\$((n_ok+1))
    else
        echo "    FAIL: old_style_eval.py errored"
        n_fail=\$((n_fail+1))
    fi
done < "\$EXPS_TMP"
rm -f "\$EXPS_TMP"

echo ""
echo "============================================================"
echo "BATCH SUMMARY: ok=\$n_ok skip=\$n_skip fail=\$n_fail"
echo "Finished: \$(date)"
echo ""
echo "Inspect per-experiment JSON at:"
echo "  <exp>/results_old_style/old_style_eval.json"
echo ""
echo "Build a side-by-side summary with:"
echo "  python ${CIPHER_DIR}/scripts/analysis/old_style_eval_summary.py"
SBATCH_EOF

echo "============================================================"
echo "OLD-STYLE EVAL BATCH"
echo "  N experiments: $(printf '%s\n' "${EXPS}" | wc -l)"
echo "  Job script:    ${JOB_SCRIPT}"
echo "  Log:           ${LOG}"
echo "============================================================"
echo ""
echo "First 10 experiments to score:"
printf '%s\n' "${EXPS}" | head -10 | sed 's/^/  /'
echo ""

DRY_RUN="${DRY_RUN:-0}"
if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Job script written to ${JOB_SCRIPT} but not submitted."
    echo "Submit with: sbatch ${JOB_SCRIPT}"
    exit 0
fi

JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $NF}')
echo "Submitted ${JOB_ID}"
echo "Log: ${CIPHER_DIR}/logs/${NAME}_${JOB_ID}.log"
