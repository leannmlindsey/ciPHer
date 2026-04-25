#!/usr/bin/env bash
#
# Post-reproduction diagnostics. Two SLURM jobs that answer:
#
#   Q1 — Does our current best model close the 0.291 PHL gap once we
#        switch to RAW-probability merging (the old eval's actual
#        scoring rule)?
#
#   Q2 — Did per-head training (v2 + v3) give us a stronger O head
#        than v1, even if combined-eval doesn't show it?
#
# Both jobs run on Delta inside SLURM (no head-node Python).
#
# JOB A — raw + raw+arbitrary re-eval of highconf_pipeline_K_prott5_mean.
#   Two new evals on the same trained checkpoints:
#     results_raw/evaluation.json           (raw zscore-off, default ties)
#     results_raw_arbitrary/evaluation.json (raw + arbitrary ties)
#   If raw or raw+arbitrary PHL HR@1 ≈ 0.29, the gap is just our default
#   zscore aggregation. No retraining needed to claim parity.
#
# JOB B — eval_per_head over all v2/v3 runs.
#   Produces K-only / O-only / combined HR@1 per dataset for each run.
#   Tells us whether per-head training (HC_K + HC_O lists) actually
#   strengthened the O head, which is where the old model's PHL
#   accuracy came from.
#
# Usage:
#   bash scripts/analysis/submit_post_repro_diagnostics.sh
#   DRY_RUN=1 bash scripts/analysis/submit_post_repro_diagnostics.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

BEST_RUN="${CIPHER_DIR}/experiments/attention_mlp/highconf_pipeline_K_prott5_mean"

# v2/v3 runs to eval_per_head against. All under main worktree.
V2_V3_RUNS=(
    "experiments/attention_mlp/v2_strict_prott5_mean"
    "experiments/attention_mlp/v2_uat_prott5_mean"
    "experiments/attention_mlp/v3_strict_prott5_mean"
    "experiments/attention_mlp/v3_uat_prott5_mean"
    "experiments/attention_mlp/v3_k_v2o_prott5_mean"
)

DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "POST-REPRODUCTION DIAGNOSTICS"
echo "  Cipher dir: ${CIPHER_DIR}"
echo ""
echo "  JOB A — raw / raw+arbitrary re-eval of:"
echo "          ${BEST_RUN}"
echo ""
echo "  JOB B — eval_per_head on ${#V2_V3_RUNS[@]} v2/v3 runs"
echo "============================================================"
echo ""

# ============================================================
# JOB A — raw merge re-evaluation
# ============================================================
JOBA_NAME="diag_raw_merge_eval"
JOBA_LOG="${CIPHER_DIR}/logs/${JOBA_NAME}_%j.log"

JOBA_SCRIPT="#!/bin/bash
#SBATCH --job-name=${JOBA_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=${JOBA_LOG}
#SBATCH --error=${JOBA_LOG}

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

echo \"=== JOB A: RAW-MERGE RE-EVAL OF EXISTING BEST PHL RUN ===\"
echo \"  Run: ${BEST_RUN}\"
echo \"  Started: \$(date)\"

echo \"\"
echo \"=== A.1 — score-norm raw, default ties (competition) ===\"
mkdir -p ${BEST_RUN}/results_raw
python -m cipher.evaluation.runner ${BEST_RUN} \\
    --score-norm raw \\
    -o ${BEST_RUN}/results_raw/evaluation.json

echo \"\"
echo \"=== A.2 — score-norm raw + tie-method arbitrary (full old-style) ===\"
mkdir -p ${BEST_RUN}/results_raw_arbitrary
python -m cipher.evaluation.runner ${BEST_RUN} \\
    --score-norm raw \\
    --tie-method arbitrary \\
    -o ${BEST_RUN}/results_raw_arbitrary/evaluation.json

echo \"\"
echo \"=== Quick PHL HR@1 dump ===\"
python -c \"
import json
for variant in ['results', 'results_raw', 'results_raw_arbitrary', 'results_arbitrary']:
    p = '${BEST_RUN}/' + variant + '/evaluation.json'
    try:
        d = json.load(open(p))
        rh = d['PhageHostLearn']['rank_hosts']['hr_at_k']['1']
        print(f'  {variant:<25} PHL rh@1 = {rh:.4f}')
    except FileNotFoundError:
        print(f'  {variant:<25} (not present)')
\"

echo \"\"
echo \"Done JOB A: \$(date)\"
"

# ============================================================
# JOB B — eval_per_head on v2/v3 runs
# ============================================================
JOBB_NAME="diag_evh_v2_v3"
JOBB_LOG="${CIPHER_DIR}/logs/${JOBB_NAME}_%j.log"

# Build the per-run loop body
JOBB_RUN_LIST=""
for r in "${V2_V3_RUNS[@]}"; do
    JOBB_RUN_LIST+="${r} "
done

JOBB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${JOBB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=${JOBB_LOG}
#SBATCH --error=${JOBB_LOG}

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

echo \"=== JOB B: eval_per_head ON v2/v3 RUNS ===\"
echo \"  Started: \$(date)\"
echo \"  Looking for whether per-head training delivered stronger O heads.\"
echo \"  Reference O-only PHL HR@1: old=0.282 (target), v1 highconf=0.135.\"

for run in ${JOBB_RUN_LIST}; do
    full=${CIPHER_DIR}/\${run}
    if [ ! -d \"\$full\" ]; then
        echo \"\"
        echo \">>> SKIP \${run} — directory not found\"
        continue
    fi
    echo \"\"
    echo \"========================================\"
    echo \">>> \${run}\"
    echo \"========================================\"
    python ${CIPHER_DIR}/scripts/analysis/eval_per_head.py \"\$full\"
done

echo \"\"
echo \"Done JOB B: \$(date)\"
"

# ============================================================
# Submit both
# ============================================================
if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would submit:"
    echo "  JOB A — ${JOBA_NAME}  (raw / raw+arbitrary re-eval of best PHL run)"
    echo "  JOB B — ${JOBB_NAME}  (eval_per_head on ${#V2_V3_RUNS[@]} v2/v3 runs)"
    exit 0
fi

mkdir -p "${CIPHER_DIR}/logs"

JOBA_ID=$(echo "$JOBA_SCRIPT" | sbatch | awk '{print $NF}')
JOBB_ID=$(echo "$JOBB_SCRIPT" | sbatch | awk '{print $NF}')

echo "Submitted ${JOBA_ID} — ${JOBA_NAME}  (raw / raw+arbitrary re-eval)"
echo "Submitted ${JOBB_ID} — ${JOBB_NAME}  (eval_per_head over v2/v3)"
echo ""
echo "Logs:"
echo "  ${CIPHER_DIR}/logs/${JOBA_NAME}_${JOBA_ID}.log"
echo "  ${CIPHER_DIR}/logs/${JOBB_NAME}_${JOBB_ID}.log"
echo ""
echo "Each job ends with a self-contained PHL HR@1 dump in its log."
echo "When both finish, read the logs (cat/tail OK on the head node)."
