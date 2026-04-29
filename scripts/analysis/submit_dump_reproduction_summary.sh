#!/usr/bin/env bash
#
# Submit a SLURM job that assembles the side-by-side summary of the
# three reproduction stages from
# `submit_reproduce_old_phl_run.sh`:
#
#   STAGE 1a: arbitrary-tie re-eval of the existing best-PHL run
#             (highconf_pipeline_K_prott5_mean)
#   STAGE 1b: eval_per_head (K / O / combined) on the same run
#   STAGE 2 : retrained attention_mlp with old-v3-config approximation
#             (repro_old_v3_esm2_650m_mean), plus its three evals
#
# Prints a compact comparison table so we can tell at-a-glance whether
# eval-methodology alone (STAGE 1) closed the old-repo 0.291 gap, or
# whether training-recipe (STAGE 2) was also needed.
#
# Everything runs inside SLURM — don't invoke the underlying Python
# directly on a login node. Even fast dumps can degrade queue priority
# if many of them accumulate.
#
# Usage:
#   bash scripts/analysis/submit_dump_reproduction_summary.sh
#   DRY_RUN=1 bash scripts/analysis/submit_dump_reproduction_summary.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

NAME="repro_summary"
LOG="${CIPHER_DIR}/logs/${NAME}_%j.log"

STAGE1_RUN="${CIPHER_DIR}/experiments/attention_mlp/highconf_pipeline_K_prott5_mean"
STAGE2_RUN="${CIPHER_DIR}/experiments/attention_mlp/repro_old_v3_esm2_650m_mean"

DRY_RUN="${DRY_RUN:-0}"

JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=${LOG}
#SBATCH --error=${LOG}

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

echo \"=========================================================\"
echo \"REPRODUCTION SUMMARY — comparing 3 stages to old-repo 0.291\"
echo \"  Target:   host_prediction_local_test_v3  PHL rh@1 = 0.291\"
echo \"  Started:  \$(date)\"
echo \"=========================================================\"

echo \"\"
echo \"== STAGE 0 (baseline) — current best PHL run, default eval ==\"
python ${CIPHER_DIR}/scripts/analysis/dump_eval_summary.py ${STAGE1_RUN}

echo \"\"
echo \"== STAGE 1a — same model, arbitrary-tie eval ==\"
if [ -f ${STAGE1_RUN}/results_arbitrary/evaluation.json ]; then
    python -c \"
import json, sys
d = json.load(open('${STAGE1_RUN}/results_arbitrary/evaluation.json'))
print(f'{\\\"Dataset\\\":<16} {\\\"rh@1\\\":>7} {\\\"rp@1\\\":>7} {\\\"n\\\":>6}')
print('-' * 40)
for ds in ['CHEN','GORODNICHIV','UCSD','PBIP','PhageHostLearn']:
    rh = d.get(ds,{}).get('rank_hosts',{}).get('hr_at_k',{}).get('1')
    rp = d.get(ds,{}).get('rank_phages',{}).get('hr_at_k',{}).get('1')
    n  = d.get(ds,{}).get('rank_hosts',{}).get('n_pairs','?')
    print(f'{ds:<16} {rh if rh is not None else \\\"—\\\":>7.3f if isinstance(rh, float) else \\\"—\\\":>7} {rp if rp is not None else \\\"—\\\":>7.3f if isinstance(rp, float) else \\\"—\\\":>7} {str(n):>6}')
\" 2>/dev/null || echo \"  (unable to parse arbitrary-tie eval json)\"
else
    echo \"  MISSING: ${STAGE1_RUN}/results_arbitrary/evaluation.json\"
    echo \"           STAGE 1a did not complete. Check logs/repro_stage1_arbitrary_*.log\"
fi

echo \"\"
echo \"== STAGE 1b — same model, K/O/combined three-way ==\"
python ${CIPHER_DIR}/scripts/analysis/eval_per_head.py ${STAGE1_RUN} 2>&1 | tail -40

echo \"\"
echo \"== STAGE 2 — retrained with old-v3 config approximation ==\"
if [ -d ${STAGE2_RUN} ]; then
    echo \"-- default eval (competition, combined) --\"
    python ${CIPHER_DIR}/scripts/analysis/dump_eval_summary.py ${STAGE2_RUN}
    echo \"\"
    echo \"-- arbitrary-tie eval --\"
    if [ -f ${STAGE2_RUN}/results_arbitrary/evaluation.json ]; then
        python -c \"
import json
d = json.load(open('${STAGE2_RUN}/results_arbitrary/evaluation.json'))
print(f'{\\\"Dataset\\\":<16} {\\\"rh@1\\\":>7} {\\\"rp@1\\\":>7}')
print('-' * 32)
for ds in ['CHEN','GORODNICHIV','UCSD','PBIP','PhageHostLearn']:
    rh = d.get(ds,{}).get('rank_hosts',{}).get('hr_at_k',{}).get('1')
    rp = d.get(ds,{}).get('rank_phages',{}).get('hr_at_k',{}).get('1')
    print(f'{ds:<16} {rh if rh is not None else \\\"—\\\":>7.3f if isinstance(rh, float) else \\\"—\\\":>7} {rp if rp is not None else \\\"—\\\":>7.3f if isinstance(rp, float) else \\\"—\\\":>7}')
\" 2>/dev/null
    else
        echo \"  (results_arbitrary not present)\"
    fi
    echo \"\"
    echo \"-- eval_per_head (K/O/combined) --\"
    python ${CIPHER_DIR}/scripts/analysis/eval_per_head.py ${STAGE2_RUN} 2>&1 | tail -40
else
    echo \"  STAGE 2 not yet landed at ${STAGE2_RUN}\"
fi

echo \"\"
echo \"=========================================================\"
echo \"DIAGNOSIS GUIDE\"
echo \"=========================================================\"
echo \"  1. If STAGE 1a PHL rh@1 ≈ 0.29  →  tie-handling explains most of the gap.\"
echo \"  2. If STAGE 1b PHL O-only HR@1 ≈ 0.29  →  O-only eval explains it.\"
echo \"  3. If STAGE 2 default PHL rh@1 ≈ 0.29  →  training recipe also matters;\"
echo \"     identify which knob (ESM-2, single_label, min_sources=3) is the win.\"
echo \"  4. If STAGE 2 arbitrary PHL rh@1 ≈ 0.29  →  eval + training together reproduce.\"
echo \"  5. If none of the above reach 0.29  →  keep_null_classes=true is the remaining gap.\"
echo \"\"
echo \"Done: \$(date)\"
"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would submit: ${NAME}"
    echo "  Reads from: ${STAGE1_RUN}/"
    echo "              ${STAGE2_RUN}/"
    echo "  Output log: ${CIPHER_DIR}/logs/${NAME}_<jobid>.log"
    exit 0
fi

mkdir -p "${CIPHER_DIR}/logs"
JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
echo "Submitted ${JOB_ID} — ${NAME}"
echo ""
echo "Log:  ${CIPHER_DIR}/logs/${NAME}_${JOB_ID}.log"
echo "Tail after it finishes:"
echo "  tail -200 ${CIPHER_DIR}/logs/${NAME}_${JOB_ID}.log"
