#!/usr/bin/env bash
#
# Replace the singleton-cluster shortcut in candidates_clusters_v4.tsv
# with proper cluster assignments via mmseqs2 best-hit + threshold-aware
# inheritance.
#
# Workflow on Delta:
#   1. mmseqs2 easy-search: candidates_v4_additions.faa (1067 queries)
#      vs candidates.faa (143240 targets), output identity per hit.
#   2. Python: for each new protein, look up best hit's cluster IDs at
#      9 thresholds; inherit at thresholds where identity ≥ T, else
#      genuine singleton.
#   3. Output candidates_clusters_v4.tsv (overwrite the singleton version)
#      and a manifest documenting assignment provenance.
#
# Total wall time on Delta: ~5-15 min (mmseqs2 is fast on this scale).
#
# After this finishes, re-run STEP=3 + STEP=4 of the v4 training pipeline
# (the embedding NPZs are already built, so only training needs to redo).
#
# Env overrides:
#   ACCOUNT, PARTITION, CIPHER_DIR
#   DRY_RUN=1   render sbatch but do not submit
#
# Usage:
#   bash scripts/run_v4_proper_clusters.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"
DRY_RUN="${DRY_RUN:-0}"

V4_DIR="${CIPHER_DIR}/data/training_data/metadata/highconf_v4"
EXISTING_FAA="${CIPHER_DIR}/data/training_data/metadata/candidates.faa"
EXISTING_CLUSTERS="${CIPHER_DIR}/data/training_data/metadata/candidates_clusters.tsv"
V4_ADDITIONS_FAA="${V4_DIR}/candidates_v4_additions.faa"
OUT_CLUSTERS="${V4_DIR}/candidates_clusters_v4.tsv"
OUT_MANIFEST="${V4_DIR}/cluster_assignment_manifest.json"

# Working dir for mmseqs intermediates
WORK_DIR="/work/hdd/bfzj/llindsey1/v4_mmseqs"
MMSEQS_RESULTS="${WORK_DIR}/v4_additions_vs_candidates.m8"

mkdir -p "${CIPHER_DIR}/logs"

NAME="v4_proper_clusters_$(date +%Y%m%d_%H%M%S)"
SBATCH_FILE="${CIPHER_DIR}/logs/${NAME}.sbatch"
LOG="${CIPHER_DIR}/logs/${NAME}_%j.log"

cat > "$SBATCH_FILE" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=v4_proper_clusters
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=${LOG}
#SBATCH --error=${LOG}

set -euo pipefail
source \$(conda info --base)/etc/profile.d/conda.sh

# Try common conda envs that may have mmseqs2; fall back to module load
if conda activate mmseqs2 2>/dev/null; then
    echo "Using conda env: mmseqs2"
elif conda activate esmfold2 2>/dev/null && command -v mmseqs >/dev/null; then
    echo "Using conda env: esmfold2 (mmseqs available)"
elif command -v module >/dev/null && module load mmseqs2 2>/dev/null; then
    echo "Using module: mmseqs2"
else
    echo "ERROR: mmseqs2 not found in any expected env or module"
    echo "Try: conda install -c bioconda mmseqs2"
    exit 1
fi

cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

mkdir -p ${WORK_DIR}
echo "=== mmseqs2 easy-search ==="
echo "Query:  ${V4_ADDITIONS_FAA} (1067 v4 additions)"
echo "Target: ${EXISTING_FAA}     (143240 cipher candidates)"
echo "Output: ${MMSEQS_RESULTS}"
echo "Started: \$(date)"
echo ""

# easy-search: query, target_db, output, tmp_dir
# Default outfmt is BLAST tabular (m8): query target fident alnlen
#   mismatch gapopen qstart qend tstart tend evalue bits
# --min-seq-id 0.30 — we won't use hits below cl30 anyway
# --max-seqs 5 — keep top-5 hits per query (we only need best, but a few
#                more lets us inspect ambiguity later)
# -s 7.5 — high sensitivity (we want to find homologs across the full
#          identity range to ensure proper threshold-aware assignment)
mmseqs easy-search \\
    ${V4_ADDITIONS_FAA} \\
    ${EXISTING_FAA} \\
    ${MMSEQS_RESULTS} \\
    ${WORK_DIR}/tmp \\
    --min-seq-id 0.30 \\
    --max-seqs 5 \\
    -s 7.5 \\
    --threads \${SLURM_CPUS_PER_TASK:-16}

n_hits=\$(wc -l < ${MMSEQS_RESULTS})
n_queries_with_hits=\$(cut -f1 ${MMSEQS_RESULTS} | sort -u | wc -l)
echo ""
echo "mmseqs2 done. Total hits: \${n_hits}, queries with ≥1 hit: \${n_queries_with_hits}/1067"
echo ""

# Run the assignment script
echo "=== Cluster assignment (Python) ==="
python ${CIPHER_DIR}/scripts/data_prep/assign_v4_clusters_from_mmseqs.py \\
    --mmseqs-results "${MMSEQS_RESULTS}" \\
    --existing-clusters "${EXISTING_CLUSTERS}" \\
    --v4-additions "${V4_ADDITIONS_FAA}" \\
    --out "${OUT_CLUSTERS}" \\
    --manifest "${OUT_MANIFEST}"

echo ""
echo "============================================================"
echo "Done: \$(date)"
echo "Updated:  ${OUT_CLUSTERS}"
echo "Manifest: ${OUT_MANIFEST}"
echo ""
echo "Next: re-run v4 training (steps 3 + 4) with the corrected cluster file:"
echo "  STEP=3 NO_DEP=1 bash scripts/run_v4_training_pipeline.sh"
echo "  STEP=4 NO_DEP=1 bash scripts/run_v4_training_pipeline.sh"
echo "(NO_DEP=1 skips the embedding-step dependency since those are done.)"
echo "============================================================"
SBATCH_EOF

echo "============================================================"
echo "v4 proper-cluster assignment job"
echo "  Script: ${SBATCH_FILE}"
echo "  Log:    ${LOG}"
echo "  Output: ${OUT_CLUSTERS}"
echo "============================================================"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Job script written, not submitted."
    echo "Submit with: sbatch ${SBATCH_FILE}"
    exit 0
fi

JOB_ID=$(sbatch "${SBATCH_FILE}" | awk '{print $NF}')
echo "Submitted ${JOB_ID}"
echo ""
echo "When this job finishes, the singleton-cluster file will be"
echo "replaced with proper cluster assignments. Then re-run v4 training:"
echo "  STEP=3 NO_DEP=1 bash scripts/run_v4_training_pipeline.sh"
echo "  STEP=4 NO_DEP=1 bash scripts/run_v4_training_pipeline.sh"
