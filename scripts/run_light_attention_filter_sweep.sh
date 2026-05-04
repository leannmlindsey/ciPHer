#!/usr/bin/env bash
#
# LA filter sweep on the v3_uat + cap=300 baseline. Trains three filter
# variants of `la_v3_uat_prott5_xl_seg8_cap300` (the current best LA),
# matching leann's 2026-05-04-2330 RBP-filter-sweep ask:
#
#   variant  | RBP filter rule                                         | training flags
#   ---------+---------------------------------------------------------+----------------
#   all      | every protein in glycan_binders_custom.tsv               | --protein_set all_glycan_binders --min_sources 1
#   tools3   | any of {SpikeHunter, DePP_85, PhageRBPdetect}            | --tools SpikeHunter,DePP_85,PhageRBPdetect
#   tools4   | any of {SpikeHunter, DePP_85, PhageRBPdetect, DepoScope} | --tools SpikeHunter,DePP_85,PhageRBPdetect,DepoScope
#
# (`pipeline_positive` is already in the harvest as the existing
# `la_v3_uat_prott5_xl_seg8_cap300` run — that variant uses K/O lists
# derived from pipeline_positive. No need to re-train it.)
#
# Hold OTHER axes constant vs the cap=300 baseline:
#   - architecture: light_attention
#   - embedding: prott5_xl_seg8 (n_segments=8)
#   - max_samples_per_k: 300  (LA's sweet spot from the cap sweep)
#   - cluster_threshold: 70
#   - label_strategy: multi_label_threshold + min_class_samples=25
#   - SLURM walltime: 24h training (per the generous-timeout rule)
#
# After training, two evals run; the second OVERWRITES the first so the
# canonical `per_head_strict_eval.json` ends up filter-matched, identical
# in behavior to agent 1's MLP filter sweep (`run_filter_sweep_12.sh` +
# `submit_training_variant.py`).
#   1. cipher.cli.train_runner's mandatory auto-strict-eval — initially
#      writes unfiltered `per_head_strict_eval.json`.
#   2. Explicit filter-matched strict-eval — OVERWRITES the same JSON
#      with filter-matched data (auto-mirrored from config.yaml via the
#      --glycan-binders flag agent 1 added on 2026-05-04).
# End state: a single `per_head_strict_eval.json` containing the
# filter-matched headline numbers, matching the harvest's expectation
# and agent 1's MLP filter-sweep convention.
#
# Naming convention follows agent 1's note (la_<emb>_<filter>[_<extras>]):
#   la_prott5seg8_all_cap300
#   la_prott5seg8_tools3_cap300
#   la_prott5seg8_tools4_cap300
#
# Env overrides:
#   FILTERS         space-separated list of filters to run (default: "all tools3 tools4")
#   CIPHER_DIR      LA worktree on Delta
#   DATA_DIR        ciPHer data dir (default: /projects/bfzj/llindsey1/PHI_TSP/ciPHer/data)
#   VAL_GB          validation glycan_binders TSV (default: per leann's broadcast)
#   ACCOUNT, PARTITION, CONDA_ENV   (Delta defaults)
#   DRY_RUN=1       render but do not submit
#
# Usage:
#   bash scripts/run_light_attention_filter_sweep.sh                        # all 3 filters
#   bash scripts/run_light_attention_filter_sweep.sh tools3                 # one filter
#   FILTERS="all tools3" bash scripts/run_light_attention_filter_sweep.sh   # subset
#   DRY_RUN=1 bash scripts/run_light_attention_filter_sweep.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="${CIPHER_DIR:-$(dirname "$SCRIPT_DIR")}"
DATA_DIR="${DATA_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer/data}"

ASSOC_MAP="${DATA_DIR}/training_data/metadata/host_phage_protein_map.tsv"
GLYCAN_BINDERS="${DATA_DIR}/training_data/metadata/glycan_binders_custom.tsv"
VAL_FASTA="${DATA_DIR}/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${DATA_DIR}/validation_data/HOST_RANGE"

# Validation glycan_binders TSV — distinct from training glycan_binders.
# Path from leann's 2026-05-04-2330 broadcast.
VAL_GB="${VAL_GB:-/projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/validation_data/combined/validation_inputs/glycan_binders_custom.tsv}"

TRAIN_EMB="/work/hdd/bfzj/llindsey1/embeddings/prott5_xl_segments8/candidates_prott5_xl_segments8_md5.npz"
VAL_EMB="/work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments8/validation_prott5_xl_segments8_md5.npz"

MODEL="light_attention"
EMB_TYPE="prott5_xl_seg8"
N_SEGMENTS=8
MAX_K_CAP=300
CLUSTER_TH=70

# Generous SLURM allocation per the no-tight-timeouts rule.
GPUS=1
CPUS=8
MEM="160G"
TIME="24:00:00"

# ============================================================
# Filter variants
# ============================================================

FILTERS_DEFAULT="all tools3 tools4"
if [ -n "${1:-}" ]; then
    FILTERS="$1"
elif [ -n "${FILTERS:-}" ]; then
    FILTERS="${FILTERS}"
else
    FILTERS="${FILTERS_DEFAULT}"
fi

DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "LA filter sweep (cap=${MAX_K_CAP} baseline)"
echo "  Cipher dir:    ${CIPHER_DIR}"
echo "  Data dir:      ${DATA_DIR}"
echo "  Embedding:     ${EMB_TYPE} (n_segments=${N_SEGMENTS})"
echo "  Cap:           ${MAX_K_CAP}"
echo "  Cluster th:    ${CLUSTER_TH}"
echo "  Validation GB: ${VAL_GB}"
echo "  Filters:       ${FILTERS}"
echo "  SLURM:         ${TIME}, ${GPUS} GPU, ${CPUS} CPU, ${MEM}"
echo "============================================================"
echo ""

if [ ! -f "$GLYCAN_BINDERS" ]; then
    echo "ERROR: training glycan_binders not found: $GLYCAN_BINDERS" >&2; exit 1
fi
if [ ! -f "$TRAIN_EMB" ]; then
    echo "ERROR: training embeddings NPZ not found: $TRAIN_EMB" >&2; exit 1
fi

mkdir -p "${CIPHER_DIR}/logs"

N_SUBMITTED=0

for FILTER in $FILTERS; do
    case "$FILTER" in
        all)
            NAME="la_prott5seg8_all_cap${MAX_K_CAP}"
            FILTER_FLAGS="--protein_set all_glycan_binders --min_sources 1"
            ;;
        tools3)
            NAME="la_prott5seg8_tools3_cap${MAX_K_CAP}"
            FILTER_FLAGS="--tools SpikeHunter,DePP_85,PhageRBPdetect"
            ;;
        tools4)
            NAME="la_prott5seg8_tools4_cap${MAX_K_CAP}"
            FILTER_FLAGS="--tools SpikeHunter,DePP_85,PhageRBPdetect,DepoScope"
            ;;
        *)
            echo "  SKIP: unknown filter '${FILTER}' (valid: all tools3 tools4)" >&2
            continue
            ;;
    esac

    EXP_DIR="${CIPHER_DIR}/experiments/${MODEL}/${NAME}"

    # NOTE: cipher.cli.train_runner runs per_head_strict_eval automatically
    # after training (writes per_head_strict_eval.json — UNFILTERED, the
    # pre-2026-05-04 default). The launcher then runs strict-eval AGAIN
    # explicitly with --glycan-binders so the filter mirrors training,
    # producing per_head_strict_eval_rbp_fm.json. The _rbp_fm suffix is
    # agent 1's convention from the 2026-05-04 broadcast.
    #
    # Once train_runner is updated to pass --glycan-binders by default,
    # the explicit filter-matched eval here becomes redundant; remove it
    # then.

    TRAIN_CMD="python -u -m cipher.cli.train_runner \
        --model ${MODEL} \
        ${FILTER_FLAGS} \
        --embedding_type ${EMB_TYPE} \
        --embedding_file ${TRAIN_EMB} \
        --n_segments ${N_SEGMENTS} \
        --association_map ${ASSOC_MAP} \
        --glycan_binders ${GLYCAN_BINDERS} \
        --val_fasta ${VAL_FASTA} \
        --val_datasets_dir ${VAL_DATASETS_DIR} \
        --val_embedding_file ${VAL_EMB} \
        --label_strategy multi_label_threshold \
        --min_class_samples 25 \
        --max_samples_per_k ${MAX_K_CAP} \
        --cluster_threshold ${CLUSTER_TH} \
        --name ${NAME}"

    # Writes to the CANONICAL per_head_strict_eval.json (overwriting the
    # auto-eval's unfiltered version) so the harvest sees filter-matched
    # numbers — matching agent 1's MLP-sweep pattern, not creating a
    # parallel _rbp_fm.json filename. Per-phage TSV also uses the plain
    # convention (no _rbp_fm suffix).
    FILTER_EVAL_CMD="python -u ${CIPHER_DIR}/scripts/analysis/per_head_strict_eval.py ${EXP_DIR} \
        --val-embedding-file ${VAL_EMB} \
        --val-fasta ${VAL_FASTA} \
        --val-datasets-dir ${VAL_DATASETS_DIR} \
        --glycan-binders ${VAL_GB} \
        --per-phage-out ${CIPHER_DIR}/results/analysis/per_phage/per_phage_${NAME}.tsv"

    JOB_SCRIPT="#!/bin/bash
#SBATCH --job-name=${NAME:0:32}
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

mkdir -p ${CIPHER_DIR}/results/analysis/per_phage

echo \"======================================\"
echo \"LA filter sweep: ${NAME}  (filter=${FILTER})\"
echo \"  filter flags: ${FILTER_FLAGS}\"
echo \"  embedding:    ${EMB_TYPE}\"
echo \"  cap:          ${MAX_K_CAP}\"
echo \"  cluster th:   ${CLUSTER_TH}\"
echo \"  val GB:       ${VAL_GB}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

echo \"\"
echo \"=== TRAIN + AUTO STRICT-EVAL (unfiltered) ===\"
${TRAIN_CMD}

echo \"\"
echo \"=== EXPLICIT FILTER-MATCHED STRICT-EVAL ===\"
${FILTER_EVAL_CMD}

echo \"\"
echo \"======================================\"
echo \"Done: ${NAME} at \$(date)\"
echo \"Result JSON:  ${EXP_DIR}/results/per_head_strict_eval.json (filter-matched, headline)\"
echo \"Per-phage TSV: ${CIPHER_DIR}/results/analysis/per_phage/per_phage_${NAME}.tsv\"
echo \"======================================\"
"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] ${NAME}  (filter=${FILTER})"
        echo ""
    else
        JOB_ID=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $NF}')
        echo "  Submitted ${JOB_ID} — ${NAME}  (filter=${FILTER})"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    fi
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN complete. Set DRY_RUN=0 to submit."
else
    echo "Submitted ${N_SUBMITTED} filter-sweep job(s)."
    echo "Monitor: squeue -u \$USER"
    echo "Once they finish, refresh harvest in main ciPHer worktree:"
    echo "  cd /projects/bfzj/llindsey1/PHI_TSP/ciPHer && python scripts/analysis/harvest_results.py \\"
    echo "    --experiments-dirs experiments \\"
    echo "    ../cipher-light-attention/experiments \\"
    echo "    ../cipher-light-attention-binary/experiments \\"
    echo "    ../cipher-binary-onevsrest/experiments"
fi
echo "============================================================"
