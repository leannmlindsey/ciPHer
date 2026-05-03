#!/usr/bin/env bash
#
# 2026-05-01 morning experiment batch — disentangle architecture vs
# training-data, fill obvious embedding × architecture gaps.
#
# Phases:
#   1. kmer_aa20_k4 across training datasets (architecture-vs-data sweep)
#   2. ESM-2 3B + segments (seg4, seg8) — training (extraction is separate)
#   3. concat ESM-2 3B + kmer (li10_k5, aa20_k4)
#   4. kmer_aa20_k4 hyperparameter sweep
#   5. ESM-2 3B with v2 / v3 per-head highconf
#
# Each phase = a few sbatch jobs. Set PHASE=N to run only one phase, or
# leave PHASE=all (default) to fire every job.
#
# Usage:
#   bash scripts/run_2026-05-01_morning_experiments.sh                # all
#   PHASE=2 bash scripts/run_2026-05-01_morning_experiments.sh        # just 3B segs
#   DRY_RUN=1 bash scripts/run_2026-05-01_morning_experiments.sh
#
# IMPORTANT — Phase 2 and Phase 3 require ESM-2 3B segment NPZs that
# don't exist yet. Submit those FIRST:
#     bash scripts/extract_embeddings/submit_extractions.sh esm2_3b_segments4
#     bash scripts/extract_embeddings/submit_extractions.sh esm2_3b_segments8
# When those finish (~12-18h), Phase 2 + 3 jobs will find the files.

set -euo pipefail

CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"
PHASE="${PHASE:-all}"
DRY_RUN="${DRY_RUN:-0}"
HELPER="${CIPHER_DIR}/scripts/cli/submit_training_variant.py"

DRY=""
if [ "$DRY_RUN" = "1" ]; then DRY="--dry-run"; fi

# ============================================================
# Path constants (Delta-canonical)
# ============================================================
HC_V1_LIST="${CIPHER_DIR}/data/training_data/metadata/pipeline_positive.list"
HC_V2_DIR="${CIPHER_DIR}/data/training_data/metadata/highconf_v2"
HC_V3_DIR="${CIPHER_DIR}/data/training_data/metadata/highconf_v3_multitop"

# ESM-2 3B segment embeddings — produced by submit_extractions.sh.
# If extraction hasn't finished, Phase 2 / Phase 3 jobs will fail at
# embedding load (then can be re-submitted).
EMB_3B_SEG4_TRAIN="/work/hdd/bfzj/llindsey1/embeddings/esm2_3b_segments4/candidates_esm2_3b_segments4_md5.npz"
EMB_3B_SEG4_VAL="/work/hdd/bfzj/llindsey1/validation_embeddings/esm2_3b_segments4/validation_esm2_3b_segments4_md5.npz"
EMB_3B_SEG8_TRAIN="/work/hdd/bfzj/llindsey1/embeddings/esm2_3b_segments8/candidates_esm2_3b_segments8_md5.npz"
EMB_3B_SEG8_VAL="/work/hdd/bfzj/llindsey1/validation_embeddings/esm2_3b_segments8/validation_esm2_3b_segments8_md5.npz"

# kmer_aa20_k4 embeddings (training + validation), already on disk
EMB_KMER_AA20_K4_TRAIN="/work/hdd/bfzj/llindsey1/kmer_features/candidates_aa20_k4.npz"
EMB_KMER_AA20_K4_VAL="/work/hdd/bfzj/llindsey1/kmer_features/validation_aa20_k4.npz"

# Bases for variant submission
KMER_BASE="experiments/attention_mlp/sweep_kmer_aa20_k4"
ESM2_3B_BASE="experiments/attention_mlp/sweep_esm2_3b_mean_cl70"

submit_variant() {
    local base="$1"; local name="$2"; shift 2
    python "${HELPER}" \
        --base-exp "${base}" \
        --name "${name}" \
        --cipher-dir "${CIPHER_DIR}" \
        ${DRY} \
        "$@"
    echo ""
}

# ────────────────────────────────────────────────────────────────────
# PHASE 1 — kmer_aa20_k4 across training datasets
# ────────────────────────────────────────────────────────────────────
if [ "$PHASE" = "all" ] || [ "$PHASE" = "1" ]; then
echo "============================================================"
echo "PHASE 1 — kmer_aa20_k4 × training datasets (4 jobs)"
echo "  Disentangles architecture-vs-data: same model+embedding, different curated data."
echo "  Existing:   sweep_kmer_aa20_k4 = tools-filter (PHL OR HR@1 = 0.560)"
echo "============================================================"

# v1 highconf: --positive_list points to the broader pipeline_positive list,
# and the highconf_pipeline_K convention narrows to the K-curated subset.
submit_variant "${KMER_BASE}" "highconf_pipeline_K_kmer_aa20_k4" \
    --override "positive_list=${CIPHER_DIR}/data/training_data/metadata/highconf_pipeline_positive_K.list" \
                "tools="

# v2 strict — per-head positive lists
submit_variant "${KMER_BASE}" "v2_strict_kmer_aa20_k4" \
    --override "positive_list_k=${HC_V2_DIR}/HC_K_cl95.list" \
                "positive_list_o=${HC_V2_DIR}/HC_O_cl95_full_coverage.list" \
                "tools="

# v3 strict — newer per-head lists (path may differ; common location)
submit_variant "${KMER_BASE}" "v3_strict_kmer_aa20_k4" \
    --override "positive_list_k=${HC_V3_DIR}/HC_K_cl95_multitop.list" \
                "positive_list_o=${HC_V3_DIR}/HC_O_cl95_multitop_full_coverage.list" \
                "tools="

# v4 (only if the v4 list is on disk; prints SKIP otherwise)
V4_K_LIST="${CIPHER_DIR}/data/training_data/metadata/highconf_v4/HC_K_v4.list"
V4_O_LIST="${CIPHER_DIR}/data/training_data/metadata/highconf_v4/HC_O_v4.list"
if [ -f "$V4_K_LIST" ] && [ -f "$V4_O_LIST" ]; then
    submit_variant "${KMER_BASE}" "v4_kmer_aa20_k4" \
        --override "positive_list_k=${V4_K_LIST}" \
                    "positive_list_o=${V4_O_LIST}" \
                    "tools="
else
    echo "  SKIP v4_kmer_aa20_k4 — v4 lists not yet on disk."
    echo "    expected: ${V4_K_LIST}"
    echo "              ${V4_O_LIST}"
fi
fi

# ────────────────────────────────────────────────────────────────────
# PHASE 2 — ESM-2 3B + segments (seg4, seg8). Needs extracted NPZs.
# ────────────────────────────────────────────────────────────────────
if [ "$PHASE" = "all" ] || [ "$PHASE" = "2" ]; then
echo "============================================================"
echo "PHASE 2 — ESM-2 3B segments (2 jobs; needs extracted seg4/seg8 NPZs)"
echo "  3B is the K-source in the current best hybrid (esm2_3b K + kmer O = 0.620 PHL)"
echo "  seg4 lifted 650M from 0.475 to 0.470 OR; if that pattern transfers to 3B"
echo "  the K-head improves and the hybrid lifts further."
echo "============================================================"

submit_variant "${ESM2_3B_BASE}" "sweep_esm2_3b_seg4" \
    --override "embedding_type=esm2_3b_seg4" \
                "embedding_file=${EMB_3B_SEG4_TRAIN}" \
                "val_embedding_file=${EMB_3B_SEG4_VAL}"

submit_variant "${ESM2_3B_BASE}" "sweep_esm2_3b_seg8" \
    --override "embedding_type=esm2_3b_seg8" \
                "embedding_file=${EMB_3B_SEG8_TRAIN}" \
                "val_embedding_file=${EMB_3B_SEG8_VAL}"
fi

# ────────────────────────────────────────────────────────────────────
# PHASE 3 — concat ESM-2 3B + kmer (li10_k5, aa20_k4)
# ────────────────────────────────────────────────────────────────────
if [ "$PHASE" = "all" ] || [ "$PHASE" = "3" ]; then
echo "============================================================"
echo "PHASE 3 — concat ESM-2 3B + kmer (2 jobs)"
echo "  prott5 + kmer_li10_k5 concat is #2 PHL. 3B has stronger standalone K@1,"
echo "  so 3B + kmer should be at least competitive."
echo "============================================================"

KMER_LI10_TRAIN="/work/hdd/bfzj/llindsey1/kmer_features/candidates_li10_k5.npz"
KMER_LI10_VAL="/work/hdd/bfzj/llindsey1/kmer_features/validation_li10_k5.npz"

# concat with kmer_li10_k5 (already partially exists per harvest but had no PHL OR)
submit_variant "${ESM2_3B_BASE}" "concat_esm2_3b_mean+kmer_li10_k5" \
    --override "embedding_type_2=kmer_li10_k5" \
                "embedding_file_2=${KMER_LI10_TRAIN}" \
                "val_embedding_file_2=${KMER_LI10_VAL}"

# concat with kmer_aa20_k4 (the better kmer per leaderboard) — NEW combination
submit_variant "${ESM2_3B_BASE}" "concat_esm2_3b_mean+kmer_aa20_k4" \
    --override "embedding_type_2=kmer_aa20_k4" \
                "embedding_file_2=${EMB_KMER_AA20_K4_TRAIN}" \
                "val_embedding_file_2=${EMB_KMER_AA20_K4_VAL}"
fi

# ────────────────────────────────────────────────────────────────────
# PHASE 4 — kmer_aa20_k4 hyperparameter sweep (5 jobs)
# Mirrors the prott5_mean hp variants that lifted PHL by 1-3 pp.
# ────────────────────────────────────────────────────────────────────
if [ "$PHASE" = "all" ] || [ "$PHASE" = "4" ]; then
echo "============================================================"
echo "PHASE 4 — kmer_aa20_k4 hyperparameter sweep (5 jobs)"
echo "  We have only one kmer_aa20_k4 run. Mirror prott5_mean's variants."
echo "============================================================"

submit_variant "${KMER_BASE}" "sweep_kmer_aa20_k4_lr5e-5" \
    --override "lr=5e-5"
submit_variant "${KMER_BASE}" "sweep_kmer_aa20_k4_dropout0.2" \
    --override "dropout=0.2"
for CAP in 300 500 750; do
    submit_variant "${KMER_BASE}" "sweep_kmer_aa20_k4_cap${CAP}" \
        --override "max_samples_per_k=${CAP}"
done
fi

# ────────────────────────────────────────────────────────────────────
# PHASE 5 — ESM-2 3B with v2 / v3 per-head highconf (2 jobs)
# Cleans the K-source's training data; should lift the hybrid further.
# ────────────────────────────────────────────────────────────────────
if [ "$PHASE" = "all" ] || [ "$PHASE" = "5" ]; then
echo "============================================================"
echo "PHASE 5 — ESM-2 3B with v2 / v3 per-head highconf (2 jobs)"
echo "  3B is the K-source in the best hybrid; per-head highconf training"
echo "  may lift its K@1 from 0.300 toward 0.40."
echo "============================================================"

submit_variant "${ESM2_3B_BASE}" "v2_strict_esm2_3b_mean" \
    --override "positive_list_k=${HC_V2_DIR}/HC_K_cl95.list" \
                "positive_list_o=${HC_V2_DIR}/HC_O_cl95_full_coverage.list" \
                "tools="

submit_variant "${ESM2_3B_BASE}" "v3_strict_esm2_3b_mean" \
    --override "positive_list_k=${HC_V3_DIR}/HC_K_cl95_multitop.list" \
                "positive_list_o=${HC_V3_DIR}/HC_O_cl95_multitop_full_coverage.list" \
                "tools="
fi

echo ""
echo "============================================================"
echo "Done. After jobs finish, refresh harvest:"
echo "  python scripts/analysis/harvest_results.py --experiments-dirs \\"
echo "      experiments \$(ls -d ../cipher-*/experiments 2>/dev/null)"
echo "Then re-run the figure scripts."
echo "============================================================"
