#!/usr/bin/env bash
#
# Cap sweep + HP ablation submitter for the top 2 attention_mlp models.
# (light_attention cap sweep is delegated to agent 2 — they own that
# worktree and will run la_v3_uat_prott5_xl_seg8 cap variants in
# cipher-light-attention themselves.)
#
# Cap sweep: 6 jobs
#   {sweep_prott5_mean_cl70, concat_prott5_mean+kmer_li10_k5}
#     × {max_samples_per_k = 300, 500, 750}
# HP ablation: 3 jobs (top model only — sweep_prott5_mean_cl70)
#   - dropout 0.1 → 0.2
#   - lr 1e-5 → 5e-5
#   - hidden_dims [1280,640,320,160] → [1536,768,384,192], attention_dim 640 → 768
#
# Total: 9 jobs. Each ~30-90 min on Delta ghx4. Wall ~5-9 hours.
#
# Each job inherits all hyperparameters from its base experiment's
# experiment.json and only changes the requested flag(s) and --name.
# After training, runs default eval + per_head_strict_eval so results
# land in the next harvest pull.
#
# Env overrides:
#   CIPHER_DIR  default /projects/bfzj/llindsey1/PHI_TSP/ciPHer
#   PHASE       all (default) | cap | hp
#   DRY_RUN=1   render sbatch but do not submit
#
# Usage:
#   bash scripts/run_top3_cap_hp_sweeps.sh
#   PHASE=cap bash scripts/run_top3_cap_hp_sweeps.sh
#   DRY_RUN=1 bash scripts/run_top3_cap_hp_sweeps.sh

set -euo pipefail

CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"
PHASE="${PHASE:-all}"
DRY_RUN="${DRY_RUN:-0}"

DRY=""
if [ "$DRY_RUN" = "1" ]; then DRY="--dry-run"; fi

HELPER="${CIPHER_DIR}/scripts/cli/submit_training_variant.py"

# Top 2 attention_mlp models (LA cap sweep is agent 2's job)
SWEEP_BASE="experiments/attention_mlp/sweep_prott5_mean_cl70"
CONCAT_BASE="experiments/attention_mlp/concat_prott5_mean+kmer_li10_k5"

submit_variant() {
    local base="$1"
    local name="$2"
    shift 2
    python "${HELPER}" \
        --base-exp "${base}" \
        --name "${name}" \
        --cipher-dir "${CIPHER_DIR}" \
        ${DRY} \
        "$@"
}

# ────────────────────────────────────────────────────────────────────
# PHASE: cap sweep — 9 jobs
# ────────────────────────────────────────────────────────────────────
if [ "$PHASE" = "all" ] || [ "$PHASE" = "cap" ]; then
echo "============================================================"
echo "CAP SWEEP — 6 jobs (2 attention_mlp models × 3 caps)"
echo "(LA cap sweep is delegated to agent 2)"
echo "============================================================"
for CAP in 300 500 750; do
    submit_variant "${SWEEP_BASE}" \
        "sweep_prott5_mean_cl70_cap${CAP}" \
        --override "max_samples_per_k=${CAP}"
    echo ""
    submit_variant "${CONCAT_BASE}" \
        "concat_prott5_mean+kmer_li10_k5_cap${CAP}" \
        --override "max_samples_per_k=${CAP}"
    echo ""
done
fi

# ────────────────────────────────────────────────────────────────────
# PHASE: HP ablation on top model — 3 jobs
# ────────────────────────────────────────────────────────────────────
if [ "$PHASE" = "all" ] || [ "$PHASE" = "hp" ]; then
echo "============================================================"
echo "HP ABLATION — 3 jobs (sweep_prott5_mean_cl70 only)"
echo "============================================================"

# HP1: dropout 0.1 -> 0.2
submit_variant "${SWEEP_BASE}" \
    "sweep_prott5_mean_cl70_dropout0.2" \
    --override "dropout=0.2"
echo ""

# HP2: lr 1e-5 -> 5e-5
submit_variant "${SWEEP_BASE}" \
    "sweep_prott5_mean_cl70_lr5e-5" \
    --override "lr=5e-5"
echo ""

# HP3: bigger model
submit_variant "${SWEEP_BASE}" \
    "sweep_prott5_mean_cl70_bigmodel" \
    --override "hidden_dims=1536,768,384,192" "attention_dim=768"
echo ""
fi

echo "============================================================"
echo "Submitted. Tail logs in ${CIPHER_DIR}/logs/variant_*_<jobid>.log"
echo "After all jobs finish, refresh harvest:"
echo "  python scripts/analysis/harvest_results.py --experiments-dirs \\"
echo "      experiments \\"
echo "      ../cipher-light-attention/experiments \\"
echo "      ../cipher-light-attention-binary/experiments"
echo "============================================================"
