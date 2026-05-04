#!/usr/bin/env bash
#
# Filter sweep — 4 attention_mlp base models × 3 RBP filter rules = 12
# new training runs. Holds architecture / embedding / hyperparameters
# constant within each row of the matrix; the only thing changing is
# the training-side RBP filter. Eval auto-runs with the matched filter
# (per the patched submit_training_variant.py + per_head_strict_eval.py).
#
# Per CLAUDE.md naming convention: <arch>_<emb>_<filter>[_<extras>].
#
# Matrix:
#   base_run                              | filter_all          | filter_tools3            | filter_tools4
#   ------                                | ----------          | ------------             | -------------
#   sweep_kmer_aa20_k4                    | mlp_kmer_aa20_k4_all| mlp_kmer_aa20_k4_tools3  | mlp_kmer_aa20_k4_tools4
#   sweep_posList_esm2_3b_mean_cl70       | mlp_esm23b_all_cl70 | mlp_esm23b_tools3_cl70   | mlp_esm23b_tools4_cl70
#   sweep_posList_esm2_650m_seg4_cl70     | mlp_esm2650mseg4_all_cl70 | mlp_esm2650mseg4_tools3_cl70 | mlp_esm2650mseg4_tools4_cl70
#   sweep_prott5_mean_cl70                | mlp_prott5_all_cl70 | mlp_prott5_tools3_cl70   | mlp_prott5_tools4_cl70
#
# Usage:
#   bash scripts/run_filter_sweep_12.sh                    # all 12
#   FILTERS="all"     bash scripts/run_filter_sweep_12.sh  # subset of filters
#   BASES="sweep_kmer_aa20_k4" bash scripts/run_filter_sweep_12.sh  # subset of bases
#   DRY_RUN=1         bash scripts/run_filter_sweep_12.sh  # render sbatch, don't submit

set -euo pipefail

CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"
SUBMIT="${CIPHER_DIR}/scripts/cli/submit_training_variant.py"

BASES="${BASES:-sweep_kmer_aa20_k4 sweep_posList_esm2_3b_mean_cl70 sweep_posList_esm2_650m_seg4_cl70 sweep_prott5_mean_cl70}"
FILTERS="${FILTERS:-all tools3 tools4}"
DRY_RUN_FLAG=""
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN_FLAG="--dry-run"
fi

# Map base run -> embedding tag for the new name (per CLAUDE.md convention)
emb_tag_for() {
    case "$1" in
        sweep_kmer_aa20_k4)                  echo "mlp_kmer_aa20_k4" ;;
        sweep_posList_esm2_3b_mean_cl70)     echo "mlp_esm23b" ;;
        sweep_posList_esm2_650m_seg4_cl70)   echo "mlp_esm2650mseg4" ;;
        sweep_prott5_mean_cl70)              echo "mlp_prott5" ;;
        *)                                   echo "mlp_unknown" ;;
    esac
}

# Map base run -> "_cl70" suffix or "" (kmer base has no cluster threshold)
extras_tag_for() {
    case "$1" in
        sweep_kmer_aa20_k4)                  echo "" ;;
        *)                                   echo "_cl70" ;;
    esac
}

# Map filter rule -> override list (cipher-train flags) for submit_training_variant
overrides_for() {
    case "$1" in
        all)
            # ALL = no tool filter; clear positive_list and protein_set
            echo "protein_set=all_glycan_binders tools= positive_list="
            ;;
        tools3)
            echo "tools=SpikeHunter,DePP_85,PhageRBPdetect protein_set= positive_list="
            ;;
        tools4)
            echo "tools=SpikeHunter,DePP_85,PhageRBPdetect,DepoScope protein_set= positive_list="
            ;;
        *)
            echo "ERROR: unknown filter rule: $1" >&2
            exit 1
            ;;
    esac
}

submit_one() {
    local BASE="$1"
    local FILTER="$2"
    local EMB_TAG=$(emb_tag_for "$BASE")
    local EXTRAS=$(extras_tag_for "$BASE")
    local NAME="${EMB_TAG}_${FILTER}${EXTRAS}"
    local OVERRIDES=$(overrides_for "$FILTER")

    echo ""
    echo "----------------------------------------"
    echo "  base:      $BASE"
    echo "  filter:    $FILTER"
    echo "  new name:  $NAME"
    echo "  overrides: $OVERRIDES"
    echo "----------------------------------------"

    python "$SUBMIT" \
        --base-exp "experiments/attention_mlp/${BASE}" \
        --name "$NAME" \
        --override $OVERRIDES \
        --cipher-dir "$CIPHER_DIR" \
        $DRY_RUN_FLAG
}

for BASE in $BASES; do
    for FILTER in $FILTERS; do
        submit_one "$BASE" "$FILTER"
    done
done
