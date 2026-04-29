#!/usr/bin/env bash
#
# Score a ciPHer attention_mlp experiment under the OLD klebsiella eval
# methodology (class-ranking: rank K + O serotypes merged, hit if top-1
# matches host's K or O). Produces apples-to-apples comparison vs OLD
# klebsiella's reported numbers.
#
# Why this exists
# ---------------
# Cipher's `cipher.evaluation.runner` ranks candidate hosts (host-ranking)
# with z-score normalization. OLD klebsiella ranks K+O serotype classes
# with raw probability merge. They produce different HR@1 from the same
# model — see notes on eval methodology.
#
# This wrapper runs OLD's `evaluate_validation_host_prediction.py`
# (unmodified) against a cipher experiment's model_k/ and model_o/
# checkpoints. We've verified the cipher AttentionMLP weights are
# bit-exact compatible with OLD's, so OLD's eval script loads them
# directly with no porting.
#
# Outputs land in <cipher_exp_dir>/results_old_method/{raw,o_only}/
# with host_prediction_results.json + per-pair details TSV + HR@K plots.
#
# Usage:
#   bash scripts/analysis/eval_with_old_method.sh <cipher_exp_dir>
#   bash scripts/analysis/eval_with_old_method.sh <cipher_exp_dir> --mode raw
#
# Env overrides:
#   OLD_REPO     path to phi_tsp/klebsiella checkout
#                (default: ../phi_tsp/klebsiella relative to cipher repo)
#   VAL_EMB      validation embeddings NPZ
#   VAL_FASTA    validation RBP FASTA
#   SEROTYPES    serotypes TSV (genome_id, K, O)
#   HR_DIR       OLD-format HOST_RANGE dir (top-level mapping CSV)
#   DATASETS     space-separated dataset list
#                (default: CHEN GORODNICHIV UCSD PBIP PhageHostLearn)

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <cipher_exp_dir> [--mode raw|o_only|both]" >&2
    exit 1
fi

EXP="$1"
shift || true
MODE="both"
while [ $# -gt 0 ]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Resolve cipher repo from this script's location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CIPHER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

OLD_REPO="${OLD_REPO:-$(cd "$CIPHER_DIR/../phi_tsp/klebsiella" 2>/dev/null && pwd || echo '')}"
if [ -z "$OLD_REPO" ] || [ ! -d "$OLD_REPO" ]; then
    echo "ERROR: OLD_REPO not found. Set OLD_REPO env var to path of phi_tsp/klebsiella checkout." >&2
    exit 1
fi

OLD_EVAL="$OLD_REPO/scripts/evaluate/evaluate_validation_host_prediction.py"
OLD_TRAIN_DIR="$OLD_REPO/scripts/train"  # for `import train_serotype_model`

VAL_EMB="${VAL_EMB:-$OLD_REPO/validation_data/combined/validation_inputs/validation_embeddings_md5.npz}"
VAL_FASTA="${VAL_FASTA:-$CIPHER_DIR/data/validation_data/metadata/validation_rbps_all.faa}"
SEROTYPES="${SEROTYPES:-$OLD_REPO/validation_data/combined/validation_inputs/serotypes.tsv}"
HR_DIR="${HR_DIR:-$OLD_REPO/validation_data/combined/HOST_RANGE}"
DATASETS="${DATASETS:-CHEN GORODNICHIV UCSD PBIP PhageHostLearn}"

# Resolve EXP to absolute path
EXP="$(cd "$EXP" && pwd)"
K_MODEL="$EXP/model_k"
O_MODEL="$EXP/model_o"

for p in "$OLD_EVAL" "$VAL_EMB" "$VAL_FASTA" "$SEROTYPES" "$HR_DIR" \
         "$K_MODEL/best_model.pt" "$K_MODEL/config.json" \
         "$O_MODEL/best_model.pt" "$O_MODEL/config.json"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: missing $p" >&2
        exit 1
    fi
done

OUT_BASE="$EXP/results_old_method"
mkdir -p "$OUT_BASE"

echo "============================================================"
echo "OLD-method eval on cipher checkpoint"
echo "  Cipher exp:   $EXP"
echo "  OLD eval:     $OLD_EVAL"
echo "  Embeddings:   $VAL_EMB"
echo "  HOST_RANGE:   $HR_DIR"
echo "  Datasets:     $DATASETS"
echo "  Modes:        $MODE"
echo "============================================================"

export PYTHONPATH="$OLD_TRAIN_DIR:${PYTHONPATH:-}"

run_mode() {
    local mode="$1"
    local out="$OUT_BASE/$mode"
    mkdir -p "$out"
    echo ""
    echo "=== merge-strategy: $mode ==="
    ( cd "$OLD_REPO" && python3 "$OLD_EVAL" \
        --k-model "$K_MODEL" \
        --o-model "$O_MODEL" \
        --embeddings "$VAL_EMB" \
        --protein-fasta "$VAL_FASTA" \
        --serotypes "$SEROTYPES" \
        --host-range-dir "$HR_DIR" \
        --datasets $DATASETS \
        --merge-strategy "$mode" \
        --output "$out" )
}

case "$MODE" in
    raw|o_only) run_mode "$MODE" ;;
    both) run_mode raw; run_mode o_only ;;
    *) echo "ERROR: unknown --mode $MODE (expected raw|o_only|both)" >&2; exit 1 ;;
esac

echo ""
echo "============================================================"
echo "SUMMARY (HR@1 per dataset)"
echo "============================================================"
python3 - <<PYEOF
import json, os, sys
out_base = "$OUT_BASE"
modes = ["raw", "o_only"] if "$MODE" == "both" else ["$MODE"]
datasets = "$DATASETS".split() + ["OVERALL"]
header = ["dataset"] + [f"{m} HR@1" for m in modes] + [f"{m} HR@5" for m in modes] + ["n"]
print("  " + "  ".join(f"{h:>16}" for h in header))
for ds in datasets:
    row = [ds]
    n = ""
    for m in modes:
        p = os.path.join(out_base, m, "host_prediction_results.json")
        if not os.path.exists(p):
            row.append("-")
            continue
        d = json.load(open(p))
        if ds in d:
            row.append(f"{d[ds]['hr_at_k']['1']:.4f}")
            n = str(d[ds]['n_evaluated'])
        else:
            row.append("-")
    for m in modes:
        p = os.path.join(out_base, m, "host_prediction_results.json")
        if not os.path.exists(p):
            row.append("-")
            continue
        d = json.load(open(p))
        row.append(f"{d[ds]['hr_at_k']['5']:.4f}" if ds in d else "-")
    row.append(n)
    print("  " + "  ".join(f"{c:>16}" for c in row))
PYEOF

echo ""
echo "Results: $OUT_BASE/{raw,o_only}/host_prediction_results.json"
