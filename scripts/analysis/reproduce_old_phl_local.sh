#!/usr/bin/env bash
#
# Re-run the OLD-repo evaluation locally on this laptop, against the
# already-trained `host_prediction_local_test_v3` model checkpoints.
# No retraining — we just exercise the old eval pipeline against the
# old saved weights and compare against the saved results.json.
#
# This answers two questions in one shot:
#   1. Is the old result (PHL HR@1 = 0.291) reproducible bit-for-bit
#      from the saved checkpoints? If yes → the saved JSON is a
#      reliable reference point.
#   2. Does our laptop environment (torch 2.11, numpy 2.4, MPS) load
#      and score the old PyTorch state_dicts cleanly? If yes → we can
#      iterate locally without Delta.
#
# Runs entirely on the laptop. Outputs land in a fresh sibling
# directory so we don't overwrite the original April 11 results.
#
# Usage:
#   bash scripts/analysis/reproduce_old_phl_local.sh
#   MERGE_STRATEGY=o_only bash scripts/analysis/reproduce_old_phl_local.sh
#                                  # to test the script's documented default
#                                  # vs the apparent `raw` used in the saved
#                                  # results.json

set -euo pipefail

OLD_REPO="${OLD_REPO:-/Users/leannmlindsey/WORK/PHI_TSP/phi_tsp/klebsiella}"
MERGE_STRATEGY="${MERGE_STRATEGY:-raw}"
OUTPUT_DIR="${OUTPUT_DIR:-${OLD_REPO}/output/validation/host_prediction_local_test_v3_repro_$(date +%Y%m%d_%H%M%S)_${MERGE_STRATEGY}}"

K_MODEL="${OLD_REPO}/output/local_test_v3/model_k_single_label_all_glycan_binders_seed42_20260411_094352"
O_MODEL="${OLD_REPO}/output/local_test_v3/model_o_single_label_all_glycan_binders_seed42_20260411_095108"

EMB="${OLD_REPO}/validation_data/combined/validation_inputs/validation_embeddings_md5.npz"
FASTA="${OLD_REPO}/validation_data/combined/validation_inputs/validation_rbps_all.faa"
SEROTYPES="${OLD_REPO}/validation_data/combined/validation_inputs/serotypes.tsv"
HOST_RANGE_DIR="${OLD_REPO}/validation_data/combined/HOST_RANGE"

EVAL_PY="${OLD_REPO}/scripts/evaluate/evaluate_validation_host_prediction.py"
ORIG_RESULTS="${OLD_REPO}/output/validation/host_prediction_local_test_v3/host_prediction_results.json"

echo "============================================================"
echo "REPRODUCE OLD PHL EVAL — LAPTOP LOCAL"
echo "  Old repo:        ${OLD_REPO}"
echo "  K model:         $(basename ${K_MODEL})"
echo "  O model:         $(basename ${O_MODEL})"
echo "  Merge strategy:  ${MERGE_STRATEGY}"
echo "  Output dir:      ${OUTPUT_DIR}"
echo "  Original result: ${ORIG_RESULTS}"
echo "============================================================"
echo ""

# Sanity checks — bail loudly if anything's missing
for path in "$K_MODEL/best_model.pt" "$O_MODEL/best_model.pt" \
            "$EMB" "$FASTA" "$SEROTYPES" "$HOST_RANGE_DIR" "$EVAL_PY"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: missing required input: $path"
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

# Run the eval. Stay inside the old repo dir so any relative imports
# resolve correctly.
cd "${OLD_REPO}"
python scripts/evaluate/evaluate_validation_host_prediction.py \
    --k-model "$K_MODEL" \
    --o-model "$O_MODEL" \
    --embeddings "$EMB" \
    --protein-fasta "$FASTA" \
    --serotypes "$SEROTYPES" \
    --host-range-dir "$HOST_RANGE_DIR" \
    --datasets CHEN GORODNICHIV UCSD PBIP PhageHostLearn \
    --merge-strategy "$MERGE_STRATEGY" \
    --output "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "COMPARISON vs original (April 11) saved results"
echo "============================================================"
python3 - <<PYEOF
import json, sys

orig = json.load(open("${ORIG_RESULTS}"))
new  = json.load(open("${OUTPUT_DIR}/host_prediction_results.json"))

print(f'{"Dataset":<16} {"orig HR@1":>10} {"repro HR@1":>11} {"orig K":>8} {"repro K":>9} {"orig O":>8} {"repro O":>9} {"match":>7}')
print('-' * 100)
for ds in ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']:
    o_d = orig.get(ds, {})
    n_d = new.get(ds, {})
    o_m = o_d.get('hr_at_k', {}).get('1')
    n_m = n_d.get('hr_at_k', {}).get('1')
    o_k = o_d.get('k_hr_at_k', {}).get('1')
    n_k = n_d.get('k_hr_at_k', {}).get('1')
    o_o = o_d.get('o_hr_at_k', {}).get('1')
    n_o = n_d.get('o_hr_at_k', {}).get('1')

    def fmt(v): return f'{v:.4f}' if isinstance(v, (int, float)) else '—'
    match = '✓' if (o_m is not None and n_m is not None
                    and abs(o_m - n_m) < 1e-4) else '✗'
    print(f'{ds:<16} {fmt(o_m):>10} {fmt(n_m):>11} {fmt(o_k):>8} {fmt(n_k):>9} {fmt(o_o):>8} {fmt(n_o):>9} {match:>7}')

orig_phl = orig['PhageHostLearn']['hr_at_k'].get('1')
new_phl  = new['PhageHostLearn']['hr_at_k'].get('1')
print()
if orig_phl is not None and new_phl is not None and abs(orig_phl - new_phl) < 1e-4:
    print(f'✓ PHL HR@1 reproduced exactly: {new_phl:.4f}')
elif orig_phl is not None and new_phl is not None:
    delta = new_phl - orig_phl
    print(f'≈ PHL HR@1 close but not exact: orig={orig_phl:.4f}, repro={new_phl:.4f}, Δ={delta:+.4f}')
    print(f'  → likely environment-version drift (torch/numpy float ops)')
else:
    print(f'PHL HR@1 — orig={orig_phl}, repro={new_phl}')
PYEOF

echo ""
echo "Done. Repro results: ${OUTPUT_DIR}/host_prediction_results.json"
