#!/usr/bin/env bash
#
# Run the OLD klebsiella `host_prediction_local_test_v3` recipe end-to-end
# on this LAPTOP using the OLD codebase. Verifies whether retraining the
# OLD model from scratch reproduces the saved-checkpoint eval results
# (PHL HR@1 = 0.291, O-only = 0.282).
#
# Three possible outcomes:
#   1. New eval matches saved checkpoint's eval bit-exact → the recipe
#      is fully reproducible. Any deviation we see in our ciPHer port
#      is a real bug or methodology difference, not GPU non-determinism.
#   2. New eval is close (within ~0.02) → reproducible up to
#      training-time numerical noise. Confirms the recipe is sound;
#      the gap to our ciPHer numbers is methodology, not luck.
#   3. New eval is far off (PHL HR@1 < 0.25 or O-only < 0.22) → even
#      the original codebase doesn't reproduce its own checkpoint when
#      retrained. Hardware non-determinism is the dominant factor;
#      multi-seed averaging is the right comparison going forward.
#
# Outputs to a TIMESTAMPED dir to avoid clobbering the saved
# checkpoints + eval that we depend on as the gold reference.
#
# Usage:
#   bash scripts/analysis/reproduce_old_v3_training_local.sh
#
# Time: ~30–60 min on MPS (200 epochs × 2 heads).
# Disk: ~50 MB for new checkpoints; eval outputs are tiny.

set -euo pipefail

OLD_REPO="${OLD_REPO:-/Users/leannmlindsey/WORK/PHI_TSP/phi_tsp/klebsiella}"
CONFIG="${CONFIG:-config_local_v3.yaml}"
SEED="${SEED:-42}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

EVAL_OUT="output/validation/host_prediction_local_test_v3_retrain_${TIMESTAMP}"
ORIG_EVAL="${OLD_REPO}/output/validation/host_prediction_local_test_v3/host_prediction_results.json"

if [ ! -d "$OLD_REPO" ]; then
    echo "ERROR: OLD_REPO not found: $OLD_REPO"
    exit 1
fi

cd "$OLD_REPO"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $OLD_REPO/$CONFIG"
    exit 1
fi

echo "============================================================"
echo "RETRAIN OLD V3 RECIPE — LAPTOP LOCAL"
echo "  Old repo: $OLD_REPO"
echo "  Config:   $CONFIG"
echo "  Seed:     $SEED"
echo "  Eval out: $EVAL_OUT (timestamped — won't clobber original)"
echo "  Original: $ORIG_EVAL"
echo "============================================================"
echo ""

echo "=== Step 1: prepare_training_data ==="
python scripts/data_prep/prepare_training_data.py --config "$CONFIG"

echo ""
echo "=== Step 2: create_canonical_split ==="
python scripts/data_prep/create_canonical_split.py --config "$CONFIG"

echo ""
echo "=== Step 3: train_serotype_model (K head + O head, ~30–60 min) ==="
python scripts/train/train_serotype_model.py --config "$CONFIG" --seed "$SEED"

# Find the freshly-trained model dirs by mtime (most recent that match
# the seed pattern). `ls -t` sorts by mtime newest-first; head -1 picks
# the most recent.
K_MODEL=$(ls -dt output/local_test_v3/model_k_*_seed${SEED}_*/ 2>/dev/null | head -1)
O_MODEL=$(ls -dt output/local_test_v3/model_o_*_seed${SEED}_*/ 2>/dev/null | head -1)

if [ -z "$K_MODEL" ] || [ -z "$O_MODEL" ]; then
    echo "ERROR: could not locate freshly trained K/O model dirs"
    exit 1
fi

echo ""
echo "=== Step 4: evaluate (merge-strategy raw, matching saved-result mode) ==="
echo "  K model: $K_MODEL"
echo "  O model: $O_MODEL"
echo "  Output:  $EVAL_OUT"

python scripts/evaluate/evaluate_validation_host_prediction.py \
    --k-model "$K_MODEL" \
    --o-model "$O_MODEL" \
    --embeddings validation_data/combined/validation_inputs/validation_embeddings_md5.npz \
    --protein-fasta validation_data/combined/validation_inputs/validation_rbps_all.faa \
    --serotypes validation_data/combined/validation_inputs/serotypes.tsv \
    --host-range-dir validation_data/combined/HOST_RANGE \
    --datasets CHEN GORODNICHIV UCSD PBIP PhageHostLearn \
    --merge-strategy raw \
    --output "$EVAL_OUT"

echo ""
echo "============================================================"
echo "COMPARISON: retrain-from-scratch vs saved checkpoint"
echo "============================================================"

python3 - <<PYEOF
import json
orig = json.load(open("$ORIG_EVAL"))
new  = json.load(open("$EVAL_OUT/host_prediction_results.json"))

print(f'{"Dataset":<16} {"orig HR@1":>10} {"new HR@1":>10} {"orig K":>8} {"new K":>8} {"orig O":>8} {"new O":>8}  match')
print('-' * 105)

def fmt(v): return f'{v:.4f}' if isinstance(v, (int, float)) else '   —  '

bit_exact = True
within_002 = True
phl_o_only_match = True

for ds in ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']:
    o_d = orig.get(ds, {}); n_d = new.get(ds, {})
    om = o_d.get('hr_at_k', {}).get('1'); nm = n_d.get('hr_at_k', {}).get('1')
    ok = o_d.get('k_hr_at_k', {}).get('1'); nk = n_d.get('k_hr_at_k', {}).get('1')
    oo = o_d.get('o_hr_at_k', {}).get('1'); no = n_d.get('o_hr_at_k', {}).get('1')

    eq = (om is not None and nm is not None and abs(om - nm) < 1e-4)
    near = (om is not None and nm is not None and abs(om - nm) < 0.02)

    marker = '✓' if eq else ('≈' if near else '✗')
    print(f'{ds:<16} {fmt(om):>10} {fmt(nm):>10} {fmt(ok):>8} {fmt(nk):>8} {fmt(oo):>8} {fmt(no):>8}  {marker}')

    if not eq: bit_exact = False
    if not near: within_002 = False

# Specific PHL O-only check (the key failure mode for our ciPHer port)
phl_oo_orig = orig['PhageHostLearn']['o_hr_at_k'].get('1')
phl_oo_new  = new['PhageHostLearn']['o_hr_at_k'].get('1')
phl_oo_match = (phl_oo_orig is not None and phl_oo_new is not None
                and abs(phl_oo_orig - phl_oo_new) < 0.02)

print()
print('Headline:')
print(f'  PHL HR@1:    orig={fmt(orig["PhageHostLearn"]["hr_at_k"].get("1"))}  '
      f'new={fmt(new["PhageHostLearn"]["hr_at_k"].get("1"))}')
print(f'  PHL O-only:  orig={fmt(phl_oo_orig)}  new={fmt(phl_oo_new)}  '
      f'(match={phl_oo_match})')
print()
if bit_exact:
    print('VERDICT: bit-exact reproduction. Old recipe IS deterministic on this laptop.')
    print('         Any gap in our ciPHer port is a real methodology difference.')
elif within_002:
    print('VERDICT: close (within ±0.02). Reproducible up to numerical noise.')
    print('         Methodology is the dominant factor; ciPHer port can match.')
elif phl_oo_match:
    print('VERDICT: most metrics drift, but PHL O-only matches. Per-dataset variance.')
else:
    print('VERDICT: training is non-deterministic. PHL O-only varies between')
    print('         retraining runs of the SAME recipe. Multi-seed sweep is the')
    print('         right comparison; expect ±0.05 spread.')

print()
print(f'New eval file: $EVAL_OUT/host_prediction_results.json')
PYEOF
