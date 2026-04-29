#!/usr/bin/env bash
#
# Run OUR ciPHer attention_mlp training with the OLD-recipe config,
# but on THIS LAPTOP (MPS), to test whether the gap between our port
# and the old model is hardware-driven (Delta CUDA vs laptop MPS) or
# a real code/methodology bug.
#
# If this run matches the old saved checkpoint's eval results, the
# Delta vs laptop hardware difference explains everything and we can
# stop searching for code bugs.
#
# Outputs to a unique experiment dir so it doesn't clobber Delta runs.
#
# Usage:
#   bash scripts/analysis/run_cipher_repro_local.sh
#
# Time: ~30–60 min on MPS (200 epochs × 2 heads).

set -euo pipefail

CIPHER_DIR="${CIPHER_DIR:-/Users/leannmlindsey/WORK/PHI_TSP/cipher}"
OLD_REPO="${OLD_REPO:-/Users/leannmlindsey/WORK/PHI_TSP/phi_tsp/klebsiella}"

ASSOC_MAP="${CIPHER_DIR}/data/training_data/metadata/host_phage_protein_map.tsv"
GLYCAN_BINDERS="${CIPHER_DIR}/data/training_data/metadata/glycan_binders_custom.tsv"
POSITIVE_LIST="${CIPHER_DIR}/data/training_data/metadata/pipeline_positive.list"

# ESM-2 650M mean embedding files — same data file used by both old and new.
# Point at the laptop copies (same md5 keys regardless of which file).
TRAIN_EMB="${TRAIN_EMB:-${OLD_REPO}/data/candidates_embeddings_md5.npz}"
VAL_EMB="${VAL_EMB:-${OLD_REPO}/validation_data/combined/validation_inputs/validation_embeddings_md5.npz}"

# Validation paths
VAL_FASTA="${VAL_FASTA:-${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa}"
VAL_DATASETS_DIR="${VAL_DATASETS_DIR:-${CIPHER_DIR}/data/validation_data/HOST_RANGE}"

NAME="repro_old_v3_in_cipher_LAPTOP_$(date +%Y%m%d_%H%M%S)"

echo "============================================================"
echo "RUN CIPHER REPRO ON LAPTOP — hardware-determinism test"
echo "  Cipher dir:     ${CIPHER_DIR}"
echo "  Train embed:    ${TRAIN_EMB}"
echo "  Val embed:      ${VAL_EMB}"
echo "  Experiment:     ${NAME}"
echo "============================================================"

# Sanity checks
for path in "$ASSOC_MAP" "$GLYCAN_BINDERS" "$POSITIVE_LIST" \
            "$TRAIN_EMB" "$VAL_EMB" "$VAL_FASTA" "$VAL_DATASETS_DIR"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: missing $path"
        exit 1
    fi
done

cd "$CIPHER_DIR"
export PYTHONPATH="${CIPHER_DIR}/src:${PYTHONPATH:-}"

python -m cipher.cli.train_runner \
    --model attention_mlp \
    --positive_list "$POSITIVE_LIST" \
    --min_sources 3 \
    --max_k_types 3 \
    --max_o_types 3 \
    --label_strategy single_label \
    --split-style canonical \
    --lr 1e-05 \
    --batch_size 64 \
    --epochs 200 \
    --patience 30 \
    --embedding_type esm2_650m_mean \
    --embedding_file "$TRAIN_EMB" \
    --association_map "$ASSOC_MAP" \
    --glycan_binders "$GLYCAN_BINDERS" \
    --val_fasta "$VAL_FASTA" \
    --val_datasets_dir "$VAL_DATASETS_DIR" \
    --val_embedding_file "$VAL_EMB" \
    --name "$NAME"

EXP_DIR="${CIPHER_DIR}/experiments/attention_mlp/${NAME}"

echo ""
echo "=== EVAL — default (zscore + competition) ==="
python -m cipher.evaluation.runner "$EXP_DIR" --val-embedding-file "$VAL_EMB"

echo ""
echo "=== EVAL — raw merge, default ties (matches old eval mode) ==="
mkdir -p "$EXP_DIR/results_raw"
python -m cipher.evaluation.runner "$EXP_DIR" \
    --val-embedding-file "$VAL_EMB" \
    --score-norm raw \
    -o "$EXP_DIR/results_raw/evaluation.json"

echo ""
echo "=== EVAL — per-head (K-only / O-only / combined) ==="
python "$CIPHER_DIR/scripts/analysis/eval_per_head.py" "$EXP_DIR"

echo ""
echo "============================================================"
echo "PHL HR@1 across eval modes (compare to old: 0.291 raw / 0.282 O-only)"
echo "============================================================"
python3 - <<PYEOF
import json, os
exp = "$EXP_DIR"
for variant in ['results', 'results_raw']:
    p = os.path.join(exp, variant, 'evaluation.json')
    try:
        d = json.load(open(p))
        rh = d['PhageHostLearn']['rank_hosts']['hr_at_k']['1']
        print(f'  {variant:<25} PHL rh@1 = {rh:.4f}')
    except FileNotFoundError:
        print(f'  {variant:<25} (not present)')
PYEOF

echo ""
echo "If PHL rh@1 raw ≈ 0.29, hardware was the issue (Delta CUDA vs MPS)."
echo "If still ≈ 0.16, real code/methodology bug remains."
