#!/usr/bin/env bash
# Run a planned matrix of cipher-train experiments + evaluate + analyze each.
#
# Designed to be safe to interrupt and re-run:
#   - Skips experiments whose evaluation.json already exists
#   - Logs all output to scripts/_logs/{run_name}.log
#
# Usage:
#   bash scripts/run_experiments.sh                  # do everything
#   bash scripts/run_experiments.sh evaluate_only    # just evaluate trained models
#   bash scripts/run_experiments.sh new_only         # only train new experiments
#
# Run in the background (recommended for long-running):
#   nohup bash scripts/run_experiments.sh > scripts/_run.log 2>&1 &

set -u  # error on undefined vars
# Note: not using set -e — we want to continue past failures

cd "$(dirname "$0")/.."  # cd to repo root
mkdir -p scripts/_logs

MODE="${1:-all}"

# -------- Helper: evaluate + analyze if not already done --------
post_train() {
    local exp_dir="$1"
    local name="$(basename "$exp_dir")"

    if [ ! -f "$exp_dir/results/evaluation.json" ]; then
        echo "  [evaluate] $name"
        cipher-evaluate "$exp_dir" --quiet >> "scripts/_logs/${name}.log" 2>&1 \
            && echo "    OK" \
            || echo "    FAILED (see scripts/_logs/${name}.log)"
    fi

    if [ ! -f "$exp_dir/analysis/per_serotype_test.json" ]; then
        echo "  [analyze] $name"
        cipher-analyze "$exp_dir" --quiet >> "scripts/_logs/${name}.log" 2>&1 \
            && echo "    OK" \
            || echo "    FAILED (see scripts/_logs/${name}.log)"
    fi
}

# -------- Helper: train a new experiment --------
# Args: name, then cipher-train args (use --name $name so dir is predictable)
run_experiment() {
    local name="$1"
    shift
    local exp_dir="experiments/attention_mlp/${name}"

    if [ -f "$exp_dir/results/evaluation.json" ]; then
        echo "[skip] $name (already evaluated)"
        return 0
    fi

    if [ ! -f "$exp_dir/model_k/best_model.pt" ]; then
        echo "[train] $name"
        cipher-train --model attention_mlp --name "$name" "$@" \
            >> "scripts/_logs/${name}.log" 2>&1 \
            || { echo "    train FAILED (see scripts/_logs/${name}.log)"; return 1; }
        echo "    train OK"
    else
        echo "[have] $name (already trained, will evaluate)"
    fi

    post_train "$exp_dir"
}

# ============================================================
# STEP 1: Evaluate + analyze every existing trained experiment
# ============================================================
if [ "$MODE" = "all" ] || [ "$MODE" = "evaluate_only" ]; then
    echo "================================================================"
    echo "STEP 1: Evaluating existing trained experiments"
    echo "================================================================"
    for exp_dir in experiments/attention_mlp/*/; do
        name="$(basename "${exp_dir%/}")"
        # Only those with both heads trained
        if [ -f "$exp_dir/model_k/best_model.pt" ] && [ -f "$exp_dir/model_o/best_model.pt" ]; then
            echo "Processing: $name"
            post_train "${exp_dir%/}"
        else
            echo "[skip] $name (missing trained models)"
        fi
    done
    echo
fi

# ============================================================
# STEP 2: Train a focused matrix of new experiments
# ============================================================
# Common training params (matched to current best known config)
COMMON=(--lr 1e-05 --batch_size 512 --epochs 1000
        --min_sources 2 --max_k_types 3 --max_o_types 3
        --max_samples_per_k 1000 --max_samples_per_o 3000)

# Suffix to make names unique to this batch
TAG="batch1"

if [ "$MODE" = "all" ] || [ "$MODE" = "new_only" ]; then
    echo "================================================================"
    echo "STEP 2: Training new experiments"
    echo "================================================================"

    # ---- Group A: label strategy comparison (TSP only, no class drop) ----
    run_experiment "${TAG}_SpikeHunter_singleLabel" \
        --tools SpikeHunter --label_strategy single_label "${COMMON[@]}"

    run_experiment "${TAG}_SpikeHunter_multiLabel" \
        --tools SpikeHunter --label_strategy multi_label "${COMMON[@]}"

    run_experiment "${TAG}_SpikeHunter_multiLabelThr" \
        --tools SpikeHunter --label_strategy multi_label_threshold "${COMMON[@]}"

    run_experiment "${TAG}_SpikeHunter_weightedMultiLabel" \
        --tools SpikeHunter --label_strategy weighted_multi_label "${COMMON[@]}"

    # ---- Group B: same as A but with min_class_samples=25 ----
    run_experiment "${TAG}_SpikeHunter_singleLabel_mcs25" \
        --tools SpikeHunter --label_strategy single_label \
        --min_class_samples 25 "${COMMON[@]}"

    run_experiment "${TAG}_SpikeHunter_multiLabelThr_mcs25" \
        --tools SpikeHunter --label_strategy multi_label_threshold \
        --min_class_samples 25 "${COMMON[@]}"

    run_experiment "${TAG}_SpikeHunter_weightedMultiLabel_mcs25" \
        --tools SpikeHunter --label_strategy weighted_multi_label \
        --min_class_samples 25 "${COMMON[@]}"

    # ---- Group C: protein set variations (best strategy: multiLabelThr + mcs25) ----
    run_experiment "${TAG}_allTools_multiLabelThr_mcs25" \
        --label_strategy multi_label_threshold --min_class_samples 25 \
        "${COMMON[@]}"

    run_experiment "${TAG}_RBPtools_multiLabelThr_mcs25" \
        --tools PhageRBPdetect,DepoScope,DepoRanker \
        --label_strategy multi_label_threshold --min_class_samples 25 \
        "${COMMON[@]}"

    run_experiment "${TAG}_noSpikeHunter_multiLabelThr_mcs25" \
        --exclude_tools SpikeHunter \
        --label_strategy multi_label_threshold --min_class_samples 25 \
        "${COMMON[@]}"

    echo
fi

# ============================================================
# STEP 3: Generate cross-experiment comparison plots
# ============================================================
if [ "$MODE" = "all" ] || [ "$MODE" = "compare_only" ]; then
    echo "================================================================"
    echo "STEP 3: Generating cross-experiment comparison"
    echo "================================================================"
    mkdir -p experiments/_comparison
    python scripts/compare_experiments.py
    echo
fi

echo "================================================================"
echo "Done. Logs in scripts/_logs/, comparison in experiments/_comparison/"
echo "================================================================"
