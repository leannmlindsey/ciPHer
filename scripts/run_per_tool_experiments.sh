#!/usr/bin/env bash
# Run one experiment per tool (+ all tools baseline) with matched config.
# Uses multi_label_threshold, min_class_samples=25, same hyperparams.

set -euo pipefail

TOOLS=(
    DePP_85
    PhageRBPdetect
    DepoScope
    DepoRanker
    SpikeHunter
    dbCAN
    IPR
    phold_glycan_tailspike
)

COMMON_ARGS=(
    --model attention_mlp
    --lr 1e-05
    --batch_size 512
    --epochs 1000
    --patience 30
    --label_strategy multi_label_threshold
    --min_class_samples 25
    --max_samples_per_k 1000
    --max_samples_per_o 3000
    --min_sources 1
)

TOTAL=$((${#TOOLS[@]} + 1))  # 8 tools + 1 baseline
CURRENT=0
FAILED=0
SUCCEEDED=0
START_TIME=$SECONDS

echo "============================================"
echo "Per-tool experiment sweep"
echo "  $TOTAL experiments to run"
echo "  Started: $(date)"
echo "============================================"

run_one() {
    local name="$1"
    shift
    local extra_args=("$@")

    CURRENT=$((CURRENT + 1))
    echo ""
    echo "============================================"
    echo "[$CURRENT/$TOTAL] Training: $name"
    echo "  Started: $(date)"
    echo "============================================"

    local exp_start=$SECONDS

    if python -m cipher.cli.train_runner "${COMMON_ARGS[@]}" \
        "${extra_args[@]}" \
        --name "$name" 2>&1; then
        local elapsed=$(( SECONDS - exp_start ))
        local mins=$(( elapsed / 60 ))
        local secs=$(( elapsed % 60 ))
        echo ""
        echo "  DONE: $name (${mins}m ${secs}s)"
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        echo ""
        echo "  FAILED: $name"
        FAILED=$((FAILED + 1))
    fi
}

# All tools baseline (min_sources=2 to match previous best)
run_one "per_tool_ALL" --min_sources 2

# One experiment per tool
for TOOL in "${TOOLS[@]}"; do
    run_one "per_tool_${TOOL}" --tools "$TOOL"
done

echo ""
echo "============================================"
echo "Training complete. $SUCCEEDED succeeded, $FAILED failed."
echo "============================================"

# Evaluate all
echo ""
echo "Evaluating all per-tool experiments..."
EVAL_COUNT=0
for DIR in experiments/attention_mlp/per_tool_*; do
    if [ -d "$DIR" ] && [ ! -f "$DIR/evaluation.json" ]; then
        EVAL_COUNT=$((EVAL_COUNT + 1))
        echo ""
        echo "  [$EVAL_COUNT] Evaluating: $(basename $DIR)"
        python -m cipher.evaluation.runner "$DIR" 2>&1
    fi
done

TOTAL_ELAPSED=$(( SECONDS - START_TIME ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))

echo ""
echo "============================================"
echo "All done!"
echo "  $SUCCEEDED trained, $EVAL_COUNT evaluated, $FAILED failed"
echo "  Total time: ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "  Finished: $(date)"
echo "============================================"
echo ""
echo "Next: python scripts/compare_experiments.py"
