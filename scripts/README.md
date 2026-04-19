# Experiment runner scripts

## What's here

- `run_experiments.sh` — runs a planned matrix of training + evaluation + analysis
- `compare_experiments.py` — aggregates results from all evaluated experiments
- `_logs/` — per-experiment training/evaluation logs
- `_run.log` — top-level script output (when run with `nohup`)

## Quick start

```bash
# Run everything (recommended for first run, in background)
nohup bash scripts/run_experiments.sh > scripts/_run.log 2>&1 &

# Check progress
tail -f scripts/_run.log

# Or run interactively (blocks the terminal)
bash scripts/run_experiments.sh
```

## Modes

```bash
bash scripts/run_experiments.sh                    # all 3 steps
bash scripts/run_experiments.sh evaluate_only      # just evaluate trained models
bash scripts/run_experiments.sh new_only           # only train new experiments
bash scripts/run_experiments.sh compare_only       # only generate comparison
```

## Safety

- **Idempotent**: re-running skips experiments whose `evaluation.json` already exists
- **Continues on failure**: one failed run doesn't stop the rest
- **Logs everything**: each experiment's output goes to `scripts/_logs/{name}.log`

## What gets run

### Step 1: Catch-up
Evaluate + analyze every experiment in `experiments/attention_mlp/` that has
trained K and O models but no evaluation results.

### Step 2: New experimental matrix (10 experiments)

All use the current best known training params:
- LR 1e-5, batch_size 512, 1000 epochs
- min_sources=2, max_k_types=3, max_o_types=3
- max_samples_per_k=1000, max_samples_per_o=3000

**Group A — label strategy comparison** (TSP only, no class drop):
- single_label (matches old baseline)
- multi_label
- multi_label_threshold (default thresholds)
- weighted_multi_label

**Group B — same as A but with min_class_samples=25**:
- single_label + mcs25
- multi_label_threshold + mcs25
- weighted_multi_label + mcs25

**Group C — protein set variations** (best strategy: multi_label_threshold + mcs25):
- allTools (no tool filter)
- RBPtools (PhageRBPdetect, DepoScope, DepoRanker — RBP-detection tools)
- noSpikeHunter (exclude SpikeHunter)

### Step 3: Comparison
Runs `compare_experiments.py` to:
- Print sortable table of all evaluated experiments
- Save `experiments/_comparison/experiment_summary.csv`
- Generate HR@k curve plots for top 10 experiments

## After running

Open the comparison CSV in Excel:
```
experiments/_comparison/experiment_summary.csv
```

Or view the comparison plots:
```
experiments/_comparison/hr_curves_rank_hosts_*.png
experiments/_comparison/hr_curves_rank_phages_*.png
```

For per-serotype analysis on a specific experiment:
```bash
cipher-analyze experiments/attention_mlp/{run_name}/
```

## Adjusting the matrix

Edit `run_experiments.sh` — the `run_experiment` calls in Step 2 are
self-explanatory. Each new experiment needs a unique `--name`.

To re-run a single experiment after changing code, delete the experiment
directory and re-run the script (it will retrain).
