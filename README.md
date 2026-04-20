# ciPHer

**Computational Inference of Phage-Host Entry Range**

Predicts phage-host interactions for Klebsiella bacteria by matching phage
receptor binding proteins (RBPs/TSPs) to host capsular serotypes (K-type,
O-type).

## Motivation

Predicting which Klebsiella hosts a phage can infect requires matching
phage receptor binding proteins (RBPs/TSPs) to host capsular serotypes.
This repository provides a systematic framework for training and evaluating
serotype classifiers under standardized conditions. The shared `cipher`
library handles data loading, evaluation, and visualization so that each
model only needs to implement its architecture and a standard prediction
interface.

## Quick Start

```bash
pip install -e ".[test]"

# Train a model (DepoScope + PhageRBPdetect proteins, multi-label)
cipher-train --model attention_mlp \
    --tools DepoScope,PhageRBPdetect \
    --label_strategy multi_label_threshold

# Evaluate against all validation datasets
cipher-evaluate experiments/attention_mlp/{run_name}/

# Train with a different embedding file (e.g., k-mer features)
cipher-train --model attention_mlp \
    --embedding_file /path/to/kmer_features.npz \
    --embedding_type kmer_murphy8_k5

# Evaluate with custom validation embeddings
cipher-evaluate experiments/attention_mlp/{run_name}/ \
    --val-embedding-file /path/to/val_embeddings.npz

# Run tests
pytest
```

## Repository Structure

```
cipher/
├── src/
│   └── cipher/                # Shared library (pip install -e .)
│       ├── data/              # Data loading and preparation
│       ├── evaluation/        # Standardized evaluation + bio-scoring
│       └── visualization/     # Plotting and comparison figures
├── models/                    # One directory per model architecture
│   └── {model_name}/
│       ├── model.py           # Model definition (nn.Module, etc.)
│       ├── train.py           # Training loop
│       ├── predict.py         # Predictor subclass + get_predictor()
│       └── base_config.yaml   # Default hyperparameters
├── experiments/               # One directory per run (config + results)
│   └── {model_name}/
│       └── {run_name}/
│           ├── config.yaml    # Full config (merged base + overrides)
│           ├── experiment.json
│           └── results/
├── data/                      # All data (training + validation, not in git)
│   ├── training_data/
│   └── validation_data/
├── data_exploration/          # Analysis scripts and output
│   ├── scripts/               # Tool overlap, error analysis, etc.
│   └── output/                # CSVs and figures
├── scripts/                   # Batch experiment and sweep scripts
├── tests/                     # Unit tests
│   └── test_data/             # Small fixture files
├── docs/                      # Reference documentation
└── setup.py
```

## Key Design Principles

1. **Shared library, pluggable models**: `src/cipher/` is stable shared code.
   `models/` holds model-specific code. `experiments/` holds configs and results.
2. **Standard interface**: Every model implements `predict.py` with a
   `Predictor` class. Evaluation code never needs to know model internals.
3. **No data duplication**: Experiment configs control data filtering
   (protein set, min_sources, downsampling). Heavy data files live once
   in `data/`.
4. **Reproducible**: Every experiment records its full config, git commit,
   and seed.

## Shared Library (`src/cipher/`)

### `cipher.data` — Data loading and preparation

| Module | Key Functions | Purpose |
|---|---|---|
| `training.py` | `prepare_training_data(config, assoc_path, glycan_path)` | Full filtering/labeling pipeline. Returns `TrainingData` with md5_list, label arrays, class lists |
| `training.py` | `TrainingConfig` | Dataclass with all filtering knobs (protein_set, min_sources, max_k_types, downsampling, etc.) |
| `embeddings.py` | `load_embeddings(path, md5_filter=None)` | Load NPZ embeddings, optionally filtered to a set of MD5s |
| `interactions.py` | `load_interaction_matrix(dataset_dir)` | Load validation interaction matrix as `{phage: {host: label}}` |
| `interactions.py` | `load_phage_protein_mapping(filepath)` | Load phage-to-protein mapping as `{phage: set(protein_ids)}` |
| `proteins.py` | `load_fasta(path)`, `load_fasta_md5(path)` | Parse FASTA, compute MD5 hashes |
| `serotypes.py` | `load_serotypes(path)` | Load serotype annotations |
| `splits.py` | `create_stratified_split(ids, labels)` | Stratified train/val/test split |

### Training data flow

```
glycan_binders_custom.tsv ──┐
(protein index, tool flags) │    TrainingConfig
                            ├──> prepare_training_data() ──> TrainingData
host_phage_protein_map.tsv ─┘    (filter, label, downsample)   (md5_list, labels)
                                                                     │
esm2_650m_md5.npz ─────────────> load_embeddings(md5_filter) ──> embeddings
                                                                     │
                                                              train.py uses both
```

### `cipher.evaluation` — Standardized evaluation

| Module | Key Functions | Purpose |
|---|---|---|
| `predictor.py` | `Predictor` (ABC) | Standard interface: `predict_protein(embedding)` and `score_pair()` |
| `ranking.py` | `evaluate_rankings(predictor, ds_name, ds_dir, emb_dict, pid_md5)` | Run both ranking modes for one validation dataset |
| `ranking.py` | `rank_hosts()`, `rank_phages()` | Score and rank candidates |
| `metrics.py` | `hr_at_k()`, `mrr()`, `hr_curve()` | Hit Rate @ k, Mean Reciprocal Rank |
| `runner.py` | `main()` | `cipher-evaluate` CLI entry point |

### Evaluation flow

```
cipher-evaluate experiments/attention_mlp/v4_downsample/
    │
    ├── find predict.py → import get_predictor(run_dir) → Predictor instance
    │
    ├── load validation embeddings + protein MD5 mapping
    │
    └── for each dataset (CHEN, GORODNICHIV, UCSD, PBIP, PhageHostLearn):
            load interaction_matrix.tsv (has host_K, host_O inline)
            load phage_protein_mapping.csv
            rank_hosts(): given phage, rank candidate hosts
            rank_phages(): given host, rank candidate phages
            compute HR@k and MRR for both directions
```

## Training

```bash
# Recommended baseline: DepoScope + PhageRBPdetect, multi-label with threshold
cipher-train --model attention_mlp \
    --tools DepoScope,PhageRBPdetect \
    --label_strategy multi_label_threshold \
    --min_class_samples 25 \
    --max_samples_per_k 1000 \
    --max_samples_per_o 3000

# Single tool
cipher-train --model attention_mlp --tools SpikeHunter

# Multiple tools (union: flagged by either)
cipher-train --model attention_mlp --tools DepoScope,DepoRanker

# Exclude a tool
cipher-train --model attention_mlp --exclude_tools SpikeHunter

# Custom embedding file (e.g., k-mer features on HPC)
cipher-train --model attention_mlp \
    --embedding_file /path/to/kmer_murphy8_k5.npz \
    --embedding_type kmer_murphy8_k5

# Combined features: pLM + k-mer concatenated per MD5
cipher-train --model attention_mlp \
    --embedding_type esm2_3b_mean \
    --embedding_file /path/to/esm2_3b.npz \
    --embedding_type_2 kmer_aa20_k4 \
    --embedding_file_2 /path/to/kmer_aa20_k4.npz \
    --val_embedding_file /path/to/val_esm2_3b.npz \
    --val_embedding_file_2 /path/to/val_kmer_aa20_k4.npz
```

With `--embedding_file_2` set, features for each MD5 become
`concatenate([pLM_vec, kmer_vec])`. Every MD5 must be present in both
files — a coverage mismatch raises an error. Evaluation via
`cipher-evaluate` automatically picks up the second file from the saved
`config.yaml`, or accept `--val-embedding-file-2` explicitly.

Valid tool names: `DePP_85`, `PhageRBPdetect`, `DepoScope`, `DepoRanker`,
`SpikeHunter`, `dbCAN`, `IPR`, `phold_glycan_tailspike`.

#### Training-set filter: `--tools` vs `--positive_list`

Two mutually exclusive ways to decide which candidate proteins enter training:

| Filter | Flag | Notes |
|---|---|---|
| Tool flags | `--tools DepoScope,PhageRBPdetect` (default) | Keep proteins flagged by **any** listed tool |
| Pipeline-positive list | `--positive_list data/training_data/metadata/pipeline_positive.list` | Intersect candidates with this file only; ignore tool flags |

```bash
# Use the positive list (broader — includes PhageRBPdetect-only adhesins
# that the DepoScope tool filter would exclude):
cipher-train --model attention_mlp \
    --positive_list data/training_data/metadata/pipeline_positive.list \
    --label_strategy multi_label_threshold --min_class_samples 25
```

#### Cluster-stratified downsampling (`--cluster_file`)

`--max_samples_per_k` / `--max_samples_per_o` cap per-class sample counts.
By default the cap is filled by **random sampling**, which tends to retain
near-duplicate sequences in over-represented classes. Passing a cluster
file switches to **round-robin across clusters** — it takes one protein
per cluster until the cap is met, looping back to refill from larger
clusters. This maximises sequence diversity in the training sample.

```bash
# Cluster-stratified sampling at 70% identity:
cipher-train --model attention_mlp \
    --cluster_file data/training_data/metadata/candidates_clusters.tsv \
    --cluster_threshold 70 \
    --max_samples_per_k 1000 --max_samples_per_o 3000
```

`candidates_clusters.tsv` has columns `protein_id, cl30_X, cl40_X, ...,
cl95_X` (no header). It ships with the training-data zip; regenerate
with `scripts/build_candidates_cluster_file.py` if candidates change.

Each run creates an experiment directory containing:
- `config.yaml` — merged config (base defaults + CLI overrides)
- `model_k/` — trained K-type head (`best_model.pt`, `config.json`, `training_history.json`)
- `model_o/` — trained O-type head (same structure)
- `experiment.json` — metadata + provenance (git commit, host, SLURM job id, user, argv)

#### Label strategies

Controls how per-protein observation counts map to training labels. Biology:
some RBPs are **specific** (target one K-type), others are **polyspecific**
(target multiple K-types).

| Strategy | Labels | Loss | Best for |
|---|---|---|---|
| `single_label` | One-hot (majority vote) | CrossEntropyLoss (softmax) | Strictly specific RBPs |
| `multi_label` | Binary per class | BCEWithLogitsLoss | Polyspecific RBPs (treats 1 obs = 100 obs) |
| `multi_label_threshold` | Binary with `count>=N AND fraction>=X` filter | BCEWithLogitsLoss | **Recommended default** — polyspecific with noise filtering |
| `weighted_soft` | Fractional, sums to 1 | KL-divergence (softmax) | Distribution matching (classes compete) |
| `weighted_multi_label` | Fractional per class | BCEWithLogitsLoss | Polyspecific with strength encoding (no competition) |

```bash
# multi_label_threshold: drop singleton associations (noise)
cipher-train --model attention_mlp \
    --label_strategy multi_label_threshold \
    --min_label_count 2 \
    --min_label_fraction 0.1

# weighted_multi_label: soft targets, independent per-class
cipher-train --model attention_mlp --label_strategy weighted_multi_label
```

### Evaluate

```bash
# Standard evaluation
cipher-evaluate experiments/attention_mlp/{run_name}/

# With custom validation embeddings
cipher-evaluate experiments/attention_mlp/{run_name}/ \
    --val-embedding-file /path/to/val_kmer.npz
```

Runs both ranking modes against 5 validation datasets (CHEN, GORODNICHIV,
UCSD, PBIP, PhageHostLearn):
- **Rank hosts given phage**: for each phage, score all candidate hosts
- **Rank phages given host**: for each host, score all candidate phages

Scoring: `pair_score = max over proteins of max(zscore(P(K_host)), zscore(P(O_host)))`

Uses competition tie ranking (hosts with identical serotypes get the same rank).

Results saved to `{experiment_dir}/results/evaluation.json`.

### 3. Visualize

```python
from cipher.visualization import plot_single_model, plot_model_comparison

# Single model: HR@k curves per dataset + average
plot_single_model('experiments/attention_mlp/{run_name}/', mode='rank_hosts')
plot_single_model('experiments/attention_mlp/{run_name}/', mode='rank_phages')

# Compare multiple models
plot_model_comparison(
    ['experiments/attention_mlp/run1/', 'experiments/attention_mlp/run2/'],
    labels=['v4 downsample', 'tsp only'],
    mode='rank_hosts',
    output_path='results/comparison',
)
```

`plot_single_model` produces a 2x3 figure (5 datasets + average).
`plot_model_comparison` produces two figures matching the presentation style:
- Average HR@k across datasets (one line per model)
- Per-dataset HR@k (one subplot per dataset, all models overlaid)

Line styles encode protein set: solid = all, dotted = TSP only, dashed = RBP only.

### Embedding Sweep (HPC)

```bash
# Dry run — print all jobs without submitting
DRY_RUN=1 bash scripts/run_embedding_sweep.sh

# Submit all embedding experiments to SLURM
bash scripts/run_embedding_sweep.sh

# Submit a single embedding
bash scripts/run_embedding_sweep.sh esm2_3b_mean
```

Two env vars toggle the training-set filter and the sampling strategy.
They combine freely; run names reflect the combination so results coexist
in `experiments/attention_mlp/`:

```bash
# Pipeline-positive filter instead of --tools (adds 'posList' to run name)
FILTER_MODE=positive_list bash scripts/run_embedding_sweep.sh

# Cluster-stratified downsampling at 70% (adds 'cl70' suffix)
USE_CLUSTERS=1 bash scripts/run_embedding_sweep.sh

# Both together — 4 variants per embedding possible
FILTER_MODE=positive_list USE_CLUSTERS=1 bash scripts/run_embedding_sweep.sh
```

### Concat Sweep (pLM + k-mer)

`scripts/run_concat_sweep.sh` iterates over `(pLM, k-mer)` embedding
pairs, training on the per-MD5 concatenation. Pairs are configured in
the `EMBEDDING_PAIRS` array at the top of the script — edit them to
match the winners of the single-embedding sweeps. The same
`FILTER_MODE` and `USE_CLUSTERS` env vars apply. Run names use the
`concat_<plm>+<kmer>` (+`posList_`, +`_cl70`) convention.

```bash
DRY_RUN=1 bash scripts/run_concat_sweep.sh
bash scripts/run_concat_sweep.sh                               # all pairs
bash scripts/run_concat_sweep.sh esm2_3b_mean+kmer_aa20_k4     # single pair
FILTER_MODE=positive_list USE_CLUSTERS=1 bash scripts/run_concat_sweep.sh
```

### Persistent results log + figures

```bash
# Harvest every experiment's metrics + provenance into one wide CSV
python scripts/harvest_results.py
# -> results/experiment_log.csv (one row per run, sorted by PHL+PBIP HR@1)

# Generate SVG figures for slide decks (PHL+PBIP + all 5 datasets)
python scripts/plot_sweep_results.py
# -> results/figures/sweep_phl_pbip_hrk.svg
# -> results/figures/sweep_all_datasets_hrk.svg
```

The CSV is rebuilt idempotently from `experiment.json` +
`results/evaluation.json` in every run dir, so it's safe to re-run after
new experiments finish. Provenance (git commit, host, SLURM job id,
user, argv) is captured automatically at training time via
`cipher.provenance.capture_provenance()`.

## Testing

```bash
pytest           # run all tests
pytest -v        # verbose output
pytest -x        # stop on first failure
```

## References

See `ONBOARDING.md` for biology background, data documentation, and known
issues from prior work.
