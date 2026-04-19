# ciPHer

**Capsule Interaction Prediction of Phage-Host Relationships**

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
```

Valid tool names: `DePP_85`, `PhageRBPdetect`, `DepoScope`, `DepoRanker`,
`SpikeHunter`, `dbCAN`, `IPR`, `phold_glycan_tailspike`.

Each run creates an experiment directory containing:
- `config.yaml` — merged config (base defaults + CLI overrides)
- `model_k/` — trained K-type head (`best_model.pt`, `config.json`, `training_history.json`)
- `model_o/` — trained O-type head (same structure)
- `experiment.json` — metadata (timestamp, data summary)

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

## Testing

```bash
pytest           # run all tests
pytest -v        # verbose output
pytest -x        # stop on first failure
```

## References

See `ONBOARDING.md` for biology background, data documentation, and known
issues from prior work.
