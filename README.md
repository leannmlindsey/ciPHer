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

# Train the baseline classifier
cipher-train --model attention_mlp \
    --tools DepoScope,PhageRBPdetect \
    --label_strategy multi_label_threshold

# Evaluate against all validation datasets
cipher-evaluate experiments/attention_mlp/{run_name}/

# Run tests
pytest
```

## Repository Structure

```
cipher/
├── src/
│   └── cipher/                # Shared library (pip install -e .)
│       ├── data/              # Data loading, filtering, clustering, sampling
│       ├── evaluation/        # Ranking, HR@k / MRR, cipher-evaluate CLI
│       ├── visualization/     # HR@k plotting primitives
│       └── provenance.py      # git commit / host / SLURM metadata capture
├── models/                    # One directory per model architecture
│   ├── attention_mlp/         # MLP + SE-attention classifier (baseline)
│   └── contrastive_encoder/   # ArcFace feature learner; produces drop-in NPZ
├── experiments/               # One dir per training run (not in git)
│   └── {model_name}/{run_name}/
│       ├── config.yaml                  # merged base_config + CLI overrides
│       ├── experiment.json              # metadata + provenance
│       ├── model_k/ model_o/            # trained heads (attention_mlp)
│       ├── encoder.pt                   # encoder weights (contrastive_encoder)
│       ├── contrastive_{train,val}_md5.npz  # drop-in embeddings (contrastive_encoder)
│       └── results/evaluation.json      # cipher-evaluate output
├── data/                      # Training + validation inputs (not in git; see dist/ zips)
├── dist/                      # Generated data zips (cipher_training_data.zip, cipher_validation_data.zip)
├── scripts/
│   ├── run_embedding_sweep.sh       # single-embedding sweep (attention_mlp)
│   ├── run_concat_sweep.sh          # (pLM, k-mer) concat sweep
│   ├── run_experiments.sh           # legacy batch runner
│   ├── run_per_tool_experiments.sh  # tool-by-tool sweep
│   ├── analysis/                    # Result harvesting, plotting, diagnostics
│   ├── utils/                       # Data packaging, cluster-file build
│   └── extract_embeddings/          # ESM-2 / ProtT5 extraction + SLURM runner
├── results/                   # Generated — experiment_log.csv, figures, analyses
├── tests/                     # pytest suite
└── setup.py
```

## Key Design Principles

1. **Shared library, pluggable models.** `src/cipher/` is stable shared
   code. `models/` holds model-specific code. `experiments/` holds configs
   and results.
2. **Standard interface.** Every classifier implements a `Predictor` class
   with `predict_protein(embedding)` returning `{'k_probs': {...}, 'o_probs': {...}}`.
   Evaluation code never needs to know model internals.
3. **No data duplication.** Experiment configs control data filtering
   (protein set, min_sources, downsampling, cluster stratification). Heavy
   data files live once in `data/` and ship together in `dist/*.zip`.
4. **Reproducible.** Every run records git commit, host, SLURM job id, user,
   and full command line via `cipher.provenance.capture_provenance()`.

## Shared Library (`src/cipher/`)

### `cipher.data` — Data loading and preparation

| Module | Key Functions | Purpose |
|---|---|---|
| `training.py` | `prepare_training_data(config, assoc_path, glycan_path)` | Full filtering + labeling pipeline. Returns `TrainingData`. |
| `training.py` | `TrainingConfig` | Dataclass for all filtering knobs (tools, positive_list, min_sources, cluster_file, downsampling). |
| `embeddings.py` | `load_embeddings(path, md5_filter=None)` | Load NPZ embeddings, optionally filtered. |
| `embeddings.py` | `load_embeddings_concat(p1, p2, md5_filter)` | Load two NPZs and concatenate per MD5 (requires full coverage). |
| `interactions.py` | `load_interaction_matrix(dataset_dir)` | Validation interactions as `{phage: {host: label}}`. |
| `interactions.py` | `load_phage_protein_mapping(filepath)` | Phage → set of protein IDs. |
| `proteins.py` | `load_fasta`, `load_fasta_md5`, `load_positive_list` | FASTA parsing + positive-list loading. |
| `serotypes.py` | `load_serotypes(path)` | Serotype annotations. |
| `splits.py` | `create_stratified_split(ids, labels)` | Stratified train/val/test split. |

### Training data flow

```
glycan_binders_custom.tsv ──┐
(protein index, tool flags) │    TrainingConfig
                            ├──> prepare_training_data() ──> TrainingData
host_phage_protein_map.tsv ─┘    (filter, label, [cluster-strat downsample])
candidates_clusters.tsv ────>    (optional — cluster-stratified sampling)
                                                                     │
esm2_650m_md5.npz ─────────────> load_embeddings(md5_filter) ──> embeddings
                                                                     │
                                                              train.py consumes both
```

### `cipher.evaluation` — Standardized evaluation

| Module | Key Functions | Purpose |
|---|---|---|
| `predictor.py` | `Predictor` (ABC) | Standard interface: `predict_protein(embedding)` + `score_pair()`. |
| `ranking.py` | `evaluate_rankings(predictor, ds_name, ...)` | Both ranking modes for one validation dataset. |
| `ranking.py` | `rank_hosts()`, `rank_phages()` | Score and rank candidates. |
| `metrics.py` | `hr_at_k()`, `mrr()`, `hr_curve()` | Hit Rate @ k, Mean Reciprocal Rank. |
| `runner.py` | `main()` | `cipher-evaluate` CLI entry point. |

### Evaluation flow

```
cipher-evaluate experiments/attention_mlp/{run_name}/
    │
    ├── find predict.py → get_predictor(run_dir) → Predictor instance
    ├── load validation embeddings + protein MD5 mapping
    └── for each dataset (CHEN, GORODNICHIV, UCSD, PBIP, PhageHostLearn):
            load interaction_matrix.tsv, phage_protein_mapping.csv
            rank_hosts(): given phage, rank candidate hosts
            rank_phages(): given host, rank candidate phages
            compute HR@k and MRR for both directions
```

---

## Available models

Each model sits under `models/{name}/` with the same four files: `base_config.yaml`,
`model.py`, `train.py`, `predict.py`. `cipher-train --model {name}` dynamically
loads `models/{name}/train.py::train(experiment_dir, config)`.

### `attention_mlp` — Baseline classifier

The default model. Two independent classifiers (K-type, O-type) trained in
sequence on top of pre-extracted protein embeddings.

#### Architecture

```
Input embedding (e.g. 1280-d ESM-2 650M)
    │
    ├─▶ SE attention block (bottleneck → sigmoid gate, learnable mix weight)
    │       input → Linear(D, 640) → LN → ReLU → Dropout → Linear(640, D) → σ
    │       gated output = (1-α)·x + α·x·attn     (α learnable, init 0.5)
    │
    ├─▶ MLP: Linear → BN → ReLU → Dropout   (dims 1280 → 640 → 320 → 160)
    │
    └─▶ Output head: Linear(160, n_classes)    # n_classes = number of K- or O-types
```

Two instances of this network are trained independently — one for K-type
classification, one for O-type. At evaluation time their probabilities are
z-scored per protein and max-pooled across a phage's RBPs for the final
score.

#### Key hyperparameters (`models/attention_mlp/base_config.yaml`)

| Knob | Default | Meaning |
|---|---|---|
| `model.hidden_dims` | `[1280, 640, 320, 160]` | MLP layer widths (post-attention) |
| `model.attention_dim` | `640` | SE bottleneck dim. Set 0 to disable attention. |
| `model.dropout` | `0.1` | Dropout on each hidden layer |
| `training.batch_size` | `64` | — |
| `training.learning_rate` | `1e-5` | AdamW |
| `training.epochs` | `200` | Max epochs |
| `training.patience` | `30` | Early-stop patience (on val micro-F1) |

Loss depends on `label_strategy` (see [Label strategies](#label-strategies)):
`single_label` uses softmax + cross-entropy; all `multi_label*` strategies
use BCE with per-class positive-weight scaling.

#### How to run

```bash
# Baseline: DepoScope + PhageRBPdetect proteins, multi-label with threshold
cipher-train --model attention_mlp \
    --tools DepoScope,PhageRBPdetect \
    --label_strategy multi_label_threshold \
    --min_class_samples 25 \
    --max_samples_per_k 1000 --max_samples_per_o 3000

# Single tool
cipher-train --model attention_mlp --tools SpikeHunter

# Multiple tools (union — flagged by any)
cipher-train --model attention_mlp --tools DepoScope,DepoRanker

# Exclude a tool
cipher-train --model attention_mlp --exclude_tools SpikeHunter

# Custom embedding file (e.g., k-mer features)
cipher-train --model attention_mlp \
    --embedding_type kmer_murphy8_k5 \
    --embedding_file /path/to/kmer_murphy8_k5.npz

# Concatenate two embeddings per MD5 (e.g., pLM + k-mer)
cipher-train --model attention_mlp \
    --embedding_type esm2_3b_mean      --embedding_file   /path/to/esm2_3b.npz \
    --embedding_type_2 kmer_aa20_k4    --embedding_file_2 /path/to/kmer_aa20_k4.npz \
    --val_embedding_file   /path/to/val_esm2_3b.npz \
    --val_embedding_file_2 /path/to/val_kmer_aa20_k4.npz
```

With `--embedding_file_2` set, features become `concatenate([vec_1, vec_2])`
per MD5. Every MD5 must be present in both files — a coverage mismatch
raises an error. Evaluation picks up the second file from saved `config.yaml`
automatically, or pass `--val-embedding-file-2` to override.

#### Output artefacts

```
experiments/attention_mlp/{run_name}/
├── config.yaml
├── experiment.json      # timestamp + provenance + data summary
├── model_k/             # K-type head
│   ├── best_model.pt
│   ├── config.json
│   └── training_history.json
├── model_o/             # O-type head (same structure)
├── splits_k.json        # MD5 split used for K head
├── splits_o.json        # MD5 split used for O head
├── label_encoders.json
├── training_data.npz
└── results/evaluation.json     # populated by cipher-evaluate
```

### `contrastive_encoder` — ArcFace feature learner

Produces a drop-in replacement NPZ with learned embeddings that cluster by
serotype, then downstream training uses `attention_mlp` on top.

Motivated by an analysis finding: raw ESM-2 mean embeddings don't separate
K-types (within-class / between-class cosine gap is ~0.004, top-1 nearest-neighbour
K-match is 11.3%). The encoder learns to pull same-K-type proteins together
and push different-K-type apart.

#### Architecture

```
Input embedding (D_in, e.g. 1280 for ESM-2 650M)
    │
    ├─▶ MLP backbone: Linear → BN → ReLU → Dropout × 3
    │   (dims 1280 → 1024 → 1024)
    │
    └─▶ Linear(1024, output_dim) → L2-normalize
        representation z ∈ S^(output_dim-1)      # unit hypersphere

Two ArcFace heads consume z during training (discarded at inference):
    ArcFaceHead(z, K-centroids, margin=0.5, scale=30) → CE loss on K
    ArcFaceHead(z, O-centroids, margin=0.5, scale=30) → CE loss on O
    total_loss = λ_k · L_K + λ_o · L_O
```

ArcFace (Deng et al. 2019, face recognition) adds an additive angular margin
to the ground-truth class's cosine logit, forcing the model to separate
classes by at least that margin. This is more stable than SupCon at the
tiny cosine deltas we have in ESM-2 space.

#### PK + cluster-stratified batching

Each mini-batch is **P K-types × K samples per K-type** (default 32 × 8 = 256).
Within each K-type's K samples, the sampler round-robins across 70%-identity
clusters so the within-class pairs are sequence-diverse. The sampler lives at
`models/contrastive_encoder/sampler.py` and has its own unit tests
(`tests/test_contrastive_sampler.py`).

Why this matters: with random sampling, K1 (1000 proteins) dominates batches
over KL151 (450). PK sampling gives every K-type equal per-batch weight
regardless of its training frequency.

#### Key hyperparameters (`models/contrastive_encoder/base_config.yaml`)

| Knob | Default | Meaning |
|---|---|---|
| `model.hidden_dims` | `[1280, 1024, 1024]` | Backbone widths |
| `model.output_dim` | `1280` | Encoder output dim (drop-in replacement size) |
| `arcface.margin` | `0.5` | Additive angular margin (radians) |
| `arcface.scale` | `30.0` | Logit scale |
| `training.lambda_k` | `1.0` | Weight on K-ArcFace loss |
| `training.lambda_o` | `1.0` | Weight on O-ArcFace loss |
| `training.learning_rate` | `1e-4` | AdamW |
| `training.weight_decay` | `1e-4` | — |
| `training.epochs` | `100` | — |
| `training.patience` | `20` | Early-stop on within/between cosine-gap stall |
| `sampler.P` | `32` | K-types per batch |
| `sampler.K` | `8` | Samples per K-type per batch |

#### How to run

```bash
# 1) Train the encoder. Reuses all of cipher.data's filter + cluster options.
cipher-train --model contrastive_encoder \
    --embedding_file data/training_data/embeddings/esm2_650m_md5.npz \
    --val_embedding_file data/validation_data/embeddings/esm2_650m_md5.npz \
    --positive_list data/training_data/metadata/pipeline_positive.list \
    --cluster_file  data/training_data/metadata/candidates_clusters.tsv \
    --cluster_threshold 70 \
    --name posList_cl70_arcface

# 2) Train a downstream classifier on the learned embeddings (drop-in NPZs).
cipher-train --model attention_mlp \
    --embedding_file experiments/contrastive_encoder/posList_cl70_arcface/contrastive_train_md5.npz \
    --val_embedding_file experiments/contrastive_encoder/posList_cl70_arcface/contrastive_val_md5.npz \
    --positive_list data/training_data/metadata/pipeline_positive.list \
    --cluster_file  data/training_data/metadata/candidates_clusters.tsv --cluster_threshold 70 \
    --name downstream_on_contrastive

# 3) Evaluate the downstream attention_mlp run as usual.
cipher-evaluate experiments/attention_mlp/downstream_on_contrastive/
```

#### Output artefacts

```
experiments/contrastive_encoder/{run_name}/
├── config.yaml
├── experiment.json
├── encoder.pt                         # encoder state_dict (no ArcFace heads)
├── contrastive_train_md5.npz          # learned training embeddings (same dim, keyed by MD5)
├── contrastive_val_md5.npz            # learned validation embeddings
├── splits_k.json splits_o.json
└── training_data.npz
```

Note: `cipher-evaluate` on a `contrastive_encoder` run **intentionally**
raises `NotImplementedError` with instructions — the encoder is a feature
transformer, not a classifier. Always pair it with a downstream
`attention_mlp` run (step 2 above) for evaluation.

#### Quality gate (recommended before downstream training)

After step 1, check that the encoder actually improved separation:

```bash
python scripts/analysis/phl_neighbor_labels.py \
    --train-emb experiments/contrastive_encoder/{run}/contrastive_train_md5.npz \
    --val-emb   experiments/contrastive_encoder/{run}/contrastive_val_md5.npz \
    --out-dir   experiments/contrastive_encoder/{run}/analysis \
    --restrict-to-labeled

python scripts/analysis/within_between_class_cosine.py \
    --train-emb experiments/contrastive_encoder/{run}/contrastive_train_md5.npz \
    --label contrastive_{run}
```

Targets: top-1 K-match rate > 20% (raw ESM-2 baseline is 11.3%); within/between
cosine gap > 0.05 (raw ESM-2 baseline is +0.004). If both move, downstream
training is worth the GPU time.

---

## Training conventions (shared across models)

### Training-set filter: `--tools` vs `--positive_list`

Two mutually exclusive ways to decide which candidate proteins enter training:

| Filter | Flag | Notes |
|---|---|---|
| Tool flags | `--tools DepoScope,PhageRBPdetect` | Keep proteins flagged by **any** listed tool |
| Pipeline-positive list | `--positive_list data/training_data/metadata/pipeline_positive.list` | Intersect candidates with this file only; ignore tool flags |

```bash
# Use the positive list (broader — includes PhageRBPdetect-only adhesins
# that the DepoScope tool filter would exclude):
cipher-train --model attention_mlp \
    --positive_list data/training_data/metadata/pipeline_positive.list \
    --label_strategy multi_label_threshold --min_class_samples 25
```

Valid tool names: `DePP_85`, `PhageRBPdetect`, `DepoScope`, `DepoRanker`,
`SpikeHunter`, `dbCAN`, `IPR`, `phold_glycan_tailspike`.

### Cluster-stratified downsampling (`--cluster_file`)

`--max_samples_per_k` / `--max_samples_per_o` cap per-class sample counts.
By default the cap is filled by **random sampling**, which tends to retain
near-duplicate sequences in over-represented classes. Passing a cluster
file switches to **round-robin across clusters** — one protein per cluster
until the cap is met. Maximises sequence diversity in the training sample.

```bash
cipher-train --model attention_mlp \
    --cluster_file data/training_data/metadata/candidates_clusters.tsv \
    --cluster_threshold 70 \
    --max_samples_per_k 1000 --max_samples_per_o 3000
```

`candidates_clusters.tsv` has columns `protein_id  cl30_X  cl40_X  ...  cl95_X`
(no header). It ships with `cipher_training_data.zip`; regenerate with
`scripts/utils/build_candidates_cluster_file.py` if candidates change.

### Label strategies

Controls how per-protein observation counts map to training labels. Some
RBPs are **specific** (target one K-type), others are **polyspecific**
(target multiple K-types).

| Strategy | Labels | Loss | Best for |
|---|---|---|---|
| `single_label` | One-hot (majority vote) | CrossEntropy (softmax) | Strictly specific RBPs |
| `multi_label` | Binary per class | BCE | Polyspecific (treats 1 obs = 100 obs) |
| `multi_label_threshold` | Binary with `count≥N AND fraction≥X` filter | BCE | **Recommended default** — polyspecific with noise filtering |
| `weighted_soft` | Fractional, sums to 1 | KL-divergence (softmax) | Distribution matching (classes compete) |
| `weighted_multi_label` | Fractional per class | BCE | Polyspecific with strength encoding |

```bash
cipher-train --model attention_mlp \
    --label_strategy multi_label_threshold \
    --min_label_count 2 --min_label_fraction 0.1
```

### Provenance

Every trained run captures at training time:

- `git_commit`, `git_dirty`
- `host`, `slurm_job_id`, `user`
- `cli_argv` (full command)
- `timestamp`

Stored under `experiment.json["provenance"]`. To reproduce any run: check
out `git_commit`, apply the saved `config.yaml`, rerun `cipher-train` with
the same `--name`.

---

## Evaluation

```bash
# Standard
cipher-evaluate experiments/{model}/{run_name}/

# With custom validation embeddings (override saved config)
cipher-evaluate experiments/{model}/{run_name}/ \
    --val-embedding-file /path/to/val_kmer.npz

# Concat / dual-NPZ setups
cipher-evaluate experiments/{model}/{run_name}/ \
    --val-embedding-file /path/to/val_pLM.npz \
    --val-embedding-file-2 /path/to/val_kmer.npz
```

Runs both ranking modes against 5 validation datasets (CHEN, GORODNICHIV,
UCSD, PBIP, PhageHostLearn; KlebPhaCol excluded — proteins aren't capsular):

- **Rank hosts given phage** — for each phage, score all candidate hosts
- **Rank phages given host** — for each host, score all candidate phages

Scoring: `pair_score = max over phage's proteins of max(zscore(P(K_host)), zscore(P(O_host)))`.
Uses competition tie ranking (hosts with identical serotypes get the same rank).

Results saved to `{experiment_dir}/results/evaluation.json`.

### Visualize a single run

```python
from cipher.visualization import plot_single_model, plot_model_comparison

# Per-dataset HR@k + average
plot_single_model('experiments/attention_mlp/{run_name}/', mode='rank_hosts')

# Compare multiple runs
plot_model_comparison(
    ['experiments/attention_mlp/run1/', 'experiments/attention_mlp/run2/'],
    labels=['v4', 'tsp_only'],
    mode='rank_hosts',
    output_path='results/comparison',
)
```

---

## HPC sweeps

All sweep scripts submit SLURM jobs on Delta-AI with sensible per-row
memory / time settings.

### Embedding sweep (single-embedding attention_mlp variants)

```bash
DRY_RUN=1 bash scripts/run_embedding_sweep.sh         # preview
bash scripts/run_embedding_sweep.sh                    # submit all
bash scripts/run_embedding_sweep.sh esm2_3b_mean       # single embedding
```

Two env-var toggles compose freely. Run names encode the combination so
variants coexist in `experiments/attention_mlp/`:

```bash
FILTER_MODE=positive_list bash scripts/run_embedding_sweep.sh     # posList_ prefix
USE_CLUSTERS=1 bash scripts/run_embedding_sweep.sh                 # _cl70 suffix
FILTER_MODE=positive_list USE_CLUSTERS=1 bash scripts/run_embedding_sweep.sh
```

### Concat sweep (pLM + k-mer)

`scripts/run_concat_sweep.sh` iterates over `(pLM, k-mer)` pairs and trains
on the per-MD5 concatenation. Pairs live in the `EMBEDDING_PAIRS` array at
the top of the script — edit to match the winners of the single-embedding
sweep. Same env-var toggles as above; run names use
`concat_<plm>+<kmer>[_cl70]` (+`posList_` if posList filter).

```bash
DRY_RUN=1 bash scripts/run_concat_sweep.sh
bash scripts/run_concat_sweep.sh                               # all pairs
bash scripts/run_concat_sweep.sh esm2_3b_mean+kmer_aa20_k4     # single pair
```

---

## Analysis & diagnostics (`scripts/analysis/`)

### Result harvest + figures

```bash
# Harvest every experiment's metrics + provenance into one wide CSV
python scripts/analysis/harvest_results.py
# -> results/experiment_log.csv (one row per run, sorted by PHL+PBIP HR@1)

# Plot HR@k curves across experiments (SVG — slide / paper appendix ready)
python scripts/analysis/plot_sweep_results.py
# -> results/figures/sweep_phl_pbip_hrk.svg
# -> results/figures/sweep_all_datasets_hrk.svg

# Focused PHL+PBIP table sorted by combined HR@1
python scripts/analysis/compare_primary_datasets.py --filter sweep_
```

The CSV is rebuilt idempotently from `experiment.json` +
`results/evaluation.json` in every run dir, so it's safe to re-run after
new experiments finish.

### Representation diagnostics (PHL ceiling investigation)

| Script | Measures | Output |
|---|---|---|
| `phl_training_distance.py` | For each PHL RBP, cosine similarity to nearest training protein. Reports distribution + within-training baseline. | `phl_training_distance.{csv,svg}` |
| `phl_neighbor_labels.py` | Top-K nearest-training-neighbour K-type match rate (top-1/5/10/50 vs random baseline). | `phl_neighbor_labels.{csv,svg}` |
| `phl_label_audit.py` | Buckets failing PHL proteins into legitimate-novelty vs representation-failure vs label-noise. | `phl_label_audit.txt`, `phl_label_audit_failed.csv` |
| `within_between_class_cosine.py` | Within-class vs between-class cosine pairs for K-type AND O-type. Reports separation gap + Cohen's d. | Appended to `within_between_summary.tsv` |
| `plot_within_between_summary.py` | Renders per-embedding table (Markdown) + bar charts for the paper appendix. | `within_between_summary.md`, `.svg` figures |

Delta-AI SLURM wrappers for the two analyses that need non-local NPZs:

```bash
bash scripts/analysis/submit_neighbor_analyses.sh          # phl_neighbor_labels for ProtT5 / seg4 / kmer_li10_k5
bash scripts/analysis/submit_within_between_analyses.sh    # within/between cosine for all 16 embeddings
```

---

## Embedding extraction (`scripts/extract_embeddings/`)

Scripts to regenerate ESM-2 / ProtT5 embeddings from raw FASTAs on Delta.

| File | Purpose |
|---|---|
| `esm2_extract.py` | ESM-2 (any size) extraction. `--pooling mean|segmentsN|full`. |
| `prott5_extract.py` | ProtT5 (XL / XL-BFD / XXL) extraction. `--pooling mean|segmentsN`. `--half_precision` for XXL. Supports checkpoint resume. |
| `submit_extractions.sh` | SLURM fan-out: one job per (model, pooling) train+val pair. Per-family conda envs (esmfold2 for ESM-2, prott5 for ProtT5). |

```bash
DRY_RUN=1 bash scripts/extract_embeddings/submit_extractions.sh   # preview
bash scripts/extract_embeddings/submit_extractions.sh              # submit all
bash scripts/extract_embeddings/submit_extractions.sh prott5_xl_segments4   # single combo
```

`prott5_extract.py` checkpoints every 2000 proteins to a `.partial.npz`
sibling file and resumes automatically on restart, so a late crash loses
at most 2000 proteins of work. OOM on a single long sequence is caught
and the sequence is skipped (reported at end via `oom_skipped`).

---

## Data packaging

Raw inputs ship as two zips with sha256 manifests:

```bash
bash scripts/utils/package_data.sh
# -> dist/cipher_training_data.zip   (raw inputs: FASTA, TSVs, cluster file)
# -> dist/cipher_validation_data.zip (5 validation datasets; KlebPhaCol excluded)
```

Each zip includes `MANIFEST.txt` for byte-level verification
(`shasum -a 256 -c MANIFEST.txt`). Both zips also get uploaded to Zenodo
(embeddings provided separately — regenerate via the extraction scripts).

---

## Testing

```bash
pytest           # full suite
pytest -v        # verbose
pytest -x        # stop on first failure

# Run one target file
pytest tests/test_contrastive_sampler.py -v
```

## References

See `ONBOARDING.md` for biology background, data documentation, and known
issues from prior work.
