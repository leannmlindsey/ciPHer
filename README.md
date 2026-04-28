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
   (protein set, min_sources, downsampling, cluster stratification). Large
   input files are stored in a single canonical location under `data/` and
   are distributed together in `dist/*.zip`.
4. **Reproducible.** Every run records git commit, host, SLURM job id, user,
   and full command line via `cipher.provenance.capture_provenance()`.

## Shared Library (`src/cipher/`)

### `cipher.data` — Data loading and preparation

| Module | Key Functions | Purpose |
|---|---|---|
| `training.py` | `prepare_training_data(config, assoc_path, glycan_path)` | Full filtering + labeling pipeline. Returns `TrainingData`. |
| `training.py` | `TrainingConfig` | Dataclass containing all filtering parameters (tools, positive_list, min_sources, cluster_file, downsampling). |
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

Each model is implemented in `models/{name}/` and contains four files:
`base_config.yaml`, `model.py`, `train.py`, and `predict.py`.
`cipher-train --model {name}` dynamically loads
`models/{name}/train.py::train(experiment_dir, config)`.

### `attention_mlp` — Baseline classifier

The default classifier, provided as a baseline. Two independent classifiers
(one for K-type and one for O-type) are trained sequentially on pre-extracted
protein embeddings.

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

Two instances of this network are trained independently: one for K-type
classification and one for O-type. At evaluation time their class
probabilities are z-scored per protein and max-pooled across the receptor
binding proteins of each phage to produce the final score.

#### Key hyperparameters (`models/attention_mlp/base_config.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `model.hidden_dims` | `[1280, 640, 320, 160]` | MLP layer widths (post-attention) |
| `model.attention_dim` | `640` | SE bottleneck dimension; set to 0 to disable attention |
| `model.dropout` | `0.1` | Dropout applied after each hidden layer |
| `training.batch_size` | `64` | — |
| `training.learning_rate` | `1e-5` | AdamW optimizer |
| `training.epochs` | `200` | Maximum training epochs |
| `training.patience` | `30` | Early-stopping patience (monitored on validation micro-F1) |

The loss function is determined by `label_strategy` (see [Label strategies](#label-strategies)):
`single_label` uses softmax with cross-entropy; all `multi_label*` strategies
use BCE with per-class positive-weight scaling.

#### How to run

```bash
# Recommended baseline: DepoScope + PhageRBPdetect + SpikeHunter,
# multi-label threshold strategy.
cipher-train --model attention_mlp \
    --tools DepoScope,PhageRBPdetect,SpikeHunter \
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

When `--embedding_file_2` is set, the per-MD5 feature vector becomes
`concatenate([vec_1, vec_2])`. Every MD5 must be present in both files;
a coverage mismatch raises an error. Evaluation automatically resolves
the second file from the saved `config.yaml`, or it can be specified
explicitly via `--val-embedding-file-2`.

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

Produces a drop-in replacement NPZ containing learned embeddings that
cluster by serotype. Downstream classification is performed by a subsequent
`attention_mlp` run on these learned embeddings.

This model is motivated by a representation diagnostic: raw ESM-2 mean
embeddings do not separate K-types (the within-class vs between-class
cosine-similarity gap is approximately 0.004, and the top-1
nearest-neighbour K-type match rate is 11.3%). The encoder is trained to
bring same-K-type proteins together in the learned feature space and to
separate proteins of different K-types.

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

ArcFace (Deng et al. 2019, originally developed for face recognition) adds
an additive angular margin to the cosine logit of the ground-truth class,
requiring the model to separate classes by at least that margin in angular
space. This formulation provides more stable gradients than SupCon when the
baseline cosine similarities between classes are very small, as is the case
for raw ESM-2 representations in this setting.

#### PK + cluster-stratified batching

Each mini-batch contains **P K-types × K samples per K-type** (default
32 × 8 = 256). Within the K samples drawn from each K-type, the sampler
round-robins across 70%-identity clusters, ensuring sequence diversity
among the within-class pairs. The sampler is implemented in
`models/contrastive_encoder/sampler.py` and has dedicated unit tests in
`tests/test_contrastive_sampler.py`.

Rationale: under uniform random sampling, common classes (e.g., K1 with
1000 proteins) dominate batches relative to rare classes (e.g., KL151
with 450 proteins). PK sampling weights each K-type equally per batch,
irrespective of its prevalence in the training set.

#### Key hyperparameters (`models/contrastive_encoder/base_config.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `model.hidden_dims` | `[1280, 1024, 1024]` | Backbone layer widths |
| `model.output_dim` | `1280` | Encoder output dimension (matches input for drop-in replacement) |
| `arcface.margin` | `0.5` | Additive angular margin (radians) |
| `arcface.scale` | `30.0` | Logit scale |
| `training.lambda_k` | `1.0` | Loss weight on the K-ArcFace term |
| `training.lambda_o` | `1.0` | Loss weight on the O-ArcFace term |
| `training.learning_rate` | `1e-4` | AdamW optimizer |
| `training.weight_decay` | `1e-4` | — |
| `training.epochs` | `100` | Maximum training epochs |
| `training.patience` | `20` | Early-stopping patience; monitored on the within/between cosine gap |
| `sampler.P` | `32` | Number of K-types per batch |
| `sampler.K` | `8` | Number of samples per K-type per batch |

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
raises `NotImplementedError` with usage instructions, because the encoder
is a feature transformer rather than a classifier. Evaluation must be
performed on a downstream `attention_mlp` run trained against the learned
embeddings (step 2 above).

#### Quality gate (recommended before downstream training)

After step 1 completes, verify that the encoder has improved class
separation before committing compute resources to downstream training:

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

Target thresholds: top-1 K-match rate above 20% (the raw ESM-2 baseline is
11.3%) and within/between cosine gap above 0.05 (the raw ESM-2 baseline is
+0.004). If both metrics improve past these thresholds, the downstream
`attention_mlp` training is warranted.

---

## Training conventions (shared across models)

### Training-set filter: `--tools` vs `--positive_list` vs per-head lists

Three mutually exclusive ways to decide which candidate proteins enter training:

| Filter | Flag | Notes |
|---|---|---|
| Tool flags | `--tools DepoScope,PhageRBPdetect` | Keep proteins flagged by **any** listed tool |
| Pipeline-positive list (single, legacy) | `--positive_list data/training_data/metadata/pipeline_positive.list` | Intersect candidates with this file only; ignore tool flags. Both heads use the same list. |
| Per-head lists (v2) | `--positive_list_k <K_list>` + `--positive_list_o <O_list>` | Independent K and O lists — training pool is the UNION, each sample contributes only to the head-loss for the list it appears in (label-level masking). |

```bash
# Use the positive list (broader — includes PhageRBPdetect-only adhesins
# that the DepoScope tool filter would exclude):
cipher-train --model attention_mlp \
    --positive_list data/training_data/metadata/pipeline_positive.list \
    --label_strategy multi_label_threshold --min_class_samples 25

# v2 per-head training (cleaner labels per axis; unlocks agent 4's
# highconf_v2/ deliverables). Each protein's K labels are zeroed if
# it's not in the K list, and same for O — so each head trains only
# on proteins that are clean for that axis.
cipher-train --model attention_mlp \
    --positive_list_k data/training_data/metadata/highconf_v2/HC_K_cl95.list \
    --positive_list_o data/training_data/metadata/highconf_v2/HC_O_cl95_full_coverage.list \
    --label_strategy multi_label_threshold --min_class_samples 25
```

Valid tool names: `DePP_85`, `PhageRBPdetect`, `DepoScope`, `DepoRanker`,
`SpikeHunter`, `dbCAN`, `IPR`, `phold_glycan_tailspike`.

### Which head(s) to train: `--heads`

`--heads {both,k,o}` controls whether both heads, just K, or just O are
trained. Orthogonal to the positive-list flags — for example, you can
train just the K head against the K list as an ablation:

```bash
cipher-train --model attention_mlp \
    --positive_list_k data/training_data/metadata/highconf_v2/HC_K_cl95.list \
    --heads k
```

For `attention_mlp`, the unused head's training loop is skipped
entirely (no `model_k/` or `model_o/` subdirectory is written). For
`contrastive_encoder`, the unused head's ArcFace weight is forced to 0.

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

`candidates_clusters.tsv` contains columns `protein_id  cl30_X  cl40_X  ...  cl95_X`
(no header). It is included in `cipher_training_data.zip`; it can be
regenerated with `scripts/utils/build_candidates_cluster_file.py` if the
set of candidates changes.

### Label strategies

The K-type and O-type serotype assignment of each protein is aggregated
from observations across multiple phage-host interactions. The same
protein sequence may appear with different K-type annotations because:

1. Some receptor binding proteins are **specific** (bind a single K-type);
   observations for the sequence will largely agree.
2. Others are **polyspecific** (recognize multiple K-types); observations
   legitimately span several classes.
3. A subset of observations are noisy — for example, a single host
   annotation may be incorrect or reflect weak binding.

Five label strategies are provided, each addressing these concerns
differently. `multi_label_threshold` is recommended as the default.

| Strategy | Label encoding | Loss | Appropriate use |
|---|---|---|---|
| `single_label` | One-hot (majority vote over observations) | Cross-entropy with softmax | Datasets in which RBPs are assumed strictly specific |
| `multi_label` | Binary indicator per class | BCE (per-class independent) | Polyspecific RBPs, when observations are trusted |
| `multi_label_threshold` | Binary with `count ≥ N AND count/total ≥ X` filter | BCE (per-class independent) | **Recommended default** — polyspecific labels with noise filtering |
| `weighted_soft` | Fractional distribution summing to 1 | KL divergence to target (softmax output) | Distribution matching; classes compete for probability mass |
| `weighted_multi_label` | Fractional per class (observation frequency) | BCE on soft targets | Polyspecific with observation-strength encoding; classes independent |

#### `single_label`

Each protein is assigned a single K-type corresponding to its most
frequent host observation. The model is trained with softmax + cross-entropy,
so classes compete for probability mass. Appropriate when the assumption of
one-protein-one-K-type is acceptable, but it discards multi-target
information for genuinely polyspecific proteins.

#### `multi_label`

Every observed K-type becomes a binary positive label; all non-observed
K-types are negative. Training uses binary cross-entropy (BCE) per class,
so each K-type is predicted independently. This preserves polyspecific
information but treats a single observation identically to many — one
noisy observation produces the same label signal as dozens of consistent
observations.

#### `multi_label_threshold` (recommended default)

Identical to `multi_label`, with the refinement that a K-type is assigned
a positive label only if:

```
count(K-type) ≥ min_label_count   AND   count(K-type) / total_observations ≥ min_label_fraction
```

Default thresholds are `min_label_count=1` and `min_label_fraction=0.1`.
A stricter setting (`min_label_count=2, min_label_fraction=0.1`) drops
singleton associations as likely noise. This strategy retains legitimate
polyspecific associations while filtering sparse or inconsistent
observations.

```bash
cipher-train --model attention_mlp \
    --label_strategy multi_label_threshold \
    --min_label_count 2 --min_label_fraction 0.1
```

#### `weighted_soft`

Each protein's labels are a fractional distribution that sums to one,
reflecting the relative frequency of each observed K-type. Training uses
KL divergence between the softmax output and this target distribution,
and classes therefore compete for probability mass. This is useful when
observation frequencies can be interpreted as posterior probabilities
over target serotypes.

#### `weighted_multi_label`

Each K-type receives a fractional positive label proportional to its
relative observation count. Training uses BCE on these soft targets, so
classes are predicted independently while preserving observation
strength. This is the most faithful representation of a polyspecific
protein whose affinities differ across K-types, without forcing
probability mass to compete between classes.

### Provenance

Every trained run captures the following metadata at training time:

| Field | Description |
|---|---|
| `git_commit` | Short SHA of `HEAD` at the moment `cipher-train` was launched. |
| `git_dirty` | Boolean. `True` if `git status --porcelain` returned non-empty output, i.e. the working tree contained uncommitted modifications, staged-but-unclosed changes, or untracked files that could have influenced the run. `False` indicates the working tree matched `git_commit` exactly. |
| `host` | Hostname of the machine executing the run (login node, compute node, or laptop). |
| `slurm_job_id` | SLURM job identifier when run under `sbatch`; empty otherwise. |
| `user` | Invoking user (from `$USER`). |
| `cli_argv` | Full command line as received by the process, including all CLI flags. |
| `timestamp` | ISO-formatted start time. |

These fields are stored under `experiment.json["provenance"]`. To
reproduce a given run, check out the recorded `git_commit`, restore the
saved `config.yaml`, and rerun `cipher-train` with the same `--name`.

**Reproducibility caveat.** When `git_dirty == True`, the recorded
`git_commit` hash is only an approximation of the code state: there were
uncommitted local modifications at training time that cannot be
recovered from version control alone. Runs intended for publication or
permanent record should be executed from a clean working tree
(`git_dirty == False`). When inspecting historical runs, treat
`git_dirty == True` as a flag that the run may be imperfectly
reproducible from its commit hash.

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

Runs both ranking modes against five validation datasets: CHEN,
GORODNICHIV, UCSD, PBIP, and PhageHostLearn.

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

All sweep scripts submit SLURM jobs on the Delta-AI cluster. Each row in
the sweep array carries its own memory and wall-clock allocation,
pre-configured for the size of the embedding it trains on.

### Embedding options tested

The framework has been evaluated against a range of protein representations.
Each embedding is keyed by MD5 hash of the amino-acid sequence, enabling
shared storage across experiments and cross-model comparison.

| Family | Variant | Output dim | Pooling | Source |
|---|---|---|---|---|
| **ESM-2** | 150M mean | 640 | mean over residues | Hugging Face `facebook/esm2_t30_150M_UR50D` |
| | 650M mean | 1280 | mean over residues | `facebook/esm2_t33_650M_UR50D` |
| | 650M seg4 | 5120 | mean within each of 4 sequence segments, concatenated | as above |
| | 650M seg8 | 10 240 | mean within each of 8 segments | as above |
| | 650M seg16 | 20 480 | mean within each of 16 segments | as above |
| | 650M full | L × 1280 | per-residue (used by Light Attention) | as above |
| | 3B mean | 2560 | mean over residues | `facebook/esm2_t36_3B_UR50D` |
| | 15B mean | 5120 | mean over residues | `facebook/esm2_t48_15B_UR50D` |
| **ProtT5** | XL mean | 1024 | mean over residues | `Rostlab/prot_t5_xl_uniref50` |
| | XL seg4 / seg8 / seg16 | 4 096 / 8 192 / 16 384 | segmented pooling as above | as above |
| | XXL mean | 1024 | mean over residues (fp16 inference) | `Rostlab/prot_t5_xxl_uniref50` |
| **K-mer** | aa20_k3 | 8 000 | normalized 3-mer frequencies (20-letter alphabet) | computed from FASTA |
| | aa20_k4 | 160 000 | normalized 4-mer frequencies (20-letter alphabet) | as above |
| | murphy8_k5 | 32 768 | 5-mer frequencies (Murphy 8-letter reduction) | as above |
| | murphy10_k5 | 100 000 | 5-mer frequencies (Murphy 10-letter reduction) | as above |
| | li10_k5 | 100 000 | 5-mer frequencies (Li 10-letter reduction) | as above |

Segmented pooling (`segN`) partitions a protein into N approximately equal
segments from the C-terminus, mean-pools each segment independently, and
concatenates the results. This preserves local sequence features that are
destroyed by whole-protein mean pooling and has been observed to improve
K-type discrimination for certain model sizes.

### Embedding sweep (single-embedding `attention_mlp` variants)

`scripts/run_embedding_sweep.sh` trains the same `attention_mlp`
configuration against each embedding in the `EMBEDDINGS` array and
evaluates the resulting runs against all validation datasets. Each row
carries its own `--mem` and `--time` SLURM allocations, reflecting the
size of the embedding file.

```bash
DRY_RUN=1 bash scripts/run_embedding_sweep.sh         # preview all jobs
bash scripts/run_embedding_sweep.sh                    # submit all embeddings
bash scripts/run_embedding_sweep.sh esm2_3b_mean       # submit a single embedding by label
```

Two environment-variable toggles compose freely. Run names encode the
combination so that variants coexist under `experiments/attention_mlp/`
without naming collisions:

| Variable | Effect on training | Run-name decoration |
|---|---|---|
| `FILTER_MODE=positive_list` | Switch from `--tools` filtering to `--positive_list` | Adds `posList_` prefix |
| `USE_CLUSTERS=1` | Enable 70%-identity cluster-stratified downsampling | Adds `_cl70` suffix |

```bash
# Pipeline-positive filter only
FILTER_MODE=positive_list bash scripts/run_embedding_sweep.sh

# Cluster-stratified downsampling only
USE_CLUSTERS=1 bash scripts/run_embedding_sweep.sh

# Both combined
FILTER_MODE=positive_list USE_CLUSTERS=1 bash scripts/run_embedding_sweep.sh
```

To add a new embedding to the sweep, append a row to the `EMBEDDINGS`
array at the top of the script in the format:

```
"<label>   <path_to_training_NPZ>   <path_to_validation_NPZ>   <mem>"
```

### Concat sweep (pLM + k-mer)

`scripts/run_concat_sweep.sh` iterates over `(pLM, k-mer)` pairs and
trains on their per-MD5 concatenation. The pairs are defined in the
`EMBEDDING_PAIRS` array near the top of the script; update this array
to reflect the best-performing embeddings from the single-embedding
sweep. The same environment-variable toggles apply. Run names follow
the pattern `concat_<plm>+<kmer>[_cl70]` (prefixed with `posList_` when
the positive-list filter is active).

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

# Plot HR@k curves for top-N experiments (any-hit, phage-weighted overall)
python scripts/analysis/plot_anyhit_curves.py --top 4
# -> results/figures/anyhit_phage2host_hrk_top4.svg     (cipher primary direction)
# -> results/figures/anyhit_host2phage_hrk_top4.svg     (PhageHostLearn comparison direction)

# Focused PHL+PBIP table sorted by combined HR@1
python scripts/analysis/compare_primary_datasets.py --filter sweep_
```

The CSV is rebuilt idempotently from `experiment.json` and
`results/evaluation.json` in each run directory, so it is safe to
regenerate after additional experiments complete.

#### Aggregating across worktrees

When multiple contributors are developing in parallel git worktrees
(for example, `cipher-light_attention/` and `cipher-contrastive_encoder/`
as siblings of the main `cipher/` checkout), each worktree accumulates
its own `experiments/` directory. Pass all of them to
`harvest_results.py` via the `--experiments-dirs` flag so the output
CSV spans every worktree:

```bash
# Explicit list of experiments roots:
python scripts/analysis/harvest_results.py --experiments-dirs \
    experiments \
    /u/llindsey1/llindsey/PHI_TSP/cipher-light-attention/experiments \
    /u/llindsey1/llindsey/PHI_TSP/cipher-light-attention-binary/experiments

# Or use shell globbing to auto-discover every sibling worktree:
python scripts/analysis/harvest_results.py --experiments-dirs \
    experiments ../cipher-*/experiments

# Explicit label=path syntax to control the source column in the CSV:
python scripts/analysis/harvest_results.py --experiments-dirs \
    main=experiments la=../cipher-light-attention/experiments
```

The flag accepts any number of directories. When labels are not supplied
explicitly, each entry is auto-tagged with the name of its parent
directory (e.g. a path `../cipher-light-attention/experiments`
produces source label `cipher-light-attention`). The resulting CSV
contains two worktree-aware columns:

- `source` — the worktree label
- `exp_dir` — the full path to the run directory

so runs from different worktrees can be distinguished unambiguously,
even if run names collide.

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

Scripts in this directory regenerate ESM-2 and ProtT5 embeddings from
the raw training and validation FASTAs. They are designed for execution
on the Delta-AI cluster, where the pretrained model weights are cached
under `$TORCH_HOME`.

| File | Purpose |
|---|---|
| `esm2_extract.py` | Extracts embeddings from any ESM-2 model. Supports `--pooling mean`, `--pooling segmentsN` for any `N ≥ 2`, and `--pooling full` (per-residue output for Light Attention). Writes a `<output>.partial.npz` checkpoint every `--checkpoint_every` proteins (default 2000) and resumes automatically on restart. |
| `prott5_extract.py` | Extracts embeddings from any Rostlab ProtT5 variant (XL, XL-BFD, XXL). Supports `--pooling mean`, `--pooling full`, and `--pooling segmentsN`, `--half_precision` for reduced VRAM, and `--max_length` for long-sequence truncation. Same checkpoint/resume behaviour as `esm2_extract.py`. |
| `filter_fasta.py` | One-shot filter: given a FASTA and a list of protein IDs, write out only the matching sequences. Used to narrow an extraction run to a positive list (e.g. `pipeline_positive.list`). |
| `split_fasta_by_length.py` | Bins a FASTA into cumulative length caps (128 / 256 / 512 / 1024 / 2048 / 4096, plus an 8192 overflow). Writes sequences within each bin in length-ascending order so per-batch padding inside a bin is small. |
| `submit_extractions.sh` | SLURM fan-out: submits one training-FASTA job and one validation-FASTA job per `(model, pooling)` entry in the `EXTRACTIONS` array. Selects the appropriate conda environment per family (`esmfold2` for ESM-2, `prott5` for ProtT5). |
| `submit_prott5_full_binned.sh` | End-to-end length-binned ProtT5-XL extraction pipeline: filter FASTA → split by length → submit one SLURM job per bin. Idempotent; skips bins whose NPZ already exists. |
| `resubmit_missing_bins.sh` | ESM-2 counterpart: given an already-split directory, submits only the length bins whose output NPZ is missing. Use to recover from partial failures without re-running completed bins. |
| `merge_split_embeddings.py` | Stream-merges many per-bin NPZs into a single NPZ (see "Merging" below). O(largest single array) memory. |

#### Invocation

```bash
# Preview all extraction jobs without submitting:
DRY_RUN=1 bash scripts/extract_embeddings/submit_extractions.sh

# Submit every (model, pooling) pair in the EXTRACTIONS array:
bash scripts/extract_embeddings/submit_extractions.sh

# Submit a single (model, pooling) pair by its derived label:
bash scripts/extract_embeddings/submit_extractions.sh prott5_xl_segments4
```

Missing NPZ files (extractions not yet run) are skipped at submission
time with a warning rather than triggering a failure. This allows the
script to be re-run safely as additional extractions become available.

#### Direct invocation of the extractors

Each extractor can also be invoked directly within a SLURM job script:

```bash
# ESM-2 650M mean (the original baseline embedding)
python scripts/extract_embeddings/esm2_extract.py \
    data/training_data/metadata/candidates.faa \
    /work/output/esm2_650m_mean.npz \
    --model esm2_t33_650M_UR50D --layer 33 --pooling mean --key_by_md5

# ESM-2 650M segmented pooling (N=8)
python scripts/extract_embeddings/esm2_extract.py \
    data/training_data/metadata/candidates.faa \
    /work/output/esm2_650m_segments8.npz \
    --model esm2_t33_650M_UR50D --layer 33 --pooling segments8 --key_by_md5

# ProtT5-XL mean
python scripts/extract_embeddings/prott5_extract.py \
    data/training_data/metadata/candidates.faa \
    /work/output/prott5_xl_mean.npz \
    --model_name Rostlab/prot_t5_xl_uniref50 --pooling mean --key_by_md5

# ProtT5-XXL mean (11B parameters; half precision required on a single H100)
python scripts/extract_embeddings/prott5_extract.py \
    data/training_data/metadata/candidates.faa \
    /work/output/prott5_xxl_mean.npz \
    --model_name Rostlab/prot_t5_xxl_uniref50 --pooling mean --key_by_md5 \
    --half_precision --max_length 3000
```

#### Crash resilience

Both `esm2_extract.py` and `prott5_extract.py` write a partial checkpoint
(`<output>.partial.npz`) every `--checkpoint_every` proteins (default 2000)
and resume automatically when restarted with the same output path, so a
late-stage failure forfeits at most ~2000 proteins of computation.
`prott5_extract.py` additionally catches an out-of-memory error on a
single long sequence, skips it, and reports the total `oom_skipped`
count at the end of the run.

#### Length-binned full per-residue extraction

For `--pooling full` (per-residue output, for Light Attention), memory
and time scale with L². Running one big extraction over the whole FASTA
fails for two reasons:

1. Inference-time VRAM on long sequences (ProtT5-XL attention is O(L²)).
2. Output size: the in-memory `{md5 -> (L, D) array}` dict before the
   final `np.savez_compressed` call is often >100 GB for our datasets,
   and the per-sequence SLURM job can't hold it.

The fix is to split the input FASTA into length bins (128 / 256 / 512 /
1024 / 2048 / 4096, plus 8192 overflow), extract each bin in its own
SLURM job with memory scaled to the bin, and merge the resulting NPZs at
the end. The per-bin jobs also stay within Delta-AI's 48-hour wall
limit for the larger bins.

**Running the pipeline end-to-end (ProtT5-XL, pipeline_positive subset):**

```bash
# One command filters candidates.faa to the positive list, splits by
# length, and submits one SLURM job per bin. Idempotent — rerun to
# resubmit only missing bins.
bash scripts/extract_embeddings/submit_prott5_full_binned.sh

# Preview without submitting:
DRY_RUN=1 bash scripts/extract_embeddings/submit_prott5_full_binned.sh
```

**ESM-2 counterpart (run after an earlier extraction left some bins missing):**

```bash
# Assumes split_fasta/ + (partially) split_embeddings/ under ROOT.
# Resubmits only bins whose NPZ is missing. maxlen1024 runs full-node
# (mem=0, 4 GPUs) because its in-memory output exceeds any per-job mem.
ROOT=/work/hdd/bfzj/llindsey1/embeddings_full \
    bash scripts/extract_embeddings/resubmit_missing_bins.sh
```

Per-bin outputs land under `<root>/split_embeddings/`, one NPZ per
length bin.

#### Merging split outputs

Once every bin's NPZ is on disk, merge them into a single drop-in NPZ
consumable by `np.load(...)`:

```bash
python scripts/extract_embeddings/merge_split_embeddings.py \
    -i /work/hdd/bfzj/llindsey1/prott5_xl_full/split_embeddings \
    -o /work/hdd/bfzj/llindsey1/prott5_xl_full/candidates_prott5_xl_full_md5.npz
```

The merge is **streaming**: each array is read from its source NPZ and
written directly as an `.npy` member inside the output zip, one at a
time. Peak memory is O(largest single array), not O(total merged data) —
this matters for the ESM-2 650M full merge where the naive in-memory
approach would need ~250 GB of RAM. The output format is a standard
NPZ; downstream code continues to use `np.load(...)` unchanged.

Pass `--no-compress` for a faster-to-write, larger-on-disk output (the
`savez` vs `savez_compressed` trade-off). Duplicate keys are kept once
(first-write wins) with a count logged at the end. Mixed last-dim
embeddings trigger a warning but don't abort.

#### Verifying coverage + sanity

Before passing a merged per-residue NPZ to a downstream model:

```bash
# Coverage against the canonical positive-ID lists + dim/NaN sanity on
# a random sample. --expected-dim asserts shape[-1] matches on every
# sampled key (exit 1 on mismatch or NaN/Inf).
python scripts/utils/check_full_npz_coverage.py \
    --npz /work/hdd/bfzj/llindsey1/prott5_xl_full/candidates_prott5_xl_full_md5.npz \
    --expected-dim 1024
```

The coverage section prints, for `candidates.faa`, `pipeline_positive.list`,
`highconf_tsp_K.list`, and `highconf_pipeline_positive_K.list`, the
number of MD5s present in the NPZ vs missing. For an extraction that
was filtered to `pipeline_positive` upstream, expect ~100% coverage on
the pipeline + highconf rows and partial coverage on the "all
candidates" row.

The sanity section samples `--sample-size` keys (default 200) and
reports: unique last-dims observed, L (first-dim) range + median, and
any NaN/Inf. Use `--expected-dim 1024` for ProtT5-XL, `--expected-dim
1280` for ESM-2 650M.

---

## Data packaging

Raw input files are distributed as two archives, each with an
accompanying sha256 manifest:

```bash
bash scripts/utils/package_data.sh
# -> dist/cipher_training_data.zip   (raw inputs: FASTA, TSVs, cluster file)
# -> dist/cipher_validation_data.zip (5 validation datasets)
```

Each archive contains a `MANIFEST.txt` file enabling byte-level
verification via `shasum -a 256 -c MANIFEST.txt`. Both archives are also
deposited on Zenodo. Embeddings are published separately and can be
regenerated from source FASTAs using the scripts under
`scripts/extract_embeddings/`.

---

## Testing

The project ships with a comprehensive `pytest` suite (111 tests across
nine modules) covering the data-preparation pipeline, evaluation
primitives, and model-specific components. All tests execute in under
two seconds on a modern laptop and require no GPU.

### Running the suite

```bash
pytest                                              # full suite (111 tests)
pytest -v                                           # per-test output
pytest -x                                           # stop on first failure
pytest -k "sampler"                                 # filter by name substring
pytest --collect-only -q                            # list all tests without executing
pytest tests/test_contrastive_sampler.py -v         # run one module
pytest tests/test_training.py::TestClusterStratifiedSample  # run one test class
```

### Coverage

| Test module | Tests | What it verifies |
|---|---:|---|
| `test_training.py` | 43 | `TrainingConfig` construction and deprecation paths; `prepare_training_data` filtering (tools, positive_list, min_sources); multi-label / single-label / threshold label building; downsampling (including cluster-stratified); `min_class_samples` enforcement; weighted-soft label arithmetic. |
| `test_metrics.py` | 13 | `hr_at_k`, `mrr`, and `hr_curve` under ordinary and edge-case inputs (empty ranks, ties, k larger than candidate pool). |
| `test_analysis.py` | 11 | Per-serotype analysis helpers used by the diagnostic scripts under `scripts/analysis/`. |
| `test_contrastive_sampler.py` | 11 | `PKClusterSampler` batch shape, class balance, cluster stratification, usable-class filtering (strict and non-strict), determinism under fixed seed, and degenerate inputs. |
| `test_predictor.py` | 8 | `Predictor.score_pair` z-score normalization across K and O heads; tie behavior; handling of missing predictions. |
| `test_serotypes.py` | 8 | Serotype parsing: null handling, K-type short-form normalization, and identity preservation for already-normalized labels. |
| `test_ranking.py` | 7 | Host and phage ranking under competition tie-handling and arbitrary-order tie-handling. |
| `test_splits.py` | 6 | Stratified train/val/test split behavior, including minority-class preservation and deterministic seeding. |
| `test_embeddings.py` | 4 | `load_embeddings_concat` coverage-mismatch error, MD5 filtering, ordering of concatenated vectors, and correct dimensionality. |

### Test fixtures

Small synthetic fixtures are stored under `tests/test_data/`:

- `association_map.tsv` — a minimal `host_phage_protein_map.tsv` covering
  a handful of phages, proteins, and K/O serotypes.
- `glycan_binders.tsv` — the matching tool-flag table.

These fixtures support end-to-end tests of `prepare_training_data`
without requiring access to the full training corpus. The fixtures are
designed to exercise filter edge cases (rare classes, null serotypes,
proteins missing from either file) rather than to reflect production
data.

### Adding a new test

Test modules follow the `tests/test_*.py` naming convention and use
standard `pytest` discovery. New tests should:

1. Be placed in an existing module when the subject area matches, or in
   a new `tests/test_<subject>.py` file otherwise.
2. Use fixtures from `tests/test_data/` where possible; add new
   fixtures only when the required data cannot be constructed
   programmatically inside the test.
3. Run in under one second per test and require no network or GPU
   access.
4. Be deterministic — seed any random number generator explicitly.

### Continuous integration

Before committing changes to `src/cipher/`, the shared library, run the
full suite:

```bash
pytest
```

Any commit that modifies the shared library is expected to leave the
suite passing. Model-specific changes (under `models/{name}/`) should
additionally pass any model-specific tests in `tests/`.

## References

Additional documentation is available in `ONBOARDING.md`, which covers
biological background, data provenance, and known limitations inherited
from prior work.
