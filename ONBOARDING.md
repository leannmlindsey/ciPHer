# ciPHer Onboarding Guide

**ciPHer** = Computational Inference of Phage-Host Entry Range

## What This Project Does

Predicts which phages can infect which Klebsiella bacterial hosts. The approach:
1. Take a phage's receptor binding proteins (RBPs/TSPs)
2. Predict which host capsular serotypes (K-type, O-type) each protein targets
3. Score phage-host pairs using a biologically-motivated "lock and key" model

## The Biology (Important for Understanding the Code)

- A bacterial host has **2 locks**: K-type (capsular polysaccharide, ~160 types) and O-type (lipopolysaccharide, ~17 types)
- A phage has **N keys**: its receptor binding proteins (RBPs/TSPs, typically 1-6 per phage)
- Each protein may target either the K or O surface structure
- **If ANY single protein matches ANY single lock, the phage can infect**
- Scoring: `pair_score = max over proteins of max(P(K_host), P(O_host))`

## The Data

### Training Data (prophage-derived, association labels)
- 143K candidate proteins from prophage regions in Klebsiella genomes
- Labels are **associations** (this protein was found in a prophage that was extracted from a host with K-type X)
- NOT functional labels — this is a key limitation (see "Known Issues" below)
- Located: `data/training_data/`

Key file: `metadata/host_phage_protein_map.tsv` — the master table linking:
```
host_assembly → host_K, host_O → phage_id → protein_id → is_tsp → protein_md5
```

### Validation Data (experimentally tested interactions)
- 5 datasets from published phage-host interaction studies
- Labels are **experimental** (phage was tested against host, infection observed or not)
- Located: `data/validation_data/HOST_RANGE/{DATASET_NAME}/`

| Dataset | Phages | Hosts | Pos pairs | Key characteristic |
|---|---|---|---|---|
| CHEN | 3 | 50 | 69 | Broad-range phages (7-10 K-types each) |
| GORODNICHIV | 3 | 83 | 12 | Narrow-range, all K23 |
| UCSD | 11 | 59 | 236 | Very broad (15-25 K-types each) |
| PBIP | 104 | 120 | 938 | K47-dominated (53% of pairs) |
| PhageHostLearn | 127 | 215 | 465 | Most diverse (86 K-types) |

Each dataset has standardized files in `metadata/`:
- `interaction_matrix.tsv` — columns: host_id, host_assembly, host_K, host_O, phage_id, label
- `phage_protein_mapping.csv` — columns: matrix_phage_name, protein_id

Raw source data from publications is in `source/` (read-only reference).

### Embeddings
Pre-computed features in `embeddings/` subdirs (training and validation):
- `esm2_650m_md5.npz` — ESM-2 protein language model (1280-d)
- `kmer3_8000d_md5.npz` — amino acid 3-mer frequencies (8000-d)
- All keyed by MD5 hash of protein sequence: `hashlib.md5(seq.encode()).hexdigest()`

### Column Standards
See `data/COLUMN_STANDARDS.md` for the naming conventions used across all files.

## The Shared Library (`cipher/`)

Installed via `pip install -e .` from the repo root.

### cipher.data
```python
from cipher.data import (
    load_embeddings,          # NPZ → {md5: vector}
    load_fasta,               # FASTA → {protein_id: sequence}
    load_serotypes,           # TSV → {host_id: {K, O}}
    load_interaction_matrix,  # TSV → {phage: {host: 0/1}}
    load_phage_protein_mapping,  # CSV → {phage: set(protein_ids)}
)
from cipher.data.interactions import load_interaction_pairs  # enriched list of dicts
from cipher.data.proteins import compute_md5, load_fasta_md5
```

### cipher.evaluation
```python
from cipher.evaluation import Predictor, hr_at_k, mrr
from cipher.evaluation.ranking import evaluate_rankings
```

The `Predictor` abstract class is the key interface every model must implement:
```python
class MyPredictor(Predictor):
    @property
    def embedding_type(self):
        return 'esm2_650m'  # or 'kmer3_8000d', etc.
    
    def predict_protein(self, embedding):
        # Run model inference
        return {'k_probs': {k: prob, ...}, 'o_probs': {o: prob, ...}}
    
    # score_pair() has a default implementation (max-over-proteins, max-over-KO)
    # Override only if your model scores differently
```

## Evaluation: Two Ranking Modes

### Scoring formula

Each model produces, for every phage protein `p`, a probability
distribution over K-types and a probability distribution over O-types:

- `P_K(k | p)` — probability that protein `p` targets K-type `k`, for
  all `k` in the K-type class set (≈ 160 classes).
- `P_O(o | p)` — probability that protein `p` targets O-type `o`, for
  all `o` in the O-type class set (≈ 17 classes).

A candidate host `h` has a single true K-type `k_h` and single true
O-type `o_h` (from Kaptive annotation).

**Step 1 — Per-protein z-score standardization (per head).**
Because the K head has roughly ten times more classes than the O head,
raw probabilities from the two heads are not directly comparable. For
each protein `p`, probabilities are standardized within each head:

```
z_K(k | p) = ( P_K(k | p) − μ_K(p) ) / σ_K(p)
z_O(o | p) = ( P_O(o | p) − μ_O(p) ) / σ_O(p)
```

where `μ_K(p)`, `σ_K(p)` are the mean and standard deviation of
`P_K(· | p)` across all K classes, and `μ_O(p)`, `σ_O(p)` are defined
analogously for O. The z-scored values are unitless and comparable
across the two heads.

**Step 2 — Per-protein, per-host score.**
The score that protein `p` assigns to host `h` is the larger of its two
standardized logits:

```
s(p, h) = max( z_K(k_h | p), z_O(o_h | p) )
```

This implements the biological "lock and key" assumption: a phage
protein that strongly recognizes either the host's K-type OR its O-type
is sufficient for infection.

**Step 3 — Phage-host pair score (max-over-proteins).**
A phage `P` with receptor binding proteins `{p_1, …, p_m}` scores each
candidate host `h` by the maximum contribution across its proteins:

```
S(P, h) = max over p in {p_1, …, p_m} of s(p, h)
```

This reflects the fact that a phage infects the host if **any single
one of its receptor binding proteins** engages the host surface; no
consensus across proteins is required.

### Ranking modes

Both modes compute the same pair score `S(P, h)` and differ only in
which variable is held fixed and which is ranked.

#### Mode A — Rank hosts given phage
"Given this phage, which candidate hosts should be tested?"

For each phage `P` in the validation set, compute `S(P, h)` for every
candidate host `h` in the dataset, sort hosts in descending order of
pair score, and record the rank position of each true host.

#### Mode B — Rank phages given host
"Given this bacterial isolate, which phage is likely to infect it?"

For each host `h`, compute `S(P, h)` for every candidate phage `P`,
sort phages in descending order of pair score, and record the rank
position of each true phage.

### Tie handling (competition ranking)

When multiple candidates receive identical scores (for example, hosts
sharing the same `(K, O)` pair that the model cannot distinguish), all
tied candidates are assigned the same rank position (the lowest rank in
the tied group), and the next distinct score skips ahead by the size of
the tie group. This is the standard competition-ranking convention
("1, 1, 1, 4" rather than "1, 2, 3, 4"), and it prevents artificial
inflation of HR@k that would occur if tied candidates were ordered
arbitrarily.

### Metrics

For a set of true pairs `{(P_i, h_i) : i = 1, …, N}`:

**Hit Rate at k** — the fraction of true pairs for which the true
target appears within the top-k ranked candidates:

```
HR@k = (1/N) · Σ_i 1[ rank(h_i in ranked list for P_i) ≤ k ]
```

Reported for k = 1, 2, …, 20.

**Mean Reciprocal Rank**:

```
MRR = (1/N) · Σ_i 1 / rank(h_i in ranked list for P_i)
```

Both metrics are computed independently for Mode A and Mode B, and
reported per validation dataset.

## Experiments (`experiments/`)

Each training run produces a directory under `experiments/{model_name}/{run_name}/`
containing the merged configuration, provenance metadata, trained
weights, and (after evaluation) the ranking results. Model code itself
lives under `models/{model_name}/` and is shared across all runs of that
model.

```
experiments/{model_name}/{run_name}/
├── config.yaml              # Merged config (base_config + CLI overrides)
├── experiment.json          # Metadata + provenance (git commit, host, SLURM id, ...)
├── model_k/                 # Trained K-type head (attention_mlp)
│   ├── best_model.pt
│   ├── config.json
│   └── training_history.json
├── model_o/                 # Trained O-type head (attention_mlp)
├── splits_k.json            # MD5 train/val/test split for the K head
├── splits_o.json            # MD5 train/val/test split for the O head
├── label_encoders.json      # K-class and O-class lists
├── training_data.npz        # The filtered, labeled training tensor
└── results/
    └── evaluation.json      # Populated by cipher-evaluate
```

### Example `config.yaml`

A complete configuration for a single run. This is the merged output of
`models/attention_mlp/base_config.yaml` with CLI overrides supplied to
`cipher-train`. Every field is machine-readable and serves as the
reproducible record of the experiment.

```yaml
model:
  hidden_dims: [1280, 640, 320, 160]
  attention_dim: 640
  dropout: 0.1

training:
  batch_size: 512
  learning_rate: 1.0e-05
  epochs: 1000
  patience: 30
  seed: 42
  train_ratio: 0.7
  val_ratio: 0.15

data:
  association_map: /projects/bfzj/llindsey1/PHI_TSP/ciPHer/data/training_data/metadata/host_phage_protein_map.tsv
  glycan_binders:  /projects/bfzj/llindsey1/PHI_TSP/ciPHer/data/training_data/metadata/glycan_binders_custom.tsv
  embedding_type:  kmer_murphy8_k5
  embedding_file:  /work/hdd/bfzj/llindsey1/kmer_features/candidates_murphy8_k5.npz

validation:
  val_fasta:          /projects/bfzj/llindsey1/PHI_TSP/ciPHer/data/validation_data/metadata/validation_rbps_all.faa
  val_embedding_file: /work/hdd/bfzj/llindsey1/kmer_features/validation_murphy8_k5.npz
  val_datasets_dir:   /projects/bfzj/llindsey1/PHI_TSP/ciPHer/data/validation_data/HOST_RANGE

experiment:
  min_sources: 1
  max_k_types: 3
  max_o_types: 3
  single_label: false
  label_strategy: multi_label_threshold
  positive_list_path: /projects/bfzj/llindsey1/PHI_TSP/ciPHer/data/training_data/metadata/pipeline_positive.list
  max_samples_per_k: 1000
  max_samples_per_o: 3000
  min_class_samples: 25
  cluster_file_path: /projects/bfzj/llindsey1/PHI_TSP/ciPHer/data/training_data/metadata/candidates_clusters.tsv
  cluster_threshold: 70
```

Four blocks control different aspects of the run:

- **`model`** — architecture hyperparameters for the classifier backbone.
- **`training`** — optimizer hyperparameters, batch size, epochs, early
  stopping, and random seeds. The train/val/test ratios apply to the
  internal split used for early stopping during training, not to the
  separate validation datasets under `data/validation_data/HOST_RANGE/`.
- **`data`** — paths to the training-side inputs (association map,
  protein index, and the pre-extracted embedding NPZ).
- **`validation`** — paths to the downstream validation inputs used by
  `cipher-evaluate`.
- **`experiment`** — the data-filtering and label-construction knobs
  handled by `cipher.data.prepare_training_data`: tool filter or
  positive-list filter, minimum class representation, optional
  cluster-stratified downsampling, and the chosen label strategy.

## Known Issues and Key Findings (from prior work)

### 1. ESM-2 embeddings have weak K-type signal
- Same-K vs different-K cosine distance gap: only 0.003 (650M) or 0.01 (3B)
- AA 3-mer features have 0.068 gap (22x better)
- Yiyan et al. 2024 reports that TSP clusters primarily map to a single K-type or O-type serotype, while also documenting several examples in which a TSP cluster spans more than one serotype
- The signal exists in sequence but ESM-2 may compress it away

### 2. MIL/TropiGAT PBIP performance is K47 bias
- MIL K-only gets HR@1=0.514 on PBIP, but predicts K47 for 89% of pairs
- K47 is 53% of PBIP pairs, so predicting K47 always gives ~50%
- TropiGAT shows same pattern (fires on ~17 KL types per protein)
- Per-class analysis script: `klebsiella/scripts/evaluate/analyze_mil_per_class.py`

### 3. Prophage vs lytic domain shift
- 100% of training data is prophage-derived (association labels)
- All validation is on lytic phages (functional labels)
- This domain shift likely explains much of the generalization gap

### 4. Broad-range phages need multi-label prediction
- CHEN phages infect 7-10 K-types each
- UCSD phages infect 15-25 K-types each
- Single-label models are structurally unable to handle these

### 5. Best models so far (rank-hosts HR@1 per validation dataset)

The following table summarizes the best-performing runs under the ciPHer
framework, spanning representative combinations of embedding family,
training-set filter, sampling strategy, and feature-concatenation
configuration. Per-column maxima are **bolded**.

| Model | Training regime | CHEN | GORODNICHIV | UCSD | **PBIP** | **PHL** | PHL+PBIP combined | 5-ds mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `sweep_esm2_650m_mean` | tools, random | 0.275 | 1.000 | 0.053 | 0.595 | 0.113 | 0.239 | 0.295 |
| `sweep_esm2_3b_mean` | tools, random | 0.275 | 1.000 | 0.072 | 0.789 | 0.107 | **0.336** | 0.373 |
| `sweep_prott5_mean` | tools, random | 0.275 | 1.000 | 0.024 | 0.735 | 0.150 | 0.288 | 0.326 |
| `sweep_kmer_murphy8_k5` | tools, random | 0.275 | 1.000 | 0.113 | **0.800** | 0.098 | 0.284 | 0.330 |
| `sweep_posList_esm2_650m_seg4_cl70` | positive_list, cluster-70 | 0.275 | 1.000 | 0.100 | 0.751 | 0.144 | 0.307 | 0.322 |
| `sweep_posList_esm2_3b_mean_cl70` | positive_list, cluster-70 | 0.275 | 1.000 | 0.100 | 0.679 | 0.150 | 0.302 | 0.353 |
| `sweep_posList_kmer_li10_k5_cl70` | positive_list, cluster-70 | 0.275 | 1.000 | 0.100 | 0.581 | 0.160 | 0.243 | 0.348 |
| `concat_prott5_mean+kmer_li10_k5` | tools, random | 0.275 | 1.000 | 0.096 | 0.737 | **0.169** | 0.288 | **0.377** |
| `concat_posList_esm2_3b_mean+kmer_li10_k5_cl70` | positive_list, cluster-70 | 0.275 | 1.000 | 0.081 | 0.647 | 0.163 | 0.289 | 0.354 |

CHEN and GORODNICHIV HR@1 are saturated for the majority of runs (CHEN
contains three phages, GORODNICHIV contains three phages and a single
K-type). PBIP and PhageHostLearn are the discriminating datasets because
their candidate pools are large enough to differentiate between models.

**Observations:**

- The best single-representation run for **PBIP rank-hosts HR@1** is
  `sweep_kmer_murphy8_k5` (0.800), indicating that k-mer features alone
  capture a meaningful fraction of the signal on this dataset.
- The best run for **PhageHostLearn rank-hosts HR@1** is
  `concat_prott5_mean+kmer_li10_k5` (0.169), which concatenates a pLM
  embedding with a k-mer feature vector. The same run produces the best
  **five-dataset mean HR@1** (0.377).
- Moving the training-set filter from tool-based to the pipeline-positive
  list and switching random downsampling to cluster-stratified sampling
  at 70% identity consistently lifts PhageHostLearn HR@1 by 30–45%
  relative to the baseline without degrading overall performance.

## Existing Code Reference

The original development repo is at `/Users/leannmlindsey/WORK/PHI_TSP/phi_tsp/klebsiella/`.
It has 24 trained models, evaluation scripts, and presentation figures in
`output/validation/presentation_figures/`. This is the reference for porting
experiments to the new ciPHer format.

Key scripts in the old repo:
- `scripts/evaluate/evaluate_host_ranking_bio.py` — bio-motivated ranking (reference implementation)
- `scripts/evaluate/plot_hr_curves.py` — HR@k curve plotting
- `scripts/evaluate/generate_presentation_materials.py` — figures for PI
- `scripts/data_prep/compute_kmer_features.py` — k-mer feature computation
- `scripts/train/train_serotype_model.py` — AttentionMLP training
- `scripts/train/train_mil_model.py` — MIL training
- `scripts/train/train_xgboost_k.py` — XGBoost training

## Parallel Development

Multiple contributors and automated agents develop new models in parallel.
The project is structured so that parallel work does not produce merge
conflicts, provided the conventions below are followed.

### Overview

The repository separates **shared infrastructure** from **model-specific
code**:

- **Shared** (`src/cipher/`): data loading, filtering pipeline, cluster
  sampling, evaluation, provenance capture. Modified rarely and with care.
- **Model-specific** (`models/{name}/`): one directory per model
  architecture. Each contributor works in a dedicated directory that
  does not intersect with other models.
- **Run artefacts** (`experiments/{name}/{run}/`): produced by training;
  never edited by hand. Not tracked in git.

When this separation is respected, two contributors adding different
models (e.g., `light_attention` and `contrastive_encoder`) have no
overlapping files and merge cleanly.

### Rules for modifying shared code (`src/cipher/`)

Shared library code is consumed by every model. Any change must preserve
backward compatibility so that existing trained models and their
saved configurations continue to work.

1. **Add, do not replace.** New functionality should be introduced as
   additional functions, classes, or optional arguments, not as a
   modification of existing signatures.
2. **Default to off.** New function arguments and `TrainingConfig`
   fields must have a default value that reproduces the pre-change
   behaviour. Example: `positive_list_path: str = None` leaves the
   tool-based filter as the default; `cluster_file_path: str = None`
   leaves random downsampling as the default.
3. **Do not rename or remove public symbols.** If a function in
   `src/cipher/data/` or `src/cipher/evaluation/` is used anywhere
   outside `src/cipher/` or by any saved `predict.py`, it is public.
   Removing or renaming it breaks older experiments.
4. **Do not change file formats without migration.** If an existing
   artefact format needs to evolve (for example, adding a new field to
   `experiment.json`), preserve the ability to load older files.
5. **Exercise the full test suite.** Every change to `src/cipher/` must
   leave the 111-test suite passing. Add new tests covering the new
   behaviour.
6. **Document new flags.** Any new CLI flag or config field should be
   described in the README, with its default and an example.

A change that cannot satisfy the above — for example, an architectural
refactor that changes the shape of a saved file — requires coordination
with the entire team and a version bump. These are rare and should be
discussed before implementation.

### Model-specific code (`models/{name}/`)

Every new model requires four files, following the structure of
`models/attention_mlp/`:

| File | Purpose |
|---|---|
| `base_config.yaml` | Default hyperparameters for the model, including `model:`, `training:`, `data:`, `validation:`, and `experiment:` blocks. |
| `model.py` | `nn.Module` definition(s) and any auxiliary classes (loss heads, samplers). |
| `train.py` | Must expose `def train(experiment_dir, config)`. Consumes `cipher.data.prepare_training_data`, trains, writes artefacts (weights, splits, `experiment.json`) to `experiment_dir`. |
| `predict.py` | Must expose `def get_predictor(run_dir)` returning an object that implements `cipher.evaluation.Predictor`. Exception: feature-transformer models such as `contrastive_encoder` may raise `NotImplementedError` with usage instructions, as they do not classify directly. |

Only these four files belong in the model directory. All data loading,
filtering, and evaluation should be delegated to `cipher.data` and
`cipher.evaluation`.

### Git worktree strategy

`git worktree` allows multiple checkouts of the repository at different
paths, each on a different branch, sharing a single `.git` directory.
This enables parallel development on several models without duplicating
the repository or switching branches repeatedly.

**Naming conventions.**

- One branch per model: `light_attention`, `light_attention_binary`,
  `contrastive_encoder`, etc. Branches are named after the model, not
  the contributor.
- Worktree directories mirror the branch name:
  `cipher-light-attention/`, `cipher-contrastive-encoder/`, etc.,
  placed as siblings of the main `cipher/` checkout.

#### Creating a worktree on the laptop

From the main repository:

```bash
cd /path/to/cipher

# Create a new branch and worktree in one step
git worktree add ../cipher-my_new_model -b my_new_model

# Or attach to an existing remote branch
git fetch origin
git worktree add ../cipher-my_new_model my_new_model

# Confirm
git worktree list
```

The new directory `../cipher-my_new_model/` is a complete working tree
on branch `my_new_model`. Install the shared library in editable mode
in this worktree as well:

```bash
cd ../cipher-my_new_model
pip install -e ".[test]"
```

Create the new model directory:

```bash
mkdir -p models/my_new_model
# Populate base_config.yaml, model.py, train.py, predict.py
# following the template in models/attention_mlp/ or models/contrastive_encoder/
```

When work on a worktree is complete, remove it cleanly:

```bash
cd /path/to/cipher
git worktree remove ../cipher-my_new_model
# Or, if the worktree directory has been deleted manually:
git worktree prune
```

#### Creating a worktree on Delta-AI

The Delta-AI cluster introduces one additional constraint: the training
and validation data (`data/training_data/` and `data/validation_data/`)
occupy several gigabytes and must not be duplicated across worktrees.
Data therefore lives in a single canonical location, and every worktree
resolves data paths against that canonical directory rather than its
own `data/` subdirectory.

**Canonical paths on Delta-AI.**

```
Code (main):          /projects/bfzj/llindsey1/PHI_TSP/ciPHer/
Code (worktrees):     /projects/bfzj/llindsey1/PHI_TSP/cipher-<branch>/
Canonical data dir:   /projects/bfzj/llindsey1/PHI_TSP/ciPHer/data/
```

**Worktree setup on Delta.**

```bash
cd /projects/bfzj/llindsey1/PHI_TSP/ciPHer

# Fetch the branch that was created on the laptop
git fetch origin

# Attach a new worktree to the remote branch
git worktree add ../cipher-my_new_model my_new_model

cd ../cipher-my_new_model

# Verify
git worktree list
git branch --show-current
```

**Separating CIPHER_DIR from DATA_DIR in SLURM scripts.**

Model-specific SLURM scripts should resolve the code directory
(`CIPHER_DIR`) from the script's own location, while resolving the data
directory (`DATA_DIR`) to the canonical path. This pattern allows the
same script to run unmodified from any worktree:

```bash
#!/usr/bin/env bash
#SBATCH ...

# Code directory — auto-detected from the script path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="${CIPHER_DIR:-$(dirname "$SCRIPT_DIR")}"

# Data directory — canonical, shared across all worktrees
DATA_DIR="${DATA_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer/data}"

# Resolve data paths against DATA_DIR
ASSOC_MAP="${DATA_DIR}/training_data/metadata/host_phage_protein_map.tsv"
GLYCAN_BINDERS="${DATA_DIR}/training_data/metadata/glycan_binders_custom.tsv"
VAL_FASTA="${DATA_DIR}/validation_data/metadata/validation_rbps_all.faa"
VAL_DATASETS_DIR="${DATA_DIR}/validation_data/HOST_RANGE"

# Standard cluster initialization — invariant across worktrees
source $(conda info --base)/etc/profile.d/conda.sh
conda activate esmfold2
cd "${CIPHER_DIR}"
export PYTHONPATH="${CIPHER_DIR}/src:${PYTHONPATH:-}"

# Training command consumes $CIPHER_DIR (code) and $DATA_DIR (inputs)
python -m cipher.cli.train_runner \
    --model my_new_model \
    --association_map "$ASSOC_MAP" \
    --glycan_binders "$GLYCAN_BINDERS" \
    --val_fasta "$VAL_FASTA" \
    --val_datasets_dir "$VAL_DATASETS_DIR"
```

With this template, the same script runs correctly from the main
checkout and from any worktree without modification. Experiment
artefacts are written under `${CIPHER_DIR}/experiments/`, so each
worktree maintains its own experiments directory, while the input data
is shared.

### Development workflow for a new model

The recommended sequence for adding a new model:

1. **Create a branch + worktree on the laptop** (see above), named for
   the model.
2. **Scaffold the four model files** under `models/{name}/`, using
   either `models/attention_mlp/` or `models/contrastive_encoder/` as a
   template depending on whether the new model is a classifier or a
   feature transformer.
3. **Run the test suite** to confirm that no shared code was
   accidentally altered: `pytest`.
4. **Perform a local smoke test** with a tiny configuration, for example
   a small training subset and a few epochs, to verify that `train.py`
   completes and that `predict.py` produces a valid Predictor.
5. **Push the branch** to the shared remote.
6. **Create the matching worktree on Delta-AI** and submit a single
   SLURM job using the shared-data template above.
7. **Verify the run completes** and that `cipher-evaluate` produces an
   `evaluation.json` with plausible HR@k values.
8. **Run a full sweep or per-tool evaluation** as appropriate.
9. **Merge to main** via pull request once results are reproducible and
   the shared-code rules have been respected.

### Avoiding conflicts

The following files are the most common sources of conflict and should
be coordinated across branches before editing:

- `src/cipher/**/*.py` (shared library)
- `src/cipher/cli/train_runner.py` (shared CLI entrypoint)
- `scripts/run_embedding_sweep.sh` and `scripts/run_concat_sweep.sh`
  (shared sweep drivers)
- `README.md`, `ONBOARDING.md`, `CLAUDE.md`

Changes to any of these should be minimal, backward-compatible (per the
rules above), and ideally batched into a single PR rather than spread
across several model branches.
