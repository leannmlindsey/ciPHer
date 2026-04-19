# ciPHer Onboarding Guide

**ciPHer** = Capsule Interaction Prediction of Phage–Host Relationships

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
- Labels are **associations** (this protein was found in a phage near a host with K-type X)
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

### Option A: Given a phage, rank hosts
"I have this phage, which hosts should I test it against?"
- For each phage: score all candidate hosts, rank by score, check where true positives land

### Option B: Given a host, rank phages
"I have this patient's bacterium, which phage should I use?"
- For each host: score all candidate phages, rank by score, check where true positives land

Both report HR@k (hit rate at k) for k=1..20 and MRR (mean reciprocal rank).

## Experiments (`experiments/`)

Each experiment gets its own directory:
```
experiments/{experiment_id}/
├── config.yaml        # Full configuration
├── experiment.json    # Metadata (architecture, features, seed, etc.)
├── model.py           # Model definition
├── train.py           # Training loop
├── predict.py         # REQUIRED: implements Predictor interface
└── results/           # Output from training + evaluation
```

## Known Issues and Key Findings (from prior work)

### 1. ESM-2 embeddings have weak K-type signal
- Same-K vs different-K cosine distance gap: only 0.003 (650M) or 0.01 (3B)
- AA 3-mer features have 0.068 gap (22x better)
- SpikeHunter paper showed sequence clusters DO map to 1-4 K-types
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

### 5. Best models so far (pair-level K HR@1)

| Model | CHEN | GOROD | UCSD | PBIP | PHL | Mean |
|---|---|---|---|---|---|---|
| v1 O-fallback | 0.288 | 1.000 | 0.091 | 0.046 | 0.120 | 0.309 |
| v4 downsample | 0.197 | 1.000 | 0.062 | 0.064 | 0.143 | 0.293 |
| Light Attention (4-seg) | 0.197 | 0.333 | 0.182 | 0.022 | 0.179 | 0.183 |
| XGBoost (3-mer) | 0.203 | 0.000 | 0.000 | 0.127 | 0.080 | 0.082 |
| TropiGAT (competitor) | 0.000 | 0.000 | 0.005 | 0.514* | 0.007 | 0.105 |

*K47 bias — not genuine learning.

### 6. Bio-motivated ranking results (v4 downsample)

| Mode | HR@1 | HR@5 | HR@10 | HR@20 |
|---|---|---|---|---|
| Rank hosts given phage | 0.091 | 0.359 | 0.466 | 0.644 |
| Rank phages given host | 0.163 | 0.635 | 0.810 | 0.898 |

## What Needs To Be Done Next

1. **Evaluation runner** — `cipher-evaluate` CLI that loads any experiment's Predictor and runs both ranking modes
2. **First experiment template** — port v4 downsample AttentionMLP to `experiments/` format
3. **Visualization module** — `cipher.visualization` for HR@k curves, comparison plots
4. **Initial git commit** — everything is ready but uncommitted
5. **More experiments** — k-mer models, light attention, hybrid features, multi-label

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

- The `cipher/` package is shared stable code — avoid modifying unless adding new shared functionality
- Each agent works in `experiments/{their_model}/` — this is where conflicts are unlikely
- Always implement `predict.py` with the `Predictor` interface so evaluation works automatically
