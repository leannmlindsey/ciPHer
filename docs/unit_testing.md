# Unit Testing Guide

## Overview

Unit tests verify that individual functions work correctly with known inputs.
Tests live in `tests/` and are run with `pytest`.

## Running Tests

```bash
# Run all tests
pytest

# Run a specific file
pytest tests/test_metrics.py

# Run a specific test
pytest tests/test_metrics.py::test_hr_at_k_all_hits

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Directory Structure

```
tests/
├── test_data/           # Tiny fixture files (3-5 rows each)
│   ├── glycan_binders.tsv
│   ├── association_map.tsv
│   ├── interaction_matrix.tsv
│   ├── phage_protein_mapping.csv
│   ├── serotypes.tsv
│   └── proteins.faa
├── test_metrics.py      # cipher.evaluation.metrics
├── test_serotypes.py    # cipher.data.serotypes
├── test_splits.py       # cipher.data.splits
├── test_training.py     # cipher.data.training
├── test_embeddings.py   # cipher.data.embeddings
├── test_interactions.py # cipher.data.interactions
├── test_proteins.py     # cipher.data.proteins
└── test_ranking.py      # cipher.evaluation.ranking
```

## What to Test

### Tier 1: Pure functions (no fixtures needed)
- `metrics.py` -- hr_at_k, mrr, hr_curve with known rank lists
- `serotypes.py` -- normalize_k_type, is_null edge cases
- `splits.py` -- valid partitions, reproducibility, edge cases
- `training.py` -- filtering logic, label vectors, downsampling

### Tier 2: Data loading (needs small fixture files)
- `embeddings.py` -- load with/without md5_filter
- `interactions.py` -- load_interaction_matrix, load_phage_protein_mapping
- `proteins.py` -- load_fasta, compute_md5

### Tier 3: Integration (needs mock Predictor)
- `ranking.py` -- rank_hosts, rank_phages, evaluate_rankings
- `runner.py` -- end-to-end with a dummy experiment

### What NOT to test
- Model training (too slow, hardware-dependent)
- Visualization (subjective)
- Thin wrappers that just call library functions

## Writing a Test

```python
import pytest
from cipher.evaluation.metrics import hr_at_k, mrr

def test_hr_at_k_all_hits():
    """All ranks are 1, so HR@1 should be 1.0."""
    assert hr_at_k([1, 1, 1], k=1) == 1.0

def test_hr_at_k_no_hits():
    """All ranks are beyond k, so HR@k should be 0.0."""
    assert hr_at_k([5, 10, 20], k=3) == 0.0

def test_hr_at_k_partial():
    """2 of 4 ranks are <= 3."""
    assert hr_at_k([1, 3, 5, 10], k=3) == pytest.approx(0.5)

def test_mrr():
    """MRR = mean(1/rank)."""
    assert mrr([1, 2, 4]) == pytest.approx((1 + 0.5 + 0.25) / 3)

def test_empty_ranks():
    """Empty input should return 0."""
    assert hr_at_k([], k=5) == 0.0
    assert mrr([]) == 0.0
```

## Conventions

- Test file names: `test_{module}.py`
- Test function names: `test_{what_it_tests}`
- Use `pytest.approx()` for floating-point comparisons
- Use `tmp_path` fixture for tests that write files
- Use `test_data/` fixtures for tests that read files
- Each test should be independent (no shared mutable state)

## When to Run Tests

- Before committing changes to `src/cipher/`
- After modifying any shared library function
- Before opening a PR
