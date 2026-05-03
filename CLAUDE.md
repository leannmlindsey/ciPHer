# ciPHer Development Guide

## What is this?
ciPHer predicts phage-host interactions for Klebsiella bacteria. See
`ONBOARDING.md` for biology background, data docs, and known issues.

## Architecture
- `src/cipher/` -- Shared library (data loading, evaluation, visualization). Installed via `pip install -e .`
- `models/` -- One directory per model architecture. Each implements model.py, train.py, predict.py.
- `experiments/` -- One directory per run. Contains config.yaml and results. No code.
- `data/` -- All data (training + validation). Not in git.
- `tests/` -- Unit tests. Run with `pytest`.

## Key Rule: Standard Predictor Interface
Every model MUST implement `cipher.evaluation.Predictor`:
- `predict_protein(embedding)` -> `{'k_probs': {...}, 'o_probs': {...}}`
- `embedding_type` property -> string identifying what input features the model needs
- `score_pair()` can be overridden for custom scoring (default: max-over-proteins, max-over-K/O)

## Running experiments
```bash
# Train a model
cipher-train --model attention_mlp --protein_set tsp_only --lr 1e-4

# Evaluate
cipher-evaluate experiments/attention_mlp/{run_name}/

# Run tests
pytest
```

## Conventions
- Always save scripts to files, never run complex code inline
- **Never give the user `python3 -c "..."` one-liners** — multiline shell-quoted Python is fragile to copy-paste. Save to a small file under `scripts/analysis/` (or `/tmp` for one-off diagnostics) and have the user run `python <path>` instead. Same rule applies to `<<EOF` heredocs.
- Every experiment must have config.yaml with full metadata
- Use the shared `cipher.data` module for loading -- don't duplicate parsing code
- Save all results as JSON with standardized keys
- Run `pytest` before committing changes to `src/cipher/`
