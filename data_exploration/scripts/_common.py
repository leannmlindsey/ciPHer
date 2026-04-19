"""Shared utilities for data exploration scripts.

Loads the trained model, training data, and PHL validation data once.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from glob import glob

import numpy as np


# ============================================================
# Path resolution
# ============================================================

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUT_DIR = os.path.join(REPO_ROOT, 'data_exploration', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def best_experiment_dir():
    """Return the path to the experiment we're using as the reference model.

    Picks the most recently trained experiment with both K and O models
    (sorted by timestamp suffix in the directory name).
    Override with CIPHER_EXPERIMENT_DIR env var.
    """
    override = os.environ.get('CIPHER_EXPERIMENT_DIR')
    if override:
        return os.path.abspath(override)

    exp_root = os.path.join(REPO_ROOT, 'experiments', 'attention_mlp')
    candidates = []
    needed = [
        'model_k/best_model.pt', 'model_k/config.json',
        'model_o/best_model.pt', 'model_o/config.json',
        'config.yaml', 'training_data.npz', 'label_encoders.json',
    ]
    for d in glob(os.path.join(exp_root, '*')):
        if not os.path.isdir(d):
            continue
        if all(os.path.isfile(os.path.join(d, p)) for p in needed):
            mtime = os.path.getmtime(d)
            candidates.append((mtime, d))
    if not candidates:
        raise FileNotFoundError(
            'No fully-trained experiment found in experiments/attention_mlp/. '
            'Train one first.')
    candidates.sort()
    return candidates[-1][1]  # most recent fully-trained


# ============================================================
# Data loading
# ============================================================

def load_training_data(experiment_dir):
    """Load TrainingData and splits."""
    from cipher.data.training import TrainingData
    td = TrainingData.load(experiment_dir)

    splits_k_path = os.path.join(experiment_dir, 'splits_k.json')
    if os.path.exists(splits_k_path):
        with open(splits_k_path) as f:
            splits_k = json.load(f)
        with open(os.path.join(experiment_dir, 'splits_o.json')) as f:
            splits_o = json.load(f)
    else:
        # Legacy: single split
        with open(os.path.join(experiment_dir, 'splits.json')) as f:
            legacy = json.load(f)
        splits_k = legacy
        splits_o = legacy

    return td, splits_k, splits_o


def load_raw_associations(experiment_dir):
    """Re-build the raw per-MD5 K-type Counters from the association map.

    This is needed for the focal K-type deep-dive (specificity profile)
    because TrainingData only stores post-reduction labels (e.g., one-hot
    for single_label).

    Re-uses the same filtering as the experiment was trained with.
    """
    from cipher.data.training import (
        TrainingConfig, _filter_proteins, _build_md5_associations,
        _filter_non_specific,
    )
    from cipher.data.interactions import load_training_map
    from cipher.data.proteins import load_glycan_binders
    import yaml

    with open(os.path.join(experiment_dir, 'config.yaml')) as f:
        config_dict = yaml.safe_load(f)
    exp = config_dict.get('experiment', {})
    cfg = TrainingConfig.from_dict(exp)

    data_cfg = config_dict.get('data', {})
    assoc_path = data_cfg.get('association_map',
                               'data/training_data/metadata/host_phage_protein_map.tsv')
    glyc_path = data_cfg.get('glycan_binders',
                              'data/training_data/metadata/glycan_binders_custom.tsv')

    if not os.path.isabs(assoc_path):
        assoc_path = os.path.join(REPO_ROOT, assoc_path)
    if not os.path.isabs(glyc_path):
        glyc_path = os.path.join(REPO_ROOT, glyc_path)

    rows = load_training_map(assoc_path)
    glycan_dict = load_glycan_binders(glyc_path)

    all_pids = {r['protein_id'] for r in rows}
    log = lambda *_: None
    filtered_pids = _filter_proteins(all_pids, glycan_dict, cfg, log)

    md5_k_counts, md5_o_counts, md5_is_tsp = _build_md5_associations(
        rows, filtered_pids)
    _filter_non_specific(md5_k_counts, md5_o_counts, md5_is_tsp, cfg, log)

    # Note: we deliberately DON'T apply single-label reduction or downsampling
    # because we want the raw multi-K associations for specificity analysis.
    return md5_k_counts, md5_o_counts


def load_phl_data():
    """Load PhageHostLearn validation data (interaction pairs only)."""
    from cipher.data.interactions import load_interaction_pairs
    phl_dir = os.path.join(REPO_ROOT, 'data', 'validation_data',
                            'HOST_RANGE', 'PhageHostLearn')
    return load_interaction_pairs(phl_dir)


# ============================================================
# Inference helpers
# ============================================================

def load_predictor_for_inference(experiment_dir):
    """Load the trained predictor."""
    from cipher.evaluation.runner import load_predictor
    return load_predictor(experiment_dir)


def load_test_embeddings(experiment_dir, td, splits_k, splits_o):
    """Load embeddings for all MD5s in either K or O test split."""
    from cipher.data.embeddings import load_embeddings
    import yaml

    with open(os.path.join(experiment_dir, 'config.yaml')) as f:
        config_dict = yaml.safe_load(f)
    emb_file = config_dict.get('data', {}).get(
        'embedding_file',
        'data/training_data/embeddings/esm2_650m_md5.npz')
    if not os.path.isabs(emb_file):
        emb_file = os.path.join(REPO_ROOT, emb_file)

    needed = set(splits_k.get('test', [])) | set(splits_o.get('test', []))
    return load_embeddings(emb_file, md5_filter=needed)


def predict_test_split(predictor, td, splits, emb_dict, head='k'):
    """Run inference on the test split for one head.

    Returns:
        list of dicts, one per test MD5: {md5, true_class, pred_class, pred_rank, all_probs}
    """
    classes = td.k_classes if head == 'k' else td.o_classes
    labels = td.k_labels if head == 'k' else td.o_labels
    test_md5s = splits.get('test', [])
    md5_to_idx = {m: i for i, m in enumerate(td.md5_list)}

    results = []
    for md5 in test_md5s:
        if md5 not in md5_to_idx or md5 not in emb_dict:
            continue
        idx = md5_to_idx[md5]
        row = labels[idx]
        if row.sum() == 0:
            continue
        true_class = classes[int(row.argmax())]

        preds = predictor.predict_protein(emb_dict[md5])
        probs = preds['k_probs'] if head == 'k' else preds['o_probs']
        if true_class not in probs:
            continue

        sorted_items = sorted(probs.items(), key=lambda x: -x[1])
        pred_class = sorted_items[0][0]
        true_rank = next(i for i, (c, _) in enumerate(sorted_items, start=1)
                         if c == true_class)

        results.append({
            'md5': md5,
            'true_class': true_class,
            'pred_class': pred_class,
            'true_rank': true_rank,
            'top_prob': sorted_items[0][1],
        })
    return results
