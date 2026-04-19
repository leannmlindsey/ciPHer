"""Per-serotype analysis of a trained model's test-split performance.

Produces per-K-type and per-O-type statistics (top-k accuracy, mean rank,
and training frequency) that can be used to identify which serotypes the
model handles well vs poorly, and how this correlates with training data
distribution.

Output format (saved to {experiment_dir}/analysis/per_serotype_test.json):
{
    "k": {class_name: {total, top1, top5, top10, mean_rank, median_rank, train_freq}},
    "o": {class_name: {...}}
}

- `total` = number of test-split proteins with this class as true label
- `topN` = count (not fraction) of those proteins ranked <= N by the model
- `train_freq` = number of training-split proteins with this class
- Ranks are 1-based (rank 1 = correct prediction)
"""

import json
import os
from collections import Counter, defaultdict

import numpy as np


def compute_per_serotype_test(experiment_dir, max_k=10, save=True, verbose=True):
    """Run the trained K and O models on the test split and compute per-class metrics.

    Args:
        experiment_dir: path to experiment directory
        max_k: compute top-k accuracy for k=1..max_k
        save: if True, save JSON to {experiment_dir}/analysis/per_serotype_test.json
        verbose: print progress

    Returns:
        dict with 'k' and 'o' keys, each mapping class_name -> stats dict
    """
    from cipher.data.training import TrainingData
    from cipher.data.embeddings import load_embeddings
    from cipher.evaluation.runner import load_predictor

    def log(msg):
        if verbose:
            print(msg)

    log(f'Loading experiment: {experiment_dir}')

    # Load training data + splits + label encoders
    td = TrainingData.load(experiment_dir)

    # Splits are now per-head (splits_k.json, splits_o.json).
    # Fall back to legacy splits.json for older experiments.
    splits_k_path = os.path.join(experiment_dir, 'splits_k.json')
    splits_o_path = os.path.join(experiment_dir, 'splits_o.json')
    legacy_path = os.path.join(experiment_dir, 'splits.json')

    if os.path.exists(splits_k_path) and os.path.exists(splits_o_path):
        with open(splits_k_path) as f:
            splits_k = json.load(f)
        with open(splits_o_path) as f:
            splits_o = json.load(f)
    elif os.path.exists(legacy_path):
        with open(legacy_path) as f:
            legacy = json.load(f)
        splits_k = legacy
        splits_o = legacy
    else:
        raise FileNotFoundError(
            f'No splits file found in {experiment_dir} '
            f'(expected splits_k.json + splits_o.json or legacy splits.json)')

    train_k_md5s = set(splits_k['train'])
    test_k_md5s = set(splits_k['test'])
    train_o_md5s = set(splits_o['train'])
    test_o_md5s = set(splits_o['test'])
    md5_to_idx = {m: i for i, m in enumerate(td.md5_list)}

    # Training frequency per class (primary label — the one with highest count)
    log('Computing training frequencies...')
    k_train_freq = _count_primary_labels(td.md5_list, td.k_labels, td.k_classes,
                                          train_k_md5s, md5_to_idx)
    o_train_freq = _count_primary_labels(td.md5_list, td.o_labels, td.o_classes,
                                          train_o_md5s, md5_to_idx)

    # Load embeddings (config tells us where)
    with open(os.path.join(experiment_dir, 'config.yaml')) as f:
        import yaml
        config = yaml.safe_load(f)
    emb_file = config.get('data', {}).get(
        'embedding_file', 'data/training_data/embeddings/esm2_650m_md5.npz')

    # Resolve relative to project root
    if not os.path.isabs(emb_file):
        project_root = _find_project_root(experiment_dir)
        if project_root:
            emb_file = os.path.join(project_root, emb_file)

    log(f'Loading embeddings (filtered to test sets)...')
    needed = test_k_md5s | test_o_md5s
    emb_dict = load_embeddings(emb_file, md5_filter=needed)
    log(f'  {len(emb_dict)} test embeddings loaded')

    # Load predictor
    log('Loading trained models...')
    predictor = load_predictor(experiment_dir)

    # Run inference on each head's own test set
    log('Running inference on test splits...')
    k_stats = _compute_serotype_stats(
        test_k_md5s, md5_to_idx, td.k_labels, td.k_classes,
        emb_dict, predictor, 'k', max_k)
    o_stats = _compute_serotype_stats(
        test_o_md5s, md5_to_idx, td.o_labels, td.o_classes,
        emb_dict, predictor, 'o', max_k)

    # Attach training frequencies
    for cls, freq in k_train_freq.items():
        if cls in k_stats:
            k_stats[cls]['train_freq'] = freq
    for cls in k_stats:
        k_stats[cls].setdefault('train_freq', 0)

    for cls, freq in o_train_freq.items():
        if cls in o_stats:
            o_stats[cls]['train_freq'] = freq
    for cls in o_stats:
        o_stats[cls].setdefault('train_freq', 0)

    results = {'k': k_stats, 'o': o_stats}

    if save:
        out_dir = os.path.join(experiment_dir, 'analysis')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'per_serotype_test.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        log(f'Saved: {out_path}')

    return results


def load_per_serotype(experiment_dir):
    """Load a previously computed per_serotype_test.json."""
    path = os.path.join(experiment_dir, 'analysis', 'per_serotype_test.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'No per_serotype_test.json at {path}. '
            f'Run compute_per_serotype_test() first.')
    with open(path) as f:
        return json.load(f)


# ========================================================================
# Internals
# ========================================================================

def _find_project_root(start):
    path = os.path.abspath(start)
    for _ in range(10):
        if os.path.exists(os.path.join(path, 'setup.py')):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return None


def _count_primary_labels(md5_list, labels, classes, md5_subset, md5_to_idx):
    """Count primary (argmax) labels for MD5s in md5_subset.

    Returns: {class_name: count}
    """
    counts = Counter()
    for m in md5_subset:
        if m not in md5_to_idx:
            continue
        idx = md5_to_idx[m]
        row = labels[idx]
        if row.sum() == 0:
            continue
        primary = classes[int(row.argmax())]
        counts[primary] += 1
    return dict(counts)


def _compute_serotype_stats(test_md5s, md5_to_idx, labels, classes,
                             emb_dict, predictor, head, max_k):
    """Compute per-class stats for one head (k or o).

    Returns: {class_name: {total, top1..top{max_k}, mean_rank, median_rank, ranks}}
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    ranks_per_class = defaultdict(list)

    for m in test_md5s:
        if m not in md5_to_idx or m not in emb_dict:
            continue
        idx = md5_to_idx[m]
        row = labels[idx]
        if row.sum() == 0:
            continue  # null-labeled sample

        true_cls = classes[int(row.argmax())]
        preds = predictor.predict_protein(emb_dict[m])
        probs = preds['k_probs'] if head == 'k' else preds['o_probs']

        # Rank of true class: sort probs descending, find index of true class
        if true_cls not in probs:
            continue

        sorted_items = sorted(probs.items(), key=lambda x: -x[1])
        rank = next(i for i, (c, _) in enumerate(sorted_items, start=1) if c == true_cls)
        ranks_per_class[true_cls].append(rank)

    stats = {}
    for cls, ranks in ranks_per_class.items():
        ranks_arr = np.array(ranks)
        entry = {
            'total': len(ranks),
            'mean_rank': float(ranks_arr.mean()),
            'median_rank': float(np.median(ranks_arr)),
        }
        for k in range(1, max_k + 1):
            entry[f'top{k}'] = int((ranks_arr <= k).sum())
        stats[cls] = entry

    return stats
