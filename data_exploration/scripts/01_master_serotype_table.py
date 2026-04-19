"""Master per-serotype table.

For each K-type (and O-type) in the universe of (training UNION PHL):
- Train/test split counts (unique MD5s with this serotype as primary)
- PHL stats (positive pairs, phages, hosts)
- Model accuracy on test split (per-K recall)

Outputs:
    data_exploration/output/master_K_table.csv
    data_exploration/output/master_O_table.csv
"""

import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import (
    REPO_ROOT, OUTPUT_DIR, best_experiment_dir,
    load_training_data, load_phl_data,
    load_predictor_for_inference, load_test_embeddings,
    predict_test_split,
)


def primary_class_counts(md5_list, labels, classes, md5_subset):
    """Count primary (argmax) labels for MD5s in subset."""
    md5_to_idx = {m: i for i, m in enumerate(md5_list)}
    counts = Counter()
    for m in md5_subset:
        if m not in md5_to_idx:
            continue
        row = labels[md5_to_idx[m]]
        if row.sum() == 0:
            continue
        counts[classes[int(row.argmax())]] += 1
    return counts


def phl_stats_per_class(phl_pairs, head='k'):
    """For each K (or O) type in PHL, count positive pairs / phages / hosts."""
    field = 'host_K' if head == 'k' else 'host_O'
    pos_pairs = [p for p in phl_pairs if p['label'] == 1]

    pairs_per = Counter(p[field] for p in pos_pairs)
    phages_per = defaultdict(set)
    hosts_per = defaultdict(set)
    for p in pos_pairs:
        phages_per[p[field]].add(p['phage_id'])
        hosts_per[p[field]].add(p['host_id'])

    return pairs_per, phages_per, hosts_per


def per_class_accuracy(test_results):
    """Compute per-class top-1 accuracy and median rank from test predictions."""
    by_class = defaultdict(list)  # true_class -> list of (correct, true_rank)
    for r in test_results:
        correct = (r['pred_class'] == r['true_class'])
        by_class[r['true_class']].append((correct, r['true_rank']))

    stats = {}
    for cls, recs in by_class.items():
        n = len(recs)
        n_correct = sum(c for c, _ in recs)
        ranks = [rk for _, rk in recs]
        stats[cls] = {
            'n_test': n,
            'top1_acc': n_correct / n if n > 0 else float('nan'),
            'median_rank': float(np.median(ranks)),
        }
    return stats


def build_master_table(td, splits, phl_pairs, test_results, head='k'):
    """Build the master table for one head (K or O)."""
    labels = td.k_labels if head == 'k' else td.o_labels
    classes = td.k_classes if head == 'k' else td.o_classes
    trained_set = set(classes)

    # Counts on training and test splits (by primary K)
    train_counts = primary_class_counts(td.md5_list, labels, classes,
                                        splits.get('train', []))
    test_counts = primary_class_counts(td.md5_list, labels, classes,
                                       splits.get('test', []))

    # Counts on PHL
    phl_pairs_per, phl_phages_per, phl_hosts_per = phl_stats_per_class(
        phl_pairs, head=head)

    # Per-class test-split accuracy from inference
    acc_stats = per_class_accuracy(test_results)

    # Universe of classes: trained UNION PHL (so we see what's missing)
    universe = sorted(set(trained_set) | set(phl_pairs_per.keys()))

    rows = []
    for c in universe:
        in_trained = c in trained_set
        rows.append({
            f'{head}_type': c,
            'in_trained_classes': 'Y' if in_trained else 'N',
            'n_train_md5s': train_counts.get(c, 0),
            'n_test_md5s': test_counts.get(c, 0),
            'n_phl_pos_pairs': phl_pairs_per.get(c, 0),
            'n_phl_phages': len(phl_phages_per.get(c, set())),
            'n_phl_hosts': len(phl_hosts_per.get(c, set())),
            'test_top1_acc': (acc_stats[c]['top1_acc']
                               if c in acc_stats else ''),
            'median_rank': (acc_stats[c]['median_rank']
                             if c in acc_stats else ''),
        })

    # Sort: trained first, then by PHL pair count desc, then train count desc
    rows.sort(key=lambda r: (
        r['in_trained_classes'] != 'Y',
        -r['n_phl_pos_pairs'],
        -r['n_train_md5s'],
    ))
    return rows


def write_csv(rows, path):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    exp_dir = best_experiment_dir()
    print(f'Using experiment: {os.path.basename(exp_dir)}')

    print('Loading training data...')
    td, splits_k, splits_o = load_training_data(exp_dir)

    print('Loading PHL data...')
    phl_pairs = load_phl_data()

    print('Loading predictor...')
    predictor = load_predictor_for_inference(exp_dir)

    print('Loading test embeddings...')
    emb_dict = load_test_embeddings(exp_dir, td, splits_k, splits_o)

    print('Running inference on test split (K head)...')
    test_results_k = predict_test_split(predictor, td, splits_k, emb_dict, head='k')
    print(f'  {len(test_results_k)} test predictions')

    print('Running inference on test split (O head)...')
    test_results_o = predict_test_split(predictor, td, splits_o, emb_dict, head='o')
    print(f'  {len(test_results_o)} test predictions')

    print('Building K master table...')
    k_rows = build_master_table(td, splits_k, phl_pairs, test_results_k, head='k')

    print('Building O master table...')
    o_rows = build_master_table(td, splits_o, phl_pairs, test_results_o, head='o')

    k_path = os.path.join(OUTPUT_DIR, 'master_K_table.csv')
    o_path = os.path.join(OUTPUT_DIR, 'master_O_table.csv')
    write_csv(k_rows, k_path)
    write_csv(o_rows, o_path)
    print(f'\nSaved: {k_path}  ({len(k_rows)} rows)')
    print(f'Saved: {o_path}  ({len(o_rows)} rows)')


if __name__ == '__main__':
    main()
