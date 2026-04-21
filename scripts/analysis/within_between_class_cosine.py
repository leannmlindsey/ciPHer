"""Compute within-class vs between-class cosine similarity stats for a
training embedding, using K-type labels.

If embeddings cluster well by K-type, within-class cosine >> between-class
cosine. The *gap* between them is a direct proxy for how usable the
representation is for serotype classification.

Sampling:
  - Only K-labelled training MD5s are loaded (same as phl_neighbor_labels).
  - For each K-type with at least 2 samples, compute within-class pairwise
    cosines on up to `--within-sample` randomly-chosen pairs.
  - Between-class: sample `--between-sample` random pairs of MD5s from
    different K-types.
  - Reports mean/median/std and the 'class-separation gap'.

Usage:
  python scripts/analysis/within_between_class_cosine.py
  python scripts/analysis/within_between_class_cosine.py \\
      --train-emb /path/to/prott5_md5.npz --label prott5_mean
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from cipher.data.embeddings import load_embeddings


NULL_LABELS = {'null', 'N/A', 'Unknown', '', 'NA', 'n/a', 'None'}


def load_training_md5_to_k(assoc_path):
    md5_k = defaultdict(set)
    with open(assoc_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            md5 = row.get('protein_md5', '').strip()
            k = row.get('host_K', '').strip()
            if not md5 or not k or k in NULL_LABELS:
                continue
            md5_k[md5].add(k)
    return dict(md5_k)


def primary_k(ks):
    return sorted(ks)[0]


def cos_pairs(mat, pairs):
    a = mat[pairs[:, 0]]
    b = mat[pairs[:, 1]]
    return np.einsum('ij,ij->i', a, b)  # both already L2-normed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-emb',
                   default='data/training_data/embeddings/esm2_650m_md5.npz')
    p.add_argument('--assoc-map',
                   default='data/training_data/metadata/host_phage_protein_map.tsv')
    p.add_argument('--label', default=None,
                   help='Label for the representation (default: basename of --train-emb)')
    p.add_argument('--within-sample', type=int, default=5000)
    p.add_argument('--between-sample', type=int, default=20000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', default='results/analysis')
    args = p.parse_args()

    label = args.label or os.path.basename(args.train_emb).replace('.npz', '')
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f'Representation: {label}')
    print(f'Loading md5 -> K-type labels...')
    md5_k = load_training_md5_to_k(args.assoc_map)

    print('Loading embeddings (restrict to labelled)...')
    emb = load_embeddings(args.train_emb, md5_filter=set(md5_k.keys()))
    md5s = list(emb.keys())
    dim = next(iter(emb.values())).shape[-1]
    print(f'  {len(md5s):,} embedded + K-labelled MD5s, dim={dim}')

    vecs = np.stack([emb[m] for m in md5s]).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Avoid divide-by-zero (kmer can have all-zero rows)
    zeros = int((norms == 0).sum())
    if zeros:
        print(f'  WARNING: {zeros} rows had zero norm — replacing with epsilon')
        norms = np.where(norms == 0, 1.0, norms)
    vecs /= norms

    # Assign primary K per md5
    idx_to_primary_k = [primary_k(md5_k[md5]) for md5 in md5s]
    class_to_idx = defaultdict(list)
    for i, k in enumerate(idx_to_primary_k):
        class_to_idx[k].append(i)

    # === within-class pairs ===
    within_pairs = []
    eligible_classes = [k for k, ix in class_to_idx.items() if len(ix) >= 2]
    # Sample proportional to class size^2 so bigger classes contribute more,
    # but cap at within-sample total.
    weights = np.array([len(class_to_idx[k]) for k in eligible_classes], dtype=float)
    weights = weights / weights.sum()
    per_class_target = (weights * args.within_sample).astype(int)
    per_class_target = np.maximum(per_class_target, 1)

    for k, target in zip(eligible_classes, per_class_target):
        ixs = class_to_idx[k]
        n = len(ixs)
        max_pairs = n * (n - 1) // 2
        target = min(target, max_pairs)
        # Sample distinct pairs
        seen = set()
        tries = 0
        while len(seen) < target and tries < target * 5:
            a, b = rng.choice(ixs, size=2, replace=False)
            if a > b:
                a, b = b, a
            seen.add((a, b))
            tries += 1
        within_pairs.extend(seen)
    within_pairs = np.array(within_pairs)

    # === between-class pairs ===
    all_ix = np.arange(len(md5s))
    between_pairs = []
    need = args.between_sample
    while len(between_pairs) < need:
        a = rng.choice(all_ix, size=need * 2)
        b = rng.choice(all_ix, size=need * 2)
        keep = [(i, j) for i, j in zip(a, b)
                if i != j and idx_to_primary_k[i] != idx_to_primary_k[j]]
        between_pairs.extend(keep[: need - len(between_pairs)])
    between_pairs = np.array(between_pairs[:need])

    # === compute cosines ===
    within_cos = cos_pairs(vecs, within_pairs)
    between_cos = cos_pairs(vecs, between_pairs)

    # === report ===
    print(f'\n=== {label} ===')
    print(f'  n within-class pairs:  {len(within_cos):,}')
    print(f'    mean:   {within_cos.mean():.4f}')
    print(f'    median: {np.median(within_cos):.4f}')
    print(f'    std:    {within_cos.std():.4f}')
    print(f'  n between-class pairs: {len(between_cos):,}')
    print(f'    mean:   {between_cos.mean():.4f}')
    print(f'    median: {np.median(between_cos):.4f}')
    print(f'    std:    {between_cos.std():.4f}')
    gap_mean = within_cos.mean() - between_cos.mean()
    gap_median = np.median(within_cos) - np.median(between_cos)
    print(f'\n  CLASS-SEPARATION GAP')
    print(f'    mean:   {gap_mean:+.4f}')
    print(f'    median: {gap_median:+.4f}')
    # Effect size (Cohen's d)
    pooled = np.sqrt((within_cos.var() + between_cos.var()) / 2)
    d = gap_mean / pooled if pooled > 0 else 0.0
    print(f"    Cohen's d: {d:+.3f}  (larger = better class separation)")

    # Save per-representation summary TSV
    summary_path = os.path.join(args.out_dir, 'within_between_summary.tsv')
    new_row = {
        'label': label,
        'dim': dim,
        'n_md5s': len(md5s),
        'within_mean': f'{within_cos.mean():.4f}',
        'within_median': f'{np.median(within_cos):.4f}',
        'within_std': f'{within_cos.std():.4f}',
        'between_mean': f'{between_cos.mean():.4f}',
        'between_median': f'{np.median(between_cos):.4f}',
        'between_std': f'{between_cos.std():.4f}',
        'gap_mean': f'{gap_mean:+.4f}',
        'gap_median': f'{gap_median:+.4f}',
        'cohens_d': f'{d:+.3f}',
    }
    # Append or create
    rows = []
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        # Replace any existing row with same label
        rows = [r for r in rows if r.get('label') != label]
    rows.append(new_row)
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(new_row.keys()), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nAppended to {summary_path}')


if __name__ == '__main__':
    main()
