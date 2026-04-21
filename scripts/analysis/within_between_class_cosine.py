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


def load_training_md5_to_labels(assoc_path):
    """Return (md5_to_K, md5_to_O) dicts of sets."""
    md5_k = defaultdict(set)
    md5_o = defaultdict(set)
    with open(assoc_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            md5 = row.get('protein_md5', '').strip()
            if not md5:
                continue
            k = row.get('host_K', '').strip()
            o = row.get('host_O', '').strip()
            if k and k not in NULL_LABELS:
                md5_k[md5].add(k)
            if o and o not in NULL_LABELS:
                md5_o[md5].add(o)
    return dict(md5_k), dict(md5_o)


def load_training_md5_to_k(assoc_path):
    """Back-compat wrapper: K only."""
    md5_k, _ = load_training_md5_to_labels(assoc_path)
    return md5_k


def primary_k(ks):
    return sorted(ks)[0]


def cos_pairs(mat, pairs):
    a = mat[pairs[:, 0]]
    b = mat[pairs[:, 1]]
    return np.einsum('ij,ij->i', a, b)  # both already L2-normed


def _sample_within_between(md5s, vecs, class_of_idx, rng,
                            within_n, between_n):
    """Sample within-class and between-class cosine pairs. Returns (within, between) arrays."""
    class_to_idx = defaultdict(list)
    for i, c in enumerate(class_of_idx):
        if c is None:
            continue
        class_to_idx[c].append(i)

    eligible = [c for c, ix in class_to_idx.items() if len(ix) >= 2]
    if not eligible:
        return np.array([]), np.array([])

    weights = np.array([len(class_to_idx[c]) for c in eligible], dtype=float)
    weights = weights / weights.sum()
    per_class_target = np.maximum((weights * within_n).astype(int), 1)

    within_pairs = []
    for c, target in zip(eligible, per_class_target):
        ixs = class_to_idx[c]
        n = len(ixs)
        max_pairs = n * (n - 1) // 2
        target = min(target, max_pairs)
        seen = set()
        tries = 0
        while len(seen) < target and tries < target * 5:
            a, b = rng.choice(ixs, size=2, replace=False)
            if a > b:
                a, b = b, a
            seen.add((a, b))
            tries += 1
        within_pairs.extend(seen)
    within_pairs = np.array(within_pairs) if within_pairs else np.zeros((0, 2), dtype=int)

    labeled_idx = np.array([i for i, c in enumerate(class_of_idx) if c is not None])
    between_pairs = []
    while len(between_pairs) < between_n:
        a = rng.choice(labeled_idx, size=between_n * 2)
        b = rng.choice(labeled_idx, size=between_n * 2)
        keep = [(i, j) for i, j in zip(a, b)
                if i != j and class_of_idx[i] != class_of_idx[j]]
        between_pairs.extend(keep[: between_n - len(between_pairs)])
    between_pairs = np.array(between_pairs[:between_n])

    within_cos = cos_pairs(vecs, within_pairs) if len(within_pairs) else np.array([])
    between_cos = cos_pairs(vecs, between_pairs) if len(between_pairs) else np.array([])
    return within_cos, between_cos


def _summarize(within_cos, between_cos, tag):
    """Produce a stats dict for one label-type."""
    if len(within_cos) == 0 or len(between_cos) == 0:
        return {f'{tag}_{k}': '' for k in (
            'n_within_pairs', 'n_between_pairs',
            'within_mean', 'within_median', 'within_std',
            'between_mean', 'between_median', 'between_std',
            'gap_mean', 'gap_median', 'cohens_d')}
    gap_mean = within_cos.mean() - between_cos.mean()
    gap_median = float(np.median(within_cos) - np.median(between_cos))
    pooled = np.sqrt((within_cos.var() + between_cos.var()) / 2)
    d = gap_mean / pooled if pooled > 0 else 0.0
    return {
        f'{tag}_n_within_pairs': len(within_cos),
        f'{tag}_n_between_pairs': len(between_cos),
        f'{tag}_within_mean': f'{within_cos.mean():.4f}',
        f'{tag}_within_median': f'{np.median(within_cos):.4f}',
        f'{tag}_within_std': f'{within_cos.std():.4f}',
        f'{tag}_between_mean': f'{between_cos.mean():.4f}',
        f'{tag}_between_median': f'{np.median(between_cos):.4f}',
        f'{tag}_between_std': f'{between_cos.std():.4f}',
        f'{tag}_gap_mean': f'{gap_mean:+.4f}',
        f'{tag}_gap_median': f'{gap_median:+.4f}',
        f'{tag}_cohens_d': f'{d:+.3f}',
    }


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
    p.add_argument('--label-type', choices=['k', 'o', 'both'], default='both',
                   help='Which serotype label to analyze (default: both K and O).')
    args = p.parse_args()

    label = args.label or os.path.basename(args.train_emb).replace('.npz', '')
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f'Representation: {label}')
    print(f'Loading md5 -> K/O labels...')
    md5_k, md5_o = load_training_md5_to_labels(args.assoc_map)
    labelled_md5s = set(md5_k.keys()) | set(md5_o.keys())

    print('Loading embeddings (restrict to any-labelled)...')
    emb = load_embeddings(args.train_emb, md5_filter=labelled_md5s)
    md5s = list(emb.keys())
    dim = next(iter(emb.values())).shape[-1]
    print(f'  {len(md5s):,} embedded MD5s, dim={dim}')

    vecs = np.stack([emb[m] for m in md5s]).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    zeros = int((norms == 0).sum())
    if zeros:
        print(f'  WARNING: {zeros} rows had zero norm — replacing with epsilon')
        norms = np.where(norms == 0, 1.0, norms)
    vecs /= norms

    # Per-md5 primary K and O (None if that label is missing)
    k_of_idx = [primary_k(md5_k[m]) if m in md5_k else None for m in md5s]
    o_of_idx = [primary_k(md5_o[m]) if m in md5_o else None for m in md5s]
    n_k_classes = len({c for c in k_of_idx if c is not None})
    n_o_classes = len({c for c in o_of_idx if c is not None})
    print(f'  K-labelled MD5s: {sum(c is not None for c in k_of_idx):,} '
          f'across {n_k_classes} K-types')
    print(f'  O-labelled MD5s: {sum(c is not None for c in o_of_idx):,} '
          f'across {n_o_classes} O-types')

    row = {'label': label, 'dim': dim, 'n_md5s': len(md5s),
           'n_k_classes': n_k_classes, 'n_o_classes': n_o_classes}

    if args.label_type in ('k', 'both'):
        print('\nSampling K-type pairs...')
        w, b = _sample_within_between(
            md5s, vecs, k_of_idx, rng, args.within_sample, args.between_sample)
        print(f"  K: within mean {w.mean():.4f}, between mean {b.mean():.4f}, "
              f"gap {w.mean() - b.mean():+.4f}")
        row.update(_summarize(w, b, 'k'))

    if args.label_type in ('o', 'both'):
        print('\nSampling O-type pairs...')
        w, b = _sample_within_between(
            md5s, vecs, o_of_idx, rng, args.within_sample, args.between_sample)
        print(f"  O: within mean {w.mean():.4f}, between mean {b.mean():.4f}, "
              f"gap {w.mean() - b.mean():+.4f}")
        row.update(_summarize(w, b, 'o'))

    # Append to TSV (replacing any prior row with same label)
    summary_path = os.path.join(args.out_dir, 'within_between_summary.tsv')
    rows = []
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        rows = [r for r in rows if r.get('label') != label]
    # Backfill any missing columns so every row has same columns
    all_cols = set()
    for r in rows + [row]:
        all_cols.update(r.keys())
    ordered = ['label', 'dim', 'n_md5s', 'n_k_classes', 'n_o_classes'] + \
              [c for c in sorted(all_cols) if c not in
               ('label', 'dim', 'n_md5s', 'n_k_classes', 'n_o_classes')]
    for r in rows:
        for c in ordered:
            r.setdefault(c, '')
    rows.append({c: row.get(c, '') for c in ordered})
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ordered, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nAppended to {summary_path}')


if __name__ == '__main__':
    main()
