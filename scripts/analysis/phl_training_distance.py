"""For each PhageHostLearn RBP, measure how similar it is to the nearest
training protein (cosine similarity in embedding space).

Question: is PHL hard because its proteins are distant from anything in
training, or hard despite having close training neighbours? If distances
are large across the board, more embeddings won't help — we need a
different training set or a different architecture.

Outputs:
  results/analysis/phl_training_distance.csv    — per-protein stats
  results/analysis/phl_training_distance.svg    — similarity histogram

Usage:
  python scripts/analysis/phl_training_distance.py
  python scripts/analysis/phl_training_distance.py \\
      --train-emb data/training_data/embeddings/esm2_650m_md5.npz \\
      --val-emb data/validation_data/embeddings/esm2_650m_md5.npz
"""

import argparse
import csv
import os

import numpy as np

from cipher.data.embeddings import load_embeddings
from cipher.data.proteins import load_fasta_md5


def cosine_sim_matrix(queries, targets):
    """Cosine similarity between each query row and every target row."""
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    t = targets / np.linalg.norm(targets, axis=1, keepdims=True)
    return q @ t.T


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-emb',
                   default='data/training_data/embeddings/esm2_650m_md5.npz')
    p.add_argument('--val-emb',
                   default='data/validation_data/embeddings/esm2_650m_md5.npz')
    p.add_argument('--val-fasta',
                   default='data/validation_data/metadata/validation_rbps_all.faa')
    p.add_argument('--phl-mapping',
                   default='data/validation_data/HOST_RANGE/PhageHostLearn/'
                           'metadata/phage_protein_mapping.csv')
    p.add_argument('--out-dir', default='results/analysis')
    p.add_argument('--topk', type=int, default=5,
                   help='Report top-K training similarities per PHL protein')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading training embeddings...')
    train = load_embeddings(args.train_emb)
    train_md5s = list(train.keys())
    train_vecs = np.stack([train[m] for m in train_md5s]).astype(np.float32)
    print(f'  {len(train_md5s):,} training proteins, dim={train_vecs.shape[1]}')

    print('Loading validation embeddings + FASTA mapping...')
    val = load_embeddings(args.val_emb)
    pid_md5 = load_fasta_md5(args.val_fasta)
    print(f'  {len(val):,} validation embeddings, {len(pid_md5):,} pid->md5')

    print(f'Reading PHL protein mapping from {args.phl_mapping}')
    phl_pids = set()
    with open(args.phl_mapping) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # column is 'protein_id' per the file header
            phl_pids.add(row['protein_id'])
    print(f'  {len(phl_pids):,} unique PHL proteins')

    # Build query matrix (PHL proteins with embeddings)
    queries, pid_order, md5_order = [], [], []
    for pid in sorted(phl_pids):
        md5 = pid_md5.get(pid)
        if md5 and md5 in val:
            queries.append(val[md5])
            pid_order.append(pid)
            md5_order.append(md5)
    queries = np.stack(queries).astype(np.float32)
    print(f'  {len(queries):,} PHL proteins resolved to embeddings')

    # Cosine similarity against all training proteins
    print('Computing cosine similarities...')
    sims = cosine_sim_matrix(queries, train_vecs)
    # Top-K per query
    top_idx = np.argsort(-sims, axis=1)[:, :args.topk]
    top_sims = np.take_along_axis(sims, top_idx, axis=1)

    # Also check: is the protein itself in the training set (identical MD5)?
    train_md5_set = set(train_md5s)
    self_in_train = [m in train_md5_set for m in md5_order]

    # Write CSV
    csv_path = os.path.join(args.out_dir, 'phl_training_distance.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['protein_id', 'md5', 'self_in_training',
                    'nearest_sim', 'mean_top5_sim', *(f'top{k+1}_sim' for k in range(args.topk))])
        for i, pid in enumerate(pid_order):
            w.writerow([
                pid, md5_order[i], self_in_train[i],
                f'{top_sims[i, 0]:.4f}',
                f'{top_sims[i].mean():.4f}',
                *(f'{v:.4f}' for v in top_sims[i]),
            ])
    print(f'Wrote {csv_path}')

    # Reference: typical pairwise similarity among training proteins.
    # Sample a small set of training proteins and compute their pairwise
    # cosine similarities, excluding self-pairs. This tells us whether
    # "nearest neighbor" cosines are actually close or just generic for
    # ESM-2 (which often embeds proteins in a tight manifold).
    print('\nSanity baseline: pairwise similarity within training...')
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(train_md5s), size=500, replace=False)
    sample_vecs = train_vecs[sample_idx]
    ref_sims = cosine_sim_matrix(sample_vecs, sample_vecs)
    np.fill_diagonal(ref_sims, np.nan)
    ref_flat = ref_sims[~np.isnan(ref_sims)]
    ref_max_per_row = np.nanmax(ref_sims, axis=1)
    print(f'  Within-training pairwise cosine (sample of 500):')
    print(f'    mean pair:          {ref_flat.mean():.3f}')
    print(f'    median pair:        {np.median(ref_flat):.3f}')
    print(f'    mean nearest-other: {ref_max_per_row.mean():.3f}')
    print(f'    median nearest-other: {np.median(ref_max_per_row):.3f}')
    print(f'  (if PHL nearest ≈ within-training nearest-other, PHL is NOT')
    print(f'   distribution-shifted in embedding space)')

    # Summary stats
    nearest = top_sims[:, 0]
    print('\n=== Summary: nearest-training-protein cosine similarity ===')
    print(f'  n PHL proteins:       {len(nearest)}')
    print(f'  mean:                 {nearest.mean():.3f}')
    print(f'  median:               {np.median(nearest):.3f}')
    print(f'  min:                  {nearest.min():.3f}')
    print(f'  max:                  {nearest.max():.3f}')
    print(f'  self-in-training:     {sum(self_in_train)}/{len(self_in_train)}')
    for thr in (0.99, 0.95, 0.90, 0.80, 0.70, 0.50):
        frac = (nearest >= thr).mean()
        print(f'  sim >= {thr}:          {(nearest >= thr).sum():>4} / {len(nearest)}  ({frac*100:5.1f}%)')

    # Plot histogram
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(nearest, bins=40, edgecolor='black', alpha=0.75)
        ax.axvline(nearest.mean(), color='red', ls='--', lw=1.5,
                   label=f'mean = {nearest.mean():.3f}')
        ax.axvline(np.median(nearest), color='orange', ls='--', lw=1.5,
                   label=f'median = {np.median(nearest):.3f}')
        ax.set_xlabel('Cosine similarity to nearest training protein (ESM-2 650M)')
        ax.set_ylabel('# PHL proteins')
        ax.set_title(f'PHL → training nearest-neighbour similarity '
                     f'(n={len(nearest)})')
        ax.legend()
        ax.grid(alpha=0.3)
        plot_path = os.path.join(args.out_dir, 'phl_training_distance.svg')
        fig.tight_layout()
        fig.savefig(plot_path, format='svg')
        plt.close(fig)
        print(f'\nWrote {plot_path}')
    except ImportError:
        print('(matplotlib not installed — skipping plot)')


if __name__ == '__main__':
    main()
