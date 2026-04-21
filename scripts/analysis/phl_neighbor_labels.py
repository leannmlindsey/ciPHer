"""For each PHL RBP, check whether its ESM-2-nearest training neighbour
shares a K-type with it.

The prior analysis (phl_training_distance.py) showed every PHL protein has
a training neighbour at cosine ≥ 0.976. That establishes "information is
available in embedding space." The next question: does that nearest
neighbour actually have the RIGHT K-type? If not, embeddings place
differently-labelled proteins at the same point — a modelling-side
label-noise problem, not a representation problem.

Outputs:
  results/analysis/phl_neighbor_labels.csv     — per-protein top-K match stats
  results/analysis/phl_neighbor_labels.svg     — summary bar chart

Usage:
  python scripts/analysis/phl_neighbor_labels.py
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from cipher.data.embeddings import load_embeddings
from cipher.data.proteins import load_fasta_md5


NULL_LABELS = {'null', 'N/A', 'Unknown', '', 'NA', 'n/a', 'None'}


def load_training_md5_to_k(assoc_path):
    """Build md5 -> set(K-types) from host_phage_protein_map.tsv."""
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


def load_phl_protein_to_k(mapping_path, interaction_path):
    """Build PHL protein_id -> set(K-types) where the phage is positive."""
    # phage -> set of K where label=1
    phage_k = defaultdict(set)
    with open(interaction_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            label = row.get('label', '').strip()
            if label not in ('1', '1.0', 'True', 'true'):
                continue
            phage = row.get('phage_id', '').strip()
            k = row.get('host_K', '').strip()
            if not phage or not k or k in NULL_LABELS:
                continue
            phage_k[phage].add(k)

    # protein -> phages, then join through phage_k
    protein_k = defaultdict(set)
    phages_seen = set()
    with open(mapping_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            phage = row.get('matrix_phage_name', '').strip()
            protein = row.get('protein_id', '').strip()
            if not phage or not protein:
                continue
            phages_seen.add(phage)
            if phage in phage_k:
                protein_k[protein] |= phage_k[phage]
    print(f'  phages with positive interactions: {len(phage_k):,}')
    print(f'  phages in mapping:                 {len(phages_seen):,}')
    return dict(protein_k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-emb',
                   default='data/training_data/embeddings/esm2_650m_md5.npz')
    p.add_argument('--val-emb',
                   default='data/validation_data/embeddings/esm2_650m_md5.npz')
    p.add_argument('--val-fasta',
                   default='data/validation_data/metadata/validation_rbps_all.faa')
    p.add_argument('--assoc-map',
                   default='data/training_data/metadata/host_phage_protein_map.tsv')
    p.add_argument('--phl-mapping',
                   default='data/validation_data/HOST_RANGE/PhageHostLearn/'
                           'metadata/phage_protein_mapping.csv')
    p.add_argument('--phl-interactions',
                   default='data/validation_data/HOST_RANGE/PhageHostLearn/'
                           'metadata/interaction_matrix.tsv')
    p.add_argument('--topk', type=int, default=50)
    p.add_argument('--out-dir', default='results/analysis')
    p.add_argument('--restrict-to-labeled', action='store_true',
                   help='Only load training proteins that have a K-type '
                        'label. Cuts peak memory from ~143K to ~55K proteins '
                        '— essential for large kmer embeddings (e.g. li10_k5, '
                        'murphy10_k5) that dont fit in memory at full size.')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading training md5 -> K-type set...')
    train_md5_k = load_training_md5_to_k(args.assoc_map)
    print(f'  {len(train_md5_k):,} training md5s with K labels')

    print('Loading PHL protein -> K-type set...')
    phl_k = load_phl_protein_to_k(args.phl_mapping, args.phl_interactions)
    print(f'  {len(phl_k):,} PHL proteins with at least one K label')

    print('Loading embeddings...')
    md5_filter = set(train_md5_k.keys()) if args.restrict_to_labeled else None
    if args.restrict_to_labeled:
        print(f'  restrict-to-labeled: loading {len(md5_filter):,} K-labelled MD5s only')
    train = load_embeddings(args.train_emb, md5_filter=md5_filter)
    train_md5s = list(train.keys())
    train_vecs = np.stack([train[m] for m in train_md5s]).astype(np.float32)
    train_vecs /= np.linalg.norm(train_vecs, axis=1, keepdims=True)
    print(f'  training embeddings: {train_vecs.shape}')

    val = load_embeddings(args.val_emb)
    pid_md5 = load_fasta_md5(args.val_fasta)

    # Build query matrix only for PHL proteins that have K labels AND embeddings
    queries, q_pids, q_md5s, q_true_k = [], [], [], []
    for pid, true_k in phl_k.items():
        md5 = pid_md5.get(pid)
        if md5 and md5 in val:
            queries.append(val[md5])
            q_pids.append(pid)
            q_md5s.append(md5)
            q_true_k.append(true_k)
    queries = np.stack(queries).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    print(f'  resolved {len(queries):,} PHL proteins to embeddings+labels')

    print('Computing top-K nearest training neighbours per PHL protein...')
    sims = queries @ train_vecs.T  # (n_q, n_train) cosine
    top_idx = np.argpartition(-sims, args.topk, axis=1)[:, :args.topk]
    # order by sim desc within the top-K partition
    for i in range(len(top_idx)):
        order = np.argsort(-sims[i, top_idx[i]])
        top_idx[i] = top_idx[i][order]
    top_sims = np.take_along_axis(sims, top_idx, axis=1)

    # Per-protein: rank of first matching neighbour
    csv_path = os.path.join(args.out_dir, 'phl_neighbor_labels.csv')
    first_match_ranks = []
    n_top1_match = 0
    n_top5_match = 0
    n_top10_match = 0
    n_never_match = 0
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['protein_id', 'md5', 'true_k_types',
                    'nearest_sim', 'nearest_k_types', 'nearest_match',
                    'first_match_rank', 'top5_has_match', 'top10_has_match'])
        for i, pid in enumerate(q_pids):
            true_k = q_true_k[i]
            # Walk top-K, find first match
            first_rank = None
            neighbor_labels_str = []
            for j in range(args.topk):
                nbr_md5 = train_md5s[top_idx[i, j]]
                nbr_k = train_md5_k.get(nbr_md5, set())
                if j == 0:
                    nearest_k = nbr_k
                if first_rank is None and (true_k & nbr_k):
                    first_rank = j + 1  # 1-indexed
                    if j + 1 <= 5:
                        neighbor_labels_str.append(f'rank{j+1}:{",".join(sorted(nbr_k))}')
            top1_match = bool(true_k & nearest_k)
            top5_match = first_rank is not None and first_rank <= 5
            top10_match = first_rank is not None and first_rank <= 10
            if first_rank is None:
                n_never_match += 1
            if top1_match:
                n_top1_match += 1
            if top5_match:
                n_top5_match += 1
            if top10_match:
                n_top10_match += 1
            first_match_ranks.append(first_rank if first_rank is not None else args.topk + 1)
            w.writerow([
                pid, q_md5s[i],
                ','.join(sorted(true_k)),
                f'{top_sims[i, 0]:.4f}',
                ','.join(sorted(nearest_k)) if nearest_k else '',
                'Y' if top1_match else 'N',
                first_rank if first_rank is not None else '',
                'Y' if top5_match else 'N',
                'Y' if top10_match else 'N',
            ])
    print(f'\nWrote {csv_path}')

    # Random baseline — how often does a *randomly* picked training
    # protein share a K-type with a PHL protein? If ~equal to top-1 match,
    # the embedding provides zero K-type signal.
    rng = np.random.default_rng(42)
    n = len(q_pids)
    n_trials = 1000
    random_matches = 0
    labeled_train_md5s = [m for m in train_md5s if m in train_md5_k]
    for _ in range(n_trials):
        i = rng.integers(0, n)
        j = rng.choice(labeled_train_md5s)
        if q_true_k[i] & train_md5_k[j]:
            random_matches += 1
    random_rate = random_matches / n_trials
    print(f'\n  Random training protein shares K-type with random PHL: '
          f'{random_rate*100:.1f}% (baseline)')

    print(f'\n=== K-type agreement with nearest training neighbour ===')
    print(f'  n PHL proteins (with K labels): {n}')
    print(f'  top-1 nearest shares a K-type:     {n_top1_match:>4} / {n}  '
          f'({100*n_top1_match/n:5.1f}%)')
    print(f'  top-5 has at least one match:      {n_top5_match:>4} / {n}  '
          f'({100*n_top5_match/n:5.1f}%)')
    print(f'  top-10 has at least one match:     {n_top10_match:>4} / {n}  '
          f'({100*n_top10_match/n:5.1f}%)')
    print(f'  no match in top-{args.topk}:                  {n_never_match:>4} / {n}  '
          f'({100*n_never_match/n:5.1f}%)')

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        # Left: top-K cumulative match rate
        ks = list(range(1, args.topk + 1))
        cum = [sum(1 for r in first_match_ranks if r <= k) / n for k in ks]
        axes[0].plot(ks, cum, marker='o', markersize=3)
        axes[0].set_xlabel('K (top-K neighbours considered)')
        axes[0].set_ylabel('Fraction of PHL proteins with ≥1 K-matching neighbour')
        axes[0].set_ylim(0, 1.02)
        axes[0].set_xscale('log')
        axes[0].set_title('Cumulative K-type recall vs. neighbour rank')
        axes[0].grid(alpha=0.3)

        # Right: bar chart of top-1/5/10 + never
        bars = ['top-1', 'top-5', 'top-10', f'no match in top-{args.topk}']
        vals = [n_top1_match / n, n_top5_match / n, n_top10_match / n, n_never_match / n]
        axes[1].bar(bars, vals, color=['#2166ac', '#4393c3', '#92c5de', '#b2182b'])
        axes[1].set_ylabel('Fraction of PHL proteins')
        axes[1].set_ylim(0, 1.02)
        axes[1].set_title('K-type match at neighbour rank')
        for i, v in enumerate(vals):
            axes[1].text(i, v + 0.01, f'{v*100:.1f}%', ha='center', fontsize=9)
        axes[1].grid(alpha=0.3, axis='y')

        fig.suptitle(f'PHL RBP → training neighbour K-type agreement '
                     f'(n={n}, ESM-2 650M mean)', fontweight='bold')
        fig.tight_layout()
        plot_path = os.path.join(args.out_dir, 'phl_neighbor_labels.svg')
        fig.savefig(plot_path, format='svg')
        plt.close(fig)
        print(f'Wrote {plot_path}')
    except ImportError:
        print('(matplotlib not available — skipping plot)')


if __name__ == '__main__':
    main()
