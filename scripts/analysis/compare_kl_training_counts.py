"""Compare per-KL training counts: cipher vs DpoTropiSearch (TropiGAT/SEQ).

Both projects train on prophage-derived data labelled by Klebsiella
capsule type (K / KL). Cipher labels with the *current host* K type;
DpoTropi labels with the *infected ancestor's* K type, computed via
LCA over the host phylogeny. Same proteins (largely) — different
labelling rules.

This script counts how many training proteins each corpus assigns to
each KL type, normalises naming (K1 ≡ KL1; KL101 unchanged), and
emits:
  results/dpotropi_vs_cipher_KL_counts.csv   — full table
  results/figures/dpotropi_vs_cipher_KL_counts.svg/.png — bar plot

Cipher labels are multi-label per protein (`k_labels` is a binary
matrix shape (n_proteins, n_K_classes)). For each K class we sum
`k_labels[:, k_idx]` to get the count of proteins assigned that
class. DpoTropi has multi-KL labels too (`KL_type_LCA` may be
"KLa|KLb|KLc" — phylogenetic ambiguity); we split on `|` and count
each label.

Usage:
    python scripts/analysis/compare_kl_training_counts.py
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np


import argparse

# Default to current best-performing experiment, NOT the LAPTOP repro
# (which uses the narrow OLD klebsiella recipe with min_sources=3 +
# single-label filtering). Override via CLI.
DEFAULT_CIPHER_EXP = ('/Users/leannmlindsey/WORK/PHI_TSP/cipher/experiments/'
                      'attention_mlp/sweep_prott5_mean_cl70')
DPOTROPI_TSV = ('/Users/leannmlindsey/WORK/PHI_TSP/DpoTropiSearch_zenoto_data/'
                'TropiGATv2.final_df_v2.tsv')
OUT_CSV = 'results/dpotropi_vs_cipher_KL_counts.csv'
OUT_SVG = 'results/figures/dpotropi_vs_cipher_KL_counts.svg'


def to_kl(name):
    """Canonicalize K↔KL: K1→KL1, KL101→KL101, null/N/A→None."""
    if name in (None, '', 'null', 'N/A'):
        return None
    n = name.strip()
    if n.startswith('KL'):
        return n
    if n.startswith('K') and n[1:].isdigit():
        return 'KL' + n[1:]
    return n  # other (rare e.g. odd label)


def load_cipher_counts(exp_dir):
    import json
    import os
    enc = json.load(open(os.path.join(exp_dir, 'label_encoders.json')))
    classes = enc['k_classes']
    npz = np.load(os.path.join(exp_dir, 'training_data.npz'), allow_pickle=True)
    k_labels = npz['k_labels']
    counts = k_labels.sum(axis=0)
    out = defaultdict(int)
    for cls, n in zip(classes, counts):
        kl = to_kl(cls)
        if kl is None:
            continue
        out[kl] += int(n)
    return dict(out), len(npz['md5_list'])


def load_dpotropi_counts():
    """Stream the 555 MB TSV; only need column 3 (KL_type_LCA)."""
    out = defaultdict(int)
    n_rows = 0
    n_multi = 0
    with open(DPOTROPI_TSV) as f:
        header = next(f).rstrip('\n').split('\t')
        kl_col = header.index('KL_type_LCA')
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= kl_col:
                continue
            n_rows += 1
            label = parts[kl_col].strip()
            if not label:
                continue
            if '|' in label:
                n_multi += 1
                # Multi-KL: count each label (phylogenetic ambiguity)
                # We attribute 1 protein to EACH KL in the multi-set
                for kl in label.split('|'):
                    kl = to_kl(kl.strip())
                    if kl:
                        out[kl] += 1
            else:
                kl = to_kl(label)
                if kl:
                    out[kl] += 1
    return dict(out), n_rows, n_multi


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cipher-exp', default=DEFAULT_CIPHER_EXP,
                   help='Cipher experiment dir (training_data.npz + label_encoders.json)')
    p.add_argument('--out-csv', default=OUT_CSV)
    p.add_argument('--out-svg', default=OUT_SVG)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.out_svg) or '.', exist_ok=True)

    print(f'Loading cipher training counts from {args.cipher_exp} ...')
    cipher_counts, n_cipher_rows = load_cipher_counts(args.cipher_exp)
    print(f'  cipher: {n_cipher_rows:,} unique proteins, '
          f'{len(cipher_counts)} unique KL types after K→KL canonicalization')

    print('Loading DpoTropi training counts ...')
    dpotropi_counts, n_dpotropi_rows, n_dpotropi_multi = load_dpotropi_counts()
    print(f'  dpotropi: {n_dpotropi_rows:,} rows, {len(dpotropi_counts)} unique KL types  '
          f'({n_dpotropi_multi} multi-KL rows split into per-KL contributions)')

    # Build joint table
    all_kls = sorted(set(cipher_counts) | set(dpotropi_counts),
                     key=lambda x: -dpotropi_counts.get(x, 0))
    rows = []
    for kl in all_kls:
        c = cipher_counts.get(kl, 0)
        d = dpotropi_counts.get(kl, 0)
        rows.append({
            'KL': kl,
            'cipher_n': c,
            'dpotropi_n': d,
            'cipher_only': 1 if (c > 0 and d == 0) else 0,
            'dpotropi_only': 1 if (d > 0 and c == 0) else 0,
            'both': 1 if (c > 0 and d > 0) else 0,
            'log10_ratio_d_over_c': (np.log10((d + 1) / (c + 1)) if (c + d) > 0
                                      else 0.0),
        })

    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nWrote {args.out_csv}  ({len(rows)} rows)')

    # Quick stdout summary
    n_cipher_only = sum(r['cipher_only'] for r in rows)
    n_dpot_only = sum(r['dpotropi_only'] for r in rows)
    n_both = sum(r['both'] for r in rows)
    print(f'\nVocab overlap (KL types with ≥1 training protein):')
    print(f'  in BOTH:        {n_both}')
    print(f'  cipher only:    {n_cipher_only}')
    print(f'  dpotropi only:  {n_dpot_only}')

    # Top-15 by DpoTropi count
    print(f'\nTop 15 KL types by DpoTropi training count:')
    print(f'  {"KL":<8}  {"DpoTropi":>10}  {"cipher":>10}  {"ratio":>10}')
    print('-' * 46)
    for r in rows[:15]:
        ratio = r['dpotropi_n'] / max(r['cipher_n'], 1)
        print(f'  {r["KL"]:<8}  {r["dpotropi_n"]:>10,}  {r["cipher_n"]:>10,}  '
              f'{ratio:>10.2f}x')

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Top 30 KLs by combined count for readability
        top30 = sorted(rows, key=lambda r: -(r['dpotropi_n'] + r['cipher_n']))[:30]
        labels = [r['KL'] for r in top30]
        d = [r['dpotropi_n'] for r in top30]
        c = [r['cipher_n'] for r in top30]
        x = np.arange(len(labels))
        w = 0.4

        fig, ax = plt.subplots(figsize=(13, 5.5))
        ax.bar(x - w/2, d, width=w, color='#d62728',
               label=f'DpoTropi (n_proteins={n_dpotropi_rows:,})')
        ax.bar(x + w/2, c, width=w, color='#1f77b4',
               label=f'cipher (n_proteins={n_cipher_rows:,})')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=9)
        ax.set_ylabel('# training proteins assigned this KL type')
        ax.set_title('Per-KL training-data counts: DpoTropiSearch vs cipher\n'
                     f'(top 30 of {len(rows)} KL types by combined count; '
                     f'multi-KL rows in DpoTropi split into per-KL contributions)',
                     fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_svg, format='svg', bbox_inches='tight')
        fig.savefig(args.out_svg.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'\nWrote {args.out_svg}')
    except ImportError:
        print('matplotlib not available, skipping plot')


if __name__ == '__main__':
    main()
