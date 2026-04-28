"""Cross-tab cipher's per-phage top-1 vs DpoTropiSearch's Combined
top-1 — to answer "which K-types does each tool catch that the
other misses".

Joins:
  - cipher per-phage TSV (from generate_per_phage_top1.py)
  - DpoTropiSearch 4-way TSV (agent 6's per_phage_diagnostics_4way.tsv)

Outputs:
  - results/figures/cipher_vs_tropi_cross_tab.svg/.png — Blues
    grouped-bar / heatmap showing per-K-type hits for each tool
    (cipher top-1, TropiSEQ top-1, TropiGAT top-1, Combined top-1).
  - results/analysis/cipher_vs_tropi_per_phage.tsv — joined per-phage
    table.
  - Stdout — cross-tab counts and the K-types where the two
    methods most strongly disagree.

Usage:
    python scripts/analysis/cross_tab_cipher_vs_tropi.py \\
        results/analysis/per_phage_top1/per_phage_top1_sweep_prott5_mean_cl70.tsv
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


TROPI_4WAY = ('/Users/leannmlindsey/WORK/PHI_TSP/cipher-depolymerase-domain/'
              'data/recall_at_k_4way/per_phage_diagnostics_4way.tsv')

OUT_DIR = 'results/figures'
OUT_TSV = 'results/analysis/cipher_vs_tropi_per_phage.tsv'


def load_cipher_tsv(path):
    out = {}
    for r in csv.DictReader(open(path), delimiter='\t'):
        out[(r['dataset'], r['phage_id'])] = r
    return out


def load_tropi_tsv(path):
    out = {}
    for r in csv.DictReader(open(path), delimiter='\t'):
        out[(r['dataset'], r['phage_id'])] = r
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cipher_tsv', help='per-phage cipher TSV from generate_per_phage_top1.py')
    p.add_argument('--tropi-tsv', default=TROPI_4WAY)
    p.add_argument('--out-dir', default=OUT_DIR)
    p.add_argument('--out-tsv', default=OUT_TSV)
    p.add_argument('--cipher-label', default='cipher (sweep_prott5_mean_cl70)',
                   help='display label for cipher run')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_tsv) or '.', exist_ok=True)

    cipher = load_cipher_tsv(args.cipher_tsv)
    tropi = load_tropi_tsv(args.tropi_tsv)

    # Inner join on (dataset, phage_id)
    common_keys = sorted(set(cipher.keys()) & set(tropi.keys()))
    print(f'Cipher TSV phages: {len(cipher)}')
    print(f'Tropi  TSV phages: {len(tropi)}')
    print(f'Joined (intersect): {len(common_keys)}')

    # Joined per-phage table
    joined_rows = []
    for k in common_keys:
        cp = cipher[k]
        tp = tropi[k]
        joined_rows.append({
            'dataset': k[0],
            'phage_id': k[1],
            'positive_K_types': cp['positive_K_types'],
            'cp_top1': cp['cp_top1_set'],
            'cp_hit@1': cp['cp_hit@1'],
            'ts_top1': tp['ts_top1_set'],
            'ts_hit@1': tp['ts_hit@1'],
            'tg_top1': tp['tg_top1_set'],
            'tg_hit@1': tp['tg_hit@1'],
            'combined_hit@1': tp['combined_hit@1'],
        })
    with open(args.out_tsv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(joined_rows[0].keys()))
        w.writeheader()
        w.writerows(joined_rows)
    print(f'Wrote joined per-phage TSV: {args.out_tsv}')

    # Cross-tab: cipher × Combined-Tropi
    matrix = defaultdict(int)
    for r in joined_rows:
        cp = int(r['cp_hit@1'] or 0)
        co = int(r['combined_hit@1'] or 0)
        matrix[f'{cp}{co}'] += 1
    n = len(joined_rows)
    print()
    print(f'Cross-tab cipher × Combined-Tropi (n={n}):')
    print(f'  both hit:      {matrix["11"]:>4}')
    print(f'  cipher only:   {matrix["10"]:>4}')
    print(f'  Tropi only:    {matrix["01"]:>4}')
    print(f'  both miss:     {matrix["00"]:>4}')

    # Per ground-truth K-type
    per_k = defaultdict(lambda: {'n':0,'cp':0,'ts':0,'tg':0,'comb':0})
    for r in joined_rows:
        pos = r['positive_K_types'].split(';')[0].strip()
        if not pos:
            continue
        per_k[pos]['n'] += 1
        per_k[pos]['cp']   += int(r['cp_hit@1'] or 0)
        per_k[pos]['ts']   += int(r['ts_hit@1'] or 0)
        per_k[pos]['tg']   += int(r['tg_hit@1'] or 0)
        per_k[pos]['comb'] += int(r['combined_hit@1'] or 0)

    # Print per-K table for K-types with n>=3
    print()
    print('Per ground-truth K-type (n>=3):')
    print(f'{"K":<8} {"n":>4}  {"cipher":>7}  {"TS":>5}  {"TG":>5}  {"Combined":>9}')
    print('-' * 60)
    for k, d in sorted(per_k.items(), key=lambda x: -x[1]['n']):
        if d['n'] < 3:
            continue
        def hr(field): return d[field] / d['n']
        print(f'{k:<8} {d["n"]:>4}  '
              f'{d["cp"]:>3}/{d["n"]:<3} '
              f'{d["ts"]:>2}/{d["n"]:<2} '
              f'{d["tg"]:>2}/{d["n"]:<2} '
              f'{d["comb"]:>3}/{d["n"]:<3}')

    # Plot: per-K hit rates as grouped bars (Blues palette).
    # Show top 15 K-types by n.
    top_ks = [k for k, d in sorted(per_k.items(), key=lambda x: -x[1]['n'])][:15]
    if not top_ks:
        print('No K-types with sufficient data; skipping plot.')
        return

    labels = top_ks
    n_per = [per_k[k]['n'] for k in top_ks]
    cipher_rates = [per_k[k]['cp'] / per_k[k]['n'] for k in top_ks]
    ts_rates     = [per_k[k]['ts'] / per_k[k]['n'] for k in top_ks]
    tg_rates     = [per_k[k]['tg'] / per_k[k]['n'] for k in top_ks]
    comb_rates   = [per_k[k]['comb'] / per_k[k]['n'] for k in top_ks]

    x = np.arange(len(labels))
    w = 0.20
    fig, ax = plt.subplots(figsize=(11, 4.5))
    # Cipher = navy (matching plot_recall_at_k); Tropi = warm tones
    ax.bar(x - 1.5*w, cipher_rates, w, label=args.cipher_label, color='#08306b')
    ax.bar(x - 0.5*w, ts_rates,     w, label='TropiSEQ',         color='#fec44f')
    ax.bar(x + 0.5*w, tg_rates,     w, label='TropiGAT',         color='#fd8d3c')
    ax.bar(x + 1.5*w, comb_rates,   w, label='Combined Tropi',   color='#a63603')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{k}\n(n={n})' for k, n in zip(labels, n_per)],
                       fontsize=8)
    ax.set_ylabel('Hit @ 1 rate', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Per-K-type top-1 hit rate: cipher vs DpoTropiSearch '
                 f'(top {len(labels)} K-types by phage count, n={sum(n_per)} '
                 f'phages)',
                 fontsize=10, pad=8)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    out_svg = os.path.join(args.out_dir, 'cipher_vs_tropi_per_K.svg')
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    fig.savefig(out_svg.replace('.svg', '.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'\nWrote {out_svg}')


if __name__ == '__main__':
    main()
