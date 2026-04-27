"""Plot any-hit HR@k=1..20 curves for the top-N experiments.

Reads `results/experiment_log.csv` ALONE — no per-experiment JSON
lookup. The harvest now emits full HR@k=1..20 curves for the headline
metric (best-of-three head-modes, any-hit) per dataset per direction.

Two figures saved (each: panel per dataset, line per top-N model):
  results/figures/anyhit_phage2host_hrk_top<N>.svg
       — for each phage, did ≥1 positive host land at rank ≤ k?
       (cipher's primary use case: predict host given phage)
  results/figures/anyhit_host2phage_hrk_top<N>.svg
       — for each host, did ≥1 positive phage land at rank ≤ k?
       (PhageHostLearn comparison direction)

Top-N picked by `phl_pbip_best_anyhit_HR1` from the harvest. Family
colors (hue per embedding family, shade by intra-family rank).

Usage:
    python scripts/analysis/plot_anyhit_curves.py --top 4
    python scripts/analysis/plot_anyhit_curves.py --top 4 --datasets all
    python scripts/analysis/plot_anyhit_curves.py --top 4 --include-or
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PRIMARY = ['PhageHostLearn', 'PBIP']
ALL_DS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']
MAX_K = 20

FAMILY_CMAPS = {
    'prott5_mean': 'Greens', 'prott5_xl_full': 'BuGn',
    'prott5_xl_seg4': 'GnBu', 'prott5_xl_seg8': 'PuBu', 'prott5_xl_seg16': 'PuBuGn',
    'esm2_3b_mean': 'Blues', 'esm2_650m_seg4': 'Oranges',
    'esm2_650m_seg8': 'YlOrBr', 'esm2_650m_seg16': 'OrRd',
    'esm2_650m_mean': 'Purples', 'esm2_650m_full': 'RdPu',
    'esm2_650m': 'Purples', 'esm2_150m_mean': 'BuPu',
    'kmer_aa20_k3': 'pink_r', 'kmer_aa20_k4': 'YlOrBr',
    'kmer_li10_k5': 'YlGn', 'kmer_murphy8_k5': 'YlOrRd',
    'kmer_murphy10_k5': 'PuRd',
}
_FALLBACK_CMAPS = ['cool', 'spring', 'summer', 'cividis', 'plasma']


def fnum(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def top_n_anyhit(n, harvest_csv='results/experiment_log.csv'):
    with open(harvest_csv) as f:
        rows = [r for r in csv.DictReader(f)
                if fnum(r.get('phl_pbip_best_anyhit_HR1')) is not None]
    rows.sort(key=lambda r: -fnum(r['phl_pbip_best_anyhit_HR1']))
    return rows[:n]


def assign_colors(rows, label_key='embedding_type'):
    by_family = {}
    for r in rows:
        by_family.setdefault(r.get(label_key, '?'), []).append(r['run_name'])
    out = {}
    fallback_iter = iter(_FALLBACK_CMAPS)
    for fam, names in by_family.items():
        cmap_name = FAMILY_CMAPS.get(fam) or next(fallback_iter, 'gray')
        cmap = plt.get_cmap(cmap_name)
        n = len(names)
        for i, name in enumerate(names):
            t = 0.85 - (0.45 * i / max(n - 1, 1)) if n > 1 else 0.75
            out[name] = cmap(t)
    return out


def get_curve(row, dataset, direction, include_or=False):
    """Return list of (k, hr@k) from harvest row.

    direction: 'phage2host' or 'host2phage'.
    Returns: [(k, value), ...] for k=1..20 where value is non-empty.
    """
    field = f'{dataset}_best_{direction}_anyhit_HR'
    out = []
    for k in range(1, MAX_K + 1):
        v = fnum(row.get(f'{field}{k}'))
        if v is not None:
            out.append((k, v))
    return out


def get_or_curve(row, dataset):
    out = []
    for k in range(1, MAX_K + 1):
        v = fnum(row.get(f'{dataset}_OR_phage2host_anyhit_HR{k}'))
        if v is not None:
            out.append((k, v))
    return out


def plot_grid(rows, datasets, direction, out_path, title, include_or=False):
    colors = assign_colors(rows)
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(4.5 * len(datasets), 4.2),
                             squeeze=False)

    plotted_any = False
    for j, ds in enumerate(datasets):
        ax = axes[0, j]
        for r in rows:
            curve = get_curve(r, ds, direction)
            if not curve:
                continue
            ks, ys = zip(*curve)
            ax.plot(ks, ys, label=r['run_name'],
                    color=colors[r['run_name']], lw=1.8, alpha=0.95)
            plotted_any = True
        if include_or:
            # Draw OR ceiling for the BEST-ranked run as a faint dashed line
            for r in rows[:1]:
                curve = get_or_curve(r, ds)
                if curve:
                    ks, ys = zip(*curve)
                    ax.plot(ks, ys, color='#999', lw=1.5, ls='--',
                            label=f'OR ceiling ({r["run_name"][:30]})',
                            alpha=0.7)
        # n label: pull from row.<DS>_n_strict_{phage,host}
        n_field = f'{ds}_n_strict_phage' if direction == 'phage2host' else f'{ds}_n_strict_host'
        n = rows[0].get(n_field, '?') if rows else '?'
        ax.set_title(f'{ds}  (n={n})', fontsize=11)
        ax.set_xlabel('k')
        ax.set_ylabel('HR@k (any-hit, strict)')
        ax.set_xlim(0.5, MAX_K + 0.5)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=0.3)

    if not plotted_any:
        print(f'  WARNING: no curve data — re-run harvest_results.py to populate '
              f'<DS>_best_{direction}_anyhit_HR<k> columns.')
        plt.close(fig)
        return

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=max(1, min(2, len(labels))),
               bbox_to_anchor=(0.5, -0.02),
               fontsize=9, title='Run (sorted by best PHL+PBIP any-hit)',
               title_fontsize=9)
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.00)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    fig.savefig(out_path.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--top', type=int, default=4)
    p.add_argument('--datasets', choices=['primary', 'all'], default='primary')
    p.add_argument('--include-or', action='store_true',
                   help='Add the K∪O ceiling for the best run as a dashed line')
    p.add_argument('--out-dir', default='results/figures')
    args = p.parse_args()

    rows = top_n_anyhit(args.top)
    if not rows:
        print('No rows with any-hit data in harvest. Re-run harvest after Delta '
              'per_head_strict_eval batch finishes.')
        return

    print(f'Top {args.top} by phl_pbip_best_anyhit_HR1:')
    for r in rows:
        print(f'  {r["run_name"]:<55} '
              f'best_anyhit={fnum(r["phl_pbip_best_anyhit_HR1"]):.3f} '
              f'[{r.get("embedding_type", "?")}]')

    datasets = PRIMARY if args.datasets == 'primary' else ALL_DS
    suffix = f'_top{args.top}'
    if args.datasets == 'all':
        suffix += '_all5'
    if args.include_or:
        suffix += '_with_ceiling'

    os.makedirs(args.out_dir, exist_ok=True)
    plot_grid(rows, datasets, 'phage2host',
              os.path.join(args.out_dir, f'anyhit_phage2host_hrk{suffix}.svg'),
              f'HR@k — given a phage, did ≥1 positive host land at rank ≤ k?  '
              f'[top-{args.top}]',
              include_or=args.include_or)
    plot_grid(rows, datasets, 'host2phage',
              os.path.join(args.out_dir, f'anyhit_host2phage_hrk{suffix}.svg'),
              f'HR@k — given a host, did ≥1 positive phage land at rank ≤ k?  '
              f'[top-{args.top}, comparable to PhageHostLearn HR@k]',
              include_or=False)


if __name__ == '__main__':
    main()
