"""Recall@k=1..20 for one cipher run, overlaying every validation dataset
on a single panel. Companion to plot_recall_at_k_vs_competitors.py — same
axis, same metric (phage-level any-hit, OR mode, strict denominator) — but
the curve dimension is dataset rather than competitor.

Reads exclusively from the harvest CSV (results/experiment_log.csv).

Output: results/figures/recall_at_k_one_model_all_datasets.svg/.png

Usage:
    python scripts/analysis/plot_recall_at_k_one_model_all_datasets.py
    python scripts/analysis/plot_recall_at_k_one_model_all_datasets.py \
        --run sweep_esm2_650m_mean_cl70
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


HARVEST_CSV = 'results/experiment_log.csv'

DEFAULT_RUN = 'sweep_kmer_aa20_k4'   # current best by PHL OR HR@1 (0.560) and overall (0.773)
DEFAULT_MODE = 'OR'  # OR | K | O — chooses the per-head any-hit column family
DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']

# Fixed denominators per project policy (memory: feedback_strict_denominator).
# Use these — not the per-run n_strict_phage in the harvest CSV. Convert
# harvested HR by multiplying by n_strict_phage to recover the numerator,
# then dividing by the value below.
FIXED_DENOM_PHAGE = {'CHEN': 3, 'GORODNICHIV': 3, 'UCSD': 11,
                     'PBIP': 103, 'PhageHostLearn': 100}  # paper-equivalent (excludes no-genome phages: 27 PHL, 1 PBIP)
FIXED_DENOM_TOTAL = sum(FIXED_DENOM_PHAGE.values())  # 220

# One color per dataset (qualitative). 'overall' uses a thicker black line.
DATASET_COLOR = {
    'CHEN':           '#1f77b4',
    'GORODNICHIV':    '#2ca02c',
    'UCSD':           '#d62728',
    'PBIP':           '#ff7f0e',
    'PhageHostLearn': '#9467bd',
}
OVERALL_COLOR = '#222222'

OUT_SVG = 'results/figures/recall_at_k_one_model_all_datasets.svg'


def f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def load_run_row(run_name):
    with open(HARVEST_CSV) as fh:
        for r in csv.DictReader(fh):
            if r.get('run_name') == run_name:
                return r
    return None


def hits_for(row, dataset, mode):
    """Recover numerator (hits per k) from harvested HR_strict × n_strict_phage."""
    n_strict = f(row.get(f'{dataset}_n_strict_phage'))
    if not n_strict:
        return {k: None for k in range(1, 21)}
    n_strict = int(n_strict)
    hits = {}
    for k in range(1, 21):
        v = f(row.get(f'{dataset}_{mode}_phage2host_anyhit_HR{k}'))
        hits[k] = round(v * n_strict) if v is not None else None
    return hits


def per_dataset_curves(row, mode):
    """{dataset: {k: hr}} for the chosen mode (K, O, OR), using the fixed
    per-dataset phage count as the denominator (not row['<ds>_n_strict_phage'])."""
    out = {}
    for ds in DATASETS:
        hits = hits_for(row, ds, mode)
        denom = FIXED_DENOM_PHAGE[ds]
        out[ds] = {k: (hits[k] / denom if hits[k] is not None else None)
                   for k in range(1, 21)}
    return out


def overall_curve(row, mode):
    """Phage-weighted overall {k: hr} = sum(hits) / sum(fixed_denom) over the
    five datasets. Returns (curve, total_denom)."""
    num = defaultdict(int)
    has_any = False
    for ds in DATASETS:
        hits = hits_for(row, ds, mode)
        for k in range(1, 21):
            if hits[k] is not None:
                num[k] += hits[k]
                has_any = True
    if not has_any:
        return {k: None for k in range(1, 21)}, FIXED_DENOM_TOTAL
    return ({k: num[k] / FIXED_DENOM_TOTAL for k in range(1, 21)},
            FIXED_DENOM_TOTAL)


def n_phages(row, ds):
    """Fixed-denominator phage count for the dataset (project policy)."""
    return FIXED_DENOM_PHAGE.get(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=DEFAULT_RUN,
                    help=f'run_name in {HARVEST_CSV} (default: {DEFAULT_RUN})')
    ap.add_argument('--mode', default=DEFAULT_MODE, choices=('OR', 'K', 'O'),
                    help='per-head any-hit column family (default: OR)')
    ap.add_argument('--out', default=OUT_SVG)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    row = load_run_row(args.run)
    if row is None:
        raise SystemExit(f'ERROR: no row in {HARVEST_CSV} for run_name={args.run}')

    per_ds = per_dataset_curves(row, args.mode)
    overall, derived_total = overall_curve(row, args.mode)
    total_n = derived_total if derived_total is not None else sum(
        n_phages(row, ds) or 0 for ds in DATASETS)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ks = list(range(1, 21))

    for ds in DATASETS:
        ys = [per_ds[ds].get(k) for k in ks]
        if all(v is None for v in ys):
            continue
        n = n_phages(row, ds)
        n_str = f'n={n}' if n is not None else 'n=?'
        ax.plot(ks, ys, color=DATASET_COLOR[ds], lw=1.8,
                marker='o', markersize=4,
                label=f'{ds} ({n_str})')

    ys = [overall.get(k) for k in ks]
    if not all(v is None for v in ys):
        ax.plot(ks, ys, color=OVERALL_COLOR, lw=2.8,
                marker='D', markersize=4.5, linestyle='--',
                label=f'phage-weighted overall (n={total_n})')

        # Annotate overall at k=1, 10, 20.
        for k_ann in (1, 10, 20):
            v = overall.get(k_ann)
            if v is not None:
                ax.annotate(f'{v:.3f}', xy=(k_ann, v),
                            xytext=(0, 8), textcoords='offset points',
                            ha='center', fontsize=9, fontweight='bold',
                            color=OVERALL_COLOR)

    ax.set_title(f'Recall@k per dataset for cipher run "{args.run}"\n'
                 f'phage-level any-hit ({args.mode} mode), '
                 f'FIXED denominator per project policy',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('k')
    ax.set_ylabel('Recall@k')
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(args.out, format='svg', bbox_inches='tight')
    fig.savefig(args.out.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {args.out}')

    # Headline values for the lab notebook
    print()
    print(f'Run: {args.run}   mode: {args.mode}')
    print(f'  {"dataset":<22}  {"n":>5}  {"HR@1":>6}  {"HR@5":>6}  {"HR@10":>6}  {"HR@20":>6}')
    for ds in DATASETS:
        c = per_ds[ds]
        n = n_phages(row, ds)
        cells = [f'{c.get(k):.3f}' if c.get(k) is not None else '   --'
                 for k in (1, 5, 10, 20)]
        print(f'  {ds:<22}  {(n if n is not None else "?"):>5}  '
              + '  '.join(f'{x:>6}' for x in cells))
    cells = [f'{overall.get(k):.3f}' if overall.get(k) is not None else '   --'
             for k in (1, 5, 10, 20)]
    print(f'  {"phage-weighted overall":<22}  {total_n:>5}  '
          + '  '.join(f'{x:>6}' for x in cells))


if __name__ == '__main__':
    main()
