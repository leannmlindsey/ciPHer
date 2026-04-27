"""Generate SVG figures for an embedding sweep.

Produces two figures, saved as SVG (vector) for slide decks and paper
appendix use:

  results/figures/sweep_phl_pbip_hrk.svg       — 2 datasets x 2 directions
  results/figures/sweep_all_datasets_hrk.svg   — 5 datasets x 2 directions

pLM embeddings use cool colors (viridis), k-mer features use warm colors
(autumn). Legend is sorted by combined PHL+PBIP HR@1 so the best runs
appear first.

Usage:
    python scripts/analysis/plot_sweep_results.py
    python scripts/analysis/plot_sweep_results.py --filter sweep_
    python scripts/analysis/plot_sweep_results.py --model attention_mlp --filter sweep_
"""

import argparse
import json
import os
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DATASETS_ALL = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']
DATASETS_PRIMARY = ['PhageHostLearn', 'PBIP']
DIRECTIONS = [('rank_hosts', 'Rank hosts given phage'),
              ('rank_phages', 'Rank phages given host')]
MAX_K = 20


def load_runs(model, name_filter, only_names=None):
    """Load runs from experiments/{model}/*/results/evaluation.json.

    If `only_names` is provided (set/list of run names), keep only runs
    whose basename is in that set — overrides the substring filter.
    """
    runs = []
    only_names = set(only_names) if only_names else None
    for eval_path in sorted(glob(f'experiments/{model}/*/results/evaluation.json')):
        exp_dir = os.path.dirname(os.path.dirname(eval_path))
        name = os.path.basename(exp_dir)
        if only_names is not None:
            if name not in only_names:
                continue
        elif name_filter and name_filter not in name:
            continue
        with open(eval_path) as f:
            ev = json.load(f)
        try:
            with open(os.path.join(exp_dir, 'experiment.json')) as f:
                meta = json.load(f)
            label = meta['config']['data'].get('embedding_type', name)
        except (FileNotFoundError, KeyError):
            label = name
        runs.append({'name': name, 'label': label, 'data': ev})
    return runs


def top_n_from_harvest(n, model=None,
                        harvest_csv='results/experiment_log.csv'):
    """Return list of run names — top N by `phl_pbip_combined_hr1`."""
    import csv as _csv
    with open(harvest_csv) as f:
        rows = [r for r in _csv.DictReader(f) if r.get('phl_pbip_combined_hr1')]
    def fnum(v):
        try: return float(v)
        except (TypeError, ValueError): return None
    rows = [r for r in rows if fnum(r['phl_pbip_combined_hr1']) is not None]
    if model:
        rows = [r for r in rows if r.get('model') == model]
    rows.sort(key=lambda r: -fnum(r['phl_pbip_combined_hr1']))
    return [r['run_name'] for r in rows[:n]]


def combined_score(run):
    vals = []
    for ds in DATASETS_PRIMARY:
        r = run['data'].get(ds, {})
        for direction, _ in DIRECTIONS:
            v = r.get(direction, {}).get('hr_at_k', {}).get('1')
            if v is not None:
                vals.append(v)
    return sum(vals) / max(len(vals), 1)


# Each embedding family gets a distinct sequential colormap.
# Within a family, intra-family rank → shade (best gets darkest).
# Goal: viewer's eye groups runs by family, distinguishes within.
FAMILY_CMAPS = {
    'prott5_mean':       'Greens',
    'prott5_xl_full':    'BuGn',
    'prott5_xl_seg4':    'GnBu',
    'prott5_xl_seg8':    'PuBu',
    'prott5_xl_seg16':   'PuBuGn',
    'esm2_3b_mean':      'Blues',
    'esm2_650m_seg4':    'Oranges',
    'esm2_650m_seg8':    'YlOrBr',
    'esm2_650m_seg16':   'OrRd',
    'esm2_650m_mean':    'Purples',
    'esm2_650m_full':    'RdPu',
    'esm2_650m':         'Purples',
    'esm2_150m_mean':    'BuPu',
    'kmer_aa20_k3':      'pink_r',
    'kmer_aa20_k4':      'YlOrBr',
    'kmer_li10_k5':      'YlGn',
    'kmer_murphy8_k5':   'YlOrRd',
    'kmer_murphy10_k5':  'PuRd',
}
_FALLBACK_CMAPS = ['cool', 'spring', 'summer', 'cividis', 'plasma']


def assign_colors(runs):
    """Hue-per-family, shade-by-intra-rank. Returns {run_name: color}.

    Runs within the same `label` (embedding_type) family share a base
    sequential colormap; the best-ranked run in that family gets the
    darkest shade. Unknown families fall back to a generic colormap.
    """
    # Preserve input ordering (which the caller sorts by combined_score desc)
    by_family = {}
    for r in runs:
        by_family.setdefault(r['label'], []).append(r)

    out = {}
    fallback_iter = iter(_FALLBACK_CMAPS)
    for family, family_runs in by_family.items():
        cmap_name = FAMILY_CMAPS.get(family) or next(fallback_iter, 'gray')
        cmap = plt.get_cmap(cmap_name)
        n = len(family_runs)
        for i, r in enumerate(family_runs):
            # i=0 is the best in this family → darkest shade.
            # Span the colormap from 0.85 (darkest) down to 0.40 (lightest).
            t = 0.85 - (0.45 * i / max(n - 1, 1)) if n > 1 else 0.75
            out[r['name']] = cmap(t)
    return out


def hr_curve(run, ds, direction, max_k=MAX_K):
    hrk = run['data'].get(ds, {}).get(direction, {}).get('hr_at_k', {})
    if not hrk:
        return [], []
    ks = sorted(int(k) for k in hrk.keys() if int(k) <= max_k)
    ys = [hrk[str(k)] for k in ks]
    return ks, ys


def n_pairs(run, ds, direction):
    return run['data'].get(ds, {}).get(direction, {}).get('n_pairs', '?')


def avg_hr_curve(run, datasets, direction, max_k=MAX_K):
    """Average HR@k across the given datasets, per k (skipping missing)."""
    ks = list(range(1, max_k + 1))
    ys = []
    for k in ks:
        vals = []
        for ds in datasets:
            hrk = run['data'].get(ds, {}).get(direction, {}).get('hr_at_k', {})
            v = hrk.get(str(k), hrk.get(k))
            if v is not None:
                vals.append(float(v))
        ys.append(sum(vals) / len(vals) if vals else None)
    # Drop trailing Nones, keep first stretch
    out_k, out_y = [], []
    for k, y in zip(ks, ys):
        if y is not None:
            out_k.append(k); out_y.append(y)
    return out_k, out_y


def plot_grid(runs, datasets, out_path, title):
    """One subplot per (dataset, direction). Rows=datasets, cols=directions."""
    runs_sorted = sorted(runs, key=combined_score, reverse=True)
    colors = assign_colors(runs_sorted)

    nrows, ncols = len(datasets), len(DIRECTIONS)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 3.2 * nrows),
                             squeeze=False)

    for i, ds in enumerate(datasets):
        for j, (direction, dir_title) in enumerate(DIRECTIONS):
            ax = axes[i, j]
            for r in runs_sorted:
                ks, ys = hr_curve(r, ds, direction)
                if not ks:
                    continue
                ax.plot(ks, ys, label=f"{r['name']} [{r['label']}]",
                        color=colors[r['name']], lw=1.7, alpha=0.95)
            n = n_pairs(runs_sorted[0], ds, direction)
            ax.set_title(f'{ds} — {dir_title}  (n={n})', fontsize=10)
            ax.set_xlabel('k')
            ax.set_ylabel('HR@k')
            ax.set_xlim(0.5, MAX_K + 0.5)
            ax.set_ylim(0, 1.02)
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.grid(alpha=0.3)

    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, leg_labels,
               loc='lower center', ncol=min(5, len(leg_labels)),
               bbox_to_anchor=(0.5, -0.01), fontsize=9,
               title='Embedding (sorted by PHL+PBIP HR@1)', title_fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.00)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')


def plot_average(runs, datasets, out_path, title):
    """One subplot per ranking direction, x=k, y=avg HR@k across datasets,
    one curve per experiment."""
    runs_sorted = sorted(runs, key=combined_score, reverse=True)
    colors = assign_colors(runs_sorted)

    fig, axes = plt.subplots(1, len(DIRECTIONS),
                             figsize=(5.5 * len(DIRECTIONS), 4.2),
                             squeeze=False)

    for j, (direction, dir_title) in enumerate(DIRECTIONS):
        ax = axes[0, j]
        for r in runs_sorted:
            ks, ys = avg_hr_curve(r, datasets, direction)
            if not ks:
                continue
            ax.plot(ks, ys, label=f"{r['name']} [{r['label']}]",
                    color=colors[r['name']], lw=1.8, alpha=0.95)
        ax.set_title(f'{dir_title}\n(avg across {", ".join(datasets)})',
                     fontsize=10)
        ax.set_xlabel('k')
        ax.set_ylabel('avg HR@k')
        ax.set_xlim(0.5, MAX_K + 0.5)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=0.3)

    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, leg_labels,
               loc='lower center', ncol=min(5, len(leg_labels)),
               bbox_to_anchor=(0.5, -0.01), fontsize=9,
               title='Embedding (sorted by PHL+PBIP HR@1)', title_fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.00)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='attention_mlp')
    p.add_argument('--filter', default='sweep_',
                   help='substring to match in run name (ignored if --top is set)')
    p.add_argument('--top', type=int, default=None,
                   help='pick top-N from results/experiment_log.csv by '
                        'phl_pbip_combined_hr1 (overrides --filter)')
    p.add_argument('--all-models', action='store_true',
                   help='with --top: pick across all models, not just --model')
    p.add_argument('--name-suffix', default='',
                   help='suffix appended to output filenames (useful for variants)')
    p.add_argument('--out-dir', default='results/figures')
    args = p.parse_args()

    only_names = None
    if args.top is not None:
        only_names = top_n_from_harvest(
            args.top, model=None if args.all_models else args.model)
        print(f'Top {args.top} from harvest: {len(only_names)} run names')
        for n in only_names:
            print(f'  {n}')
    runs = load_runs(args.model, args.filter, only_names=only_names)
    if not runs:
        print('No runs found.')
        return
    print(f'Loaded {len(runs)} runs with local evaluation.json')

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = args.name_suffix
    plot_grid(runs, DATASETS_PRIMARY,
              os.path.join(args.out_dir, f'sweep_phl_pbip_hrk{suffix}.svg'),
              ('Top-{} runs by PHL+PBIP HR@1 — PhageHostLearn & PBIP'
               .format(args.top) if args.top
               else 'Embedding sweep — PhageHostLearn & PBIP (primary datasets)'))
    plot_grid(runs, DATASETS_ALL,
              os.path.join(args.out_dir, f'sweep_all_datasets_hrk{suffix}.svg'),
              ('Top-{} runs by PHL+PBIP HR@1 — all 5 validation datasets'
               .format(args.top) if args.top
               else 'Embedding sweep — all 5 validation datasets'))

    # Average-across-datasets companion figures (1 row × 2 directions).
    plot_average(runs, DATASETS_PRIMARY,
                 os.path.join(args.out_dir, f'sweep_avg_phl_pbip_hrk{suffix}.svg'),
                 ('Top-{} runs — avg HR@k over PHL+PBIP'.format(args.top)
                  if args.top else 'Embedding sweep — avg HR@k over PHL+PBIP'))
    plot_average(runs, DATASETS_ALL,
                 os.path.join(args.out_dir, f'sweep_avg_all_datasets_hrk{suffix}.svg'),
                 ('Top-{} runs — avg HR@k over all 5 datasets'.format(args.top)
                  if args.top else 'Embedding sweep — avg HR@k over all 5 datasets'))


if __name__ == '__main__':
    main()
