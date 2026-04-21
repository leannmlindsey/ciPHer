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


def load_runs(model, name_filter):
    runs = []
    for eval_path in sorted(glob(f'experiments/{model}/*/results/evaluation.json')):
        exp_dir = os.path.dirname(os.path.dirname(eval_path))
        name = os.path.basename(exp_dir)
        if name_filter and name_filter not in name:
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


def combined_score(run):
    vals = []
    for ds in DATASETS_PRIMARY:
        r = run['data'].get(ds, {})
        for direction, _ in DIRECTIONS:
            v = r.get(direction, {}).get('hr_at_k', {}).get('1')
            if v is not None:
                vals.append(v)
    return sum(vals) / max(len(vals), 1)


def assign_colors(labels):
    """pLM -> viridis, kmer -> autumn. Best-ranked first gets darkest shade."""
    plm = [l for l in labels if not l.startswith('kmer')]
    kmer = [l for l in labels if l.startswith('kmer')]
    out = {}
    plm_cmap = plt.get_cmap('viridis')
    kmer_cmap = plt.get_cmap('autumn')
    for i, l in enumerate(plm):
        t = i / max(len(plm) - 1, 1)
        out[l] = plm_cmap(0.1 + 0.75 * t)
    for i, l in enumerate(kmer):
        t = i / max(len(kmer) - 1, 1)
        out[l] = kmer_cmap(0.05 + 0.6 * t)
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


def plot_grid(runs, datasets, out_path, title):
    """One subplot per (dataset, direction). Rows=datasets, cols=directions."""
    runs_sorted = sorted(runs, key=combined_score, reverse=True)
    labels = [r['label'] for r in runs_sorted]
    colors = assign_colors(labels)

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
                ax.plot(ks, ys, label=r['label'], color=colors[r['label']],
                        lw=1.7, alpha=0.95)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='attention_mlp')
    p.add_argument('--filter', default='sweep_')
    p.add_argument('--out-dir', default='results/figures')
    args = p.parse_args()

    runs = load_runs(args.model, args.filter)
    if not runs:
        print('No runs found.')
        return

    os.makedirs(args.out_dir, exist_ok=True)
    plot_grid(runs, DATASETS_PRIMARY,
              os.path.join(args.out_dir, 'sweep_phl_pbip_hrk.svg'),
              'Embedding sweep — PhageHostLearn & PBIP (primary datasets)')
    plot_grid(runs, DATASETS_ALL,
              os.path.join(args.out_dir, 'sweep_all_datasets_hrk.svg'),
              'Embedding sweep — all 5 validation datasets')


if __name__ == '__main__':
    main()
