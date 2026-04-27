"""Bar chart of mean(HR@k) across k=1..20, per experiment.

For each experiment with results/evaluation.json present, computes:
    mean_hr_at_k = mean over k=1..20 of HR@k, averaged across
                   PHL & PBIP × rank_hosts + rank_phages

This is a more robust single-number summary than HR@1 alone — it
rewards models that get to the right answer somewhere in the top 20,
not just the very top.

Output: results/figures/mean_hrk_leaderboard_top<N>.svg

Companion to plot_combined_leaderboard.py (HR@1 only).

Usage:
    python scripts/analysis/plot_mean_hrk_leaderboard.py
    python scripts/analysis/plot_mean_hrk_leaderboard.py --top 25
    python scripts/analysis/plot_mean_hrk_leaderboard.py --datasets all  # 5 datasets
    python scripts/analysis/plot_mean_hrk_leaderboard.py --max-k 10      # k=1..10 instead
"""

import argparse
import json
import os
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PRIMARY = ['PhageHostLearn', 'PBIP']
ALL_DS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']
DIRECTIONS = ['rank_hosts', 'rank_phages']

MODEL_COLORS = {
    'attention_mlp': '#1f77b4',
    'light_attention': '#2ca02c',
    'light_attention_binary': '#9467bd',
    'contrastive_encoder': '#ff7f0e',
}


def mean_hr_at_k(ev, datasets, max_k):
    """Across the requested datasets and both directions, return mean HR@k
    over k=1..max_k. None if no data."""
    vals = []
    for ds in datasets:
        d = ev.get(ds, {})
        for direction in DIRECTIONS:
            hrk = d.get(direction, {}).get('hr_at_k', {})
            for k in range(1, max_k + 1):
                v = hrk.get(str(k))
                if v is None:
                    v = hrk.get(k)
                if v is not None:
                    vals.append(float(v))
    return sum(vals) / len(vals) if vals else None


def load_runs(experiments_glob, datasets, max_k):
    out = []
    for path in sorted(glob(experiments_glob)):
        ev_path = os.path.join(path, 'results', 'evaluation.json')
        if not os.path.exists(ev_path):
            continue
        with open(ev_path) as f:
            ev = json.load(f)
        m = mean_hr_at_k(ev, datasets, max_k)
        if m is None:
            continue
        # Per-dataset means too, for the side stripe
        per_ds = {}
        for ds in datasets:
            per_ds[ds] = mean_hr_at_k(ev, [ds], max_k)
        # Pull metadata
        run_name = os.path.basename(path)
        model = os.path.basename(os.path.dirname(path))
        emb = ''
        meta_p = os.path.join(path, 'experiment.json')
        if os.path.exists(meta_p):
            try:
                meta = json.load(open(meta_p))
                emb = meta.get('config', {}).get('data', {}).get('embedding_type', '') or ''
            except Exception:
                pass
        out.append({'run': run_name, 'model': model, 'emb': emb,
                    'mean': m, 'per_ds': per_ds})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--top', type=int, default=20)
    p.add_argument('--max-k', type=int, default=20)
    p.add_argument('--datasets', choices=['primary', 'all'], default='primary',
                   help='primary = PHL+PBIP only; all = 5 validation datasets')
    p.add_argument('--experiments-glob', default='experiments/*/*',
                   help='glob for experiment dirs')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    datasets = PRIMARY if args.datasets == 'primary' else ALL_DS
    runs = load_runs(args.experiments_glob, datasets, args.max_k)
    runs.sort(key=lambda r: -r['mean'])
    runs = runs[:args.top]

    if not runs:
        print('No runs with evaluation.json found.')
        return

    labels = [f'{r["run"]}  [{r["emb"]}]' if r["emb"] else r['run'] for r in runs]
    values = [r['mean'] for r in runs]
    colors = [MODEL_COLORS.get(r['model'], '#999') for r in runs]

    fig, ax = plt.subplots(figsize=(10.5, max(4, 0.36 * len(runs) + 1.5)))
    y = list(range(len(runs)))
    ax.barh(y, values, color=colors, edgecolor='#333', linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel(f'mean HR@k for k=1..{args.max_k}, '
                  f'avg over {",".join(datasets)} × rank_hosts + rank_phages')
    ds_label = 'PHL+PBIP' if args.datasets == 'primary' else 'all 5 datasets'
    ax.set_title(f'Top {len(runs)} cipher experiments by mean HR@k=1..{args.max_k}\n'
                 f'({ds_label}, both ranking directions; NEW eval, '
                 'z-score, competition ties)',
                 fontsize=11, fontweight='bold')

    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)

    # Color legend by model
    from matplotlib.patches import Patch
    seen = sorted({r['model'] for r in runs if r['model'] in MODEL_COLORS})
    if seen:
        handles = [Patch(facecolor=MODEL_COLORS[m], edgecolor='#333', label=m)
                   for m in seen]
        ax.legend(handles=handles, loc='lower right', fontsize=8, title='model')

    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()

    suffix = '_all5' if args.datasets == 'all' else ''
    out = args.out or (f'results/figures/mean_hrk_leaderboard_top{len(runs)}'
                       f'_k{args.max_k}{suffix}.svg')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')

    # Stdout leaderboard
    print()
    print(f'{"#":>2}  {"mean":>6}  {"model":<22}  {"run":<55}  {"emb":<22}')
    print('-' * 115)
    for i, r in enumerate(runs, 1):
        print(f'{i:>2}  {r["mean"]:>6.4f}  {r["model"][:22]:<22}  '
              f'{r["run"][:55]:<55}  {r["emb"][:22]:<22}')


if __name__ == '__main__':
    main()
