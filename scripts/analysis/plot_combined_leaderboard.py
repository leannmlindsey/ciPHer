"""Horizontal bar chart of the top-N experiments by PHL+PBIP combined HR@1.

Reads results/experiment_log.csv (the harvest output) and produces:
  results/figures/combined_leaderboard_top<N>.svg

One bar per experiment, length = `phl_pbip_combined_hr1` (mean of
PHL & PBIP × rank_hosts + rank_phages HR@1 from cipher's NEW eval).
Color by model architecture. Reference line at OLD klebsiella's
baseline (0.245 under strict denominator).

Usage:
    python scripts/analysis/plot_combined_leaderboard.py
    python scripts/analysis/plot_combined_leaderboard.py --top 25
    python scripts/analysis/plot_combined_leaderboard.py --metric five_ds_mean_hr1
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


MODEL_COLORS = {
    'attention_mlp': '#1f77b4',         # blue
    'light_attention': '#2ca02c',       # green
    'light_attention_binary': '#9467bd', # purple
    'contrastive_encoder': '#ff7f0e',   # orange
}

# OLD klebsiella PHL HR@1 = 0.291 over their lenient n=326 → 0.245 strict.
# That's a CLASS-rank metric, not directly comparable to NEW host-rank
# numbers; shown as a reference for what the original codebase achieved
# on PHL alone.
OLD_KLEB_PHL_HR1_LENIENT = 0.291
OLD_KLEB_PHL_HR1_STRICT = 0.245


def f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--top', type=int, default=20)
    p.add_argument('--metric', default='phl_pbip_combined_hr1',
                   help='column from experiment_log.csv to sort/plot by')
    p.add_argument('--harvest', default='results/experiment_log.csv')
    p.add_argument('--out', default=None,
                   help='default: results/figures/combined_leaderboard_top<N>.svg')
    args = p.parse_args()

    with open(args.harvest) as fh:
        rows = [r for r in csv.DictReader(fh) if f(r.get(args.metric)) is not None]
    rows.sort(key=lambda r: -f(r[args.metric]))
    rows = rows[:args.top]

    if not rows:
        print(f'No rows with non-null {args.metric}.')
        return

    labels = [r['run_name'] for r in rows]
    values = [f(r[args.metric]) for r in rows]
    models = [r.get('model', '') for r in rows]
    embs = [r.get('embedding_type', '') for r in rows]

    # Tick labels: "<run_name> [<embedding>]"
    tick_labels = [f'{lab}  [{e}]' if e else lab for lab, e in zip(labels, embs)]
    colors = [MODEL_COLORS.get(m, '#999999') for m in models]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(rows) + 1.5)))
    y = list(range(len(rows)))
    ax.barh(y, values, color=colors, edgecolor='#333', linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.invert_yaxis()  # best on top
    ax.set_xlim(0, max(max(values), OLD_KLEB_PHL_HR1_LENIENT) * 1.15)
    ax.set_xlabel(args.metric.replace('_', ' '))
    ax.set_title(f'Top {len(rows)} cipher experiments by {args.metric}\n'
                  f'(NEW eval — host & phage ranking, z-score, competition ties)',
                  fontsize=11, fontweight='bold')

    # Annotate each bar with the value
    for i, v in enumerate(values):
        ax.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=8)

    # Reference lines (only meaningful when metric is PHL-related)
    if 'phl' in args.metric.lower() or args.metric == 'phl_pbip_combined_hr1':
        ax.axvline(OLD_KLEB_PHL_HR1_LENIENT, color='gray', ls=':', lw=1,
                   label=f'OLD klebsiella PHL HR@1 = {OLD_KLEB_PHL_HR1_LENIENT} '
                         '(class-rank, lenient denom)')
        ax.axvline(OLD_KLEB_PHL_HR1_STRICT, color='gray', ls='--', lw=1,
                   label=f'OLD klebsiella PHL HR@1 = {OLD_KLEB_PHL_HR1_STRICT} '
                         '(class-rank, strict denom)')
        ax.legend(loc='lower right', fontsize=8)

    # Color legend by model
    from matplotlib.patches import Patch
    seen = sorted({m for m in models if m in MODEL_COLORS})
    if seen:
        handles = [Patch(facecolor=MODEL_COLORS[m], edgecolor='#333', label=m)
                   for m in seen]
        ax.legend(handles=handles + ax.get_legend().legend_handles
                  if ax.get_legend() else handles,
                  loc='lower right', fontsize=8)

    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()

    out = args.out or f'results/figures/combined_leaderboard_top{len(rows)}.svg'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')

    # Also print the leaderboard to stdout
    print()
    print(f'{"#":>2}  {"value":>7}  {"model":<24}  {"run":<55}  {"embed":<22}')
    print('-' * 120)
    for i, (lab, v, m, e) in enumerate(zip(labels, values, models, embs), 1):
        print(f'{i:>2}  {v:>7.4f}  {m[:24]:<24}  {lab[:55]:<55}  {e[:22]:<22}')


if __name__ == '__main__':
    main()
