"""Regenerate the top-results landing page (results/README.md).

Reads results/experiment_log.csv, dedups to one row per distinct
(model, embedding_type) combination, keeps the best `phl_pbip_combined_hr1`
within each group, and writes:

  results/top_results.md             top-N markdown table
  results/figures/top_results_hrk.{svg,png}
                                     6-panel HR@1 figure (5 datasets + average)
  results/README.md                  landing page that embeds both

Run after scripts/analysis/harvest_results.py. Example:

    python scripts/analysis/harvest_results.py
    python scripts/analysis/build_top_results.py --top 10
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']
RANK_METRIC = 'phl_pbip_combined_hr1'


def infer_filter_sampling(row):
    """Return a short human-readable 'filter + sampling' string from a log row.

    Inference rules based on run_name prefix and known sweep conventions:
      sweep_<emb>                  -> "tools + random"
      sweep_posList_<emb>          -> "positive_list + random"
      sweep_<emb>_cl70 (or ..cl**) -> "tools + cluster<N>"
      sweep_posList_<emb>_cl70     -> "positive_list + cluster<N>"
      highconf_tsp_*               -> "highconf (TSP only)"
      highconf_pipeline_*          -> "highconf (pipeline_positive)"
      concat_*                     -> "concat" (falls back; filter not encoded)

    Anything unrecognised yields "—".
    """
    name = str(row.get('run_name', ''))
    low = name.lower()

    if low.startswith('highconf_tsp'):
        return 'highconf (TSP)'
    if low.startswith('highconf_pipeline'):
        return 'highconf (pipeline_positive)'

    pieces = []
    if '_poslist' in low or low.startswith('sweep_poslist_'):
        pieces.append('positive_list')
    elif low.startswith('sweep_') or low.startswith('concat_'):
        pieces.append('tools')

    cluster_token = next((tok for tok in low.split('_') if tok.startswith('cl') and tok[2:].isdigit()), None)
    if cluster_token:
        pieces.append(f'cluster{cluster_token[2:]}')
    elif pieces:
        pieces.append('random')

    return ' + '.join(pieces) if pieces else '—'


def load_and_dedup(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df[RANK_METRIC].notna()].copy()
    df[RANK_METRIC] = pd.to_numeric(df[RANK_METRIC], errors='coerce')
    df = df[df[RANK_METRIC].notna()]

    df['filter_sampling'] = df.apply(infer_filter_sampling, axis=1)
    df = df.sort_values(RANK_METRIC, ascending=False)
    dedup_cols = ['model', 'embedding_type', 'filter_sampling']
    df_best = df.drop_duplicates(subset=dedup_cols, keep='first')
    return df_best.reset_index(drop=True)


def write_table(df, top_n, out_path):
    top = df.head(top_n).copy()
    top.insert(0, 'rank', range(1, len(top) + 1))

    cols = [
        ('rank', 'Rank'),
        ('model', 'Model'),
        ('embedding_type', 'Embedding'),
        ('filter_sampling', 'Filter + Sampling'),
        ('PhageHostLearn_rh1', 'PHL rh@1'),
        ('PBIP_rh1', 'PBIP rh@1'),
        (RANK_METRIC, 'PHL+PBIP HR@1'),
        ('five_ds_mean_hr1', '5-ds mean HR@1'),
    ]

    header = '| ' + ' | '.join(h for _, h in cols) + ' |'
    sep = '|' + '|'.join(['---'] * len(cols)) + '|'
    lines = [header, sep]
    for _, r in top.iterrows():
        cells = []
        for key, _ in cols:
            v = r.get(key, '')
            if isinstance(v, float):
                cells.append(f'{v:.3f}' if pd.notna(v) else '—')
            else:
                cells.append(str(v) if pd.notna(v) else '—')
        lines.append('| ' + ' | '.join(cells) + ' |')

    table_md = '\n'.join(lines)
    out_path.write_text(table_md + '\n')
    print(f'Wrote {out_path}')
    return table_md, top


def plot_top_hrk(top, out_dir):
    n = len(top)
    datasets_with_avg = DATASETS + ['Average (5 datasets)']
    nrows, ncols = 2, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    cmap = plt.get_cmap('viridis')
    colors = [cmap(0.1 + 0.8 * i / max(n - 1, 1)) for i in range(n)]

    labels = [f"{r['embedding_type']} · {r['filter_sampling']}"
              for _, r in top.iterrows()]

    for idx, ds in enumerate(datasets_with_avg):
        ax = axes_flat[idx]
        if ds == 'Average (5 datasets)':
            vals = top['five_ds_mean_hr1'].fillna(0).values
        else:
            col = f'{ds}_rh1'
            vals = top[col].fillna(0).values if col in top.columns else np.zeros(n)

        ax.barh(range(n), vals, color=colors, edgecolor='black', linewidth=0.3)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f'#{i+1}' for i in range(n)], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('HR@1', fontsize=9)
        ax.set_title(ds, fontsize=10, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=7.5)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(n)]
    fig.legend(legend_handles, labels,
               loc='lower center', ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.08),
               title=f'Top {n} by PHL+PBIP HR@1 (rank 1 = best)',
               title_fontsize=9)

    fig.suptitle('HR@1 across validation datasets — top implementations',
                 fontsize=12, fontweight='bold', y=1.00)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])

    svg_path = out_dir / 'top_results_hrk.svg'
    png_path = out_dir / 'top_results_hrk.png'
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    fig.savefig(png_path, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {svg_path}')
    print(f'Wrote {png_path}')
    return png_path, svg_path


def write_landing_page(table_md, png_rel, svg_rel, n_total, top_n, out_path):
    content = f"""# ciPHer — current results

Auto-generated snapshot of the best runs so far. Regenerate with:

```bash
python scripts/analysis/harvest_results.py
python scripts/analysis/build_top_results.py --top {top_n}
```

## Top {top_n} implementations — ranked by PHL+PBIP HR@1

Deduplicated to one row per `(model, embedding, filter + sampling)` combo.
`PHL+PBIP HR@1` is the mean of rank_hosts and rank_phages HR@1 across the two
primary datasets (the advisor's headline metric). `5-ds mean HR@1` averages
HR@1 across all five validation datasets.

{table_md}

## HR@1 across datasets

Per-dataset `rank_hosts` HR@1 for the same top-{top_n} runs, plus the 5-dataset
average in the bottom-right panel.

![Top HR@1 across datasets]({png_rel})

SVG version: [`{svg_rel}`]({svg_rel})

## Full result log

All {n_total} runs (without dedup) live in [`experiment_log.csv`](experiment_log.csv).
`scripts/analysis/harvest_results.py` rebuilds it idempotently by scanning
`experiments/<model>/*/results/evaluation.json` and their accompanying
`experiment.json` provenance.
"""
    out_path.write_text(content)
    print(f'Wrote {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='results/experiment_log.csv')
    p.add_argument('--out-dir', default='results')
    p.add_argument('--top', type=int, default=10)
    args = p.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise SystemExit(f'ERROR: {csv_path} not found. Run harvest_results.py first.')

    df_best = load_and_dedup(csv_path)
    if df_best.empty:
        raise SystemExit('ERROR: no rows with phl_pbip_combined_hr1 in the log.')

    table_md, top = write_table(df_best, args.top, out_dir / 'top_results.md')
    png_path, svg_path = plot_top_hrk(top, figures_dir)

    png_rel = png_path.relative_to(out_dir).as_posix()
    svg_rel = svg_path.relative_to(out_dir).as_posix()
    write_landing_page(table_md, png_rel, svg_rel, len(df_best), args.top,
                       out_dir / 'README.md')


if __name__ == '__main__':
    main()
