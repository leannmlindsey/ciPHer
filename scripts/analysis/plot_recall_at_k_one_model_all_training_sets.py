"""Recall@k=1..20 for one model architecture + one embedding, overlaying
every training-dataset recipe on a single panel. Companion to
plot_recall_at_k_vs_competitors.py — same axis, same metric (phage-level
any-hit, OR mode, strict denominator) — but the curve dimension is the
training-dataset recipe (tools / pipeline_positive / highconf v1 (hc) /
v2 / v3 / v4) rather than the competitor.

Holds architecture and embedding fixed so the only varying axis is the
training-data filter. Default fixed pair: attention_mlp + prott5_mean,
which has the largest spread of training-dataset recipes in the harvest.

Reads exclusively from the harvest CSV (results/experiment_log.csv).

Output: results/figures/recall_at_k_one_model_all_training_sets.svg/.png

Usage:
    python scripts/analysis/plot_recall_at_k_one_model_all_training_sets.py
    python scripts/analysis/plot_recall_at_k_one_model_all_training_sets.py \
        --dataset overall
    python scripts/analysis/plot_recall_at_k_one_model_all_training_sets.py \
        --runs sweep_prott5_mean_cl70 highconf_pipeline_K_prott5_mean v4_attention_mlp_prott5_mean
"""

import argparse
import csv
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


HARVEST_CSV = 'results/experiment_log.csv'
DEFAULT_DATASET = 'PhageHostLearn'
DEFAULT_MODE = 'OR'
DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']

# Fixed denominators per project policy (memory: feedback_strict_denominator).
FIXED_DENOM_PHAGE = {'CHEN': 3, 'GORODNICHIV': 3, 'UCSD': 11,
                     'PBIP': 103, 'PhageHostLearn': 100}  # paper-equivalent (excludes no-genome phages: 27 PHL, 1 PBIP)
FIXED_DENOM_TOTAL = sum(FIXED_DENOM_PHAGE.values())

# Default fixed pair (model architecture + embedding family) and the curated
# list of training-dataset recipes for that pair. Each entry is
# (recipe_label, run_name). Edit the run_name to swap embeddings/architecture
# for a different cross-section. Order is preserved in the legend.
#
# Pair: attention_mlp + prott5_mean — chosen because it has the broadest
# training-dataset coverage in the harvest CSV.
DEFAULT_RECIPES = [
    ('tools (sweep)',           'sweep_prott5_mean_cl70'),
    ('pipeline_positive',       'sweep_posList_prott5_mean_cl70'),
    ('highconf v1 (hc)',        'highconf_pipeline_K_prott5_mean'),
    ('highconf v2 strict',      'v2_strict_prott5_mean'),
    ('highconf v3 strict',      'v3_strict_prott5_mean'),
    ('highconf v3 k_v2o',       'v3_k_v2o_prott5_mean'),
    ('highconf v3 uat (no cap)','v3uat_nocap_attention_mlp_prott5_mean'),
    ('highconf v4',             'v4_attention_mlp_prott5_mean'),
]

# Color per training-dataset family — cooler shades for upstream/permissive
# regimes, warmer for the highconf-gated v1..v4 succession.
RECIPE_COLOR = {
    'tools (sweep)':            '#1f77b4',
    'pipeline_positive':        '#17becf',
    'highconf v1 (hc)':         '#ffbb78',
    'highconf v2 strict':       '#ff7f0e',
    'highconf v3 strict':       '#d62728',
    'highconf v3 k_v2o':        '#e377c2',
    'highconf v3 uat (no cap)': '#9467bd',
    'highconf v4':              '#2ca02c',
}

OUT_SVG = 'results/figures/recall_at_k_one_model_all_training_sets.svg'


def f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def load_rows():
    with open(HARVEST_CSV) as fh:
        return list(csv.DictReader(fh))


def hits_for(row, dataset, mode):
    n_strict = f(row.get(f'{dataset}_n_strict_phage'))
    if not n_strict:
        return {k: None for k in range(1, 21)}
    n_strict = int(n_strict)
    return {k: (round(f(row.get(f'{dataset}_{mode}_phage2host_anyhit_HR{k}'))
                       * n_strict)
                if f(row.get(f'{dataset}_{mode}_phage2host_anyhit_HR{k}'))
                   is not None else None)
            for k in range(1, 21)}


def curve_for(row, dataset, mode):
    """HR@k under the FIXED per-dataset denominator."""
    if dataset == 'overall':
        num = {k: 0 for k in range(1, 21)}
        any_data = False
        for ds in DATASETS:
            hits = hits_for(row, ds, mode)
            for k in range(1, 21):
                if hits[k] is not None:
                    num[k] += hits[k]
                    any_data = True
        if not any_data:
            return {k: None for k in range(1, 21)}
        return {k: num[k] / FIXED_DENOM_TOTAL for k in range(1, 21)}
    hits = hits_for(row, dataset, mode)
    denom = FIXED_DENOM_PHAGE[dataset]
    return {k: (hits[k] / denom if hits[k] is not None else None)
            for k in range(1, 21)}


def n_phages(row, dataset):
    if dataset == 'overall':
        return FIXED_DENOM_TOTAL
    return FIXED_DENOM_PHAGE.get(dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=DEFAULT_DATASET,
                    choices=DATASETS + ['overall'])
    ap.add_argument('--mode', default=DEFAULT_MODE, choices=('OR', 'K', 'O'))
    ap.add_argument('--runs', nargs='+',
                    help=('explicit run_name list (overrides default recipe '
                          'list). Legend label falls back to run_name.'))
    ap.add_argument('--out', default=OUT_SVG)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    rows = load_rows()
    by_name = {r['run_name']: r for r in rows}

    if args.runs:
        recipes = [(rn, rn) for rn in args.runs]
    else:
        recipes = DEFAULT_RECIPES

    # First pass: gather rows so we can decide on uniform arch/embedding.
    gathered = []  # (label, run_name, row, curve, arch, emb)
    missing = []
    for label, run_name in recipes:
        r = by_name.get(run_name)
        if r is None:
            missing.append(run_name)
            continue
        curve = curve_for(r, args.dataset, args.mode)
        ys = [curve.get(k) for k in list(range(1, 21))]
        if all(v is None for v in ys):
            print(f'(skipped — no {args.mode} data on {args.dataset}: {run_name})')
            continue
        gathered.append((label, run_name, r, curve,
                         r.get('model', '?'), r.get('embedding_type', '?')))

    archs = {g[4] for g in gathered}
    embs  = {g[5] for g in gathered}
    uniform_pair = (len(archs) == 1 and len(embs) == 1)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ks = list(range(1, 21))
    plotted = []  # (label, hr1, hr20, run_name, model, embedding)
    n_phages_seen = None

    for i, (label, run_name, r, curve, arch, emb) in enumerate(gathered):
        ys = [curve.get(k) for k in ks]
        color = RECIPE_COLOR.get(label)
        if color is None:
            cmap = plt.get_cmap('tab10')
            color = cmap(i % cmap.N)
        # When arch + embedding are the same on every curve, omit them
        # from the legend and surface them in the title instead.
        if uniform_pair:
            leg = f'{label}  ({run_name})'
        else:
            leg = f'{label}  [{arch} / {emb}]  ({run_name})'
        ax.plot(ks, ys, color=color, lw=2.0,
                marker='o', markersize=4, label=leg)
        plotted.append((label, curve.get(1), curve.get(20), run_name, arch, emb))
        if n_phages_seen is None:
            n_phages_seen = n_phages(r, args.dataset)

    if missing:
        print(f'(skipped — not in harvest CSV: {", ".join(missing)})')
    if not plotted:
        raise SystemExit('ERROR: none of the requested runs have curve data.')

    n_str = f'n={n_phages_seen}' if n_phages_seen is not None else 'n=?'
    title_ds = ('phage-weighted overall' if args.dataset == 'overall'
                else args.dataset)
    fixed = (f'{next(iter(archs))} / {next(iter(embs))}'
             if uniform_pair else 'mixed')

    ax.set_title(f'Recall@k for {title_ds} ({n_str} phages, FIXED denom)\n'
                 f'one curve per training-dataset recipe — {fixed}, mode={args.mode}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('k')
    ax.set_ylabel('Recall@k')
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=7.5, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(args.out, format='svg', bbox_inches='tight')
    fig.savefig(args.out.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {args.out}')

    # Headline values for the lab notebook (sorted by HR@1 desc)
    print()
    print(f'Dataset: {args.dataset}   Mode: {args.mode}   Pair: {fixed}')
    print(f'  {"HR@1":>6}  {"HR@20":>6}  {"recipe":<28}  run_name')
    plotted.sort(key=lambda x: -(x[1] or -1))
    for label, hr1, hr20, run_name, _arch, _emb in plotted:
        s1  = f'{hr1:.3f}'  if hr1  is not None else '   --'
        s20 = f'{hr20:.3f}' if hr20 is not None else '   --'
        print(f'  {s1:>6}  {s20:>6}  {label:<28}  {run_name}')


if __name__ == '__main__':
    main()
