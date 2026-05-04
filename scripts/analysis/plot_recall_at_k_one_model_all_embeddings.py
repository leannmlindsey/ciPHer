"""Recall@k=1..20 for one model architecture, overlaying every protein
representation (embedding family) on a single panel. Companion to
plot_recall_at_k_vs_competitors.py — same axis, same metric (phage-level
any-hit, OR mode, strict denominator) — but the curve dimension is the
embedding family rather than the competitor.

The curve for each embedding family is the best-OR-HR@1 run available
for that embedding in the harvest CSV. Because the harvest contains
multiple training regimes for some embeddings (tools / posList,
random / cluster70), the chosen run name is shown in the legend so the
regime is transparent. Use --runs to pin a specific list, or --regime
to restrict to a single regime.

Reads exclusively from the harvest CSV (results/experiment_log.csv).

Output: results/figures/recall_at_k_one_model_all_embeddings.svg/.png

Usage:
    python scripts/analysis/plot_recall_at_k_one_model_all_embeddings.py
    python scripts/analysis/plot_recall_at_k_one_model_all_embeddings.py \
        --dataset PBIP
    python scripts/analysis/plot_recall_at_k_one_model_all_embeddings.py \
        --dataset overall
    python scripts/analysis/plot_recall_at_k_one_model_all_embeddings.py \
        --regime cl70                 # restrict to sweep_<emb>_cl70 runs
    python scripts/analysis/plot_recall_at_k_one_model_all_embeddings.py \
        --runs sweep_prott5_mean_cl70 sweep_kmer_murphy8_k5
"""

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


HARVEST_CSV = 'results/experiment_log.csv'
DEFAULT_MODEL = 'attention_mlp'
DEFAULT_DATASET = 'PhageHostLearn'
DEFAULT_MODE = 'OR'
DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']

# Fixed denominators per project policy (memory: feedback_strict_denominator).
FIXED_DENOM_PHAGE = {'CHEN': 3, 'GORODNICHIV': 3, 'UCSD': 11,
                     'PBIP': 103, 'PhageHostLearn': 100}  # paper-equivalent (excludes no-genome phages: 27 PHL, 1 PBIP)
FIXED_DENOM_TOTAL = sum(FIXED_DENOM_PHAGE.values())

# Order of embedding families in the legend. Listed embeddings missing
# from the harvest are simply skipped.
EMBEDDING_ORDER = [
    'kmer_aa20_k3', 'kmer_aa20_k4',
    'kmer_murphy8_k5', 'kmer_murphy10_k5', 'kmer_li10_k5',
    'esm2_150m_mean', 'esm2_650m_mean', 'esm2_650m_seg4',
    'esm2_3b_mean',
    'prott5_mean',
]

# Color per embedding family. Cool family for k-mers, warm for ESM-2,
# distinct colour for ProtT5.
EMBEDDING_COLOR = {
    'kmer_aa20_k3':       '#a6cee3',
    'kmer_aa20_k4':       '#1f78b4',
    'kmer_murphy8_k5':    '#33a02c',
    'kmer_murphy10_k5':   '#b2df8a',
    'kmer_li10_k5':       '#6a3d9a',
    'esm2_150m_mean':     '#fdbf6f',
    'esm2_650m_mean':     '#ff7f00',
    'esm2_650m_seg4':     '#fb9a99',
    'esm2_3b_mean':       '#e31a1c',
    'prott5_mean':        '#000000',
}

OUT_SVG = 'results/figures/recall_at_k_one_model_all_embeddings.svg'


def f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def load_rows():
    with open(HARVEST_CSV) as fh:
        return list(csv.DictReader(fh))


def regime_of(run_name):
    """Classify a run by training-set filter / sampling regime.

    sweep_posList_<emb>_cl70   → posList_cl70
    sweep_posList_<emb>        → posList_random
    sweep_<emb>_cl70           → tools_cl70   (a.k.a. "cl70")
    sweep_<emb>                → tools_random (a.k.a. "default")
    other                      → other
    """
    if not run_name.startswith('sweep_'):
        return 'other'
    is_pos = '_posList_' in run_name or run_name.startswith('sweep_posList_')
    is_cl70 = run_name.endswith('_cl70')
    if is_pos and is_cl70: return 'posList_cl70'
    if is_pos:             return 'posList_random'
    if is_cl70:            return 'tools_cl70'
    return 'tools_random'


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


def candidate_sweep_runs(rows, model):
    """attention_mlp single-embedding sweep_* runs (excludes concat)."""
    out = []
    for r in rows:
        if r.get('model') != model: continue
        rn = r.get('run_name', '')
        if not rn.startswith('sweep_'): continue
        if r.get('embedding_type_2'): continue       # exclude concat
        if '+' in rn or 'concat' in rn: continue     # belt & braces
        et = r.get('embedding_type', '')
        if not et: continue
        out.append((r, rn, et))
    return out


def auto_select_per_embedding(rows, model, dataset, mode, regime):
    """For each embedding family, pick the run with the highest HR@1
    on `dataset` in the given regime. Returns ordered list of run rows."""
    by_emb = defaultdict(list)
    for r, rn, et in candidate_sweep_runs(rows, model):
        if regime != 'any' and regime_of(rn) != regime:
            continue
        v1 = curve_for(r, dataset, mode).get(1)
        if v1 is None:
            continue
        by_emb[et].append((v1, rn, r))

    selected = []
    for et in EMBEDDING_ORDER:
        cands = by_emb.get(et)
        if not cands: continue
        cands.sort(reverse=True)
        _, _, best_row = cands[0]
        selected.append(best_row)
    # Append any embedding families not in EMBEDDING_ORDER (forward-compat)
    for et, cands in by_emb.items():
        if et in EMBEDDING_ORDER: continue
        cands.sort(reverse=True)
        selected.append(cands[0][2])
    return selected


def color_for(et, fallback_idx):
    if et in EMBEDDING_COLOR:
        return EMBEDDING_COLOR[et]
    cmap = plt.get_cmap('tab20')
    return cmap(fallback_idx % cmap.N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL,
                    help='filter by model column (default: attention_mlp)')
    ap.add_argument('--dataset', default=DEFAULT_DATASET,
                    choices=DATASETS + ['overall'])
    ap.add_argument('--mode', default=DEFAULT_MODE, choices=('OR', 'K', 'O'))
    ap.add_argument('--regime', default='any',
                    choices=('any', 'tools_random', 'tools_cl70',
                             'posList_random', 'posList_cl70'),
                    help=('restrict auto-selection to one training regime; '
                          '"any" picks the best-OR run per embedding regardless '
                          'of regime (default).'))
    ap.add_argument('--runs', nargs='+',
                    help='explicit run_name list (overrides auto selection)')
    ap.add_argument('--out', default=OUT_SVG)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    rows = load_rows()
    if args.runs:
        by_name = {r['run_name']: r for r in rows}
        selected = []
        missing = []
        for n in args.runs:
            if n in by_name: selected.append(by_name[n])
            else:            missing.append(n)
        if missing:
            print(f'(skipped — not in harvest CSV: {", ".join(missing)})')
    else:
        selected = auto_select_per_embedding(rows, args.model, args.dataset,
                                              args.mode, args.regime)

    if not selected:
        raise SystemExit('ERROR: no matching runs found.')

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ks = list(range(1, 21))
    plotted = []  # rows for headline table

    for i, r in enumerate(selected):
        et = r.get('embedding_type', '?')
        rn = r.get('run_name', '?')
        curve = curve_for(r, args.dataset, args.mode)
        ys = [curve.get(k) for k in ks]
        if all(v is None for v in ys):
            print(f'(skipped — no {args.mode} data on {args.dataset}: {rn})')
            continue
        color = color_for(et, i)
        regime = regime_of(rn)
        label = f'{et}  [{regime}]  ({rn})'
        ax.plot(ks, ys, color=color, lw=2.0,
                marker='o', markersize=4, label=label)
        plotted.append((et, regime, rn, curve.get(1), curve.get(20)))

    n = n_phages(selected[0], args.dataset)
    n_str = f'n={n}' if n is not None else 'n=?'
    title_ds = ('phage-weighted overall' if args.dataset == 'overall'
                else args.dataset)
    regime_note = ('best-of-regime per embedding' if args.regime == 'any'
                   else f'regime={args.regime}')
    ax.set_title(f'Recall@k for {title_ds} ({n_str} phages, FIXED denom)\n'
                 f'one curve per embedding family — model={args.model}, '
                 f'mode={args.mode}, {regime_note}',
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
    print(f'Model: {args.model}   Dataset: {args.dataset}   '
          f'Mode: {args.mode}   Regime filter: {args.regime}')
    print(f'  {"HR@1":>6}  {"HR@20":>6}  {"embedding":<20}  '
          f'{"regime":<16}  run_name')
    plotted.sort(key=lambda x: -(x[3] or -1))
    for et, regime, rn, hr1, hr20 in plotted:
        s1  = f'{hr1:.3f}'  if hr1  is not None else '   --'
        s20 = f'{hr20:.3f}' if hr20 is not None else '   --'
        print(f'  {s1:>6}  {s20:>6}  {et:<20}  {regime:<16}  {rn}')


if __name__ == '__main__':
    main()
