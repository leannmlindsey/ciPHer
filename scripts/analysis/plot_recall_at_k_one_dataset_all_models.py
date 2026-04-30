"""Recall@k=1..20 for one validation dataset, overlaying every cipher run
in a curated default list (or a CLI-specified list). Companion to
plot_recall_at_k_vs_competitors.py — same axis, same metric (phage-level
any-hit, OR mode, strict denominator) — but the curve dimension is the
cipher run rather than the competitor.

Reads exclusively from the harvest CSV (results/experiment_log.csv).

Output: results/figures/recall_at_k_one_dataset_all_models.svg/.png

Usage:
    python scripts/analysis/plot_recall_at_k_one_dataset_all_models.py
    python scripts/analysis/plot_recall_at_k_one_dataset_all_models.py \
        --dataset PBIP
    python scripts/analysis/plot_recall_at_k_one_dataset_all_models.py \
        --dataset PhageHostLearn \
        --runs sweep_prott5_mean_cl70 la_v3_uat_prott5_xl_seg8
    python scripts/analysis/plot_recall_at_k_one_dataset_all_models.py \
        --top 10
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


HARVEST_CSV = 'results/experiment_log.csv'
DEFAULT_DATASET = 'PhageHostLearn'
DEFAULT_MODE = 'OR'
DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']

# Fixed denominators per project policy (memory: feedback_strict_denominator).
# Convert harvested HR by multiplying by n_strict_phage to recover the
# numerator, then dividing by the value below.
FIXED_DENOM_PHAGE = {'CHEN': 3, 'GORODNICHIV': 3, 'UCSD': 11,
                     'PBIP': 103, 'PhageHostLearn': 100}  # paper-equivalent (excludes no-genome phages: 27 PHL, 1 PBIP)
FIXED_DENOM_TOTAL = sum(FIXED_DENOM_PHAGE.values())  # 220

# Default curated list — a representative cross-section of architectures,
# embeddings, and training regimes. Edit freely. Order is preserved in
# the legend.
DEFAULT_RUNS = [
    'concat_prott5_mean+kmer_li10_k5',
    'sweep_esm2_650m_mean_cl70',
    'sweep_prott5_mean_cl70',
    'sweep_esm2_3b_mean_cl70',
    'sweep_kmer_murphy8_k5',
    'la_seg4_posList_cl70',
    'highconf_pipeline_K_prott5_mean',
    'lab_esm2_650m_full_highconf_pipeline',
]

OUT_SVG = 'results/figures/recall_at_k_one_dataset_all_models.svg'


def f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def load_rows():
    with open(HARVEST_CSV) as fh:
        return list(csv.DictReader(fh))


def hits_for(row, dataset, mode):
    """Numerator = harvested HR_strict × n_strict_phage."""
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
    """Fixed-denominator phage count (ignores per-run n_strict_phage)."""
    if dataset == 'overall':
        return FIXED_DENOM_TOTAL
    return FIXED_DENOM_PHAGE.get(dataset)


def pick_runs(rows, args):
    if args.runs:
        names = args.runs
    elif args.top:
        ranked = []
        for r in rows:
            v = curve_for(r, args.dataset, args.mode).get(1)
            if v is not None:
                ranked.append((v, r['run_name']))
        ranked.sort(reverse=True)
        names = [n for _, n in ranked[:args.top]]
    else:
        names = list(DEFAULT_RUNS)
    by_name = {r['run_name']: r for r in rows}
    selected = []
    missing = []
    for n in names:
        if n in by_name:
            selected.append(by_name[n])
        else:
            missing.append(n)
    if missing:
        print(f'(skipped — not in harvest CSV: {", ".join(missing)})')
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=DEFAULT_DATASET,
                    choices=DATASETS + ['overall'])
    ap.add_argument('--mode', default=DEFAULT_MODE, choices=('OR', 'K', 'O'))
    ap.add_argument('--runs', nargs='+',
                    help='explicit run_name list (overrides --top and defaults)')
    ap.add_argument('--top', type=int,
                    help='auto-pick top-N runs by HR@1 on the chosen dataset')
    ap.add_argument('--out', default=OUT_SVG)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    rows = load_rows()
    runs = pick_runs(rows, args)
    if not runs:
        raise SystemExit('ERROR: no matching runs found.')

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ks = list(range(1, 21))

    n_curves = len(runs)
    cmap = plt.get_cmap('tab10' if n_curves <= 10 else 'tab20')
    plotted = []  # (final_label, hr1, hr20)
    for i, r in enumerate(runs):
        curve = curve_for(r, args.dataset, args.mode)
        ys = [curve.get(k) for k in ks]
        if all(v is None for v in ys):
            print(f'(skipped — no {args.mode} data on {args.dataset}: {r["run_name"]})')
            continue
        color = cmap(i % cmap.N)
        label = f'{r["run_name"]} [{r.get("model","?")}]'
        ax.plot(ks, ys, color=color, lw=2.0,
                marker='o', markersize=4, label=label)
        plotted.append((label, curve.get(1), curve.get(20), r['run_name'],
                        r.get('model', '?')))

    n = n_phages(runs[0], args.dataset)
    n_str = f'n={n}' if n is not None else 'n=?'
    title_ds = ('phage-weighted overall' if args.dataset == 'overall'
                else args.dataset)
    ax.set_title(f'Recall@k for {title_ds} ({n_str} phages, FIXED denom)\n'
                 f'phage-level any-hit ({args.mode} mode), '
                 f'project-policy fixed denominator',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('k')
    ax.set_ylabel('Recall@k')
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(args.out, format='svg', bbox_inches='tight')
    fig.savefig(args.out.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {args.out}')

    # Headline values (HR@1 / HR@20) for the lab notebook
    print()
    print(f'Dataset: {args.dataset}   mode: {args.mode}   '
          f'(sorted by HR@1 desc)')
    print(f'  {"HR@1":>6}  {"HR@20":>6}  {"model":<24}  run_name')
    plotted.sort(key=lambda x: -(x[1] or -1))
    for _, hr1, hr20, run_name, model in plotted:
        s1 = f'{hr1:.3f}' if hr1 is not None else '   --'
        s20 = f'{hr20:.3f}' if hr20 is not None else '   --'
        print(f'  {s1:>6}  {s20:>6}  {model:<24}  {run_name}')


if __name__ == '__main__':
    main()
