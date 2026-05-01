"""6-panel per-dataset figure showing the two cipher runs that feed the
best hybrid OR — for the writeup's "where does the hybrid lift come from"
diagnostic.

Per panel (one per dataset + overall), 5 curves:
    - K-source K-only        — the K-head ranks that feed the hybrid
    - K-source intra-OR      — the K-source's best by itself
    - O-source O-only        — the O-head ranks that feed the hybrid
    - O-source intra-OR      — the O-source's best by itself
    - Hybrid OR              — cross-model min(K_rank_A, O_rank_B)

Reads from the harvest CSV + a hybrid-curves JSON. Default pair is the
2026-04-30 leaderboard winner: K from sweep_posList_esm2_3b_mean_cl70,
O from sweep_kmer_aa20_k4.

Output: results/figures/recall_at_k_per_dataset_hybrid_pair.svg/.png

Usage:
    python scripts/analysis/plot_recall_at_k_per_dataset_hybrid_pair.py
    # Override the pair:
    python scripts/analysis/plot_recall_at_k_per_dataset_hybrid_pair.py \
        --k-run la_v3_uat_prott5_xl_seg8 --o-run sweep_prott5_mean_cl70 \
        --hybrid-json results/analysis/hybrid_or_la_K_sweep_O_curves.json
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


HARVEST_CSV = 'results/experiment_log.csv'
DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']
FIXED_DENOM_PHAGE = {'CHEN': 3, 'GORODNICHIV': 3, 'UCSD': 11,
                     'PBIP': 103, 'PhageHostLearn': 100}
FIXED_DENOM_TOTAL = sum(FIXED_DENOM_PHAGE.values())  # 220

DEFAULT_K_RUN = 'sweep_posList_esm2_3b_mean_cl70'
DEFAULT_O_RUN = 'sweep_kmer_aa20_k4'
DEFAULT_HYBRID_JSON = ('results/analysis/'
                       'hybrid_or_esm2_3b_K_kmer_aa20_O_curves.json')

OUT_SVG = 'results/figures/recall_at_k_per_dataset_hybrid_pair.svg'


def f(v):
    try: return float(v) if v else None
    except (TypeError, ValueError): return None


def load_run(run_name):
    with open(HARVEST_CSV) as fh:
        for r in csv.DictReader(fh):
            if r.get('run_name') == run_name:
                return r
    return None


def hits(row, ds, mode_col):
    n = f(row.get(f'{ds}_n_strict_phage'))
    hr = f(row.get(f'{ds}_{mode_col}_phage2host_anyhit_HR1'))
    return round(hr * n) if (n is not None and hr is not None) else None


def curve(row, ds, mode_col, ks=range(1, 21)):
    n = f(row.get(f'{ds}_n_strict_phage'))
    if n is None: return {k: None for k in ks}
    n = int(n)
    out = {}
    denom = FIXED_DENOM_PHAGE[ds]
    for k in ks:
        hr = f(row.get(f'{ds}_{mode_col}_phage2host_anyhit_HR{k}'))
        out[k] = (round(hr * n) / denom) if hr is not None else None
    return out


def overall_curve(row, mode_col, ks=range(1, 21)):
    out = {k: 0 for k in ks}
    has = False
    for ds in DATASETS:
        n = f(row.get(f'{ds}_n_strict_phage'))
        if n is None: continue
        n = int(n)
        for k in ks:
            hr = f(row.get(f'{ds}_{mode_col}_phage2host_anyhit_HR{k}'))
            if hr is not None:
                out[k] += round(hr * n); has = True
    return ({k: out[k] / FIXED_DENOM_TOTAL for k in ks}
            if has else {k: None for k in ks})


def load_hybrid(path):
    """Returns {ds_or_overall: {k: hr}} re-divided to fixed denom.
    Works for either schema (with 'hybrid_hr@k' top-level)."""
    with open(path) as fh:
        d = json.load(fh)
    out = {}
    overall_num = defaultdict(int)
    for ds, c in d.get('datasets', {}).items():
        if ds == 'overall': continue
        n_buggy = c.get('n')
        denom = FIXED_DENOM_PHAGE.get(ds)
        out[ds] = {}
        for k, v in c.get('hybrid_hr@k', {}).items():
            hits = round(v * n_buggy) if n_buggy else None
            if hits is not None and denom is not None:
                out[ds][int(k)] = hits / denom
                overall_num[int(k)] += hits
            else:
                out[ds][int(k)] = v
    out['overall'] = {k: overall_num[k] / FIXED_DENOM_TOTAL
                      for k in overall_num}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k-run', default=DEFAULT_K_RUN)
    ap.add_argument('--o-run', default=DEFAULT_O_RUN)
    ap.add_argument('--hybrid-json', default=DEFAULT_HYBRID_JSON)
    ap.add_argument('--out', default=OUT_SVG)
    args = ap.parse_args()

    k_row = load_run(args.k_run)
    o_row = load_run(args.o_run)
    if k_row is None: raise SystemExit(f'no harvest row for {args.k_run}')
    if o_row is None: raise SystemExit(f'no harvest row for {args.o_run}')
    hybrid = load_hybrid(args.hybrid_json) if os.path.exists(args.hybrid_json) else None

    # Style — K-source = blue family, O-source = green family, hybrid = purple
    style = {
        'k_src_K':  ('#9ecae1', '-',  1.6, 's', f'{args.k_run} K-only (feeds hybrid)'),
        'k_src_OR': ('#08306b', '-',  2.4, 'o', f'{args.k_run} intra-OR'),
        'o_src_O':  ('#a1d99b', '-',  1.6, '^', f'{args.o_run} O-only (feeds hybrid)'),
        'o_src_OR': ('#00441b', '-',  2.4, 'o', f'{args.o_run} intra-OR'),
        'hybrid':   ('#6a3d9a', '-',  2.8, 'D', f'hybrid OR (cross-model)'),
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    panels = list(DATASETS) + ['overall']
    ks = list(range(1, 21))

    for i, ds in enumerate(panels):
        ax = axes[i // 3, i % 3]

        # Five cipher curves
        if ds == 'overall':
            ksrc_K  = overall_curve(k_row, 'K')
            ksrc_OR = overall_curve(k_row, 'OR')
            osrc_O  = overall_curve(o_row, 'O')
            osrc_OR = overall_curve(o_row, 'OR')
            n_label = FIXED_DENOM_TOTAL
            title = 'Phage-weighted overall (5 datasets)'
        else:
            ksrc_K  = curve(k_row, ds, 'K')
            ksrc_OR = curve(k_row, ds, 'OR')
            osrc_O  = curve(o_row, ds, 'O')
            osrc_OR = curve(o_row, ds, 'OR')
            n_label = FIXED_DENOM_PHAGE[ds]
            title = ds

        for key, data in (('k_src_K', ksrc_K), ('k_src_OR', ksrc_OR),
                          ('o_src_O', osrc_O), ('o_src_OR', osrc_OR)):
            color, ls, lw, marker, lab = style[key]
            ys = [data.get(k) for k in ks]
            if all(v is None for v in ys): continue
            ax.plot(ks, ys, color=color, ls=ls, lw=lw, marker=marker,
                    markersize=4, label=lab if i == 0 else None)

        if hybrid:
            hyb_curve = hybrid.get('overall' if ds == 'overall' else ds, {})
            ys = [hyb_curve.get(k) for k in ks]
            if not all(v is None for v in ys):
                color, ls, lw, marker, lab = style['hybrid']
                ax.plot(ks, ys, color=color, ls=ls, lw=lw, marker=marker,
                        markersize=4.5, label=lab if i == 0 else None)

        ax.set_title(f'{title}  (n={n_label} phages)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('k')
        if i % 3 == 0:
            ax.set_ylabel('Recall@k (phage-level any-hit, fixed denom)')
        ax.set_xlim(0.5, 20.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=0.3)

    # Single legend below all panels
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'Recall@k per dataset — best hybrid OR sources\n'
                 f'K from {args.k_run}, O from {args.o_run}',
                 fontsize=12, fontweight='bold', y=1.00)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, format='svg', bbox_inches='tight')
    fig.savefig(args.out.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {args.out}')

    # Headline numbers
    print()
    print(f'Per-dataset HR@1 (hybrid pair: K={args.k_run}, O={args.o_run})')
    print(f'  {"dataset":<22s}  {"n":>4s}  {"K-src K@1":>10s}  '
          f'{"K-src OR@1":>11s}  {"O-src O@1":>10s}  {"O-src OR@1":>11s}  {"hybrid@1":>9s}')
    for ds in panels:
        if ds == 'overall':
            cells = [overall_curve(k_row, 'K')[1], overall_curve(k_row, 'OR')[1],
                     overall_curve(o_row, 'O')[1], overall_curve(o_row, 'OR')[1]]
            n = FIXED_DENOM_TOTAL
        else:
            cells = [curve(k_row, ds, 'K')[1], curve(k_row, ds, 'OR')[1],
                     curve(o_row, ds, 'O')[1], curve(o_row, ds, 'OR')[1]]
            n = FIXED_DENOM_PHAGE[ds]
        hyb = (hybrid or {}).get('overall' if ds == 'overall' else ds, {}).get(1)
        cells.append(hyb)
        print(f'  {ds:<22s}  {n:>4d}  ' +
              '  '.join(f'{c:>10.3f}' if c is not None else '       --'
                       for c in cells))


if __name__ == '__main__':
    main()
