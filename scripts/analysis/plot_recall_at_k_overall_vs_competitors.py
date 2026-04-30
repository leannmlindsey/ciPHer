"""Single-panel companion to plot_recall_at_k_vs_competitors.py — only
the right (phage-weighted overall) panel, suitable for slides where the
left (PHL-only) panel isn't useful.

Same colors, same labels, same metric (phage-level any-hit, OR mode,
strict denominator). Curves:
    - Cipher K-only / O-only / OR (sweep_prott5_mean_cl70)
    - TropiSEQ / TropiGAT / Combined Tropi (phage-weighted overall)
    - Optional cipher hybrid OR (LA K + MLP O) if the JSON exists

Reads the harvest CSV + agent 6's 4-way TSV, post-corrects to the
project-policy fixed denominators (PHL=100, PBIP=103, sum=220).

Output: results/figures/recall_at_k_overall_vs_competitors.svg/.png

Usage:
    python scripts/analysis/plot_recall_at_k_overall_vs_competitors.py
"""

import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


AGENT6_TSV = ('/Users/leannmlindsey/WORK/PHI_TSP/cipher-depolymerase-domain/'
              'data/recall_at_k_4way/recall_at_k_4way.tsv')
HARVEST_CSV = 'results/experiment_log.csv'

CIPHER_RUN_NAME = 'sweep_prott5_mean_cl70'
DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']

FIXED_DENOM_PHAGE = {'CHEN': 3, 'GORODNICHIV': 3, 'UCSD': 11,
                     'PBIP': 103, 'PhageHostLearn': 100}
FIXED_DENOM_TOTAL = sum(FIXED_DENOM_PHAGE.values())  # 220

HYBRID_CURVES_JSON = ('results/analysis/'
                       'hybrid_or_la_K_sweep_O_curves.json')
HYBRID_COLOR = '#6a3d9a'
HYBRID_LW    = 2.6
HYBRID_LABEL = 'cipher hybrid OR (LA K + MLP O)'

OUT_SVG = 'results/figures/recall_at_k_overall_vs_competitors.svg'


def f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def load_agent6_tsv():
    """{(dataset, model): {'n_phages': int, 'hr_at_k': dict<int,float>}}.
    Re-divides the TSV's HR to the project-policy headline denominator.
    """
    out = {}
    with open(AGENT6_TSV) as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            ds, model = row['dataset'], row['model']
            tsv_n = int(row['n_phages'])
            denom = FIXED_DENOM_PHAGE.get(ds, tsv_n)
            hrk = {}
            for k in row:
                if not k.startswith('R@'): continue
                idx = int(k.split('@')[1])
                hits = round(float(row[k]) * tsv_n)
                hrk[idx] = hits / denom
            out[(ds, model)] = {'n_phages': denom, 'hr_at_k': hrk}
    return out


def _hits_for(row, dataset, col_key):
    n_strict = f(row.get(f'{dataset}_n_strict_phage'))
    if not n_strict:
        return {k: None for k in range(1, 21)}
    n_strict = int(n_strict)
    return {k: (round(f(row.get(f'{dataset}_{col_key}_phage2host_anyhit_HR{k}'))
                       * n_strict)
                if f(row.get(f'{dataset}_{col_key}_phage2host_anyhit_HR{k}'))
                   is not None else None)
            for k in range(1, 21)}


def cipher_overall_curves():
    """K-only / O-only / OR curves, phage-weighted overall under the
    fixed denominator (220). Pinned to CIPHER_RUN_NAME."""
    with open(HARVEST_CSV) as fh:
        rows = list(csv.DictReader(fh))
    matches = [r for r in rows if r.get('run_name') == CIPHER_RUN_NAME]
    if not matches:
        return None
    r = matches[0]
    out = {}
    for mode_key, col_key in (('k_only', 'K'), ('o_only', 'O'), ('or', 'OR')):
        num = {k: 0 for k in range(1, 21)}
        any_data = False
        for ds in DATASETS:
            hits = _hits_for(r, ds, col_key)
            for k in range(1, 21):
                if hits[k] is not None:
                    num[k] += hits[k]
                    any_data = True
        out[mode_key] = ({k: num[k] / FIXED_DENOM_TOTAL for k in range(1, 21)}
                         if any_data else
                         {k: None for k in range(1, 21)})
    return out


def weighted_overall_from_agent6(agent6, model):
    """Phage-weighted overall for one Tropi model — sums hits and
    divides by FIXED_DENOM_TOTAL."""
    num = defaultdict(int)
    have_data = False
    for ds in DATASETS:
        c = agent6.get((ds, model))
        if not c: continue
        n = c['n_phages']
        for k, v in c['hr_at_k'].items():
            num[k] += round(v * n)
            have_data = True
    if not have_data:
        return {}
    return {k: num[k] / FIXED_DENOM_TOTAL for k in num}


def load_hybrid_curves(path):
    """{dataset_or_overall: {k: hr}} for hybrid OR; re-divided to fixed denoms.
    Returns None if the JSON is absent."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    out = {}
    overall_num = defaultdict(int)
    for ds, c in d.get('datasets', {}).items():
        if ds == 'overall':
            continue
        n_buggy = c.get('n')
        denom = FIXED_DENOM_PHAGE.get(ds)
        if n_buggy is None or denom is None:
            out[ds] = {int(k): v for k, v in c.get('hybrid_hr@k', {}).items()}
            continue
        out[ds] = {}
        for k, v in c.get('hybrid_hr@k', {}).items():
            hits = round(v * n_buggy)
            out[ds][int(k)] = hits / denom
            overall_num[int(k)] += hits
    out['overall'] = {k: overall_num[k] / FIXED_DENOM_TOTAL
                      for k in overall_num}
    return out


def main():
    os.makedirs(os.path.dirname(OUT_SVG) or '.', exist_ok=True)

    agent6 = load_agent6_tsv()
    cipher_overall = cipher_overall_curves()
    hybrid = load_hybrid_curves(HYBRID_CURVES_JSON)

    tropi_seq_w  = weighted_overall_from_agent6(agent6, 'TropiSEQ')
    tropi_gat_w  = weighted_overall_from_agent6(agent6, 'TropiGAT')
    tropi_comb_w = weighted_overall_from_agent6(agent6, 'Combined')

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ks = list(range(1, 21))

    cipher_color = {'k_only': '#3182bd', 'o_only': '#31a354', 'or': '#08306b'}
    cipher_lw    = {'k_only': 2.0,        'o_only': 2.0,        'or': 2.6}
    cipher_label = {
        'k_only': f'cipher K-only ({CIPHER_RUN_NAME})',
        'o_only': f'cipher O-only',
        'or':     f'cipher OR (K∪O ceiling)',
    }
    tropi_color = {'TropiSEQ': '#fec44f', 'TropiGAT': '#fd8d3c',
                   'Combined': '#a63603'}
    tropi_lw    = {'TropiSEQ': 1.8,        'TropiGAT': 1.8,
                   'Combined': 2.4}

    if cipher_overall:
        for mode in ('k_only', 'o_only', 'or'):
            ys = [cipher_overall[mode].get(k) for k in ks]
            if all(v is None for v in ys):
                continue
            ax.plot(ks, ys, color=cipher_color[mode], lw=cipher_lw[mode],
                    marker='o', markersize=4, label=cipher_label[mode])

    for model, curve in (('TropiSEQ', tropi_seq_w),
                         ('TropiGAT', tropi_gat_w),
                         ('Combined', tropi_comb_w)):
        if not curve:
            continue
        ys = [curve.get(k) for k in ks]
        label = ('TropiSEQ ∪ TropiGAT (combined)' if model == 'Combined' else model)
        ax.plot(ks, ys, color=tropi_color[model], lw=tropi_lw[model],
                marker='s', markersize=3.5, label=label)

    if hybrid is not None and 'overall' in hybrid:
        ys = [hybrid['overall'].get(k) for k in ks]
        if not all(v is None for v in ys):
            ax.plot(ks, ys, color=HYBRID_COLOR, lw=HYBRID_LW,
                    marker='D', markersize=4.5, label=HYBRID_LABEL)

    # Annotations on the headline curves at k=10, k=20
    if cipher_overall and cipher_overall.get('or'):
        for k_ann in (10, 20):
            v = cipher_overall['or'].get(k_ann)
            if v is not None:
                ax.annotate(f'{v:.3f}', xy=(k_ann, v),
                            xytext=(0, -16), textcoords='offset points',
                            ha='center', fontsize=9, fontweight='bold',
                            color=cipher_color['or'])
    if hybrid is not None and 'overall' in hybrid:
        for k_ann in (10, 20):
            v = hybrid['overall'].get(k_ann)
            if v is not None:
                ax.annotate(f'{v:.3f}', xy=(k_ann, v),
                            xytext=(0, 8), textcoords='offset points',
                            ha='center', fontsize=9, fontweight='bold',
                            color=HYBRID_COLOR)

    ax.set_title(f'Phage-weighted overall recall@k across 5 cipher val datasets '
                 f'(n={FIXED_DENOM_TOTAL} phages)\n'
                 f'cipher vs DpoTropiSearch — phage-level any-hit, strict denominator',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('k')
    ax.set_ylabel('Recall@k (phage-level any-hit, strict denominator)')
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
    fig.savefig(OUT_SVG.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT_SVG}')

    # Headline numbers for the lab notebook
    print()
    print(f'Phage-weighted overall (n={FIXED_DENOM_TOTAL}) — k=1, 5, 10, 20:')
    if cipher_overall:
        for mode in ('k_only', 'o_only', 'or'):
            c = cipher_overall[mode]
            if any(c.get(k) is not None for k in (1, 5, 10, 20)):
                print(f'    cipher {mode:<8}: ' + '  '.join(
                    f'k={k}={c[k]:.3f}' for k in (1, 5, 10, 20)
                    if c.get(k) is not None))
    for model, curve in (('TropiSEQ', tropi_seq_w),
                         ('TropiGAT', tropi_gat_w),
                         ('Combined', tropi_comb_w)):
        if curve:
            print(f'    {model:<14}: ' + '  '.join(
                f'k={k}={curve[k]:.3f}' for k in (1, 5, 10, 20)
                if k in curve))
    if hybrid is not None and 'overall' in hybrid:
        c = hybrid['overall']
        print(f'    hybrid OR     : ' + '  '.join(
            f'k={k}={c[k]:.3f}' for k in (1, 5, 10, 20) if k in c))


if __name__ == '__main__':
    main()
