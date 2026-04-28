"""MRR (Mean Reciprocal Rank) ranking heatmap: cipher (K/O/OR) vs
DpoTropiSearch (TropiSEQ + TropiGAT + Combined Tropi) across all 5
cipher validation datasets + a phage-weighted overall column.

MRR is computed from the HR@k=1..20 curve, truncated at k=20:
    MRR ≈ Σ_{k=1..20} (HR@k − HR@(k−1)) / k    (HR@0 = 0)
This treats any phage whose correct host ranks beyond k=20 as
contributing 0 to the MRR (rank=∞). Truncation error is small for
small candidate sets (≤ ~50 hosts per dataset).

Output: results/figures/mrr_ranking_heatmap.svg/.png + a text table
to stdout for the lab notebook.

Reads from harvest CSV (cipher) + agent 6 TSV (Tropi) — the harvest
must have been refreshed via per_head_strict_eval on all 5 datasets.

Usage:
    python scripts/analysis/plot_mrr_ranking.py
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


AGENT6_TSV = ('/Users/leannmlindsey/WORK/PHI_TSP/cipher-depolymerase-domain/'
              'data/recall_at_k_4way/recall_at_k_4way.tsv')
HARVEST_CSV = 'results/experiment_log.csv'
CIPHER_RUN_NAME = 'highconf_pipeline_K_prott5_mean'
DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']

OUT_SVG = 'results/figures/mrr_ranking_heatmap.svg'

MODEL_ROWS = [
    ('cipher OR (K∪O ceiling)', 'cipher', 'or'),
    ('cipher O-only',           'cipher', 'o_only'),
    ('cipher K-only',           'cipher', 'k_only'),
    ('TropiSEQ ∪ TropiGAT',     'tropi',  'Combined'),
    ('TropiGAT',                'tropi',  'TropiGAT'),
    ('TropiSEQ',                'tropi',  'TropiSEQ'),
]


def f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def mrr_from_hrk(curve):
    """MRR ≈ Σ_{k=1..20} (HR@k − HR@(k-1))/k. None values treated as 0."""
    prev = 0.0
    mrr = 0.0
    for k in range(1, 21):
        v = curve.get(k)
        cur = float(v) if v is not None else 0.0
        cur = max(cur, prev)  # HR@k must be monotone non-decreasing
        delta = cur - prev
        if delta > 0:
            mrr += delta / k
        prev = cur
    return mrr


def load_agent6_tsv():
    out = {}
    with open(AGENT6_TSV) as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        for row in reader:
            ds, model = row['dataset'], row['model']
            n = int(row['n_phages'])
            hrk = {int(k.split('@')[1]): float(row[k])
                   for k in row if k.startswith('R@')}
            out[(ds, model)] = {'n_phages': n, 'hr_at_k': hrk}
    return out


def load_cipher_curves():
    with open(HARVEST_CSV) as fh:
        rows = list(csv.DictReader(fh))
    matches = [r for r in rows if r.get('run_name') == CIPHER_RUN_NAME]
    if not matches:
        return None
    r = matches[0]
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for mode_key, col_key in (('k_only', 'K'), ('o_only', 'O'), ('or', 'OR')):
            out[ds][mode_key] = {
                k: f(r.get(f'{ds}_{col_key}_phage2host_anyhit_HR{k}'))
                for k in range(1, 21)
            }
    out['overall'] = {}
    for mode_key, col_prefix in (('k_only', 'overall_K_anyhit_HR'),
                                   ('o_only', 'overall_O_anyhit_HR'),
                                   ('or',     'overall_OR_anyhit_HR')):
        out['overall'][mode_key] = {k: f(r.get(f'{col_prefix}{k}'))
                                    for k in range(1, 21)}
    return out


def tropi_overall_curve(agent6, model):
    num = defaultdict(float)
    den = 0
    for ds in DATASETS:
        key = (ds, model)
        if key not in agent6:
            continue
        n = agent6[key]['n_phages']
        if n == 0:
            continue
        den += n
        for k, v in agent6[key]['hr_at_k'].items():
            num[k] += v * n
    if den == 0:
        return {}
    return {k: v / den for k, v in num.items()}


def main():
    os.makedirs(os.path.dirname(OUT_SVG) or '.', exist_ok=True)

    agent6 = load_agent6_tsv()
    cipher = load_cipher_curves()
    if cipher is None:
        print(f'ERROR: harvest CSV has no row for run_name={CIPHER_RUN_NAME}')
        return

    columns = DATASETS + ['overall']
    matrix = np.full((len(MODEL_ROWS), len(columns)), np.nan)

    for i, (label, family, key) in enumerate(MODEL_ROWS):
        for j, ds in enumerate(columns):
            if family == 'cipher':
                curve = cipher.get(ds, {}).get(key, {})
            else:
                if ds == 'overall':
                    curve = tropi_overall_curve(agent6, key)
                else:
                    curve = agent6.get((ds, key), {}).get('hr_at_k', {})
            if not curve or all(v is None for v in curve.values()):
                continue
            matrix[i, j] = mrr_from_hrk(curve)

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.viridis
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1.0)

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c if c != 'overall' else 'Overall\n(weighted)'
                        for c in columns], rotation=0, fontsize=10)
    ax.set_yticks(range(len(MODEL_ROWS)))
    ax.set_yticklabels([row[0] for row in MODEL_ROWS], fontsize=10)

    # Annotate cells
    for i in range(len(MODEL_ROWS)):
        for j in range(len(columns)):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, '—', ha='center', va='center',
                        color='gray', fontsize=10)
            else:
                color = 'white' if v < 0.55 else 'black'
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('MRR (truncated at k=20)', fontsize=10)

    # Light divider lines between cipher and Tropi rows
    ax.axhline(y=2.5, color='black', lw=1.0)
    ax.axvline(x=len(DATASETS) - 0.5, color='black', lw=1.0)

    ax.set_title('MRR ranking across cipher validation datasets\n'
                 f'cipher run: {CIPHER_RUN_NAME}  |  '
                 'phage-level any-hit, strict denominator',
                 fontsize=11, fontweight='bold', pad=10)
    fig.tight_layout()
    fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
    fig.savefig(OUT_SVG.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT_SVG}')

    # Print table for the lab notebook
    print()
    print('MRR table:')
    header = f'  {"model":<28} ' + ' '.join(f'{c:>14}' for c in columns)
    print(header)
    for i, (label, _, _) in enumerate(MODEL_ROWS):
        row_str = f'  {label:<28} '
        for j in range(len(columns)):
            v = matrix[i, j]
            row_str += f'{"--":>14} ' if np.isnan(v) else f'{v:>14.3f} '
        print(row_str.rstrip())


if __name__ == '__main__':
    main()
