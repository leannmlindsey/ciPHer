"""MRR (Mean Reciprocal Rank) ranking heatmap-table: cipher (K/O/OR)
vs DpoTropiSearch (TropiSEQ + TropiGAT + Combined Tropi) across all 5
cipher validation datasets + a phage-weighted overall column.

Renders compact "table with heatmap in each cell" in two formats:
  - Matplotlib PNG/SVG (Blues colormap, table-like spacing)
  - LaTeX .tex using colortbl + xcolor for native typesetting

MRR is computed from the HR@k=1..20 curve, truncated at k=20:
    MRR ≈ Σ_{k=1..20} (HR@k − HR@(k−1)) / k    (HR@0 = 0)
This treats any phage whose correct host ranks beyond k=20 as
contributing 0 to the MRR.

Outputs:
  - results/figures/mrr_ranking_heatmap.svg/.png
  - results/figures/mrr_ranking_heatmap.tex   (standalone LaTeX)

Reads from harvest CSV (cipher) + agent 6 TSV (Tropi) — the harvest
must have been refreshed via per_head_strict_eval on all 5 datasets.

Usage:
    python scripts/analysis/plot_mrr_ranking.py
    pdflatex results/figures/mrr_ranking_heatmap.tex   # to render the .tex
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
CIPHER_RUN_NAME = 'sweep_prott5_mean_cl70'
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


def _blue_rgb(v):
    """Map MRR value in [0,1] to an RGB triple sampled from matplotlib's
    Blues colormap. Returns 3 floats in [0,1] suitable for LaTeX
    \\definecolor{...}{rgb}{r,g,b}."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    rgba = plt.cm.Blues(float(v))
    return rgba[0], rgba[1], rgba[2]


def write_latex_table(path, matrix, columns):
    """Standalone LaTeX heatmap-table using colortbl + xcolor.

    Compiles with: pdflatex <path>
    """
    col_labels = [c if c != 'overall' else 'Overall' for c in columns]
    n_cols = len(columns)

    lines = []
    lines.append(r'\documentclass[border=4pt]{standalone}')
    lines.append(r'\usepackage[table,dvipsnames]{xcolor}')
    lines.append(r'\usepackage{colortbl}')
    lines.append(r'\usepackage{array}')
    lines.append(r'\renewcommand{\arraystretch}{1.25}')
    lines.append(r'\begin{document}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{l|' + 'c' * len(DATASETS) + '|c}')
    lines.append(r'\hline')

    header = r'\textbf{Model} & ' + ' & '.join(rf'\textbf{{{c}}}'
                                                for c in col_labels) + r' \\'
    lines.append(header)
    lines.append(r'\hline')

    for i, (label, family, _) in enumerate(MODEL_ROWS):
        if i == 3:  # divider between cipher block and Tropi block
            lines.append(r'\hline')
        cells = []
        # escape any latex specials in label
        latex_label = label.replace('∪', r'$\cup$')
        for j in range(n_cols):
            v = matrix[i, j]
            if np.isnan(v):
                cells.append(r'\multicolumn{1}{c}{\textit{--}}'
                             if j < n_cols - 1
                             else r'\multicolumn{1}{c}{\textit{--}}')
            else:
                rgb = _blue_rgb(v)
                # Use white text when cell is dark, black otherwise
                txt_color = 'white' if v >= 0.55 else 'black'
                cell_color = rf'\cellcolor[rgb]{{{rgb[0]:.3f},{rgb[1]:.3f},{rgb[2]:.3f}}}'
                cells.append(rf'{cell_color}\textcolor{{{txt_color}}}{{{v:.2f}}}')
        lines.append(rf'{latex_label} & ' + ' & '.join(cells) + r' \\')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')

    # Caption / footnote
    lines.append(r'\\[2pt]')
    lines.append(r'{\footnotesize\itshape\textcolor{gray!70}{GORODNICHIV: '
                 'O empty (no O labels in publication); OR = K by construction.}}')

    lines.append(r'\end{document}')

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


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

    # Compact "table with per-cell heatmap" — Blues palette only.
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    cmap = plt.cm.Blues
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1.0)

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c if c != 'overall' else 'Overall\n(weighted)'
                        for c in columns], fontsize=8)
    ax.set_yticks(range(len(MODEL_ROWS)))
    ax.set_yticklabels([row[0] for row in MODEL_ROWS], fontsize=8)
    ax.tick_params(axis='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Annotate cells with the value (white text on dark blue, black on light).
    for i in range(len(MODEL_ROWS)):
        for j in range(len(columns)):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, '—', ha='center', va='center',
                        color='#888', fontsize=8)
            else:
                color = 'white' if v >= 0.55 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        color=color, fontsize=8)

    # Subtle divider between cipher and Tropi blocks; column divider before Overall.
    ax.axhline(y=2.5, color='#444', lw=0.6)
    ax.axvline(x=len(DATASETS) - 0.5, color='#444', lw=0.6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.015)
    cbar.set_label('MRR', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(f'MRR — cipher ({CIPHER_RUN_NAME}) vs DpoTropiSearch',
                 fontsize=9, pad=4)

    fig.text(0.5, -0.04,
             'GORODNICHIV: O column empty (no O labels in publication); '
             'OR = K by construction.',
             ha='center', fontsize=7.5, style='italic', color='#555')

    fig.tight_layout()
    fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
    fig.savefig(OUT_SVG.replace('.svg', '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT_SVG}')

    # ---- Also emit a LaTeX heatmap-table ----
    out_tex = OUT_SVG.replace('.svg', '.tex')
    write_latex_table(out_tex, matrix, columns)
    print(f'Wrote {out_tex}')

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
