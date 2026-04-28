"""Per-dataset Recall@k=1..20 grid: cipher (K/O/OR) vs DpoTropiSearch
(TropiSEQ + TropiGAT + Combined Tropi), one panel per cipher validation
dataset.

Same 6 curves as plot_recall_at_k_vs_competitors.py, same colors and
labels — only the dataset scope differs (CHEN, GORODNICHIV, UCSD, PBIP,
PhageHostLearn, plus a 6th "weighted overall" panel). This is the
"only difference between panels is the dataset" companion plot.

Reads exclusively from the harvest CSV (results/experiment_log.csv);
the harvest must have been refreshed via per_head_strict_eval on all 5
datasets first.

Output: results/figures/recall_at_k_per_dataset.svg/.png

Usage:
    python scripts/analysis/plot_recall_at_k_per_dataset.py
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


AGENT6_TSV = ('/Users/leannmlindsey/WORK/PHI_TSP/cipher-depolymerase-domain/'
              'data/recall_at_k_4way/recall_at_k_4way.tsv')
HARVEST_CSV = 'results/experiment_log.csv'
CIPHER_RUN_NAME = 'highconf_pipeline_K_prott5_mean'
DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']

OUT_SVG = 'results/figures/recall_at_k_per_dataset.svg'

# Colors (must match plot_recall_at_k_vs_competitors.py)
CIPHER_COLOR = {'k_only': '#3182bd', 'o_only': '#31a354', 'or': '#08306b'}
CIPHER_LW    = {'k_only': 2.0,        'o_only': 2.0,        'or': 2.6}
CIPHER_LABEL = {
    'k_only': f'cipher K-only ({CIPHER_RUN_NAME})',
    'o_only': f'cipher O-only',
    'or':     f'cipher OR (K∪O ceiling)',
}
TROPI_COLOR = {'TropiSEQ': '#fec44f', 'TropiGAT': '#fd8d3c', 'Combined': '#a63603'}
TROPI_LW    = {'TropiSEQ': 1.8,        'TropiGAT': 1.8,        'Combined': 2.4}
TROPI_LABEL = {
    'TropiSEQ': 'TropiSEQ',
    'TropiGAT': 'TropiGAT',
    'Combined': 'TropiSEQ ∪ TropiGAT (combined)',
}


def f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def load_agent6_tsv():
    """{(dataset, model): {'n_phages': int, 'hr_at_k': dict<int,float>}}."""
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
    """{ds_or_overall: {mode: {k: float}}} for CIPHER_RUN_NAME from harvest."""
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
            curve = {k: f(r.get(f'{ds}_{col_key}_phage2host_anyhit_HR{k}'))
                     for k in range(1, 21)}
            out[ds][mode_key] = curve
    out['overall'] = {}
    for mode_key, col_prefix in (('k_only', 'overall_K_anyhit_HR'),
                                   ('o_only', 'overall_O_anyhit_HR'),
                                   ('or',     'overall_OR_anyhit_HR')):
        out['overall'][mode_key] = {k: f(r.get(f'{col_prefix}{k}'))
                                    for k in range(1, 21)}
    return out


def weighted_overall_from_agent6(agent6, model):
    """Phage-weighted average across 5 datasets for one Tropi model."""
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


def plot_panel(ax, title, n, cipher_modes, tropi_models):
    """Draw one panel. cipher_modes={mode:{k:v}}; tropi_models={model:{k:v}}."""
    ks = list(range(1, 21))
    for mode in ('k_only', 'o_only', 'or'):
        curve = cipher_modes.get(mode) or {}
        ys = [curve.get(k) for k in ks]
        if all(v is None for v in ys):
            continue
        ax.plot(ks, ys, color=CIPHER_COLOR[mode], lw=CIPHER_LW[mode],
                marker='o', markersize=4, label=CIPHER_LABEL[mode])
    for model in ('TropiSEQ', 'TropiGAT', 'Combined'):
        curve = tropi_models.get(model) or {}
        ys = [curve.get(k) for k in ks]
        if all(v is None for v in ys):
            continue
        ax.plot(ks, ys, color=TROPI_COLOR[model], lw=TROPI_LW[model],
                marker='s', markersize=3.5, label=TROPI_LABEL[model])
    n_str = f'n={n}' if n is not None else 'n=?'
    ax.set_title(f'{title} ({n_str} phages)', fontsize=11, fontweight='bold')
    ax.set_xlabel('k')
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.3)


def main():
    os.makedirs(os.path.dirname(OUT_SVG) or '.', exist_ok=True)

    agent6 = load_agent6_tsv()
    cipher = load_cipher_curves()
    if cipher is None:
        print(f'ERROR: harvest CSV has no row for run_name={CIPHER_RUN_NAME}')
        return

    tropi_overall = {m: weighted_overall_from_agent6(agent6, m)
                     for m in ('TropiSEQ', 'TropiGAT', 'Combined')}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    panels = list(DATASETS) + ['overall']

    for i, ds in enumerate(panels):
        ax = axes[i // 3, i % 3]
        if ds == 'overall':
            n = sum(agent6.get((d, 'TropiSEQ'), {}).get('n_phages', 0)
                    for d in DATASETS)
            tropi_models = tropi_overall
            title = 'Phage-weighted overall (5 datasets)'
        else:
            tropi_models = {m: agent6.get((ds, m), {}).get('hr_at_k', {})
                            for m in ('TropiSEQ', 'TropiGAT', 'Combined')}
            n = agent6.get((ds, 'TropiSEQ'), {}).get('n_phages')
            title = ds
        plot_panel(ax, title, n, cipher.get(ds, {}), tropi_models)
        if i % 3 == 0:
            ax.set_ylabel('Recall@k (phage-level any-hit, strict denom)')

    # Single shared legend below all panels — collect unique entries
    # across all panels (some datasets may have only Tropi or only cipher).
    seen_labels = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in seen_labels:
                seen_labels[lab] = h
    # Order: cipher (K, O, OR) first, then Tropi (SEQ, GAT, Combined)
    desired_order = [CIPHER_LABEL['k_only'], CIPHER_LABEL['o_only'],
                     CIPHER_LABEL['or'],
                     TROPI_LABEL['TropiSEQ'], TROPI_LABEL['TropiGAT'],
                     TROPI_LABEL['Combined']]
    ordered = [(lab, seen_labels[lab]) for lab in desired_order
               if lab in seen_labels]
    if ordered:
        labs, hs = zip(*[(lab, h) for lab, h in ordered])
        fig.legend(hs, labs, loc='lower center', ncol=3, fontsize=9,
                   framealpha=0.95, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Recall@k per dataset: cipher (K/O/OR) vs DpoTropiSearch\n'
                 'Phage-level any-hit, strict denominator '
                 '(cipher = host-rank any-hit; Tropi = per-protein top-k union)',
                 fontsize=11, fontweight='bold', y=1.00)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
    fig.savefig(OUT_SVG.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT_SVG}')

    # Print headline HR@1 per dataset for the lab notebook
    print()
    print('HR@1 per dataset (cipher OR vs Combined Tropi):')
    print(f'  {"dataset":<22}  {"n":>4}  {"cipher OR":>10}  {"Combined":>10}')
    for ds in DATASETS:
        cor = cipher.get(ds, {}).get('or', {}).get(1)
        ct = agent6.get((ds, 'Combined'), {}).get('hr_at_k', {}).get(1)
        n = agent6.get((ds, 'TropiSEQ'), {}).get('n_phages')
        cor_s = f'{cor:.3f}' if cor is not None else '   --'
        ct_s = f'{ct:.3f}' if ct is not None else '   --'
        print(f'  {ds:<22}  {n!s:>4}  {cor_s:>10}  {ct_s:>10}')
    cor = cipher.get('overall', {}).get('or', {}).get(1)
    ct = tropi_overall['Combined'].get(1)
    n = sum(agent6.get((d, 'TropiSEQ'), {}).get('n_phages', 0) for d in DATASETS)
    cor_s = f'{cor:.3f}' if cor is not None else '   --'
    ct_s = f'{ct:.3f}' if ct is not None else '   --'
    print(f'  {"weighted overall":<22}  {n:>4}  {cor_s:>10}  {ct_s:>10}')


if __name__ == '__main__':
    main()
