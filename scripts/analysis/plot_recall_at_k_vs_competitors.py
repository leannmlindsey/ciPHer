"""Recall@k=1..20 plot: cipher vs DpoTropiSearch (TropiSEQ + TropiGAT
+ Combined Tropi), phage-level any-hit metric on cipher's 5 validation
datasets where possible.

Boeckaerts (PhageHostLearn) is omitted: their published HR@k uses
in-silico LOGO with a different denominator, and we have not yet
replicated their inference on cipher's PHL strict denominator.

Two panels:

  Panel 1 (PhageHostLearn-only):
    - TropiSEQ, TropiGAT, Combined Tropi   (from agent 6's
      cipher-depolymerase-domain/data/recall_at_k_4way/recall_at_k_4way.tsv)
    - Cipher K-only, O-only, OR            (phage-level any-hit from
      experiments/attention_mlp/highconf_pipeline_K_prott5_mean/results/
      per_head_strict_eval.json)

  Panel 2 (phage-weighted average across 5 cipher val datasets):
    - TropiSEQ, TropiGAT, Combined Tropi   (weighted by per-dataset
      n_phages from agent 6's TSV)
    - Cipher best-of-modes                 (harvest's
      overall_phage2host_anyhit_HR<k>; max over K-only, O-only, merged
      at each k, phage-weighted across all 5 datasets)

Color scheme:
    - Cipher curves use blues/greens (cool family)
    - TropiSearch curves use yellows/oranges (warm family)

Caveats clearly labelled in the figure.

Output: results/figures/recall_at_k_vs_competitors.svg/.png

Usage:
  python scripts/analysis/plot_recall_at_k_vs_competitors.py
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

CIPHER_RUN_NAME = 'sweep_prott5_mean_cl70'  # top performer with 5/5 K coverage
DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']

OUT_SVG = 'results/figures/recall_at_k_vs_competitors.svg'


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


def cipher_phl_curves():
    """K-only, O-only, OR phage-level any-hit on PHL for CIPHER_RUN_NAME.

    Reads from harvest CSV (per-dataset PhageHostLearn_K/O/OR columns).
    """
    with open(HARVEST_CSV) as fh:
        rows = list(csv.DictReader(fh))
    matches = [r for r in rows if r.get('run_name') == CIPHER_RUN_NAME]
    if not matches:
        return None, None
    r = matches[0]
    out = {}
    for mode_key, col_key in (('k_only', 'K'), ('o_only', 'O'), ('or', 'OR')):
        out[mode_key] = {k: f(r.get(f'PhageHostLearn_{col_key}_phage2host_anyhit_HR{k}'))
                         for k in range(1, 21)}
    n_phl = r.get('PhageHostLearn_n_strict_phage')
    try: n_phl = int(n_phl) if n_phl else None
    except (TypeError, ValueError): n_phl = None
    return out, n_phl


def cipher_overall_curves():
    """K-only, O-only, OR phage-weighted overall HR@k across 5 datasets.

    Pinned to CIPHER_RUN_NAME so left and right panels show the SAME
    cipher run, with the SAME 3-mode breakdown — only the dataset scope
    differs (PHL on left, weighted across 5 on right).
    """
    with open(HARVEST_CSV) as fh:
        rows = list(csv.DictReader(fh))
    matches = [r for r in rows if r.get('run_name') == CIPHER_RUN_NAME]
    if not matches:
        return None, None
    r = matches[0]
    out = {}
    for mode_key, col_prefix in (('k_only', 'overall_K_anyhit_HR'),
                                   ('o_only', 'overall_O_anyhit_HR'),
                                   ('or',     'overall_OR_anyhit_HR')):
        out[mode_key] = {k: f(r.get(f'{col_prefix}{k}')) for k in range(1, 21)}
    return r['run_name'], out


def weighted_overall_from_agent6(agent6, model):
    """Phage-weighted average across 5 datasets for one model."""
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
    cipher_phl, phl_n = cipher_phl_curves()
    cipher_overall_run, cipher_overall = cipher_overall_curves()

    # Tropi weighted-overall curves (across 5 datasets, weighted by n_phages)
    tropi_seq_w = weighted_overall_from_agent6(agent6, 'TropiSEQ')
    tropi_gat_w = weighted_overall_from_agent6(agent6, 'TropiGAT')
    tropi_comb_w = weighted_overall_from_agent6(agent6, 'Combined')

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), squeeze=False)
    ks = list(range(1, 21))

    # ── Panel 1: PHL only — cipher (blues/greens) vs Tropi (yellows/oranges) ──
    ax = axes[0, 0]
    # Cipher curves: blues / greens (cool family). OR is the highlight (deep navy).
    cipher_color = {'k_only': '#3182bd', 'o_only': '#31a354', 'or': '#08306b'}
    cipher_lw    = {'k_only': 2.0,        'o_only': 2.0,        'or': 2.6}
    cipher_label = {
        'k_only': f'cipher K-only ({CIPHER_RUN_NAME})',
        'o_only': f'cipher O-only',
        'or':     f'cipher OR (K∪O ceiling)',
    }
    for mode in ('k_only', 'o_only', 'or'):
        ys = [cipher_phl[mode].get(k) for k in ks]
        ax.plot(ks, ys, color=cipher_color[mode], lw=cipher_lw[mode],
                marker='o', markersize=4, label=cipher_label[mode])
    # Tropi curves: yellows / oranges (warm family). Combined is the highlight (dark orange).
    tropi_color = {'TropiSEQ': '#fec44f', 'TropiGAT': '#fd8d3c',
                   'Combined': '#a63603'}
    tropi_lw    = {'TropiSEQ': 1.8,        'TropiGAT': 1.8,
                   'Combined': 2.4}
    for model in ('TropiSEQ', 'TropiGAT', 'Combined'):
        key = ('PhageHostLearn', model)
        if key not in agent6:
            continue
        ys = [agent6[key]['hr_at_k'].get(k) for k in ks]
        label = (f'TropiSEQ ∪ TropiGAT (combined)' if model == 'Combined'
                 else model)
        ax.plot(ks, ys, color=tropi_color[model], lw=tropi_lw[model],
                marker='s', markersize=3.5, label=label)

    n_phl = agent6.get(('PhageHostLearn', 'TropiSEQ'), {}).get('n_phages', '?')
    ax.set_title(f'PhageHostLearn (n={n_phl} phages)', fontsize=12, fontweight='bold')
    ax.set_xlabel('k')
    ax.set_ylabel('Recall@k (phage-level any-hit, strict denominator)')
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.95)

    # ── Panel 2: weighted average across 5 datasets ──
    # Same labels & color scheme as panel 1 — only the dataset scope differs.
    ax = axes[0, 1]
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

    total_n = sum(agent6.get((ds, 'TropiSEQ'), {}).get('n_phages', 0)
                  for ds in DATASETS)
    ax.set_title(f'Phage-weighted overall across 5 cipher val datasets '
                 f'(n={total_n} phages)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('k')
    ax.set_ylabel('Recall@k')
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.95)

    fig.suptitle('Recall@k comparison: cipher vs DpoTropiSearch\n'
                 'Phage-level any-hit, strict denominator '
                 '(cipher = host-rank any-hit; Tropi = per-protein top-k union — both K-class hit-or-miss)',
                 fontsize=10.5, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
    fig.savefig(OUT_SVG.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT_SVG}')

    # Print headline numbers for the lab notebook
    print()
    print('Headline numbers (k=1, 5, 10, 20):')
    print(f'\n  PHL panel:')
    for mode in ('k_only', 'o_only', 'or'):
        c = cipher_phl[mode]
        print(f'    cipher {mode:<8}: ' + '  '.join(
            f'k={k}={c[k]:.3f}' for k in (1, 5, 10, 20)))
    for model in ('TropiSEQ', 'TropiGAT', 'Combined'):
        key = ('PhageHostLearn', model)
        if key in agent6:
            c = agent6[key]['hr_at_k']
            print(f'    {model:<14}: ' + '  '.join(
                f'k={k}={c[k]:.3f}' for k in (1, 5, 10, 20)))

    print(f'\n  Weighted-overall panel (across 5 datasets, n={total_n}):')
    if cipher_overall:
        for mode in ('k_only', 'o_only', 'or'):
            c = cipher_overall[mode]
            ks_present = [k for k in (1, 5, 10, 20) if c.get(k) is not None]
            if not ks_present:
                continue
            print(f'    cipher {mode:<8}: ' + '  '.join(
                f'k={k}={c[k]:.3f}' for k in ks_present))
    for model, curve in (('TropiSEQ', tropi_seq_w),
                          ('TropiGAT', tropi_gat_w),
                          ('Combined', tropi_comb_w)):
        if curve:
            print(f'    {model:<14}: ' + '  '.join(
                f'k={k}={curve[k]:.3f}' for k in (1, 5, 10, 20)))


if __name__ == '__main__':
    main()
