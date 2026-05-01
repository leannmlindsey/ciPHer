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
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


AGENT6_TSV = ('/Users/leannmlindsey/WORK/PHI_TSP/cipher-depolymerase-domain/'
              'data/recall_at_k_4way/recall_at_k_4way.tsv')
HARVEST_CSV = 'results/experiment_log.csv'

CIPHER_RUN_NAME = 'sweep_kmer_aa20_k4'  # top performer with 5/5 K coverage
DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']

# Fixed denominators per project policy (memory: feedback_strict_denominator).
# Used to post-correct harvest-CSV columns that are still over the buggy
# n_strict_phage denominator until per_head_strict_eval is bulk-rerun.
FIXED_DENOM_PHAGE = {'CHEN': 3, 'GORODNICHIV': 3, 'UCSD': 11,
                     'PBIP': 103, 'PhageHostLearn': 100}  # paper-equivalent (excludes no-genome phages: 27 PHL, 1 PBIP)
FIXED_DENOM_TOTAL = sum(FIXED_DENOM_PHAGE.values())  # 220

# Optional hybrid OR curves (cross-model: K from one run + O from another)
HYBRID_CURVES_JSON = ('results/analysis/'
                       'hybrid_or_esm2_3b_K_kmer_aa20_O_curves.json')
HYBRID_COLOR = '#6a3d9a'
HYBRID_LW    = 2.6
HYBRID_LABEL = 'cipher hybrid OR (esm2_3b K + kmer_aa20_k4 O)'

OUT_SVG = 'results/figures/recall_at_k_vs_competitors.svg'


def f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def load_agent6_tsv():
    """{(dataset, model): {'n_phages': int, 'hr_at_k': dict<int,float>}}.

    The TSV's HR is over the strict-but-permissive denominator from
    score_recall_at_k_4way.py (PHL=127, PBIP=104). We re-divide to
    the project-policy headline (PHL=100, PBIP=103, the
    paper-equivalent denominators). Numerator is unchanged because
    the excluded no-genome phages contribute 0 hits either way.
    """
    out = {}
    with open(AGENT6_TSV) as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        for row in reader:
            ds, model = row['dataset'], row['model']
            tsv_n = int(row['n_phages'])
            denom = FIXED_DENOM_PHAGE.get(ds, tsv_n)
            hrk = {}
            for k in row:
                if not k.startswith('R@'):
                    continue
                idx = int(k.split('@')[1])
                tsv_hr = float(row[k])
                hits = round(tsv_hr * tsv_n)
                hrk[idx] = hits / denom
            out[(ds, model)] = {'n_phages': denom, 'hr_at_k': hrk}
    return out


def load_hybrid_curves(path):
    """Load cross_model_or_union.py output. Returns {dataset: {k: hr}}
    for the hybrid-OR mode, RE-DIVIDED by the fixed per-dataset
    denominator per project policy. The JSON's `n` is the old buggy
    n_strict_phage; we recover the numerator and re-divide.
    None if missing."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    out = {}
    # Sum buggy numerators across datasets to rebuild a correct overall.
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


def _hits_for(row, dataset, col_key):
    """Recover numerator for harvested HR (over n_strict_phage)."""
    n_strict = f(row.get(f'{dataset}_n_strict_phage'))
    if not n_strict:
        return {k: None for k in range(1, 21)}
    n_strict = int(n_strict)
    return {k: (round(f(row.get(f'{dataset}_{col_key}_phage2host_anyhit_HR{k}'))
                       * n_strict)
                if f(row.get(f'{dataset}_{col_key}_phage2host_anyhit_HR{k}'))
                   is not None else None)
            for k in range(1, 21)}


def cipher_phl_curves():
    """K-only, O-only, OR phage-level any-hit on PHL for CIPHER_RUN_NAME,
    re-divided by the FIXED PHL denominator (127) per project policy."""
    with open(HARVEST_CSV) as fh:
        rows = list(csv.DictReader(fh))
    matches = [r for r in rows if r.get('run_name') == CIPHER_RUN_NAME]
    if not matches:
        return None, None
    r = matches[0]
    out = {}
    denom = FIXED_DENOM_PHAGE['PhageHostLearn']
    for mode_key, col_key in (('k_only', 'K'), ('o_only', 'O'), ('or', 'OR')):
        hits = _hits_for(r, 'PhageHostLearn', col_key)
        out[mode_key] = {k: (hits[k] / denom if hits[k] is not None else None)
                         for k in range(1, 21)}
    return out, denom


def cipher_overall_curves():
    """K-only, O-only, OR phage-weighted overall HR@k across 5 datasets,
    using the FIXED total denominator (sum of per-dataset fixed denoms = 248).
    Pinned to CIPHER_RUN_NAME."""
    with open(HARVEST_CSV) as fh:
        rows = list(csv.DictReader(fh))
    matches = [r for r in rows if r.get('run_name') == CIPHER_RUN_NAME]
    if not matches:
        return None, None
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
    hybrid = load_hybrid_curves(HYBRID_CURVES_JSON)
    if hybrid is None:
        print(f'(no hybrid curves at {HYBRID_CURVES_JSON} — rendering without)')

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
    # Hybrid OR curve on PHL (purple)
    if hybrid is not None and 'PhageHostLearn' in hybrid:
        ys = [hybrid['PhageHostLearn'].get(k) for k in ks]
        if not all(v is None for v in ys):
            ax.plot(ks, ys, color=HYBRID_COLOR, lw=HYBRID_LW,
                    marker='D', markersize=4.5, label=HYBRID_LABEL)

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
    # Hybrid OR curve overall (purple)
    if hybrid is not None and 'overall' in hybrid:
        ys = [hybrid['overall'].get(k) for k in ks]
        if not all(v is None for v in ys):
            ax.plot(ks, ys, color=HYBRID_COLOR, lw=HYBRID_LW,
                    marker='D', markersize=4.5, label=HYBRID_LABEL)

    # Annotate cipher OR + hybrid OR at k=10 and k=20 on the weighted-
    # average panel. Hybrid is the top curve so it gets the upper label;
    # cipher OR sits below it.
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
