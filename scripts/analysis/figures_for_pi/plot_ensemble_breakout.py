"""Ensemble breakout: each of the 6 component sources plotted alongside
the 6-source ensemble OR curve, on PHL and on overall (5-dataset).

Companion to plot_three_family_or_ensemble.py — that one shows the
ensemble vs the 2-hybrid + best single. This one shows the ensemble
vs each of its 6 component models, so the PI can see exactly what
each component contributes.

Two panels:
  Panel 1: PhageHostLearn (n=100, paper-aligned denominator)
  Panel 2: Overall (n=220, sum of fixed per-dataset denominators)

For each source, plot HR@k computed from the source's per-phage TSV
under its head-mode (k_only_rank for K-sources, o_only_rank for
O-sources). Ensemble curve loaded from the 6-source HRK CSV produced
by plot_three_family_or_ensemble.py.

Output: results/figures/knob_comparisons/fig_ensemble_breakout.{svg,png}

Run:
    python scripts/analysis/figures_for_pi/plot_ensemble_breakout.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
TSV_DIR = REPO / 'results' / 'analysis' / 'per_phage'
ENSEMBLE_CSV = (REPO / 'results' / 'figures' / 'knob_comparisons'
                / 'three_family_or_ensemble_hrk.csv')
OUT_DIR = REPO / 'results' / 'figures' / 'knob_comparisons'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SVG = OUT_DIR / 'fig_ensemble_breakout.svg'
OUT_PNG = OUT_DIR / 'fig_ensemble_breakout.png'

# 6 ensemble sources — must match plot_three_family_or_ensemble.py
SOURCES = [
    ('ESM-2_K',  'sweep_posList_esm2_650m_seg4_cl70', 'k', '#3182bd'),  # blue
    ('ESM-2_O',  'sweep_posList_esm2_3b_mean_cl70',   'o', '#9ecae1'),  # light blue
    ('ProtT5_K', 'la_v3_uat_prott5_xl_seg8',          'k', '#e6550d'),  # orange
    ('ProtT5_O', 'sweep_prott5_mean_cl70',            'o', '#fdae6b'),  # light orange
    ('kmer_K',   'sweep_kmer_aa20_k4',                'k', '#31a354'),  # green
    ('kmer_O',   'sweep_kmer_aa20_k4',                'o', '#a1d99b'),  # light green
]

ENSEMBLE_COLOR = '#d73027'   # red — top performer
ENSEMBLE_LW = 3.0
ENSEMBLE_METHOD_LABEL = '6-source voting ensemble'

DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']
FIXED_DENOM_PHAGE = {'CHEN': 3, 'GORODNICHIV': 3, 'UCSD': 11,
                      'PBIP': 103, 'PhageHostLearn': 100}
FIXED_DENOM_TOTAL = sum(FIXED_DENOM_PHAGE.values())


def load_source_ranks(run_name, head):
    """Return {(dataset, phage_id): rank or None} from per-phage TSV."""
    path = TSV_DIR / f'per_phage_{run_name}.tsv'
    if not path.exists():
        print(f'  WARNING: missing {path}')
        return {}
    col = 'k_only_rank' if head == 'k' else 'o_only_rank'
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            v = r.get(col, '').strip()
            try:
                rank = int(v) if v else None
            except ValueError:
                rank = None
            out[(r['dataset'], r['phage_id'])] = rank
    return out


def hrk_for_source(ranks_by_pair, dataset=None, k_max=20):
    """Compute HR@k for k=1..k_max with FIXED denominators.

    If dataset is None: overall (sum across all 5 datasets, denom=220).
    Else: per-dataset (denom=FIXED_DENOM_PHAGE[dataset]).
    """
    if dataset is None:
        denom = FIXED_DENOM_TOTAL
        relevant = ranks_by_pair  # all (dataset, phage) pairs
        # but still restrict to the 5 fixed datasets
        relevant = {pair: r for pair, r in ranks_by_pair.items()
                    if pair[0] in FIXED_DENOM_PHAGE}
    else:
        denom = FIXED_DENOM_PHAGE[dataset]
        relevant = {pair: r for pair, r in ranks_by_pair.items()
                    if pair[0] == dataset}
    out = {}
    for k in range(1, k_max + 1):
        hits = sum(1 for r in relevant.values()
                   if r is not None and r <= k)
        out[k] = hits / denom if denom else None
    return out


def load_ensemble_hrk():
    """Load PHL + per-dataset HR@k for the 6-source ensemble; compute overall."""
    if not ENSEMBLE_CSV.exists():
        print(f'  WARNING: missing {ENSEMBLE_CSV}')
        return None, None
    per_ds = {}
    n_per_ds = {}
    with open(ENSEMBLE_CSV) as fh:
        for row in csv.DictReader(fh):
            if row['method'] != ENSEMBLE_METHOD_LABEL:
                continue
            ds = row['dataset']
            n_per_ds[ds] = int(row['n'])
            per_ds[ds] = {int(c.split('@')[1]): float(row[c])
                          for c in row if c.startswith('HR@')}
    # Overall: phage-weighted across the 5 fixed datasets
    overall = {}
    for k in range(1, 21):
        num = sum(per_ds[ds].get(k, 0) * n_per_ds[ds]
                  for ds in DATASETS if ds in per_ds)
        overall[k] = num / FIXED_DENOM_TOTAL
    return per_ds, overall


def main():
    # Load all 6 sources' rank maps
    print('[load] 6 component sources')
    source_ranks = {}
    for label, run, head, _ in SOURCES:
        print(f'  {label}: per_phage_{run}.tsv (head={head})')
        source_ranks[label] = load_source_ranks(run, head)

    # Load ensemble curves
    print('[load] ensemble curves')
    ens_per_ds, ens_overall = load_ensemble_hrk()

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), squeeze=False)
    ks = list(range(1, 21))

    # Panel 1: PhageHostLearn
    ax = axes[0, 0]
    for label, _, _, color in SOURCES:
        hr = hrk_for_source(source_ranks[label], dataset='PhageHostLearn')
        ys = [hr[k] for k in ks]
        ax.plot(ks, ys, color=color, lw=1.6, marker='o', markersize=3.5,
                label=label, alpha=0.85)
    if ens_per_ds and 'PhageHostLearn' in ens_per_ds:
        ys = [ens_per_ds['PhageHostLearn'].get(k) for k in ks]
        ax.plot(ks, ys, color=ENSEMBLE_COLOR, lw=ENSEMBLE_LW,
                marker='*', markersize=9, label='6-source voting ensemble (avg-rank vote then OR)')

    ax.set_title(f'PhageHostLearn (n={FIXED_DENOM_PHAGE["PhageHostLearn"]})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('k')
    ax.set_ylabel('Recall@k (phage-level any-hit, fixed denominator)')
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.95)

    # Panel 2: Overall (5 datasets, n=220)
    ax = axes[0, 1]
    for label, _, _, color in SOURCES:
        hr = hrk_for_source(source_ranks[label], dataset=None)
        ys = [hr[k] for k in ks]
        ax.plot(ks, ys, color=color, lw=1.6, marker='o', markersize=3.5,
                label=label, alpha=0.85)
    if ens_overall:
        ys = [ens_overall.get(k) for k in ks]
        ax.plot(ks, ys, color=ENSEMBLE_COLOR, lw=ENSEMBLE_LW,
                marker='*', markersize=9, label='6-source voting ensemble (avg-rank vote then OR)')

    ax.set_title(f'Overall — 5 cipher val datasets (n={FIXED_DENOM_TOTAL})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('k')
    ax.set_ylabel('Recall@k')
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.95)

    fig.suptitle(
        '6-source voting ensemble breakout: each component model + the consensus ensemble\n'
        'Voting = avg-rank within K-heads, avg-rank within O-heads, then OR. '
        'Fixed denominator (PHL=100, PBIP=103, UCSD=11, CHEN=3, GORODNICHIV=3).',
        fontsize=10.5, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {OUT_SVG}')
    print(f'  wrote {OUT_PNG}')

    # Headline numbers
    print()
    print('Headline (k=1, 5, 10, 20):')
    print()
    print('  PHL panel:')
    for label, _, _, _ in SOURCES:
        hr = hrk_for_source(source_ranks[label], dataset='PhageHostLearn')
        print(f'    {label:<10}: ' + '  '.join(
            f'k={k}={hr[k]:.3f}' for k in (1, 5, 10, 20)))
    if ens_per_ds and 'PhageHostLearn' in ens_per_ds:
        c = ens_per_ds['PhageHostLearn']
        print(f'    ensemble  : ' + '  '.join(
            f'k={k}={c[k]:.3f}' for k in (1, 5, 10, 20)))
    print()
    print(f'  Overall panel (n={FIXED_DENOM_TOTAL}):')
    for label, _, _, _ in SOURCES:
        hr = hrk_for_source(source_ranks[label], dataset=None)
        print(f'    {label:<10}: ' + '  '.join(
            f'k={k}={hr[k]:.3f}' for k in (1, 5, 10, 20)))
    if ens_overall:
        print(f'    ensemble  : ' + '  '.join(
            f'k={k}={ens_overall[k]:.3f}' for k in (1, 5, 10, 20)))


if __name__ == '__main__':
    main()
