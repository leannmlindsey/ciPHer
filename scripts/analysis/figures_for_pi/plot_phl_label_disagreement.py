"""Visualize the same-sequence-different-K-label finding for PHL.

Source: agent 4's analysis 32 — `phl_miss_neighbour_drilldown.csv`.
Of 38 PHL-miss phages with ≥80%-id training neighbours, what fraction
of those high-id neighbours have a K-label matching the phage's host
K-set? Spoiler: 7% (or 3% for the phage's dominant K).

Two panels:
  Panel A: Stacked bar — across all 156 (miss-phage, neighbour) pairs,
           breakdown of (matches dominant K, matches any phage K, no match).
  Panel B: Bar chart — match rate by sequence-identity bin (≥99, 95-99,
           90-95, 85-90, 80-85). Shows disagreement persists even at
           very high sequence identity.

Output: results/figures/pi_meeting_2026-05-05/02_phl_gap_diagnosis/
        fig_phl_label_disagreement.{svg,png}
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
SRC_CSV = (Path('/Users/leannmlindsey/WORK/CLAUDE_PHI_DATA_ANALYSIS/'
                'analyses/32_per_serotype_model_comparison/output/'
                'phl_miss_neighbour_drilldown.csv'))
OUT_DIR = REPO / 'results' / 'figures' / 'pi_meeting_2026-05-05' / '02_phl_gap_diagnosis'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SVG = OUT_DIR / 'fig_phl_label_disagreement.svg'
OUT_PNG = OUT_DIR / 'fig_phl_label_disagreement.png'

PIDENT_BINS = [
    ('≥99%',     99.0, 100.001),
    ('95-99%',   95.0, 99.0),
    ('90-95%',   90.0, 95.0),
    ('85-90%',   85.0, 90.0),
    ('80-85%',   80.0, 85.0),
]


def main():
    rows = list(csv.DictReader(open(SRC_CSV)))
    print(f'[load] {len(rows)} (PHL-miss-RBP, training-neighbour) pairs at ≥80% id')

    # Categorize each pair into one of 3 buckets
    n_dom_match = 0   # train K matches PHL dominant K
    n_any_match = 0   # train K matches some PHL positive host K (incl dominant)
    n_no_match = 0
    for r in rows:
        if r['train_matches_phl_dominant'] == 'yes':
            n_dom_match += 1
        elif r['train_matches_any_phl_K'] == 'yes':
            n_any_match += 1
        else:
            n_no_match += 1
    print(f'  matches PHL dominant K          : {n_dom_match} ({n_dom_match/len(rows)*100:.0f}%)')
    print(f'  matches some PHL host K (not dom): {n_any_match} ({n_any_match/len(rows)*100:.0f}%)')
    print(f'  no match to any PHL host K       : {n_no_match} ({n_no_match/len(rows)*100:.0f}%)')

    # Per-identity-bin match rate
    bin_counts = {label: {'match': 0, 'no_match': 0} for label, _, _ in PIDENT_BINS}
    for r in rows:
        try:
            pid = float(r['pident'])
        except (TypeError, ValueError):
            continue
        for label, lo, hi in PIDENT_BINS:
            if lo <= pid < hi:
                if r['train_matches_any_phl_K'] == 'yes':
                    bin_counts[label]['match'] += 1
                else:
                    bin_counts[label]['no_match'] += 1
                break

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={'width_ratios': [1, 1.3]})

    # Panel A: stacked horizontal bar
    ax = axes[0]
    cats = ['matches\nPHL dominant K', 'matches\nsome PHL host K\n(not dominant)',
            'no match to any\nPHL host K']
    counts = [n_dom_match, n_any_match, n_no_match]
    colors = ['#2c7bb6', '#abd9e9', '#d7191c']
    pcts = [c / len(rows) * 100 for c in counts]
    bars = ax.barh(cats, counts, color=colors, edgecolor='white', linewidth=2)
    for bar, pct, count in zip(bars, pcts, counts):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f'{count} ({pct:.0f}%)',
                va='center', ha='left', fontsize=10, fontweight='bold')
    ax.set_xlabel('Number of (miss-RBP, training-neighbour) pairs')
    ax.set_xlim(0, max(counts) * 1.25)
    ax.set_title('K-label agreement of high-id training neighbours\n(38 PHL-miss phages, 156 ≥80% id pairs)',
                 fontsize=10.5, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Panel B: per-identity-bin match rate
    ax = axes[1]
    bin_labels = [b[0] for b in PIDENT_BINS]
    match_pcts = [(bin_counts[b]['match'] / max(1, bin_counts[b]['match'] + bin_counts[b]['no_match'])) * 100
                  for b in bin_labels]
    bin_n = [bin_counts[b]['match'] + bin_counts[b]['no_match'] for b in bin_labels]
    xpos = np.arange(len(bin_labels))
    bars = ax.bar(xpos, match_pcts, color='#2c7bb6', edgecolor='white', linewidth=2)
    for bar, pct, n in zip(bars, match_pcts, bin_n):
        if n > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f'{pct:.0f}%\n(n={n})',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(xpos)
    ax.set_xticklabels(bin_labels, fontsize=10)
    ax.set_ylabel('% of training neighbours that\nmatch ANY PHL host K-label')
    ax.set_xlabel('Sequence identity to PHL miss-RBP')
    ax.set_ylim(0, max(match_pcts) * 1.4 if max(match_pcts) > 0 else 100)
    ax.set_title('Even at near-identical sequence,\nthe training K-label rarely matches the PHL phage\'s host K',
                 fontsize=10.5, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        'PHL-miss RBPs have high-identity training neighbours — but with the WRONG K-label',
        fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {OUT_SVG}')
    print(f'  wrote {OUT_PNG}')


if __name__ == '__main__':
    main()
