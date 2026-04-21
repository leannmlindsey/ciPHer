"""Render appendix-quality figures + a clean LaTeX/Markdown table from
results/analysis/within_between_summary.tsv.

Produces:
  results/analysis/within_between_summary.md    — Markdown table (paper-ready)
  results/analysis/within_between_gap_bars.svg  — Bar chart: K + O class-separation gap per embedding
  results/analysis/within_between_cohensd.svg   — Bar chart: Cohen's d per embedding (effect size)

Usage:
  python scripts/analysis/plot_within_between_summary.py
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _f(v, default=None):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def family(label):
    """Colour by representation family."""
    l = label.lower()
    if l.startswith('esm2'):
        return 'ESM-2'
    if l.startswith('prott5'):
        return 'ProtT5'
    if l.startswith('kmer'):
        return 'k-mer'
    return 'other'


FAMILY_COLOR = {
    'ESM-2': '#2166ac',
    'ProtT5': '#b2182b',
    'k-mer': '#d6604d',
    'other': '#888888',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', default='results/analysis/within_between_summary.tsv')
    p.add_argument('--out-dir', default='results/analysis')
    args = p.parse_args()

    if not os.path.exists(args.summary):
        raise SystemExit(f'Summary TSV not found: {args.summary}')

    with open(args.summary) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    if not rows:
        raise SystemExit('Summary is empty.')

    # Sort by K gap descending (most separating first)
    rows.sort(key=lambda r: -(_f(r.get('k_gap_mean'), -99) or -99))

    # ---- Markdown table ----
    md_path = os.path.join(args.out_dir, 'within_between_summary.md')
    with open(md_path, 'w') as f:
        f.write('# Within-class vs between-class cosine similarity\n\n')
        f.write('Per embedding representation, cosine similarity between same-K-type (within) '
                'and different-K-type (between) training pairs. Larger **gap** = better '
                'K-type separation. **Cohen\'s d** is the standardized effect size.\n\n')
        header = ('| Representation | dim | n MD5s | n K | K within | K between | '
                  '**K gap** | K d | O within | O between | **O gap** | O d |')
        sep = '|' + '|'.join(['---'] * 12) + '|'
        f.write(header + '\n')
        f.write(sep + '\n')
        for r in rows:
            f.write(
                f"| {r.get('label', ''):<22} "
                f"| {r.get('dim', ''):>6} "
                f"| {r.get('n_md5s', ''):>6} "
                f"| {r.get('n_k_classes', ''):>4} "
                f"| {r.get('k_within_mean', ''):>8} "
                f"| {r.get('k_between_mean', ''):>9} "
                f"| **{r.get('k_gap_mean', ''):>8}** "
                f"| {r.get('k_cohens_d', ''):>6} "
                f"| {r.get('o_within_mean', ''):>8} "
                f"| {r.get('o_between_mean', ''):>9} "
                f"| **{r.get('o_gap_mean', ''):>8}** "
                f"| {r.get('o_cohens_d', ''):>6} |\n"
            )
    print(f'Wrote {md_path}')

    # ---- Bar chart: gap ----
    labels = [r['label'] for r in rows]
    k_gaps = [_f(r.get('k_gap_mean'), 0.0) for r in rows]
    o_gaps = [_f(r.get('o_gap_mean'), 0.0) for r in rows]
    colors = [FAMILY_COLOR[family(l)] for l in labels]

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 5))
    x = range(len(labels))
    w = 0.38
    b1 = ax.bar([i - w/2 for i in x], k_gaps, w, label='K-type gap',
                color=colors, edgecolor='black', linewidth=0.4)
    b2 = ax.bar([i + w/2 for i in x], o_gaps, w, label='O-type gap',
                color=colors, edgecolor='black', linewidth=0.4, hatch='///',
                alpha=0.7)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Class-separation gap (within − between cosine)')
    ax.set_title('How well does each embedding cluster by serotype?\n'
                 '(larger = better separation; bar colour = family; hatched = O-type)')
    # Family legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLOR.items()
               if any(family(l) == f for l in labels)]
    handles.append(mpatches.Patch(color='white', edgecolor='black',
                                   hatch='///', label='O-type (hatched)'))
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    bar_path = os.path.join(args.out_dir, 'within_between_gap_bars.svg')
    fig.savefig(bar_path, format='svg')
    plt.close(fig)
    print(f'Wrote {bar_path}')

    # ---- Bar chart: Cohen's d ----
    k_d = [_f(r.get('k_cohens_d'), 0.0) for r in rows]
    o_d = [_f(r.get('o_cohens_d'), 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 5))
    ax.bar([i - w/2 for i in x], k_d, w, label='K-type',
           color=colors, edgecolor='black', linewidth=0.4)
    ax.bar([i + w/2 for i in x], o_d, w, label='O-type',
           color=colors, edgecolor='black', linewidth=0.4, hatch='///', alpha=0.7)
    ax.axhline(0, color='black', lw=0.5)
    # Reference lines for Cohen's d interpretation
    for thr, lbl in [(0.2, 'small'), (0.5, 'medium'), (0.8, 'large')]:
        ax.axhline(thr, color='grey', ls='--', lw=0.5, alpha=0.5)
        ax.text(len(labels) - 0.5, thr, f'  {lbl} (d={thr})',
                fontsize=7, color='grey', va='bottom')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Cohen's d  (effect size of class separation)")
    ax.set_title('Standardized class-separation effect size by embedding\n'
                 '(d ≥ 0.2 small, ≥ 0.5 medium, ≥ 0.8 large)')
    handles = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLOR.items()
               if any(family(l) == f for l in labels)]
    handles.append(mpatches.Patch(color='white', edgecolor='black',
                                   hatch='///', label='O-type (hatched)'))
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    d_path = os.path.join(args.out_dir, 'within_between_cohensd.svg')
    fig.savefig(d_path, format='svg')
    plt.close(fig)
    print(f'Wrote {d_path}')


if __name__ == '__main__':
    main()
