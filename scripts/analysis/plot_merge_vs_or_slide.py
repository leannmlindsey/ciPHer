"""Slide-ready visual: Merged vs OR — what's the difference?

A single PNG/SVG with:
  - left: tiny score table (3 hosts × K/O/merged columns) with positive marked
  - middle: the three rankings
  - right: HR@1 outcomes per method, showing why OR catches when merged misses

Output: results/figures/merge_vs_or_explainer.{png,svg}
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches


OUT = 'results/figures/merge_vs_or_explainer.svg'

# Colors (consistent with the rest of the figure set)
BLUE  = '#3182bd'   # K
GREEN = '#31a354'   # O
NAVY  = '#08306b'   # merged / OR
PURPLE = '#6a3d9a'  # OR highlight
HIT   = '#2ca02c'
MISS  = '#d62728'


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    fig = plt.figure(figsize=(14, 5.2))

    # =====================================================
    # LEFT — score table
    # =====================================================
    ax1 = fig.add_axes([0.03, 0.10, 0.27, 0.78])
    ax1.set_axis_off()
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    ax1.text(5, 9.3, 'Scores per candidate host',
             ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.text(5, 8.65, '(positive host: H1)',
             ha='center', va='center', fontsize=10, color='#555', style='italic')

    headers = ['', 'K-score', 'O-score', 'merged\n= max(K,O)']
    rows = [
        ('H1 (+)', '3',  '1',  '3', True),    # positive
        ('H2 (−)', '2',  '4',  '4', False),   # tops merged
        ('H3 (−)', '1',  '2',  '2', False),
    ]
    col_x = [0.5, 3.0, 5.3, 7.6]

    for i, h in enumerate(headers):
        ax1.text(col_x[i], 7.8, h, ha='center', va='center',
                 fontsize=10.5, fontweight='bold', color=NAVY)
    ax1.plot([0.2, 9.5], [7.3, 7.3], color='#666', lw=0.8)

    y = 6.4
    for label, ks, os_, ms, is_pos in rows:
        bg = '#e6f0fa' if is_pos else None
        if bg:
            ax1.add_patch(Rectangle((0.2, y - 0.55), 9.3, 1.0,
                                     facecolor=bg, edgecolor='none', zorder=0))
        ax1.text(col_x[0], y, label, ha='center', va='center',
                 fontsize=11, fontweight='bold' if is_pos else 'normal',
                 color=NAVY if is_pos else '#333')
        ax1.text(col_x[1], y, ks, ha='center', va='center', fontsize=11,
                 color=BLUE, fontweight='bold' if is_pos else 'normal')
        ax1.text(col_x[2], y, os_, ha='center', va='center', fontsize=11,
                 color=GREEN, fontweight='bold' if is_pos else 'normal')
        ms_color = HIT if is_pos else (MISS if ms == '4' else '#333')
        ax1.text(col_x[3], y, ms, ha='center', va='center', fontsize=11,
                 fontweight='bold', color=ms_color)
        if not is_pos and ms == '4':
            ax1.text(col_x[3] + 1.2, y, '←tops merged',
                     ha='left', va='center', fontsize=8.5, color=MISS,
                     style='italic')
        y -= 1.05

    # =====================================================
    # MIDDLE — three rankings
    # =====================================================
    ax2 = fig.add_axes([0.34, 0.10, 0.30, 0.78])
    ax2.set_axis_off()
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    ax2.text(5, 9.3, 'Resulting rankings (rank 1 = top)',
             ha='center', va='center', fontsize=12, fontweight='bold')

    rankings = [
        ('K-only',  ['H1', 'H2', 'H3'], BLUE),
        ('O-only',  ['H2', 'H3', 'H1'], GREEN),
        ('merged',  ['H2', 'H1', 'H3'], NAVY),
    ]
    y = 7.6
    for name, order, color in rankings:
        ax2.text(0.5, y, name, ha='left', va='center', fontsize=11,
                 fontweight='bold', color=color)
        for i, host in enumerate(order):
            xc = 3.6 + 1.7 * i
            is_h1 = (host == 'H1')
            face = '#e6f0fa' if is_h1 else 'white'
            edge = NAVY if is_h1 else '#888'
            box = FancyBboxPatch((xc - 0.55, y - 0.4), 1.1, 0.8,
                                  boxstyle='round,pad=0.05',
                                  facecolor=face, edgecolor=edge, lw=1.5)
            ax2.add_patch(box)
            ax2.text(xc, y, host, ha='center', va='center',
                     fontsize=11, fontweight='bold' if is_h1 else 'normal',
                     color=NAVY if is_h1 else '#444')
            ax2.text(xc, y - 0.95, f'rank {i + 1}',
                     ha='center', va='center', fontsize=8, color='#888')
        y -= 2.3

    # =====================================================
    # RIGHT — outcomes at k=1
    # =====================================================
    ax3 = fig.add_axes([0.69, 0.10, 0.28, 0.78])
    ax3.set_axis_off()
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)

    ax3.text(5, 9.3, 'Hit at k = 1 ?',
             ha='center', va='center', fontsize=12, fontweight='bold')
    ax3.text(5, 8.65, '(did the positive host H1 land at rank 1?)',
             ha='center', va='center', fontsize=9.5, color='#555',
             style='italic')

    outcomes = [
        ('K-only',  'rank 1',   HIT,   'HIT'),
        ('O-only',  'rank 3',   MISS,  'miss'),
        ('merged',  'rank 2',   MISS,  'miss'),
        ('OR',      'union of K + O hits', HIT, 'HIT'),
    ]
    y = 7.6
    for name, where, color, status in outcomes:
        # Box for the row
        box = Rectangle((0.3, y - 0.55), 9.3, 1.0,
                         facecolor='#fafafa' if name != 'OR' else '#f0e7f7',
                         edgecolor=PURPLE if name == 'OR' else '#ddd',
                         lw=1.5 if name == 'OR' else 0.8)
        ax3.add_patch(box)
        ax3.text(0.7, y, name, ha='left', va='center', fontsize=11,
                 fontweight='bold',
                 color=PURPLE if name == 'OR' else '#333')
        ax3.text(3.2, y, where, ha='left', va='center', fontsize=9.5,
                 color='#666', style='italic')
        ax3.text(8.5, y, status, ha='center', va='center', fontsize=12,
                 fontweight='bold', color=color)
        y -= 1.2

    # Bottom takeaway
    ax3.text(5, 1.0,
             'Merged misses because H2\'s strong O signal outranks H1.\n'
             'OR still catches H1 because K-only alone caught it.',
             ha='center', va='center', fontsize=9, color='#444',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffde7',
                        edgecolor='#bbb', alpha=0.9))

    # Title
    fig.suptitle(
        'Merged vs OR — why the OR ceiling can exceed the merged ranking',
        fontsize=13, fontweight='bold', y=0.97)

    fig.savefig(OUT, format='svg', bbox_inches='tight')
    fig.savefig(OUT.replace('.svg', '.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT}')
    print(f'Wrote {OUT.replace(".svg", ".png")}')


if __name__ == '__main__':
    main()
