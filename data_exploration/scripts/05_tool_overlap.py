"""Tool overlap analysis: UpSet-style intersection table for all 8 tools.

For each protein in glycan_binders_custom.tsv, records which tools flagged it.
Computes all 2^8 - 1 possible tool combinations and their sizes.

Outputs:
    data_exploration/output/tool_intersections.csv   (sorted by count)
    data_exploration/output/tool_upset.png            (UpSet-style plot)
    data_exploration/output/tool_pairwise.csv         (pairwise overlap matrix)
"""

import csv
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import REPO_ROOT, OUTPUT_DIR

TOOLS = [
    'DePP_85', 'PhageRBPdetect', 'DepoScope', 'DepoRanker',
    'SpikeHunter', 'dbCAN', 'IPR', 'phold_glycan_tailspike',
]

GLYCAN_PATH = os.path.join(
    REPO_ROOT, 'data', 'training_data', 'metadata', 'glycan_binders_custom.tsv')
ASSOC_PATH = os.path.join(
    REPO_ROOT, 'data', 'training_data', 'metadata', 'host_phage_protein_map.tsv')


def load_tool_flags(training_only=True):
    """Load per-protein tool flags.

    Args:
        training_only: If True, restrict to proteins in the training
            association table (prophage-derived with host labels).
            If False, use ALL proteins in glycan_binders_custom.tsv.

    Returns list of frozensets of tool names."""
    training_pids = None
    if training_only:
        from cipher.data.interactions import load_training_map
        rows = load_training_map(ASSOC_PATH)
        training_pids = set(r['protein_id'] for r in rows)

    proteins = []
    with open(GLYCAN_PATH) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if training_pids is not None and row['protein_id'] not in training_pids:
                continue
            flagged = frozenset(
                t for t in TOOLS if int(row.get(t, 0)) == 1)
            if flagged:
                proteins.append(flagged)
    return proteins


def compute_intersections(proteins):
    """Count exact-set intersections (UpSet-style)."""
    counts = Counter(proteins)
    # Sort by count descending
    return sorted(counts.items(), key=lambda x: -x[1])


def compute_per_tool_counts(proteins):
    """Count how many proteins each tool flags."""
    counts = Counter()
    for p in proteins:
        for t in p:
            counts[t] += 1
    return counts


def compute_pairwise(proteins):
    """Pairwise overlap matrix: how many proteins are flagged by both tool A and B."""
    n = len(TOOLS)
    matrix = np.zeros((n, n), dtype=int)
    for p in proteins:
        for i, t1 in enumerate(TOOLS):
            if t1 not in p:
                continue
            for j, t2 in enumerate(TOOLS):
                if t2 in p:
                    matrix[i, j] += 1
    return matrix


def write_intersections_csv(intersections, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = TOOLS + ['count', 'tools_list']
        writer.writerow(header)
        for toolset, count in intersections:
            row = [1 if t in toolset else 0 for t in TOOLS]
            row.append(count)
            row.append('+'.join(sorted(toolset)))
            writer.writerow(row)


def write_pairwise_csv(matrix, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + TOOLS)
        for i, t in enumerate(TOOLS):
            writer.writerow([t] + list(matrix[i]))


def plot_heatmap(matrix, per_tool, output_path, title=None):
    """Pairwise overlap heatmap with fraction-of-smaller-set normalization."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print('Skipping heatmap (matplotlib not available)')
        return

    SHORT_NAMES = {
        'DePP_85': 'DePP',
        'PhageRBPdetect': 'RBPdetect',
        'DepoScope': 'DepoScope',
        'DepoRanker': 'DepoRanker',
        'SpikeHunter': 'SpikeHunter',
        'dbCAN': 'dbCAN',
        'IPR': 'IPR',
        'phold_glycan_tailspike': 'phold',
    }

    # Sort tools by total count (largest first)
    tools_sorted = sorted(TOOLS, key=lambda t: -per_tool.get(t, 0))
    tool_short = [SHORT_NAMES[t] for t in tools_sorted]
    n = len(tools_sorted)

    # Build index mapping from original TOOLS order to sorted order
    orig_idx = {t: TOOLS.index(t) for t in TOOLS}

    # Build fraction matrix: overlap / min(tool_i_total, tool_j_total)
    frac = np.zeros((n, n))
    for i, ti in enumerate(tools_sorted):
        for j, tj in enumerate(tools_sorted):
            overlap = matrix[orig_idx[ti], orig_idx[tj]]
            min_total = min(per_tool.get(ti, 1), per_tool.get(tj, 1))
            frac[i, j] = overlap / max(min_total, 1)

    # White-to-dark-teal colormap
    cmap = LinearSegmentedColormap.from_list(
        'white_teal', ['#FFFFFF', '#3D8BA7', '#1B5E7A'], N=256)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect='equal')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tool_short, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(tool_short, fontsize=10)

    # Annotate cells with fraction and raw count
    for i in range(n):
        for j in range(n):
            overlap = matrix[orig_idx[tools_sorted[i]], orig_idx[tools_sorted[j]]]
            f = frac[i, j]
            color = 'white' if f > 0.6 else '#333333'
            if i == j:
                ax.text(j, i, f'{overlap:,}', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')
            else:
                ax.text(j, i, f'{f:.2f}\n({overlap:,})', ha='center', va='center',
                        fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Overlap fraction (of smaller set)')
    n_proteins = sum(per_tool.get(t, 0) for t in TOOLS)  # not quite right, use title
    plot_title = title or 'Pairwise Tool Overlap'
    ax.set_title(plot_title, fontsize=12, pad=12)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_upset(intersections, per_tool, output_path, subtitle=None):
    """UpSet-style plot matching standard format: set size bars on left,
    intersection size bars on top, dot matrix on bottom."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Skipping plot (matplotlib not available)')
        return

    SHORT_NAMES = {
        'DePP_85': 'DePP',
        'PhageRBPdetect': 'RBPdetect',
        'DepoScope': 'DepoScope',
        'DepoRanker': 'DepoRanker',
        'SpikeHunter': 'SpikeHunter',
        'dbCAN': 'dbCAN',
        'IPR': 'IPR',
        'phold_glycan_tailspike': 'phold',
    }

    # Sort tools by total count (highest at top)
    tools_sorted = sorted(TOOLS, key=lambda t: -per_tool.get(t, 0))
    tool_short = [SHORT_NAMES[t] for t in tools_sorted]
    # Reverse for plotting (highest at top = last y index)
    tools_plot = list(reversed(tools_sorted))
    labels_plot = list(reversed(tool_short))
    tool_idx = {t: i for i, t in enumerate(tools_plot)}
    n_tools = len(tools_plot)

    # Show all intersections with at least min_count proteins,
    # sorted by degree (1-tool, 2-tool, ...) then by count descending
    min_count = 250
    top = [(ts, c) for ts, c in intersections if c >= min_count]
    top.sort(key=lambda x: (len(x[0]), -x[1]))
    n_sets = len(top)

    # Figure layout
    fig = plt.figure(figsize=(max(16, n_sets * 0.55), 10))

    left_w = 0.18
    gap = 0.01
    right_w = 1.0 - left_w - gap - 0.02
    bot_h = 0.35
    top_h = 0.55
    mid_gap = 0.02

    ax_totals = fig.add_axes([0.01, 0.08, left_w - 0.02, bot_h])
    ax_bars   = fig.add_axes([left_w + gap, 0.08 + bot_h + mid_gap, right_w, top_h])
    ax_dots   = fig.add_axes([left_w + gap, 0.08, right_w, bot_h])

    # -- Set size bars (left, horizontal) --
    totals = [per_tool.get(t, 0) for t in tools_plot]
    y_pos = np.arange(n_tools)
    ax_totals.barh(y_pos, totals, color='#404040', height=0.6, edgecolor='none')
    ax_totals.set_yticks(y_pos)
    ax_totals.set_yticklabels(labels_plot, fontsize=10)
    ax_totals.set_ylim(-0.5, n_tools - 0.5)
    ax_totals.invert_xaxis()
    ax_totals.set_xlabel('Set Size', fontsize=10)
    ax_totals.spines['top'].set_visible(False)
    ax_totals.spines['right'].set_visible(False)

    # -- Intersection size bars (top, vertical) --
    counts = [c for _, c in top]
    degree_colors = {
        1: '#1B5E7A',  # dark teal
        2: '#3D8BA7',  # teal
        3: '#73A1A4',  # sage
        4: '#90AEA8',  # muted green
        5: '#DC4350',  # red
        6: '#DC4350',
        7: '#E3807B',  # salmon
        8: '#ECAFA8',  # light pink
    }
    colors = [degree_colors.get(len(ts), '#DC4350') for ts, _ in top]

    x = np.arange(n_sets)
    ax_bars.bar(x, counts, color=colors, edgecolor='none', width=0.7)
    ax_bars.set_ylabel('Intersection Size', fontsize=11)
    ax_bars.set_xlim(-0.5, n_sets - 0.5)
    ax_bars.set_xticks([])
    ax_bars.spines['top'].set_visible(False)
    ax_bars.spines['right'].set_visible(False)
    ax_bars.spines['bottom'].set_visible(False)

    for i, c in enumerate(counts):
        ax_bars.text(i, c + max(counts) * 0.01, f'{c:,}',
                     ha='center', va='bottom', fontsize=6.5, rotation=0)

    title_line = subtitle + '\n' if subtitle else ''
    ax_bars.set_title(f'{title_line}Showing all intersections with ≥{min_count:,} proteins\n'
                      '1 tool  |  2 tools  |  3-4 tools  |  5+ tools (red)',
                      fontsize=10, color='#666666', pad=8)

    # -- Dot matrix (bottom) --
    ax_dots.set_xlim(-0.5, n_sets - 0.5)
    ax_dots.set_ylim(-0.5, n_tools - 0.5)
    ax_dots.set_yticks([])
    ax_dots.set_xticks([])
    ax_dots.spines['top'].set_visible(False)
    ax_dots.spines['right'].set_visible(False)
    ax_dots.spines['bottom'].set_visible(False)
    ax_dots.spines['left'].set_visible(False)

    # Horizontal gridlines
    for i in range(n_tools):
        ax_dots.axhline(i, color='#f0f0f0', linewidth=0.5, zorder=0)

    for col, (toolset, _) in enumerate(top):
        active = sorted([tool_idx[t] for t in toolset])

        # Light dots for inactive
        for row in range(n_tools):
            if row not in active:
                ax_dots.scatter(col, row, s=40, color='#d8d8d8',
                                zorder=2, linewidths=0)

        # Dark dots for active
        for row in active:
            ax_dots.scatter(col, row, s=40, color='#333333',
                            zorder=3, linewidths=0)

        # Connecting line
        if len(active) > 1:
            ax_dots.plot([col, col], [min(active), max(active)],
                         color='#333333', linewidth=1.5, zorder=2)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def run_analysis(proteins, label, suffix):
    """Run full analysis (CSVs + plots) for a protein set.

    Args:
        proteins: list of frozensets of tool names
        label: human-readable label (e.g., 'All Proteins')
        suffix: file name suffix (e.g., 'all' or 'training')
    """
    n = len(proteins)
    print(f'\n{"="*60}')
    print(f'{label} ({n:,} proteins)')
    print(f'{"="*60}')

    per_tool = compute_per_tool_counts(proteins)
    print(f'\nPer-tool counts:')
    for t in TOOLS:
        print(f'  {t:<25} {per_tool.get(t, 0):>7,}')

    print(f'\nComputing intersections...')
    intersections = compute_intersections(proteins)
    print(f'  {len(intersections)} unique tool combinations')

    # Write outputs
    csv_path = os.path.join(OUTPUT_DIR, f'tool_intersections_{suffix}.csv')
    write_intersections_csv(intersections, csv_path)
    print(f'\nSaved: {csv_path}')

    pair_path = os.path.join(OUTPUT_DIR, f'tool_pairwise_{suffix}.csv')
    matrix = compute_pairwise(proteins)
    write_pairwise_csv(matrix, pair_path)
    print(f'Saved: {pair_path}')

    print('\nTop 15 intersections:')
    print(f'  {"Tools":<60} {"Count":>8}')
    print(f'  {"-"*60} {"-"*8}')
    for toolset, count in intersections[:15]:
        tool_label = ' + '.join(sorted(toolset))
        print(f'  {tool_label:<60} {count:>8,}')

    plot_path = os.path.join(OUTPUT_DIR, f'tool_upset_{suffix}.png')
    plot_upset(intersections, per_tool, plot_path,
               subtitle=f'{label} ({n:,} proteins)')
    print(f'\nSaved: {plot_path}')

    heatmap_path = os.path.join(OUTPUT_DIR, f'tool_overlap_heatmap_{suffix}.png')
    plot_heatmap(matrix, per_tool, heatmap_path,
                 title=f'Pairwise Tool Overlap — {label} ({n:,})')
    print(f'Saved: {heatmap_path}')


def main():
    print('Loading all proteins (full glycan_binders_custom.tsv)...')
    all_proteins = load_tool_flags(training_only=False)
    print(f'  {len(all_proteins)} proteins total')

    print('Loading training-only proteins (pipeline positives)...')
    training_proteins = load_tool_flags(training_only=True)
    print(f'  {len(training_proteins)} proteins in training')

    run_analysis(all_proteins, 'All Candidate Proteins', 'all')
    run_analysis(training_proteins, 'Pipeline Positive Proteins', 'training')


if __name__ == '__main__':
    main()
