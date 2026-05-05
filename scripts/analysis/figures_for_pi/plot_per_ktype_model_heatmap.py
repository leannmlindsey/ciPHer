"""Per-K-type (and per-O-type) HR@1 heatmap, broken down by single-head
model — visualizes WHICH K-types and O-types each model catches, so we
can see where the models disagree (and why a future smarter ensemble
might help where naive averaging fails).

Two panels:
  Panel 1: K-types × K-head models (PHL only)
    Rows: K-types in PHL with ≥2 positive-host phages, sorted by phage count
    Cols: ESM-2_K (esm2_650m_seg4_cl70 K-head), ProtT5_K (la_v3_uat
          prott5_xl_seg8 K-head), kmer_K (sweep_kmer_aa20_k4 K-head),
          plus the actual headline K (esm2_3b_mean_cl70) used in the
          current 2-hybrid for reference.
    Cells: HR@1 = fraction of phages with that K-type whose K-head
           ranked the true K-class at rank 1.

  Panel 2: O-types × O-head models (PHL only)
    Same shape, O-side.

Output: results/figures/knob_comparisons/fig_per_ktype_model_heatmap.{svg,png}

Run:
    python scripts/analysis/figures_for_pi/plot_per_ktype_model_heatmap.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
TSV_DIR = REPO / 'results' / 'analysis' / 'per_phage'
HOST_RANGE_DIR = REPO / 'data' / 'validation_data' / 'HOST_RANGE'
OUT_DIR = REPO / 'results' / 'figures' / 'knob_comparisons'
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALL_DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']

# K-side model columns: 3 ensemble K-heads + the headline 2-hybrid K
K_MODELS = [
    ('ESM-2 3B mean\n(2-hybrid K)', 'sweep_posList_esm2_3b_mean_cl70'),
    ('ESM-2 650M seg4', 'sweep_posList_esm2_650m_seg4_cl70'),
    ('ProtT5 LA seg8',  'la_v3_uat_prott5_xl_seg8'),
    ('kmer aa20 k4',    'sweep_kmer_aa20_k4'),
]
# O-side model columns: 3 ensemble O-heads + kmer O (which is the 2-hybrid O)
O_MODELS = [
    ('ESM-2 3B mean',  'sweep_posList_esm2_3b_mean_cl70'),
    ('ProtT5 mean',    'sweep_prott5_mean_cl70'),
    ('kmer aa20 k4\n(2-hybrid O)', 'sweep_kmer_aa20_k4'),
]


def load_phage_to_serotype(datasets):
    """{(dataset, phage_id): (host_K, host_O)} from positive interactions
    across the given list of datasets."""
    out = {}
    for ds in datasets:
        path = HOST_RANGE_DIR / ds / 'metadata' / 'interaction_matrix.tsv'
        if not path.exists():
            print(f'  WARNING: missing {path}')
            continue
        with open(path) as f:
            for r in csv.DictReader(f, delimiter='\t'):
                if r.get('label') in ('1', 1):
                    ph = r['phage_id'].strip()
                    k = (r.get('host_K') or '').strip() or 'unknown'
                    o = (r.get('host_O') or '').strip() or 'unknown'
                    out.setdefault((ds, ph), (k, o))
    return out


def load_ranks(run_name, head, datasets):
    """{(dataset, phage_id): rank_or_None} restricted to the given datasets."""
    path = TSV_DIR / f'per_phage_{run_name}.tsv'
    if not path.exists():
        print(f'  WARNING: missing {path}')
        return {}
    col = 'k_only_rank' if head == 'k' else 'o_only_rank'
    ds_set = set(datasets)
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            ds = r.get('dataset')
            if ds not in ds_set:
                continue
            v = r.get(col, '').strip()
            try:
                rank = int(v) if v else None
            except ValueError:
                rank = None
            out[(ds, r['phage_id'])] = rank
    return out


def per_serotype_hr1(phage_to_st, ranks, st_index, min_n=1):
    """For each serotype, fraction of its phages with rank ≤ 1.

    st_index: 0 for K-type, 1 for O-type.
    Returns {serotype: (hr1, n_phages)}.
    """
    by_st = defaultdict(list)
    for ph, sts in phage_to_st.items():
        by_st[sts[st_index]].append(ph)
    out = {}
    for st, phages in by_st.items():
        n = len(phages)
        if n < min_n:
            continue
        hits = sum(1 for ph in phages
                   if ranks.get(ph) is not None and ranks[ph] <= 1)
        out[st] = (hits / n, n)
    return out


def render_heatmap(ax, matrix, model_labels, serotype_labels, title, n_per_serotype):
    """Compact heatmap rotated so models are rows and serotypes are columns.

    Square cells, no in-cell annotations, Blues colormap. Serotype tick
    labels rotated 90° to fit when many columns.
    """
    # Transpose: incoming matrix is (n_serotypes, n_models); we want
    # (n_models, n_serotypes) so models are y-axis, serotypes are x-axis.
    m_t = matrix.T
    im = ax.imshow(m_t, cmap='Blues', vmin=0, vmax=1, aspect='equal')
    ax.set_aspect('equal', adjustable='box', anchor='W')
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=8)
    col_lab = [f'{st} (n={n})' for st, n in zip(serotype_labels, n_per_serotype)]
    ax.set_xticks(range(len(col_lab)))
    ax.set_xticklabels(col_lab, rotation=90, fontsize=7)
    ax.set_title(title, fontsize=11, fontweight='bold')
    # Light gridlines between cells
    ax.set_xticks(np.arange(-0.5, len(col_lab), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(model_labels), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    ax.tick_params(which='minor', length=0)
    return im


def build_panel(phage_to_st, models, st_index, datasets, top_n=50):
    """Returns (matrix, row_labels, col_labels, n_per_row).

    Includes top_n serotypes by phage count (no min-N filter).
    """
    head = 'k' if st_index == 0 else 'o'
    model_ranks = {label: load_ranks(run, head, datasets) for label, run in models}
    per_model_per_st = {label: per_serotype_hr1(phage_to_st, ranks, st_index, min_n=1)
                        for label, ranks in model_ranks.items()}
    all_sts = set()
    for label, _ in models:
        all_sts.update(per_model_per_st[label].keys())
    n_per_st = {}
    for st in all_sts:
        for label, _ in models:
            if st in per_model_per_st[label]:
                n_per_st[st] = per_model_per_st[label][st][1]
                break
    sts_sorted = sorted(all_sts, key=lambda s: (-n_per_st[s], s))[:top_n]
    matrix = np.zeros((len(sts_sorted), len(models)))
    for i, st in enumerate(sts_sorted):
        for j, (label, _) in enumerate(models):
            if st in per_model_per_st[label]:
                matrix[i, j] = per_model_per_st[label][st][0]
            else:
                matrix[i, j] = float('nan')
    return matrix, sts_sorted, [m[0] for m in models], [n_per_st[s] for s in sts_sorted]


def make_figure(datasets, scope_label, out_basename):
    """Render the K + O heatmap pair for the given dataset list and save as
    out_basename.{svg,png} under OUT_DIR."""
    print(f'\n=== Building heatmap for: {scope_label} ===')
    phage_to_st = load_phage_to_serotype(datasets)
    print(f'  {len(phage_to_st)} phages with positive (K,O) host across {datasets}')

    print('[build] K-side panel (top 50)')
    k_mat, k_rows, k_cols, k_n = build_panel(phage_to_st, K_MODELS, 0, datasets, top_n=50)
    print(f'  {len(k_rows)} K-types × {len(k_cols)} K-head models')
    print('[build] O-side panel (top 50)')
    o_mat, o_rows, o_cols, o_n = build_panel(phage_to_st, O_MODELS, 1, datasets, top_n=50)
    print(f'  {len(o_rows)} O-types × {len(o_cols)} O-head models')

    # ── Plot — rotated 90°: models are rows, serotypes are columns.
    # Use explicit axes positioning so cell pixel-size is IDENTICAL in
    # both K and O panels (gridspec would stretch the panel with fewer
    # columns to fill the same width, making its cells appear larger).
    cell_in = 0.22  # inches per cell on both axes — IDENTICAL for K and O
    n_k_models = len(k_cols)
    n_k_serotypes = len(k_rows)
    n_o_models = len(o_cols)
    n_o_serotypes = len(o_rows)

    # Margins (inches)
    left_margin = 1.6      # for y-tick model labels
    right_margin = 1.5     # for colorbar
    bottom_margin = 2.2    # for x-tick serotype labels (rotated 90°)
    top_margin = 0.7       # for K panel title
    panel_gap_top = 0.2    # title spacing for O panel
    panel_gap_label = 1.0  # x-tick label space below K panel

    k_w = n_k_serotypes * cell_in
    k_h = n_k_models * cell_in
    o_w = n_o_serotypes * cell_in
    o_h = n_o_models * cell_in

    fig_w = max(k_w, o_w) + left_margin + right_margin
    fig_h = (top_margin + k_h + panel_gap_label + panel_gap_top
             + o_h + bottom_margin)

    fig = plt.figure(figsize=(fig_w, fig_h))

    # K panel on top, anchored at left margin
    k_left = left_margin / fig_w
    k_bottom = (bottom_margin + o_h + panel_gap_label + panel_gap_top) / fig_h
    k_width = k_w / fig_w
    k_height = k_h / fig_h
    ax_k = fig.add_axes((k_left, k_bottom, k_width, k_height))

    # O panel on bottom, anchored at left margin
    o_left = left_margin / fig_w
    o_bottom = bottom_margin / fig_h
    o_width = o_w / fig_w
    o_height = o_h / fig_h
    ax_o = fig.add_axes((o_left, o_bottom, o_width, o_height))

    # Note: build_panel returns (matrix, serotype_labels, model_labels, n_per_serotype).
    # render_heatmap signature: (ax, matrix, model_labels, serotype_labels, title, n_per_serotype)
    im1 = render_heatmap(ax_k, k_mat, k_cols, k_rows,
                         f'K-head models × K-types  ({scope_label} HR@1)', k_n)
    im2 = render_heatmap(ax_o, o_mat, o_cols, o_rows,
                         f'O-head models × O-types  ({scope_label} HR@1)', o_n)

    # Colorbar — short, vertically centered to the right of the K panel
    cbar_height_in = min(2.0, k_h)
    cbar_left = (left_margin + max(k_w, o_w) + 0.3) / fig_w
    cbar_bottom = (bottom_margin + o_h + panel_gap_label + panel_gap_top
                   + (k_h - cbar_height_in) / 2) / fig_h
    cbar_width = 0.15 / fig_w
    cbar_height = cbar_height_in / fig_h
    cax = fig.add_axes((cbar_left, cbar_bottom, cbar_width, cbar_height))
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label('HR@1 within serotype', fontsize=9)

    fig.suptitle(
        f'Per-serotype HR@1 by single-head model ({scope_label})',
        fontsize=11, fontweight='bold', y=0.995)
    out_svg = OUT_DIR / f'{out_basename}.svg'
    out_png = OUT_DIR / f'{out_basename}.png'
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_svg}')
    print(f'  wrote {out_png}')

    # Headline observations
    print()
    print('K-side: per-K-type HR@1 by model (top by phage count):')
    print(f'  {"K-type":<10s} {"n":>4s}  ' + '  '.join(f'{c:>10s}' for c in k_cols))
    for i, st in enumerate(k_rows[:15]):
        vals = '  '.join(f'{k_mat[i,j]:>10.2f}' for j in range(len(k_cols)))
        print(f'  {st:<10s} {k_n[i]:>4d}  {vals}')

    print()
    print('O-side: per-O-type HR@1 by model:')
    print(f'  {"O-type":<10s} {"n":>4s}  ' + '  '.join(f'{c:>10s}' for c in o_cols))
    for i, st in enumerate(o_rows[:15]):
        vals = '  '.join(f'{o_mat[i,j]:>10.2f}' for j in range(len(o_cols)))
        print(f'  {st:<10s} {o_n[i]:>4d}  {vals}')


def main():
    # PHL-only (the historical figure)
    make_figure(['PhageHostLearn'], 'PhageHostLearn',
                'fig_per_ktype_model_heatmap')
    # All 5 cipher validation datasets
    make_figure(ALL_DATASETS, 'all 5 cipher val datasets',
                'fig_per_ktype_model_heatmap_all')


if __name__ == '__main__':
    main()
