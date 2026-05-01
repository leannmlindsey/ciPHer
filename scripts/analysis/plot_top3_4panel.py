"""4-panel grid plots: top 3 cipher models broken down by head mode
(K-only / O-only / merged / OR), in BOTH ranking directions.

Two figures:
  - phage_to_host_4panel.png — phage→host any-hit (cipher's primary
    use case; "given a phage, did one of its positive hosts rank
    ≤k among the candidate set?")
  - host_to_phage_4panel.png — host→phage any-hit (PHL-tool comparison
    direction; "given a host, did one of its positive phages rank
    ≤k among the candidate set?")

Each figure has a 2×2 panel grid:
  ┌─────────┬─────────┐
  │ K-only  │ O-only  │
  ├─────────┼─────────┤
  │ merged  │   OR    │
  └─────────┴─────────┘

Each panel shows phage-weighted overall HR@k=1..20 across all 5
cipher validation datasets, one curve per top model:
  1. sweep_prott5_mean_cl70           (MLP, ProtT5 mean)
  2. concat_prott5_mean+kmer_li10_k5  (MLP, ProtT5 + kmer)
  3. la_v3_uat_prott5_xl_seg8         (LA,  ProtT5 xl seg8)

Reads each top model's per_head_strict_eval.json directly (the
harvest CSV doesn't store per-mode host→phage curves; the JSON does).

Usage:
    python scripts/analysis/plot_top3_4panel.py
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


CIPHER_REPO = '/Users/leannmlindsey/WORK/PHI_TSP/cipher'
LA_REPO = '/Users/leannmlindsey/WORK/PHI_TSP/cipher-light-attention'

DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']

# Top 3 models — (display_name, json_path, color, line_width)
TOP3 = [
    ('sweep_prott5_mean_cl70 (MLP, ProtT5 mean)',
     f'{CIPHER_REPO}/experiments/attention_mlp/sweep_prott5_mean_cl70/results/per_head_strict_eval.json',
     '#08306b', 2.4),
    ('concat_prott5_mean+kmer_li10_k5 (MLP)',
     f'{CIPHER_REPO}/experiments/attention_mlp/concat_prott5_mean+kmer_li10_k5/results/per_head_strict_eval.json',
     '#3182bd', 2.0),
    ('la_v3_uat_prott5_xl_seg8 (LA)',
     f'{LA_REPO}/experiments/light_attention/la_v3_uat_prott5_xl_seg8/results/per_head_strict_eval.json',
     '#31a354', 2.0),
]

K_VALUES = list(range(1, 21))

# Mode order for the 2x2 grid
MODES = [('k_only', 'K-only'),
         ('o_only', 'O-only'),
         ('merged', 'merged (K+O)'),
         ('or', 'OR (K∪O ceiling)')]

OUT_PHAGE2HOST = f'{CIPHER_REPO}/results/figures/top3_4panel_phage_to_host.svg'
OUT_HOST2PHAGE = f'{CIPHER_REPO}/results/figures/top3_4panel_host_to_phage.svg'


def load_model_curves(json_path):
    """Returns {dataset: {mode: {family: {k: hr}}}} where family is
    'phage_to_host' or 'host_to_phage'. Plus 'n_strict_phage' /
    'n_strict_host' per dataset."""
    if not os.path.exists(json_path):
        print(f'  MISSING: {json_path}')
        return None
    with open(json_path) as fh:
        d = json.load(fh)
    out = {}
    for ds in DATASETS:
        if ds not in d:
            continue
        ds_data = d[ds]
        out[ds] = {
            'n_strict_phage': ds_data.get('n_strict_phage', 0),
            'n_strict_host':  ds_data.get('n_strict_host', 0),
            'modes': {}
        }
        for mode, _ in MODES:
            mdata = ds_data.get(mode, {})
            out[ds]['modes'][mode] = {
                'phage_to_host': {int(k): v for k, v in
                                   mdata.get('hr_at_k_any_hit', {}).items()},
                'host_to_phage': {int(k): v for k, v in
                                   mdata.get('hr_at_k_phage_any_hit', {}).items()},
            }
    return out


def weighted_overall_curve(model_curves, mode, direction):
    """Phage-weighted (or host-weighted) overall HR@k across 5 datasets.

    direction: 'phage_to_host' weights by n_strict_phage;
               'host_to_phage' weights by n_strict_host.
    """
    weight_field = 'n_strict_phage' if direction == 'phage_to_host' \
        else 'n_strict_host'
    out = {}
    for k in K_VALUES:
        num = 0.0
        den = 0
        for ds, ds_data in model_curves.items():
            n = ds_data.get(weight_field, 0) or 0
            v = ds_data['modes'].get(mode, {}).get(direction, {}).get(k)
            if v is not None and n > 0:
                num += v * n
                den += n
        out[k] = (num / den) if den > 0 else None
    return out


def plot_4panel(title, direction, out_path):
    """Render one 4-panel figure (K, O, merged, OR for a single
    direction) with one curve per top model."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)

    # Compute total denominator for the title (sum across datasets,
    # using first available model)
    total_n = 0
    for label, json_path, _, _ in TOP3:
        curves = load_model_curves(json_path)
        if curves:
            total_n = sum(c.get(
                'n_strict_phage' if direction == 'phage_to_host'
                else 'n_strict_host', 0) for c in curves.values())
            break

    for i, (mode, mode_label) in enumerate(MODES):
        ax = axes[i // 2, i % 2]
        # Track per-model values at k=10 and k=20 for OR-panel annotation
        annot_candidates = {10: [], 20: []}
        for label, json_path, color, lw in TOP3:
            curves = load_model_curves(json_path)
            if curves is None:
                continue
            wcurve = weighted_overall_curve(curves, mode, direction)
            ys = [wcurve.get(k) for k in K_VALUES]
            if all(v is None for v in ys):
                continue
            ax.plot(K_VALUES, ys, color=color, lw=lw,
                    marker='o', markersize=4, label=label)
            for k_ann in (10, 20):
                v = wcurve.get(k_ann)
                if v is not None:
                    annot_candidates[k_ann].append((v, color))
        # Annotate top value at k=10 and k=20 on the OR panel
        if mode == 'or':
            for k_ann in (10, 20):
                if annot_candidates[k_ann]:
                    v_top, c_top = max(annot_candidates[k_ann],
                                        key=lambda x: x[0])
                    ax.annotate(f'{v_top:.3f}', xy=(k_ann, v_top),
                                xytext=(0, 8), textcoords='offset points',
                                ha='center', fontsize=9, fontweight='bold',
                                color=c_top)
        ax.set_title(mode_label, fontsize=11, fontweight='bold')
        ax.set_xlabel('k')
        ax.set_xlim(0.5, 20.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=0.3)
        if i % 2 == 0:
            ax.set_ylabel('Recall@k (any-hit, strict denominator)')

    # Single shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=1, fontsize=9,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(f'{title}\n(phage-weighted overall across 5 cipher val '
                 f'datasets, n={total_n})',
                 fontsize=12, fontweight='bold', y=1.00)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    fig.savefig(out_path.replace('.svg', '.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')


def main():
    os.makedirs(os.path.dirname(OUT_PHAGE2HOST), exist_ok=True)

    # Sanity check that model JSONs exist
    print('Top 3 models:')
    for label, json_path, _, _ in TOP3:
        exists = os.path.exists(json_path)
        print(f'  [{ "OK" if exists else "MISSING"}] {label}')
        print(f'           {json_path}')

    print()
    plot_4panel('Top 3 cipher models — phage→host any-hit',
                'phage_to_host', OUT_PHAGE2HOST)
    plot_4panel('Top 3 cipher models — host→phage any-hit',
                'host_to_phage', OUT_HOST2PHAGE)

    # Print headline at k=1 per model per mode per direction
    print()
    print('Headline numbers (HR@1, weighted across 5 datasets):')
    for direction in ('phage_to_host', 'host_to_phage'):
        print()
        print(f'  Direction: {direction.replace("_", "→")}')
        print(f'  {"model":<48}  {"K":>6} {"O":>6} {"merged":>7} {"OR":>6}')
        for label, json_path, _, _ in TOP3:
            curves = load_model_curves(json_path)
            if curves is None:
                continue
            row = []
            for mode, _ in MODES:
                wc = weighted_overall_curve(curves, mode, direction)
                v = wc.get(1)
                row.append(f'{v:.3f}' if v is not None else '   --')
            print(f'  {label:<48}  ' + ' '.join(f'{x:>6}' for x in row))


if __name__ == '__main__':
    main()
