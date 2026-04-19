"""HR@k curve plotting for single and multi-model comparison.

Two standard figures:
1. Average HR@k across datasets — one line per model
2. Per-dataset HR@k — one subplot per dataset, all models overlaid

Line styles encode protein set:
    solid  = all proteins
    dotted = TSP only
    dashed = RBP only
"""

import json
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']
KS = list(range(1, 21))

PSET_LINESTYLE = {'all': '-', 'tsp': ':', 'rbp': '--',
                  'all_glycan_binders': '-', 'tsp_only': ':', 'rbp_only': '--'}


def _require_matplotlib():
    if not HAS_MPL:
        raise ImportError('matplotlib required for plotting. Install with: pip install matplotlib')


def load_evaluation_results(experiment_dir):
    """Load evaluation.json from an experiment directory.

    Returns:
        dict: {dataset_name: {mode: {hr_at_k: {k: val}, mrr: val, ...}}}
    """
    path = os.path.join(experiment_dir, 'results', 'evaluation.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f'No evaluation.json found at {path}')
    with open(path) as f:
        return json.load(f)


def _get_hr_curve(results, dataset, mode='rank_hosts'):
    """Extract HR@k curve from results dict."""
    ds_data = results.get(dataset, {})
    mode_data = ds_data.get(mode, {})
    hr = mode_data.get('hr_at_k', {})
    return [hr.get(str(k), hr.get(k, 0)) for k in KS]


def _mean_hr_curve(results, mode='rank_hosts'):
    """Compute mean HR@k across all available datasets."""
    curves = []
    for ds in DATASETS:
        if ds in results:
            curves.append(_get_hr_curve(results, ds, mode))
    if not curves:
        return [0] * len(KS)
    return list(np.mean(curves, axis=0))


def _get_protein_set(experiment_dir):
    """Infer protein set from experiment config."""
    for fname in ['config.yaml', 'experiment.json']:
        path = os.path.join(experiment_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                if fname.endswith('.yaml'):
                    import yaml
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            exp = data.get('experiment', data.get('config', {}).get('experiment', {}))
            pset = exp.get('protein_set', 'all')
            return pset
    return 'all'


def _get_n_pairs(results, dataset, mode='rank_hosts'):
    """Get number of pairs from results."""
    return results.get(dataset, {}).get(mode, {}).get('n_pairs', '?')


def plot_single_model(experiment_dir, mode='rank_hosts', output_path=None):
    """Plot HR@k curves for a single model across all datasets.

    Creates a 2x3 subplot figure (5 datasets + summary).

    Args:
        experiment_dir: path to experiment with results/evaluation.json
        mode: 'rank_hosts' or 'rank_phages'
        output_path: save path (default: experiment_dir/results/hr_curves_{mode}.png)
    """
    _require_matplotlib()

    results = load_evaluation_results(experiment_dir)
    model_name = os.path.basename(experiment_dir.rstrip('/'))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        if ds not in results:
            ax.set_title(f'{ds} — no data')
            continue

        curve = _get_hr_curve(results, ds, mode)
        n_pairs = _get_n_pairs(results, ds, mode)

        ax.plot(KS, curve, color='#1f77b4', linewidth=2.5)
        ax.set_title(f'{ds} (n={n_pairs} pairs)', fontsize=11, fontweight='bold')
        ax.set_xlabel('k', fontsize=10)
        ax.set_ylabel('HR@k', fontsize=10)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlim(0.5, 20.5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    # 6th subplot: average curve
    ax_avg = axes[5]
    avg_curve = _mean_hr_curve(results, mode)
    ax_avg.plot(KS, avg_curve, color='#1f77b4', linewidth=2.5)
    ax_avg.set_title('Average (all datasets)', fontsize=11, fontweight='bold')
    ax_avg.set_xlabel('k', fontsize=10)
    ax_avg.set_ylabel('Mean HR@k', fontsize=10)
    ax_avg.set_xticks([1, 5, 10, 15, 20])
    ax_avg.set_xlim(0.5, 20.5)
    ax_avg.set_ylim(0, 1.05)
    ax_avg.grid(True, alpha=0.3)

    mode_label = 'Rank Hosts Given Phage' if mode == 'rank_hosts' else 'Rank Phages Given Host'
    plt.suptitle(f'HR@k — {mode_label}\n{model_name}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_path is None:
        results_dir = os.path.join(experiment_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, f'hr_curves_{mode}.png')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def plot_model_comparison(experiment_dirs, labels=None, mode='rank_hosts',
                          output_path=None):
    """Plot HR@k comparison across multiple models.

    Creates two figures:
    1. Average HR@k across datasets (one line per model)
    2. Per-dataset HR@k (one subplot per dataset)

    Args:
        experiment_dirs: list of paths to experiment directories
        labels: list of model display names (default: directory names)
        mode: 'rank_hosts' or 'rank_phages'
        output_path: base path for output (appends _avg.png and _per_dataset.png)

    Returns:
        list of saved file paths
    """
    _require_matplotlib()

    if labels is None:
        labels = [os.path.basename(d.rstrip('/')) for d in experiment_dirs]

    # Load all results
    all_results = {}
    protein_sets = {}
    for exp_dir, label in zip(experiment_dirs, labels):
        all_results[label] = load_evaluation_results(exp_dir)
        protein_sets[label] = _get_protein_set(exp_dir)

    # Assign colors and styles
    n_models = len(labels)
    cmap = plt.cm.tab20(np.linspace(0, 1, max(n_models, 2)))

    # Rank by mean HR@5
    model_avg_curves = {label: _mean_hr_curve(res, mode)
                        for label, res in all_results.items()}
    ranked = sorted(model_avg_curves.items(), key=lambda x: x[1][4], reverse=True)
    top5 = set(m for m, _ in ranked[:5])

    color_map = {label: cmap[i] for i, (label, _) in enumerate(ranked)}

    saved = []

    # ── Figure 1: Average across datasets ──
    fig, ax = plt.subplots(figsize=(14, 8))

    for label, curve in ranked:
        pset = protein_sets.get(label, 'all')
        ls = PSET_LINESTYLE.get(pset, '-')
        is_top = label in top5
        lw = 3.0 if is_top else 1.3
        alpha = 1.0 if is_top else 0.5

        ax.plot(KS, curve, color=color_map[label], linewidth=lw,
                alpha=alpha, linestyle=ls, label=label)

    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel('Mean HR@k (across 5 datasets)', fontsize=12)
    mode_label = 'Rank Hosts Given Phage' if mode == 'rank_hosts' else 'Rank Phages Given Host'
    ax.set_title(f'Average Hit Rate @ k — {mode_label}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, max(0.75, max(c[-1] for c in model_avg_curves.values()) + 0.05))
    ax.grid(True, alpha=0.3)

    # Legend with line style key
    handles, leg_labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='none'))
    leg_labels.append('')
    for ls_label, ls_style in [('Solid = All proteins', '-'),
                                ('Dotted = TSP only', ':'),
                                ('Dashed = RBP only', '--')]:
        handles.append(Line2D([0], [0], color='gray', linewidth=1.5,
                               linestyle=ls_style))
        leg_labels.append(ls_label)

    leg = ax.legend(handles, leg_labels, loc='center left',
                    bbox_to_anchor=(1.02, 0.5), fontsize=8,
                    framealpha=0.9, ncol=1, handlelength=2.5)
    for text in leg.get_texts():
        if text.get_text() in top5:
            text.set_fontweight('bold')

    plt.tight_layout()
    base = output_path or 'hr_curves'
    avg_path = f'{base}_avg.png'
    plt.savefig(avg_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved.append(avg_path)

    # ── Figure 2: Per-dataset ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]

        for label, _ in ranked:
            res = all_results[label]
            if ds not in res:
                continue
            curve = _get_hr_curve(res, ds, mode)
            pset = protein_sets.get(label, 'all')
            ls = PSET_LINESTYLE.get(pset, '-')
            is_top = label in top5
            lw = 3.0 if is_top else 1.3
            alpha = 1.0 if is_top else 0.5

            ax.plot(KS, curve, color=color_map[label], linewidth=lw,
                    alpha=alpha, linestyle=ls, label=label)

        n_pairs = _get_n_pairs(all_results[ranked[0][0]], ds, mode)
        ax.set_title(f'{ds} (n={n_pairs} pairs)', fontsize=11, fontweight='bold')
        ax.set_xlabel('k', fontsize=10)
        ax.set_ylabel('HR@k', fontsize=10)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlim(0.5, 20.5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    # 6th subplot: legend
    ax_leg = axes[5]
    ax_leg.axis('off')

    legend_handles = []
    legend_labels = []
    for label, _ in ranked:
        pset = protein_sets.get(label, 'all')
        ls = PSET_LINESTYLE.get(pset, '-')
        lw = 2.5 if label in top5 else 1.3
        legend_handles.append(Line2D([0], [0], color=color_map[label],
                                      linewidth=lw, linestyle=ls))
        legend_labels.append(label)

    legend_handles.append(Line2D([0], [0], color='none'))
    legend_labels.append('')
    for ls_label, ls_style in [('Solid = All proteins', '-'),
                                ('Dotted = TSP only', ':'),
                                ('Dashed = RBP only', '--')]:
        legend_handles.append(Line2D([0], [0], color='gray', linewidth=1.5,
                                      linestyle=ls_style))
        legend_labels.append(ls_label)

    leg = ax_leg.legend(legend_handles, legend_labels, loc='center',
                         fontsize=8, framealpha=0.9, ncol=1, handlelength=2.5,
                         title=f'Models (bold = top 5 by HR@5)')
    for text in leg.get_texts():
        if text.get_text() in top5:
            text.set_fontweight('bold')

    mode_label = 'Rank Hosts Given Phage' if mode == 'rank_hosts' else 'Rank Phages Given Host'
    plt.suptitle(f'HR@k Per Dataset — {mode_label}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    ds_path = f'{base}_per_dataset.png'
    plt.savefig(ds_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved.append(ds_path)

    return saved
