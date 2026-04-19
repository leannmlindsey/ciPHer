"""Per-serotype bubble plots: HR@1 vs training frequency.

Each serotype gets a consistent color across all panels so you can visually
track individual serotypes (e.g., KL47) between experiments.
"""

import os

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from cipher.analysis.per_serotype import load_per_serotype


def _require_matplotlib():
    if not HAS_MPL:
        raise ImportError('matplotlib required. Install with: pip install matplotlib')


def _serotype_color_map(all_serotypes, seed=0):
    """Build a deterministic color map for serotypes.

    Uses a shuffled combination of tab20 + tab20b + tab20c palettes so
    adjacent serotypes (KL1, KL2) get visually distinct colors but the
    mapping is stable across runs.
    """
    serotypes = sorted(all_serotypes)
    rng = np.random.default_rng(seed)

    palette = np.vstack([
        plt.cm.tab20(np.linspace(0, 1, 20)),
        plt.cm.tab20b(np.linspace(0, 1, 20)),
        plt.cm.tab20c(np.linspace(0, 1, 20)),
    ])

    # Extend palette by cycling if needed
    n = len(serotypes)
    if n > len(palette):
        reps = (n // len(palette)) + 1
        palette = np.vstack([palette] * reps)

    order = rng.permutation(len(palette))
    return {s: palette[order[i]] for i, s in enumerate(serotypes)}


def _pick_highlights(all_stats, n_top=10, user_highlight=None):
    """Select serotypes to label on the plots.

    Picks the top-N by training frequency (union across all experiments)
    plus any user-provided serotypes.
    """
    # Union of all serotypes with their max training freq across experiments
    max_freq = {}
    for stats in all_stats:
        for cls, entry in stats.items():
            freq = entry.get('train_freq', 0)
            max_freq[cls] = max(max_freq.get(cls, 0), freq)

    sorted_by_freq = sorted(max_freq.items(), key=lambda x: -x[1])
    highlights = set(s for s, _ in sorted_by_freq[:n_top])
    if user_highlight:
        highlights.update(user_highlight)
    return highlights


def plot_serotype_bubble(experiment_dirs, labels=None, serotype='k',
                          output_path=None, n_highlight=10,
                          highlight=None, size_scale=2.0, max_bubble_size=250):
    """Bubble plot comparing per-serotype HR@1 across experiments.

    Args:
        experiment_dirs: list of experiment directories (must have
                         analysis/per_serotype_test.json)
        labels: list of display names (default: directory basenames)
        serotype: 'k' or 'o' — which head to plot
        output_path: save path (default: ./per_serotype_{k|o}_bubble.png)
        n_highlight: number of top-frequency serotypes to label with names
        highlight: additional serotypes to always label
        size_scale: bubble size multiplier
        max_bubble_size: cap on bubble size

    Returns:
        path to saved plot
    """
    _require_matplotlib()

    if labels is None:
        labels = [os.path.basename(d.rstrip('/')) for d in experiment_dirs]

    # Load all per-serotype data
    all_data = []
    for exp_dir in experiment_dirs:
        data = load_per_serotype(exp_dir)
        all_data.append(data[serotype])

    # Universe of serotypes across all experiments
    all_serotypes = set()
    for stats in all_data:
        all_serotypes.update(stats.keys())

    color_map = _serotype_color_map(all_serotypes)
    highlight_set = _pick_highlights(all_data, n_highlight, highlight)

    # Layout
    n = len(experiment_dirs)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5.5 * nrows),
                              sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    for idx, (stats, label) in enumerate(zip(all_data, labels)):
        ax = axes[idx]

        freqs = []
        hr1s = []
        sizes = []
        names = []
        colors = []
        for cls, entry in stats.items():
            if entry['total'] == 0:
                continue
            freq = entry.get('train_freq', 0)
            if freq <= 0:
                freq = 1  # log scale can't show 0
            freqs.append(freq)
            hr1s.append(entry['top1'] / entry['total'])
            sizes.append(min(entry['total'] * size_scale, max_bubble_size))
            names.append(cls)
            colors.append(color_map[cls])

        # Plot all serotypes
        ax.scatter(freqs, hr1s, s=sizes, c=colors, alpha=0.6, edgecolors='none')

        # Outline + label highlighted serotypes
        for i, name in enumerate(names):
            if name in highlight_set:
                ax.scatter(freqs[i], hr1s[i], s=sizes[i],
                           facecolors='none', edgecolors='black',
                           linewidths=1.2, zorder=5)
                ax.annotate(name, (freqs[i], hr1s[i]),
                            fontsize=8, fontweight='bold',
                            xytext=(6, 5), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor='white', alpha=0.85,
                                       edgecolor='gray'))

        if hr1s:
            mean_hr = float(np.mean(hr1s))
            ax.axhline(mean_hr, color='gray', linestyle='--', alpha=0.5)
            if freqs:
                ax.text(min(freqs) * 1.2, mean_hr + 0.03,
                        f'mean={mean_hr:.2f}', fontsize=9, color='gray')

        ax.set_xscale('log')
        ax.set_xlabel('Training frequency (log scale)', fontsize=11)
        if idx % ncols == 0:
            ax.set_ylabel('HR@1', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    serotype_label = 'K-type' if serotype == 'k' else 'O-type'
    plt.suptitle(f'Per-{serotype_label} HR@1 vs Training Frequency\n'
                 f'(Bubble size = test set count; same color = same serotype across panels)',
                 fontsize=13, y=1.02)
    plt.tight_layout()

    if output_path is None:
        output_path = f'per_serotype_{serotype}_bubble.png'

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    return output_path
