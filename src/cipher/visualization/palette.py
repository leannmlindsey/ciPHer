"""Project color palette for ciPHer visualizations.

Usage:
    from cipher.visualization.palette import COLORS, HEATMAP_CMAP, CATEGORICAL

    # For categorical/discrete colors:
    plt.plot(x, y, color=COLORS['dark_teal'])
    plt.scatter(x, y, color=COLORS['red'])

    # For heatmaps/continuous:
    plt.imshow(data, cmap=HEATMAP_CMAP)

    # For ordered sets (e.g., models in a comparison):
    for i, model in enumerate(models):
        plt.plot(x, y, color=CATEGORICAL[i % len(CATEGORICAL)])
"""

# Sequential colormap for heatmaps (matplotlib built-in)
HEATMAP_CMAP = 'YlGnBu'

# Named colors from the palette
COLORS = {
    'dark_teal':   '#1B5E7A',
    'teal':        '#3D8BA7',
    'sage':        '#73A1A4',
    'muted_green': '#90AEA8',
    'light_sage':  '#B8C1B0',
    'cream':       '#F2E5D6',
    'red':         '#DC4350',
    'salmon':      '#E3807B',
    'light_pink':  '#ECAFA8',
    'white':       '#FFFFFF',
}

# Ordered list for categorical use
CATEGORICAL = [
    '#1B5E7A',  # dark teal
    '#DC4350',  # red
    '#73A1A4',  # sage
    '#E3807B',  # salmon
    '#3D8BA7',  # teal
    '#B8C1B0',  # light sage
    '#90AEA8',  # muted green
    '#ECAFA8',  # light pink
    '#F2E5D6',  # cream
]

# For UpSet-style plots: color by intersection degree
DEGREE_COLORS = {
    1: '#1B5E7A',   # dark teal — single tool
    2: '#3D8BA7',   # teal — pairs
    3: '#73A1A4',   # sage — triples
    4: '#90AEA8',   # muted green — quads
    5: '#DC4350',   # red — 5+
    6: '#DC4350',
    7: '#E3807B',   # salmon — 7
    8: '#ECAFA8',   # light pink — all 8
}
