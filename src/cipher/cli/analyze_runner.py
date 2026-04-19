"""CLI runner for cipher-analyze.

Usage:
    # Compute per-serotype analysis for one experiment
    cipher-analyze experiments/attention_mlp/run1/

    # Compare multiple experiments (generates bubble plots)
    cipher-analyze experiments/attention_mlp/run1/ experiments/attention_mlp/run2/ \
        --compare --output-dir comparison_plots/

    # Only re-plot from existing JSON (skip inference)
    cipher-analyze experiments/attention_mlp/run1/ --plot-only
"""

import argparse
import os

from cipher.analysis.per_serotype import compute_per_serotype_test
from cipher.visualization.per_serotype import plot_serotype_bubble


def main():
    parser = argparse.ArgumentParser(
        description='Compute per-serotype analysis and generate plots.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cipher-analyze experiments/attention_mlp/run1/
  cipher-analyze run1/ run2/ --compare --output-dir comparison/
  cipher-analyze run1/ --plot-only
  cipher-analyze run1/ run2/ --highlight KL47,KL64,KL107
""")

    parser.add_argument('experiments', nargs='+',
                        help='One or more experiment directories')
    parser.add_argument('--compare', action='store_true',
                        help='Generate side-by-side comparison plots')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip computing; only plot from existing per_serotype_test.json')
    parser.add_argument('--output-dir',
                        help='Output directory for plots (default: first experiments analysis/ dir)')
    parser.add_argument('--serotype', choices=['k', 'o', 'both'], default='both',
                        help='Which head to plot (default: both)')
    parser.add_argument('--highlight',
                        help='Comma-separated serotypes to always label (e.g. KL47,KL64)')
    parser.add_argument('--n-highlight', type=int, default=10,
                        help='Number of top-frequency serotypes to auto-label (default: 10)')
    parser.add_argument('--max-k', type=int, default=10,
                        help='Max k for topK accuracy computation (default: 10)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()
    verbose = not args.quiet

    experiment_dirs = [os.path.abspath(d) for d in args.experiments]

    # Step 1: compute (unless --plot-only)
    if not args.plot_only:
        for exp_dir in experiment_dirs:
            if verbose:
                print(f'\n=== Computing per-serotype analysis for {exp_dir} ===')
            compute_per_serotype_test(
                exp_dir, max_k=args.max_k, save=True, verbose=verbose)

    # Step 2: plot
    highlights = None
    if args.highlight:
        highlights = [s.strip() for s in args.highlight.split(',') if s.strip()]

    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
        os.makedirs(out_dir, exist_ok=True)
    elif len(experiment_dirs) == 1:
        out_dir = os.path.join(experiment_dirs[0], 'analysis')
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = os.getcwd()

    serotypes = ['k', 'o'] if args.serotype == 'both' else [args.serotype]

    for st in serotypes:
        if args.compare or len(experiment_dirs) > 1:
            suffix = 'comparison'
        else:
            suffix = 'bubble'
        output_path = os.path.join(out_dir, f'per_serotype_{st}_{suffix}.png')

        if verbose:
            print(f'\nGenerating {st}-type bubble plot...')
        plot_serotype_bubble(
            experiment_dirs,
            serotype=st,
            output_path=output_path,
            highlight=highlights,
            n_highlight=args.n_highlight,
        )
        if verbose:
            print(f'  Saved: {output_path}')


if __name__ == '__main__':
    main()
