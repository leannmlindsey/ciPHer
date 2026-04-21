"""Compare all evaluated experiments.

Reads results/evaluation.json from every experiment that has one,
prints a sortable summary table, saves a CSV, and generates HR@k
comparison plots.

Usage:
    python scripts/analysis/compare_experiments.py
    python scripts/analysis/compare_experiments.py --top 10            # only show top 10
    python scripts/analysis/compare_experiments.py --filter SpikeHunter  # only experiments matching pattern
"""

import argparse
import csv
import json
import os
from glob import glob


def load_all_results(model='attention_mlp', filter_pattern=None):
    """Load evaluation.json from every experiment."""
    results = []
    pattern = f'experiments/{model}/*/results/evaluation.json'
    for eval_path in sorted(glob(pattern)):
        exp_dir = os.path.dirname(os.path.dirname(eval_path))
        name = os.path.basename(exp_dir)
        if filter_pattern and filter_pattern not in name:
            continue
        with open(eval_path) as f:
            data = json.load(f)
        results.append({'name': name, 'exp_dir': exp_dir, 'data': data})
    return results


def summarize(result):
    """Compute aggregate metrics from one experiment's evaluation results."""
    rh_hr = {1: [], 5: [], 10: []}
    rp_hr = {1: [], 5: [], 10: []}
    rh_mrr = []
    rp_mrr = []

    for ds, r in result['data'].items():
        if ds.startswith('_'):  # skip metadata keys like _meta
            continue
        rh = r.get('rank_hosts', {})
        rp = r.get('rank_phages', {})
        for k in (1, 5, 10):
            rh_hr[k].append(rh.get('hr_at_k', {}).get(str(k), 0))
            rp_hr[k].append(rp.get('hr_at_k', {}).get(str(k), 0))
        rh_mrr.append(rh.get('mrr', 0))
        rp_mrr.append(rp.get('mrr', 0))

    n = max(len(rh_hr[1]), 1)
    return {
        'rh_hr1': sum(rh_hr[1]) / n,
        'rh_hr5': sum(rh_hr[5]) / n,
        'rh_hr10': sum(rh_hr[10]) / n,
        'rp_hr1': sum(rp_hr[1]) / n,
        'rp_hr5': sum(rp_hr[5]) / n,
        'rp_hr10': sum(rp_hr[10]) / n,
        'rh_mrr': sum(rh_mrr) / n,
        'rp_mrr': sum(rp_mrr) / n,
        'mean_hr1': (sum(rh_hr[1]) + sum(rp_hr[1])) / (2 * n),
    }


def print_table(rows):
    print(f"\n{'Experiment':<70} {'rh@1':>6} {'rh@5':>6} {'rp@1':>6} {'rp@5':>6} {'mean':>6}")
    print('-' * 110)
    for r in rows:
        s = r['summary']
        print(f"{r['name']:<70} "
              f"{s['rh_hr1']:>6.3f} {s['rh_hr5']:>6.3f} "
              f"{s['rp_hr1']:>6.3f} {s['rp_hr5']:>6.3f} "
              f"{s['mean_hr1']:>6.3f}")


def save_csv(rows, path):
    fieldnames = ['name', 'rh_hr1', 'rh_hr5', 'rh_hr10', 'rp_hr1', 'rp_hr5',
                  'rp_hr10', 'rh_mrr', 'rp_mrr', 'mean_hr1']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = {'name': r['name']}
            row.update(r['summary'])
            writer.writerow(row)
    print(f"\nSaved CSV: {path}")


def plot_comparison(rows, mode='rank_hosts', output_path=None, top_n=10):
    """Plot HR@k curves for top N experiments."""
    try:
        from cipher.visualization import plot_model_comparison
    except ImportError:
        print('Skipping plot (matplotlib not available)')
        return

    # Take top N by mean_hr1
    top = sorted(rows, key=lambda r: -r['summary']['mean_hr1'])[:top_n]
    dirs = [r['exp_dir'] for r in top]
    labels = [r['name'][:50] for r in top]  # truncate long names

    if not dirs:
        return

    saved = plot_model_comparison(dirs, labels, mode=mode,
                                   output_path=output_path)
    for s in saved:
        print(f"Saved plot: {s}")


def main():
    parser = argparse.ArgumentParser(description='Compare evaluated experiments')
    parser.add_argument('--model', default='attention_mlp',
                        help='Model directory name (default: attention_mlp)')
    parser.add_argument('--filter', help='Only include experiments containing this string')
    parser.add_argument('--top', type=int, default=20,
                        help='Show only top N experiments by mean HR@1')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plotting')
    parser.add_argument('--output-dir', default='experiments/_comparison',
                        help='Directory for CSV and plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_all_results(args.model, args.filter)
    if not results:
        print('No evaluation results found.')
        return

    for r in results:
        r['summary'] = summarize(r)

    # Sort by mean HR@1
    results.sort(key=lambda r: -r['summary']['mean_hr1'])

    print(f"\n{len(results)} evaluated experiments")
    print_table(results[:args.top])

    save_csv(results, os.path.join(args.output_dir, 'experiment_summary.csv'))

    if not args.no_plots:
        plot_comparison(results, mode='rank_hosts',
                        output_path=os.path.join(args.output_dir, 'hr_curves_rank_hosts'),
                        top_n=min(args.top, 10))
        plot_comparison(results, mode='rank_phages',
                        output_path=os.path.join(args.output_dir, 'hr_curves_rank_phages'),
                        top_n=min(args.top, 10))


if __name__ == '__main__':
    main()
