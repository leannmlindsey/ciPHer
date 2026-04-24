#!/usr/bin/env python3
"""Side-by-side HR@k table across multiple experiment run dirs.

Reads each run's `results/evaluation{,_raw,_k_only,_o_only}.json` (based
on --variant) and prints one row per run with HR@k per dataset. Pure
file I/O; safe on login nodes.

Variant → file:
    default  -> evaluation.json          (zscore combined, cipher-evaluate default)
    raw      -> evaluation_raw.json      (--score-norm raw)
    k_only   -> evaluation_k_only.json   (--head-mode k_only)
    o_only   -> evaluation_o_only.json   (--head-mode o_only)

Usage:
    python scripts/analysis/show_eval_all.py
    python scripts/analysis/show_eval_all.py experiments/light_attention/la_*/
    python scripts/analysis/show_eval_all.py --k 5 --datasets PBIP PhageHostLearn
    python scripts/analysis/show_eval_all.py --variant k_only
    python scripts/analysis/show_eval_all.py --variant raw
    python scripts/analysis/show_eval_all.py --mode rank_phages
"""

import argparse
import glob
import json
import os
import sys


DEFAULT_DATASETS = ['PBIP', 'PhageHostLearn', 'UCSD', 'CHEN', 'GORODNICHIV']


def _short(name):
    """Shorten common repetitive prefixes for a tidier table."""
    repl = [
        ('la_highconf_pipeline_', 'la_hc_pipe_'),
        ('la_highconf_tsp_',      'la_hc_tsp_'),
        ('esm2_650m_',            'esm_'),
        ('prott5_xl_',            'p5_'),
    ]
    for src, dst in repl:
        name = name.replace(src, dst)
    return name


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('runs', nargs='*',
                   help='Experiment directories or a glob expansion (default: '
                        'all la_* under experiments/light_attention/)')
    p.add_argument('--experiments-root',
                   default='experiments/light_attention',
                   help='Root for the default glob')
    p.add_argument('--k', type=int, default=1,
                   help='HR@k to report (default: 1)')
    p.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS,
                   help='Datasets to include as columns')
    p.add_argument('--variant',
                   choices=('default', 'raw', 'k_only', 'o_only'),
                   default='default',
                   help='Which evaluation JSON to read:'
                        ' "default"->evaluation.json (zscore combined, cipher-evaluate default);'
                        ' "raw"->evaluation_raw.json (--score-norm raw);'
                        ' "k_only"->evaluation_k_only.json (--head-mode k_only);'
                        ' "o_only"->evaluation_o_only.json (--head-mode o_only).')
    p.add_argument('--mode', choices=('rank_hosts', 'rank_phages'),
                   default='rank_hosts',
                   help='Which ranking direction to show (default: rank_hosts)')
    p.add_argument('--short-names', action='store_true',
                   help='Abbreviate common prefixes in run names')
    args = p.parse_args()

    if args.runs:
        run_dirs = [d.rstrip('/') for d in args.runs]
    else:
        run_dirs = sorted(
            d.rstrip('/')
            for d in glob.glob(os.path.join(args.experiments_root, 'la_*'))
            if os.path.isdir(d)
        )

    if not run_dirs:
        print(f'No experiment directories found under {args.experiments_root}',
              file=sys.stderr)
        return 1

    eval_filename = {
        'default': 'evaluation.json',
        'raw': 'evaluation_raw.json',
        'k_only': 'evaluation_k_only.json',
        'o_only': 'evaluation_o_only.json',
    }[args.variant]
    k = str(args.k)
    datasets = args.datasets

    # Load results
    rows = []
    for d in run_dirs:
        path = os.path.join(d, 'results', eval_filename)
        name = os.path.basename(d)
        if args.short_names:
            name = _short(name)
        if not os.path.exists(path):
            rows.append((name, None, {}))
            continue
        with open(path) as f:
            r = json.load(f)
        by_ds = {}
        for ds in datasets:
            if ds in r:
                m = r[ds].get(args.mode, {})
                hr = m.get('hr_at_k', {}).get(k)
                n_pairs = m.get('n_pairs', 0)
                by_ds[ds] = (hr, n_pairs)
            else:
                by_ds[ds] = (None, 0)
        rows.append((name, r.get('_meta', {}), by_ds))

    # Print
    name_w = max(len(r[0]) for r in rows)
    name_w = max(name_w, 10)

    header = f'{"run":<{name_w}} ' + ' '.join(
        f'{ds[:9] + " ":>10}' for ds in datasets) + '  mean'
    print(f'{args.mode} HR@{args.k}  ({eval_filename})')
    print(header)
    print('-' * len(header))

    for name, meta, by_ds in rows:
        if not by_ds or all(v[0] is None for v in by_ds.values()):
            print(f'{name:<{name_w}}  (missing {eval_filename})')
            continue
        cells = []
        valid_vals = []
        for ds in datasets:
            hr, n = by_ds.get(ds, (None, 0))
            if hr is None or n == 0:
                cells.append('     -    ')
            else:
                cells.append(f'{hr:>7.3f}   ')
                valid_vals.append(hr)
        mean = sum(valid_vals) / len(valid_vals) if valid_vals else None
        mean_s = f'{mean:.3f}' if mean is not None else '  -  '
        print(f'{name:<{name_w}} ' + ''.join(cells) + f'  {mean_s}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
