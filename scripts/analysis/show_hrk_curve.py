#!/usr/bin/env python3
"""Print the HR@k curve (k=1..20) for one experiment's evaluation JSON.

Pure file I/O. Safe on login nodes. Useful when you want to see retrieval
quality at top-5 / top-10 / top-20 (not just HR@1) — for example to check
whether a "low HR@1" run still has the right answer in the top few.

Usage:
    python scripts/analysis/show_hrk_curve.py <run_dir>
    python scripts/analysis/show_hrk_curve.py <run_dir> --variant k_only
    python scripts/analysis/show_hrk_curve.py <run_dir> --mode rank_phages
    python scripts/analysis/show_hrk_curve.py <run_dir> --variant raw \\
        --datasets PBIP PhageHostLearn

Variant mapping (matches show_eval_all.py):
    default  -> evaluation.json
    raw      -> evaluation_raw.json
    k_only   -> evaluation_k_only.json
    o_only   -> evaluation_o_only.json
"""

import argparse
import json
import os
import sys


PRIMARY_ORDER = ['PBIP', 'PhageHostLearn', 'UCSD', 'CHEN', 'GORODNICHIV']

VARIANT_TO_FILE = {
    'default': 'evaluation.json',
    'raw': 'evaluation_raw.json',
    'k_only': 'evaluation_k_only.json',
    'o_only': 'evaluation_o_only.json',
}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('run_dir', help='Path to the experiment run directory')
    p.add_argument('--variant',
                   choices=tuple(VARIANT_TO_FILE.keys()),
                   default='default',
                   help='Which evaluation JSON to read')
    p.add_argument('--mode',
                   choices=('rank_hosts', 'rank_phages'),
                   default='rank_hosts',
                   help='Ranking direction')
    p.add_argument('--datasets', nargs='+', default=PRIMARY_ORDER,
                   help='Datasets to include')
    p.add_argument('--max-k', type=int, default=20,
                   help='Highest k to report (default: 20)')
    args = p.parse_args()

    path = os.path.join(args.run_dir, 'results', VARIANT_TO_FILE[args.variant])
    if not os.path.exists(path):
        print(f'ERROR: {path} not found', file=sys.stderr)
        return 1

    with open(path) as f:
        r = json.load(f)

    # Filter datasets to those present in the JSON, in canonical order then
    # any extras at the end.
    datasets = [d for d in PRIMARY_ORDER if d in args.datasets and d in r]
    datasets += [d for d in args.datasets if d in r and d not in PRIMARY_ORDER]

    if not datasets:
        print('No matching datasets found in evaluation JSON', file=sys.stderr)
        return 1

    # n_pairs row (from the mode of interest) — shows whether each dataset
    # is statistically meaningful at all.
    n_pairs = {}
    for ds in datasets:
        n_pairs[ds] = r[ds].get(args.mode, {}).get('n_pairs', 0)

    # Header
    print(f'{args.mode}  HR@k  ({path})')
    print(f'  variant:  {args.variant}')
    print()
    header_cells = ['k'] + [f'{ds[:9]:>10}' for ds in datasets] + [f'{"mean":>8}']
    print(' '.join(header_cells))
    nps_cells = ['n='] + [f'{n_pairs[ds]:>10d}' for ds in datasets] + [' ' * 8]
    print(' '.join(nps_cells))
    print('-' * len(' '.join(header_cells)))

    for k in range(1, args.max_k + 1):
        row = [f'{k:>2}']
        valid = []
        for ds in datasets:
            hr_dict = r[ds].get(args.mode, {}).get('hr_at_k', {})
            v = hr_dict.get(str(k))
            if v is None or n_pairs[ds] == 0:
                row.append(f'{"-":>10}')
            else:
                row.append(f'{v:>10.3f}')
                valid.append(v)
        if valid:
            row.append(f'{sum(valid)/len(valid):>8.3f}')
        else:
            row.append(f'{"-":>8}')
        print(' '.join(row))

    return 0


if __name__ == '__main__':
    sys.exit(main())
