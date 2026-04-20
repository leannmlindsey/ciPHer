#!/usr/bin/env python3
"""Print HR@1 (host and phage ranking) from an evaluation.json.

Usage:
    python scripts/analysis/show_eval.py <path-to-evaluation.json>
    python scripts/analysis/show_eval.py experiments/light_attention/la_seg4_match_sweep/results/evaluation.json
"""

import argparse
import json
import os


PRIMARY_ORDER = ['PBIP', 'PhageHostLearn', 'UCSD', 'CHEN', 'GORODNICHIV']


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('path', help='Path to evaluation.json')
    p.add_argument('--k', default='1',
                   help='Which HR@k to report (default: 1)')
    args = p.parse_args()

    with open(args.path) as f:
        r = json.load(f)

    print(f'File: {os.path.abspath(args.path)}')
    meta = r.get('_meta', {})
    if meta:
        print(f"  tie_method={meta.get('tie_method')}  "
              f"score_norm={meta.get('score_norm')}  "
              f"max_k={meta.get('max_k')}")
    print()

    k = args.k
    print(f'{"Dataset":<16} {"host HR@" + k:>12} {"phage HR@" + k:>13}  Pairs')
    print('-' * 60)

    datasets = [d for d in PRIMARY_ORDER if d in r]
    datasets += [d for d in r if d not in PRIMARY_ORDER and d != '_meta']

    host_vals, phage_vals = [], []
    for ds in datasets:
        v = r[ds]
        host_hr = v['rank_hosts']['hr_at_k'].get(k)
        phage_hr = v['rank_phages']['hr_at_k'].get(k)
        pairs = v['rank_hosts']['n_pairs']
        if host_hr is not None:
            host_vals.append(host_hr)
        if phage_hr is not None:
            phage_vals.append(phage_hr)
        host_s = f'{host_hr:.3f}' if host_hr is not None else '-'
        phage_s = f'{phage_hr:.3f}' if phage_hr is not None else '-'
        print(f'{ds:<16} {host_s:>12} {phage_s:>13}  {pairs:>5}')

    if host_vals:
        print('-' * 60)
        print(f'{"Mean":<16} {sum(host_vals)/len(host_vals):>12.3f} '
              f'{sum(phage_vals)/len(phage_vals):>13.3f}')


if __name__ == '__main__':
    main()
