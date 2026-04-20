"""Compare experiments focusing on PhageHostLearn and PBIP only.

The 5-dataset mean is misleading because CHEN (3 phages, 50 hosts) and
GORODNICHIV (3 phages, 83 hosts) have tiny candidate pools that inflate
rank_phages scores. PHL (~60-98 candidates) and PBIP (~9 candidates per
direction) are the meaningful tests per the project's validation priorities.

Usage:
    python scripts/analysis/compare_primary_datasets.py
    python scripts/analysis/compare_primary_datasets.py --filter sweep_
"""

import argparse
import json
import os
from glob import glob

PRIMARY = ('PhageHostLearn', 'PBIP')


def extract(eval_path):
    with open(eval_path) as f:
        data = json.load(f)
    out = {}
    for ds in PRIMARY:
        r = data.get(ds, {})
        rh = r.get('rank_hosts', {}).get('hr_at_k', {})
        rp = r.get('rank_phages', {}).get('hr_at_k', {})
        out[ds] = {
            'rh1': rh.get('1', 0.0),
            'rh5': rh.get('5', 0.0),
            'rp1': rp.get('1', 0.0),
            'rp5': rp.get('5', 0.0),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='attention_mlp')
    p.add_argument('--filter', default='')
    args = p.parse_args()

    rows = []
    pattern = f'experiments/{args.model}/*/results/evaluation.json'
    for path in sorted(glob(pattern)):
        name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if args.filter and args.filter not in name:
            continue
        ds = extract(path)
        phl_avg = (ds['PhageHostLearn']['rh1'] + ds['PhageHostLearn']['rp1']) / 2
        pbip_avg = (ds['PBIP']['rh1'] + ds['PBIP']['rp1']) / 2
        rows.append({
            'name': name,
            'phl_rh1': ds['PhageHostLearn']['rh1'],
            'phl_rh5': ds['PhageHostLearn']['rh5'],
            'phl_rp1': ds['PhageHostLearn']['rp1'],
            'pbip_rh1': ds['PBIP']['rh1'],
            'pbip_rh5': ds['PBIP']['rh5'],
            'pbip_rp1': ds['PBIP']['rp1'],
            'combined': (phl_avg + pbip_avg) / 2,
        })

    rows.sort(key=lambda r: -r['combined'])

    hdr = (f"{'Experiment':<32} "
           f"{'PHL rh@1':>9} {'PHL rh@5':>9} {'PHL rp@1':>9} | "
           f"{'PBIP rh@1':>10} {'PBIP rh@5':>10} {'PBIP rp@1':>10} | "
           f"{'combined':>9}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['name']:<32} "
              f"{r['phl_rh1']:>9.3f} {r['phl_rh5']:>9.3f} {r['phl_rp1']:>9.3f} | "
              f"{r['pbip_rh1']:>10.3f} {r['pbip_rh5']:>10.3f} {r['pbip_rp1']:>10.3f} | "
              f"{r['combined']:>9.3f}")


if __name__ == '__main__':
    main()
