"""Aggregate every experiment's old_style_eval.json into one CSV.

Walks experiments/ for results_old_style/old_style_eval.json files and
produces results/old_style_eval_summary.csv — one row per experiment,
sorted by PHL HR@1 (the headline metric used to compare against OLD
klebsiella's 0.291 baseline).

This is the OLD-eval analogue to results/experiment_log.csv (which is
sorted by NEW eval).

Usage:
    python scripts/analysis/old_style_eval_summary.py
    python scripts/analysis/old_style_eval_summary.py --experiments-dirs \\
        experiments ../cipher-light-attention/experiments
"""

import argparse
import csv
import json
import os
import sys
from glob import glob


DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']


def find_runs(exp_dirs):
    runs = []
    for root in exp_dirs:
        for path in sorted(glob(os.path.join(root, '*', '*'))):
            j = os.path.join(path, 'results_old_style', 'old_style_eval.json')
            if os.path.isfile(j):
                runs.append(path)
    return runs


def get_emb_type(exp_dir):
    p = os.path.join(exp_dir, 'experiment.json')
    if not os.path.exists(p):
        return ''
    try:
        d = json.load(open(p))
        return d.get('config', {}).get('data', {}).get('embedding_type', '') or ''
    except Exception:
        return ''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiments-dirs', nargs='+',
                   default=['experiments'],
                   help='one or more experiment-roots to scan')
    p.add_argument('--out', default='results/old_style_eval_summary.csv')
    args = p.parse_args()

    runs = find_runs(args.experiments_dirs)
    if not runs:
        sys.exit(f'No old_style_eval.json found under {args.experiments_dirs}')

    rows = []
    for exp in runs:
        d = json.load(open(os.path.join(exp, 'results_old_style', 'old_style_eval.json')))
        row = {
            'run_name': os.path.basename(exp.rstrip('/')),
            'model': os.path.basename(os.path.dirname(exp.rstrip('/'))),
            'exp_dir': os.path.abspath(exp),
            'embedding_type': get_emb_type(exp),
        }
        for ds in DATASETS + ['OVERALL']:
            r = d.get(ds, {})
            row[f'{ds}_n'] = r.get('n_pairs', '')
            row[f'{ds}_merged_HR1'] = r.get('hr_at_k', {}).get('1', '') if r else ''
            row[f'{ds}_merged_HR5'] = r.get('hr_at_k', {}).get('5', '') if r else ''
            row[f'{ds}_K_HR1'] = r.get('k_hr_at_k', {}).get('1', '') if r else ''
            row[f'{ds}_O_HR1'] = r.get('o_hr_at_k', {}).get('1', '') if r else ''
        rows.append(row)

    # Sort by PHL merged HR@1 desc
    def f(v):
        try: return float(v)
        except: return -1.0
    rows.sort(key=lambda r: -f(r.get('PhageHostLearn_merged_HR1', '')))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'Wrote {args.out} ({len(rows)} runs)')

    # Print headline table
    print()
    print(f'{"#":>2} {"run_name":<55} {"emb":<22} {"PHL_m@1":>8} {"PHL_K@1":>8} {"PHL_O@1":>8}')
    print('-' * 110)
    for i, r in enumerate(rows, 1):
        phl_m = f(r['PhageHostLearn_merged_HR1'])
        phl_k = f(r['PhageHostLearn_K_HR1'])
        phl_o = f(r['PhageHostLearn_O_HR1'])
        print(f'{i:>2} {r["run_name"][:55]:<55} {r["embedding_type"][:22]:<22} '
              f'{phl_m:>8.4f} {phl_k:>8.4f} {phl_o:>8.4f}')

    print()
    print('OLD klebsiella reference: PHL merged HR@1 = 0.291 (raw merge baseline).')


if __name__ == '__main__':
    main()
