"""Print a compact per-dataset HR@1 summary for one or more trained runs.

Reads `<run_dir>/results/evaluation.json` for each run and prints
rank_hosts HR@1 + rank_phages HR@1 + n_pairs per validation dataset.

Use this instead of `cat` + python-one-liners when comparing a handful
of runs on Delta — the shell heredoc path is fragile to paste.

Usage:
    python scripts/analysis/dump_eval_summary.py <run_dir> [<run_dir> ...]

Example — the three v3 multitop runs:
    python scripts/analysis/dump_eval_summary.py \\
        experiments/attention_mlp/v3_strict_prott5_mean \\
        experiments/attention_mlp/v3_uat_prott5_mean \\
        experiments/attention_mlp/v3_k_v2o_prott5_mean
"""

import json
import os
import sys

DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']


def fmt(x):
    return f'{x:.3f}' if isinstance(x, (int, float)) else '—'


def dump_one(run_dir):
    name = os.path.basename(os.path.normpath(run_dir))
    eval_path = os.path.join(run_dir, 'results', 'evaluation.json')
    if not os.path.exists(eval_path):
        print(f'=== {name} ===  (no evaluation.json at {eval_path})')
        return
    with open(eval_path) as f:
        d = json.load(f)

    print(f'=== {name} ===')
    print(f'{"Dataset":<16} {"rh@1":>7} {"rp@1":>7} {"n_pairs":>9}')
    print('-' * 44)
    for ds in DATASETS:
        ds_d = d.get(ds, {})
        rh = ds_d.get('rank_hosts', {}).get('hr_at_k', {}).get('1')
        rp = ds_d.get('rank_phages', {}).get('hr_at_k', {}).get('1')
        n = ds_d.get('rank_hosts', {}).get('n_pairs')
        print(f'{ds:<16} {fmt(rh):>7} {fmt(rp):>7} {str(n) if n is not None else "?":>9}')
    print()


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: dump_eval_summary.py <run_dir> [<run_dir> ...]')
    for run_dir in sys.argv[1:]:
        dump_one(run_dir)


if __name__ == '__main__':
    main()
