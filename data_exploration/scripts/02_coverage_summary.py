"""High-level coverage summary derived from master_K_table and master_O_table.

Reports:
- How many PHL serotypes are not in trained model
- How many positive PHL pairs have valid K, O, both, or neither
- Achievable HR@1 ceiling

Outputs:
    data_exploration/output/coverage_summary.txt
"""

import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from _common import OUTPUT_DIR, REPO_ROOT, load_phl_data


def load_master_table(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main():
    k_table = load_master_table(os.path.join(OUTPUT_DIR, 'master_K_table.csv'))
    o_table = load_master_table(os.path.join(OUTPUT_DIR, 'master_O_table.csv'))

    # Build trained class sets and PHL counts from the tables
    trained_k = {r['k_type'] for r in k_table if r['in_trained_classes'] == 'Y'}
    trained_o = {r['o_type'] for r in o_table if r['in_trained_classes'] == 'Y'}

    phl_k_rows = [r for r in k_table if int(r['n_phl_pos_pairs']) > 0]
    phl_o_rows = [r for r in o_table if int(r['n_phl_pos_pairs']) > 0]

    # Per-pair coverage from raw PHL data
    phl_pairs = load_phl_data()
    pos_pairs = [p for p in phl_pairs if p['label'] == 1]

    n_total = len(pos_pairs)
    n_k_valid = sum(1 for p in pos_pairs if p['host_K'] in trained_k)
    n_o_valid = sum(1 for p in pos_pairs if p['host_O'] in trained_o)
    n_both_valid = sum(1 for p in pos_pairs
                       if p['host_K'] in trained_k and p['host_O'] in trained_o)
    n_neither = sum(1 for p in pos_pairs
                    if p['host_K'] not in trained_k and p['host_O'] not in trained_o)
    n_either = n_total - n_neither

    # Threshold-based class coverage
    def coverage(threshold):
        return sum(1 for r in phl_k_rows
                   if int(r['n_train_md5s']) >= threshold)

    n_phl_k = len(phl_k_rows)
    n_phl_k_with_train = sum(1 for r in phl_k_rows if int(r['n_train_md5s']) > 0)
    n_phl_k_with_5 = coverage(5)
    n_phl_k_with_25 = coverage(25)
    n_phl_k_in_trained = sum(1 for r in phl_k_rows if r['in_trained_classes'] == 'Y')

    n_phl_o = len(phl_o_rows)
    n_phl_o_in_trained = sum(1 for r in phl_o_rows if r['in_trained_classes'] == 'Y')

    def pct(num, den):
        return f'{100.0 * num / den:.1f}%' if den > 0 else 'N/A'

    out_lines = []
    out_lines.append('=' * 70)
    out_lines.append('COVERAGE SUMMARY — PhageHostLearn')
    out_lines.append('=' * 70)
    out_lines.append('')
    out_lines.append(f'K-types appearing in PHL positive pairs:  {n_phl_k}')
    out_lines.append(f'  ...with >= 1 training MD5:               {n_phl_k_with_train}  ({pct(n_phl_k_with_train, n_phl_k)})')
    out_lines.append(f'  ...with >= 5 training MD5:               {n_phl_k_with_5}  ({pct(n_phl_k_with_5, n_phl_k)})')
    out_lines.append(f'  ...with >= 25 training MD5:              {n_phl_k_with_25}  ({pct(n_phl_k_with_25, n_phl_k)})')
    out_lines.append(f'  ...in the trained model class set:       {n_phl_k_in_trained}  ({pct(n_phl_k_in_trained, n_phl_k)})')
    out_lines.append(f'  ...NOT in trained model:                 {n_phl_k - n_phl_k_in_trained}  ({pct(n_phl_k - n_phl_k_in_trained, n_phl_k)})')
    out_lines.append('')
    out_lines.append(f'O-types appearing in PHL positive pairs:  {n_phl_o}')
    out_lines.append(f'  ...in the trained model class set:       {n_phl_o_in_trained}  ({pct(n_phl_o_in_trained, n_phl_o)})')
    out_lines.append('')
    out_lines.append(f'Total PHL positive pairs:  {n_total}')
    out_lines.append(f'  ...with valid K (in trained classes):    {n_k_valid}  ({pct(n_k_valid, n_total)})')
    out_lines.append(f'  ...with valid O (in trained classes):    {n_o_valid}  ({pct(n_o_valid, n_total)})')
    out_lines.append(f'  ...with valid K AND valid O:             {n_both_valid}  ({pct(n_both_valid, n_total)})')
    out_lines.append(f'  ...with valid K OR valid O (scoreable):  {n_either}  ({pct(n_either, n_total)})')
    out_lines.append(f'  ...with NEITHER (unscoreable):           {n_neither}  ({pct(n_neither, n_total)})')
    out_lines.append('')
    out_lines.append(f'Achievable HR@1 ceiling (perfect K classifier): ~{pct(n_either, n_total)}')
    out_lines.append('  (the unscoreable pairs cannot be ranked correctly regardless of model)')
    out_lines.append('')

    # Show top-10 PHL K-types ranked by # positive pairs
    out_lines.append('Top 15 PHL K-types by positive pair count:')
    out_lines.append('-' * 70)
    out_lines.append(f'  {"K_type":<14} {"PHL_pairs":>10} {"trained":>9} {"n_train":>9} {"test_acc":>10}')
    for r in sorted(phl_k_rows, key=lambda x: -int(x['n_phl_pos_pairs']))[:15]:
        acc = r.get('test_top1_acc', '')
        acc_str = f'{float(acc):.3f}' if acc else 'N/A'
        out_lines.append(
            f'  {r["k_type"]:<14} {r["n_phl_pos_pairs"]:>10} '
            f'{r["in_trained_classes"]:>9} {r["n_train_md5s"]:>9} {acc_str:>10}'
        )

    text = '\n'.join(out_lines)
    print(text)

    out_path = os.path.join(OUTPUT_DIR, 'coverage_summary.txt')
    with open(out_path, 'w') as f:
        f.write(text + '\n')
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
