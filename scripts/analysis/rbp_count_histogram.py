"""Per-phage RBP-count histogram for the H3 multi-RBP label-leakage test.

For each phage_id in host_phage_protein_map.tsv, count:
  - distinct protein_ids that are in pipeline_positive.list ("RBP")
  - distinct protein_ids where is_tsp == 1

Then histogram both distributions (1, 2, 3, 4+).

Usage:
    python rbp_count_histogram.py \\
        --map data/training_data/metadata/host_phage_protein_map.tsv \\
        --positive_list data/training_data/metadata/pipeline_positive.list \\
        [--out results/analysis/rbp_count_histogram.csv]

DATA_DIR (env var) is honoured as a path prefix when --map / --positive_list are
relative.
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict


def _resolve(path, base):
    if os.path.isabs(path):
        return path
    return os.path.join(base, path) if base else path


def load_positive_set(path):
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def collect_per_phage_counts(map_path, positives):
    """Return dict: phage_id -> {'rbp': set, 'tsp': set, 'host_K': str, 'host_O': str}."""
    per_phage = defaultdict(lambda: {'rbp': set(), 'tsp': set(),
                                     'host_K': None, 'host_O': None})
    with open(map_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            phage = row['phage_id']
            prot = row['protein_id']
            is_tsp = row['is_tsp'].strip() == '1'
            entry = per_phage[phage]
            if entry['host_K'] is None:
                entry['host_K'] = row['host_K']
                entry['host_O'] = row['host_O']
            if prot in positives:
                entry['rbp'].add(prot)
            if is_tsp:
                entry['tsp'].add(prot)
    return per_phage


def histogram(counts):
    """counts: iterable of ints. Returns Counter binned 0,1,2,3,'4+'."""
    h = Counter()
    for c in counts:
        if c >= 4:
            h['4+'] += 1
        else:
            h[c] += 1
    return h


def fmt_hist(h, total):
    keys = [0, 1, 2, 3, '4+']
    rows = []
    for k in keys:
        n = h.get(k, 0)
        pct = 100.0 * n / total if total else 0.0
        rows.append(f'  {str(k):>3} RBPs: {n:5d} phages  ({pct:5.1f}%)')
    return '\n'.join(rows)


def per_k_breakdown(per_phage, count_field='rbp'):
    """Per-K-type RBP-count summary. Returns list of (K, n_phages, hist)."""
    by_k = defaultdict(list)
    for phage, entry in per_phage.items():
        k = entry['host_K'] or 'null'
        by_k[k].append(len(entry[count_field]))
    rows = []
    for k in sorted(by_k.keys(), key=lambda x: (-len(by_k[x]), x)):
        counts = by_k[k]
        h = histogram(counts)
        rows.append((k, len(counts), h))
    return rows


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--map', required=True,
                    help='host_phage_protein_map.tsv')
    ap.add_argument('--positive_list', required=True,
                    help='pipeline_positive.list')
    ap.add_argument('--out', default=None,
                    help='CSV output (per-phage counts). Optional.')
    ap.add_argument('--data_dir', default=os.environ.get('DATA_DIR', ''),
                    help='Base dir for relative paths; default $DATA_DIR.')
    args = ap.parse_args()

    map_path = _resolve(args.map, args.data_dir)
    pos_path = _resolve(args.positive_list, args.data_dir)

    positives = load_positive_set(pos_path)
    print(f'Loaded {len(positives):,} pipeline_positive protein IDs', file=sys.stderr)

    per_phage = collect_per_phage_counts(map_path, positives)
    n_phages = len(per_phage)
    print(f'Loaded {n_phages:,} phages from map', file=sys.stderr)

    rbp_counts = [len(e['rbp']) for e in per_phage.values()]
    tsp_counts = [len(e['tsp']) for e in per_phage.values()]

    h_rbp = histogram(rbp_counts)
    h_tsp = histogram(tsp_counts)

    print()
    print(f'=== Per-phage pipeline_positive ("RBP") count histogram '
          f'(n_phages = {n_phages:,}) ===')
    print(fmt_hist(h_rbp, n_phages))
    print(f'  total positive proteins (sum): {sum(rbp_counts):,}')
    print(f'  median: {sorted(rbp_counts)[n_phages // 2]}')

    print()
    print(f'=== Per-phage is_tsp==1 count histogram (n_phages = {n_phages:,}) ===')
    print(fmt_hist(h_tsp, n_phages))
    print(f'  total tsp proteins (sum): {sum(tsp_counts):,}')
    print(f'  median: {sorted(tsp_counts)[n_phages // 2]}')

    n_single_rbp = sum(1 for c in rbp_counts if c == 1)
    n_zero_rbp = sum(1 for c in rbp_counts if c == 0)
    n_multi_rbp = sum(1 for c in rbp_counts if c >= 2)
    print()
    print(f'=== H3 split summary ===')
    print(f'  phages with 0 RBP:  {n_zero_rbp:5d}  ({100*n_zero_rbp/n_phages:.1f}%)  [excluded from training]')
    print(f'  phages with 1 RBP:  {n_single_rbp:5d}  ({100*n_single_rbp/n_phages:.1f}%)  [single-RBP training set]')
    print(f'  phages with ≥2 RBP: {n_multi_rbp:5d}  ({100*n_multi_rbp/n_phages:.1f}%)  [multi-RBP, dropped under H3 test]')

    n_single_tsp = sum(1 for c in tsp_counts if c == 1)
    print(f'  --- stricter is_tsp==1 cut ---')
    print(f'  phages with exactly 1 tsp:  {n_single_tsp:5d}  ({100*n_single_tsp/n_phages:.1f}%)')

    # Per-K breakdown (top-15 K-types by phage count)
    print()
    print(f'=== Per-K RBP-count breakdown (top 15 K by n_phages) ===')
    print(f'  {"K":<8} {"n_phages":>8} {"0":>5} {"1":>5} {"2":>5} {"3":>5} {"4+":>5}  {"%single":>7}')
    rows = per_k_breakdown(per_phage, 'rbp')
    for k, n, h in rows[:15]:
        c0, c1, c2, c3, c4 = (h.get(0, 0), h.get(1, 0), h.get(2, 0),
                              h.get(3, 0), h.get('4+', 0))
        pct_single = 100.0 * c1 / n if n else 0
        print(f'  {k:<8} {n:>8} {c0:>5} {c1:>5} {c2:>5} {c3:>5} {c4:>5}  {pct_single:>6.1f}%')

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['phage_id', 'host_K', 'host_O',
                        'n_pipeline_positive', 'n_is_tsp'])
            for phage in sorted(per_phage.keys()):
                e = per_phage[phage]
                w.writerow([phage, e['host_K'], e['host_O'],
                            len(e['rbp']), len(e['tsp'])])
        print(f'\nPer-phage CSV written: {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
