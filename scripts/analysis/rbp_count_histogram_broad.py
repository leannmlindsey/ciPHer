"""Per-phage RBP-count histogram using the BROAD tool-flagged universe.

RBP = any protein with total_sources >= 1 in glycan_binders_custom.tsv
(union of 8 tool predictions: DePP_85, PhageRBPdetect, DepoScope,
DepoRanker, SpikeHunter, dbCAN, IPR, phold_glycan_tailspike).

Compare against pipeline_positive.list (advisor-curated subset).

Also reports n_total_distinct_proteins_per_phage from the map directly,
so we can tell whether host_phage_protein_map.tsv is filtered to
pipeline_positive only, or whether it covers the broader universe.

Usage:
    python rbp_count_histogram_broad.py \\
        --map data/training_data/metadata/host_phage_protein_map.tsv \\
        --glycan_tools data/training_data/metadata/glycan_binders_custom.tsv \\
        --positive_list data/training_data/metadata/pipeline_positive.list \\
        [--out results/analysis/rbp_count_histogram_broad.csv]
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


def load_tool_flagged_set(path, min_sources=1):
    """Return set of protein_ids with total_sources >= min_sources."""
    flagged = set()
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                ts = int(row['total_sources'])
            except (KeyError, ValueError):
                continue
            if ts >= min_sources:
                flagged.add(row['protein_id'])
    return flagged


def collect_per_phage_counts(map_path, broad_set, narrow_set):
    """Per phage_id: counts under broad (tool-flagged) and narrow (pipeline_positive),
    plus total distinct proteins in the map (no filter)."""
    per_phage = defaultdict(lambda: {
        'broad': set(), 'narrow': set(), 'all_in_map': set(),
        'host_K': None, 'host_O': None,
    })
    with open(map_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            phage = row['phage_id']
            prot = row['protein_id']
            entry = per_phage[phage]
            if entry['host_K'] is None:
                entry['host_K'] = row['host_K']
                entry['host_O'] = row['host_O']
            entry['all_in_map'].add(prot)
            if prot in broad_set:
                entry['broad'].add(prot)
            if prot in narrow_set:
                entry['narrow'].add(prot)
    return per_phage


def histogram(counts):
    h = Counter()
    for c in counts:
        if c >= 6:
            h['6+'] += 1
        elif c >= 4:
            h['4-5'] += 1
        else:
            h[c] += 1
    return h


def fmt_hist(h, total, keys=(0, 1, 2, 3, '4-5', '6+')):
    rows = []
    for k in keys:
        n = h.get(k, 0)
        pct = 100.0 * n / total if total else 0.0
        rows.append(f'  {str(k):>4} : {n:7d} phages  ({pct:5.1f}%)')
    return '\n'.join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', required=True)
    ap.add_argument('--glycan_tools', required=True)
    ap.add_argument('--positive_list', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--data_dir', default=os.environ.get('DATA_DIR', ''))
    ap.add_argument('--min_tool_sources', type=int, default=1,
                    help='broad set = total_sources >= this (default 1)')
    args = ap.parse_args()

    map_path = _resolve(args.map, args.data_dir)
    tools_path = _resolve(args.glycan_tools, args.data_dir)
    pos_path = _resolve(args.positive_list, args.data_dir)

    broad = load_tool_flagged_set(tools_path, args.min_tool_sources)
    narrow = load_positive_set(pos_path)
    print(f'Broad tool-flagged set (total_sources >= {args.min_tool_sources}): '
          f'{len(broad):,} proteins', file=sys.stderr)
    print(f'Narrow pipeline_positive set: {len(narrow):,} proteins',
          file=sys.stderr)
    print(f'Overlap broad ∩ narrow: {len(broad & narrow):,}', file=sys.stderr)
    print(f'Narrow but not broad:  {len(narrow - broad):,}', file=sys.stderr)
    print(f'Broad but not narrow:  {len(broad - narrow):,}', file=sys.stderr)

    per_phage = collect_per_phage_counts(map_path, broad, narrow)
    n_phages = len(per_phage)
    print(f'\nLoaded {n_phages:,} phages from map', file=sys.stderr)

    broad_counts = [len(e['broad']) for e in per_phage.values()]
    narrow_counts = [len(e['narrow']) for e in per_phage.values()]
    all_counts = [len(e['all_in_map']) for e in per_phage.values()]

    print()
    print(f'=== Per-phage TOTAL distinct proteins in map (no filter) ===')
    print(f'  n_phages = {n_phages:,}')
    print(fmt_hist(histogram(all_counts), n_phages))
    print(f'  total distinct (sum): {sum(all_counts):,}')
    print(f'  median: {sorted(all_counts)[n_phages // 2]}')
    print(f'  >>> If this matches the narrow histogram, the map IS filtered '
          f'to pipeline_positive only.')

    print()
    print(f'=== Per-phage BROAD (tool-flagged, ≥{args.min_tool_sources} tool) ===')
    print(fmt_hist(histogram(broad_counts), n_phages))
    print(f'  total broad (sum): {sum(broad_counts):,}')
    print(f'  median: {sorted(broad_counts)[n_phages // 2]}')

    print()
    print(f'=== Per-phage NARROW (pipeline_positive) — for comparison ===')
    print(fmt_hist(histogram(narrow_counts), n_phages))
    print(f'  total narrow (sum): {sum(narrow_counts):,}')
    print(f'  median: {sorted(narrow_counts)[n_phages // 2]}')

    # H3 split summary under broad
    n_single_broad = sum(1 for c in broad_counts if c == 1)
    n_zero_broad = sum(1 for c in broad_counts if c == 0)
    n_multi_broad = sum(1 for c in broad_counts if c >= 2)
    print()
    print(f'=== H3 split (broad RBP definition) ===')
    print(f'  0 RBP  : {n_zero_broad:7d} ({100*n_zero_broad/n_phages:5.1f}%)')
    print(f'  1 RBP  : {n_single_broad:7d} ({100*n_single_broad/n_phages:5.1f}%)  [single-RBP]')
    print(f'  ≥2 RBP : {n_multi_broad:7d} ({100*n_multi_broad/n_phages:5.1f}%)  [multi-RBP, dropped under H3]')

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['phage_id', 'host_K', 'host_O',
                        'n_all_in_map', 'n_broad_rbp', 'n_pipeline_positive'])
            for phage in sorted(per_phage.keys()):
                e = per_phage[phage]
                w.writerow([phage, e['host_K'], e['host_O'],
                            len(e['all_in_map']), len(e['broad']),
                            len(e['narrow'])])
        print(f'\nPer-phage CSV: {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
