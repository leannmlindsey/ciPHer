"""Per-phage broad-RBP histogram against the FULL phage_protein.tsv.

Streams phage_protein.tsv (1.36M proteins × 722k phages), filters proteins
by membership in either:
  - pipeline_positive.list (NARROW), or
  - glycan_binders_custom.tsv with total_sources >= --min_tool_sources (BROAD)

Joins host accession from phage_id (regex-extract GCA_xxx.x prefix) to
20250106.K-O.tsv for K/O type, so per-K histograms are also produced.

Usage:
    python rbp_count_histogram_broad_full.py \\
        --phage_protein /Users/.../phi_tsp/klebsiella/data/phage_protein.tsv \\
        --glycan_tools  data/training_data/metadata/glycan_binders_custom.tsv \\
        --positive_list data/training_data/metadata/pipeline_positive.list \\
        --serotypes     data/training_data/metadata/20250106.K-O.tsv \\
        --out results/analysis/rbp_count_histogram_broad_full.csv
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict


HOST_RE = re.compile(r'(GCA_\d+\.\d+|GCF_\d+\.\d+)')


def _resolve(path, base):
    if os.path.isabs(path):
        return path
    return os.path.join(base, path) if base else path


def load_positive_set(path):
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def load_tool_flagged_set(path, min_sources=1):
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


def load_serotypes(path):
    serotypes = {}
    with open(path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                serotypes[parts[0].strip()] = (parts[1].strip(), parts[2].strip())
    return serotypes


def extract_host(phage_id):
    m = HOST_RE.match(phage_id)
    return m.group(1) if m else None


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
    ap.add_argument('--phage_protein', required=True,
                    help='phage_protein.tsv (full, ~1.4 GB)')
    ap.add_argument('--glycan_tools', required=True)
    ap.add_argument('--positive_list', required=True)
    ap.add_argument('--serotypes', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--data_dir', default=os.environ.get('DATA_DIR', ''))
    ap.add_argument('--min_tool_sources', type=int, default=1)
    args = ap.parse_args()

    pp_path = _resolve(args.phage_protein, args.data_dir)
    tools_path = _resolve(args.glycan_tools, args.data_dir)
    pos_path = _resolve(args.positive_list, args.data_dir)
    sero_path = _resolve(args.serotypes, args.data_dir)

    print('Loading reference sets...', file=sys.stderr)
    broad = load_tool_flagged_set(tools_path, args.min_tool_sources)
    narrow = load_positive_set(pos_path)
    serotypes = load_serotypes(sero_path)
    print(f'  broad (tools >= {args.min_tool_sources}): {len(broad):,}',
          file=sys.stderr)
    print(f'  narrow (pipeline_positive): {len(narrow):,}', file=sys.stderr)
    print(f'  serotype entries: {len(serotypes):,}', file=sys.stderr)

    # Per-phage protein sets (broad and narrow)
    per_phage_broad = defaultdict(set)
    per_phage_narrow = defaultdict(set)
    n_lines = 0
    n_skipped_broad = 0
    print(f'\nStreaming {pp_path}...', file=sys.stderr)
    with open(pp_path) as f:
        for line in f:
            n_lines += 1
            if n_lines % 5_000_000 == 0:
                print(f'  ... {n_lines:,} rows', file=sys.stderr)
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            phage, prot = parts[0], parts[1]
            if prot in broad:
                per_phage_broad[phage].add(prot)
            else:
                n_skipped_broad += 1
            if prot in narrow:
                per_phage_narrow[phage].add(prot)

    n_phages_broad = len(per_phage_broad)
    print(f'\nDone. {n_lines:,} rows read.', file=sys.stderr)
    print(f'  phages with >=1 broad-RBP: {n_phages_broad:,}', file=sys.stderr)
    print(f'  phages with >=1 narrow-RBP: {len(per_phage_narrow):,}', file=sys.stderr)

    broad_counts = [len(s) for s in per_phage_broad.values()]
    narrow_counts_aligned = [
        len(per_phage_narrow.get(p, set())) for p in per_phage_broad.keys()
    ]

    print()
    print(f'=== Per-phage BROAD RBP count (>=1 tool flagged) ===')
    print(f'  n_phages with any broad RBP = {n_phages_broad:,}')
    print(fmt_hist(histogram(broad_counts), n_phages_broad))
    print(f'  total broad RBPs (sum): {sum(broad_counts):,}')
    print(f'  median broad/phage: {sorted(broad_counts)[n_phages_broad // 2]}')
    print(f'  mean broad/phage:   {sum(broad_counts) / n_phages_broad:.2f}')

    print()
    print(f'=== Per-phage NARROW RBP count (pipeline_positive) — same phages ===')
    print(fmt_hist(histogram(narrow_counts_aligned), n_phages_broad))
    print(f'  total narrow RBPs (sum): {sum(narrow_counts_aligned):,}')
    print(f'  median narrow/phage: {sorted(narrow_counts_aligned)[n_phages_broad // 2]}')

    n_single = sum(1 for c in broad_counts if c == 1)
    n_multi = sum(1 for c in broad_counts if c >= 2)
    print()
    print(f'=== H3 split (BROAD definition) ===')
    print(f'  1 broad RBP : {n_single:7d}  ({100*n_single/n_phages_broad:.1f}%)')
    print(f'  >=2 broad RBP: {n_multi:7d}  ({100*n_multi/n_phages_broad:.1f}%)')

    # Per-K breakdown using host extraction
    print(f'\n=== Per-K BROAD RBP histogram (top 20 K by n_phages) ===')
    by_k = defaultdict(list)
    n_no_host = 0
    n_no_sero = 0
    for phage, prots in per_phage_broad.items():
        host = extract_host(phage)
        if host is None:
            n_no_host += 1
            continue
        if host not in serotypes:
            n_no_sero += 1
            continue
        k = serotypes[host][0]
        by_k[k].append(len(prots))
    print(f'  (skipped: {n_no_host:,} no-host-prefix, {n_no_sero:,} no-serotype)')
    print(f'  {"K":<10} {"n_phages":>9} {"med":>4} {"mean":>5} {"1":>6} {"2":>6} {"3":>6} {"4-5":>5} {"6+":>4}  {"%single":>7}')
    rows = sorted(by_k.items(), key=lambda kv: -len(kv[1]))
    for k, counts in rows[:20]:
        n = len(counts)
        h = histogram(counts)
        c1 = h.get(1, 0)
        med = sorted(counts)[n // 2]
        mean = sum(counts) / n
        print(f'  {k:<10} {n:>9} {med:>4} {mean:>5.2f} '
              f'{h.get(1,0):>6} {h.get(2,0):>6} {h.get(3,0):>6} '
              f'{h.get("4-5",0):>5} {h.get("6+",0):>4}  '
              f'{100*c1/n:>6.1f}%')

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['phage_id', 'host_accession', 'K', 'O',
                        'n_broad_rbp', 'n_pipeline_positive'])
            for phage in sorted(per_phage_broad.keys()):
                host = extract_host(phage) or ''
                k, o = serotypes.get(host, ('', ''))
                n_broad = len(per_phage_broad[phage])
                n_narrow = len(per_phage_narrow.get(phage, set()))
                w.writerow([phage, host, k, o, n_broad, n_narrow])
        print(f'\nPer-phage CSV: {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
