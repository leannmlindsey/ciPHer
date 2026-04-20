"""Filter the old phold.rmdup cluster file down to our current candidates.

Reads the 1.36M-row multi-threshold cluster file from the old klebsiella repo
and writes a trimmed TSV containing only proteins present in our
candidates.faa. Preserves the original tab-separated format:

    protein_id  cl30_X  cl40_X  cl50_X  cl60_X  cl70_X  cl80_X  cl85_X  cl90_X  cl95_X

One-time script — re-run only if candidates.faa changes.

Usage:
    python scripts/build_candidates_cluster_file.py
    python scripts/build_candidates_cluster_file.py \\
        --source-cluster-file /path/to/phold.rmdup.faa_cluster.txt
"""

import argparse
import os
import sys


def load_candidate_ids(fasta_path):
    ids = set()
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                ids.add(line[1:].split()[0].strip())
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidates-fasta',
                   default='data/training_data/metadata/candidates.faa')
    p.add_argument('--source-cluster-file',
                   default='/Users/leannmlindsey/WORK/PHI_TSP/phi_tsp/klebsiella/'
                           'data/proteins/phold.rmdup.faa_cluster.txt')
    p.add_argument('--out',
                   default='data/training_data/metadata/candidates_clusters.tsv')
    args = p.parse_args()

    ids = load_candidate_ids(args.candidates_fasta)
    print(f'Loaded {len(ids):,} candidate IDs from {args.candidates_fasta}')

    if not os.path.isfile(args.source_cluster_file):
        print(f'ERROR: source cluster file not found: {args.source_cluster_file}',
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    found = 0
    with open(args.source_cluster_file) as f, open(args.out, 'w') as out:
        for line in f:
            pid = line.split('\t', 1)[0]
            if pid in ids:
                out.write(line)
                found += 1

    print(f'Wrote {args.out}: {found:,} rows')
    if found < len(ids):
        print(f'WARNING: {len(ids) - found:,} candidate IDs missing from '
              f'cluster file', file=sys.stderr)


if __name__ == '__main__':
    main()
