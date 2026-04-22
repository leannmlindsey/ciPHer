"""Filter a FASTA to only those entries whose protein ID appears in a list.

Protein-ID matching uses the first whitespace-delimited token of the
FASTA header. The list file has one protein ID per line; blank lines
and lines starting with '#' are ignored.

Usage:
    python scripts/extract_embeddings/filter_fasta.py \\
        --in data/training_data/metadata/candidates.faa \\
        --list data/training_data/metadata/pipeline_positive.list \\
        --out /tmp/pipeline_positive.faa
"""

import argparse
import os
import sys


def load_id_set(path):
    ids = set()
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith('#'):
                ids.add(s)
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='inp', required=True,
                   help='Input FASTA path.')
    p.add_argument('--list', dest='idlist', required=True,
                   help='Protein ID list (one per line).')
    p.add_argument('--out', required=True,
                   help='Output FASTA path.')
    args = p.parse_args()

    keep = load_id_set(args.idlist)
    if not keep:
        sys.exit(f'ERROR: {args.idlist} is empty')

    n_in = n_out = 0
    keep_flag = False
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.inp) as f_in, open(args.out, 'w') as f_out:
        for line in f_in:
            if line.startswith('>'):
                n_in += 1
                pid = line[1:].strip().split()[0]
                keep_flag = pid in keep
                if keep_flag:
                    n_out += 1
                    f_out.write(line)
            else:
                if keep_flag:
                    f_out.write(line)

    print(f'Input sequences:  {n_in:,}')
    print(f'Kept sequences:   {n_out:,}')
    print(f'Requested IDs:    {len(keep):,}')
    print(f'IDs not in FASTA: {len(keep) - n_out:,}')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
