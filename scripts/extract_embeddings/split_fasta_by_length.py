"""Split a FASTA file into length-binned files (powers of two).

Bins are cumulative caps: <=128, <=256, <=512, <=1024, <=2048, <=4096.
Anything longer than 4096 goes into a maxlen8192 overflow bin.

Within each bin, sequences are written in length-ascending order so GPU
batches see relatively uniform-length sequences and pad less. Output
filenames follow the convention used by run_embedding_sweep.sh and the
klebsiella tooling:

    <output_dir>/<basename>_maxlen128.faa
    <output_dir>/<basename>_maxlen256.faa
    ... etc.

Usage:
    python scripts/extract_embeddings/split_fasta_by_length.py \\
        input.faa output_dir/
"""

import argparse
import os
from collections import Counter


BINS = [128, 256, 512, 1024, 2048, 4096]
OVERFLOW = 8192


def get_bin(length):
    for b in BINS:
        if length <= b:
            return b
    return OVERFLOW


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input_fasta')
    p.add_argument('output_dir')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sequences = []
    cur_id = None
    cur_seq = []
    with open(args.input_fasta) as f:
        for line in f:
            if line.startswith('>'):
                if cur_id:
                    seq = ''.join(cur_seq)
                    sequences.append((cur_id, seq, len(seq)))
                cur_id = line[1:].strip().split()[0]
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_id:
            seq = ''.join(cur_seq)
            sequences.append((cur_id, seq, len(seq)))

    print(f'Total sequences: {len(sequences):,}')

    sequences.sort(key=lambda x: x[2])
    basename = os.path.splitext(os.path.basename(args.input_fasta))[0]
    bin_counts = Counter()
    bin_files = {}

    for seq_id, seq, length in sequences:
        b = get_bin(length)
        bin_counts[b] += 1
        if b not in bin_files:
            bin_path = os.path.join(args.output_dir,
                                    f'{basename}_maxlen{b}.faa')
            bin_files[b] = open(bin_path, 'w')
        bin_files[b].write(f'>{seq_id}\n{seq}\n')

    for fh in bin_files.values():
        fh.close()

    print('\nBin summary:')
    print(f'{"max_length":<12} {"sequences":>10}  path')
    print('-' * 70)
    for b in sorted(bin_counts):
        path = os.path.join(args.output_dir, f'{basename}_maxlen{b}.faa')
        print(f'<= {b:<10} {bin_counts[b]:>10,}  {path}')
    print(f'\nTotal: {sum(bin_counts.values()):,} sequences '
          f'in {len(bin_counts)} bin(s)')


if __name__ == '__main__':
    main()
