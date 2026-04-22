"""Merge multiple per-length-bin NPZ embedding files into a single NPZ.

Reads all *.npz in a directory (or a list of explicit paths), deduplicates
by MD5 key, and writes a single compressed NPZ at the output path. Warns
on mixed embedding shapes — those can legitimately differ for per-residue
full pooling (different L per protein), but should match for mean pooling.

Usage:
    python scripts/extract_embeddings/merge_split_embeddings.py \\
        -i /work/hdd/bfzj/llindsey1/embeddings_full/split_embeddings \\
        -o /work/hdd/bfzj/llindsey1/embeddings_full/candidates_embeddings_full_md5.npz

    python scripts/extract_embeddings/merge_split_embeddings.py \\
        -i file1.npz file2.npz file3.npz \\
        -o merged.npz
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', nargs='+', required=True,
                   help='Input NPZ files, or one or more directories '
                        'containing NPZ files.')
    p.add_argument('-o', '--output', required=True,
                   help='Output merged NPZ path.')
    p.add_argument('--no-compress', action='store_true',
                   help='Use np.savez (no compression) instead of savez_compressed. '
                        'Faster to write but larger on disk. Useful for very '
                        'large per-residue outputs where compression ratio is '
                        'low and save time is significant.')
    args = p.parse_args()

    npz_files = []
    for path in args.input:
        if os.path.isdir(path):
            npz_files.extend(sorted(Path(path).glob('*.npz')))
        elif os.path.isfile(path) and path.endswith('.npz'):
            npz_files.append(Path(path))
        else:
            print(f'WARNING: skipping unrecognized input {path}', file=sys.stderr)

    if not npz_files:
        sys.exit('ERROR: no NPZ files found')

    print(f'Merging {len(npz_files)} NPZ files...')
    merged = {}
    duplicates = 0

    for npz_path in npz_files:
        data = np.load(npz_path)
        n_keys = len(data.files)
        n_new = 0
        for key in data.files:
            if key in merged:
                duplicates += 1
            else:
                merged[key] = data[key]
                n_new += 1
        size_mb = npz_path.stat().st_size / 1e6
        print(f'  {npz_path.name}: {n_keys:>7,} keys, {n_new:>7,} new '
              f'({size_mb:>7,.0f} MB on disk)')

    if duplicates:
        print(f'\nDuplicates skipped: {duplicates:,}')

    # Report dimensional consistency
    sample_shape = None
    mixed = False
    for k, v in merged.items():
        if sample_shape is None:
            sample_shape = v.shape
        elif v.shape[-1] != sample_shape[-1]:
            mixed = True
            break
    print(f'\nTotal unique embeddings: {len(merged):,}')
    if sample_shape is not None:
        dim = sample_shape[-1]
        if mixed:
            print(f'WARNING: mixed last-dim embeddings detected '
                  f'(expected {dim})')
        else:
            # per-residue outputs will have different first-dim (L) per key
            print(f'Embedding dim (last axis): {dim}')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    if args.no_compress:
        print(f'\nSaving (uncompressed) to {args.output} ...')
        np.savez(args.output, **merged)
    else:
        print(f'\nSaving (compressed) to {args.output} ...')
        np.savez_compressed(args.output, **merged)

    verify = np.load(args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f'Verified: {len(verify.files):,} keys, {size_mb:,.0f} MB on disk')


if __name__ == '__main__':
    main()
