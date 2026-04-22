"""Merge multiple per-length-bin NPZ embedding files into a single NPZ.

Reads all *.npz in a directory (or a list of explicit paths) and writes a
single NPZ at the output path. Deduplicates by key (first-write wins).

Since 2026-04-22 the merge is **streaming**: each array is read from its
source NPZ and written directly into the output zip, one at a time.
Peak memory is O(largest single array), not O(total merged data) — this
matters for the ESM-2 650M full merge where the in-memory dict approach
would need ~250 GB of RAM. The output file format is unchanged: standard
NPZ consumable by `np.load(...)`.

Dimension consistency is still checked on-the-fly (useful warning signal
for per-residue runs that accidentally mix models), and mixed last-dims
are warned about at the end.

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
import zipfile
from pathlib import Path

import numpy as np


def iter_input_npzs(inputs):
    for path in inputs:
        if os.path.isdir(path):
            for p in sorted(Path(path).glob('*.npz')):
                yield p
        elif os.path.isfile(path) and path.endswith('.npz'):
            yield Path(path)
        else:
            print(f'WARNING: skipping unrecognized input {path}', file=sys.stderr)


def stream_merge(npz_files, output_path, compress=True):
    """Stream-copy every key from each input NPZ into a single output NPZ.

    Only one array is held in memory at a time. Returns (n_unique, n_duplicates,
    sample_last_dim, mixed).
    """
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    seen_keys: set = set()
    duplicates = 0
    sample_last_dim = None
    mixed = False
    n_unique = 0

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # allowZip64 = True so we can exceed the 4 GB single-file limit.
    with zipfile.ZipFile(output_path, mode='w',
                         compression=compression,
                         allowZip64=True) as zout:
        for npz_path in npz_files:
            data = np.load(npz_path)
            n_keys = len(data.files)
            n_new = 0
            for key in data.files:
                if key in seen_keys:
                    duplicates += 1
                    continue
                seen_keys.add(key)
                arr = data[key]

                if sample_last_dim is None and arr.ndim >= 1:
                    sample_last_dim = int(arr.shape[-1])
                elif arr.ndim >= 1 and int(arr.shape[-1]) != sample_last_dim:
                    mixed = True

                # Write this single array as a .npy member inside the output
                # zip. force_zip64 protects against the rare per-protein
                # array that exceeds 4 GB (shouldn't happen at sane
                # embedding sizes, but belt-and-braces).
                member_name = f'{key}.npy'
                with zout.open(member_name, mode='w', force_zip64=True) as fp:
                    np.lib.format.write_array(fp, arr, allow_pickle=False)

                n_new += 1
                n_unique += 1

                # Release this array back to the OS before the next one.
                del arr

            # Release the whole bin before moving on.
            data.close()
            size_mb = npz_path.stat().st_size / 1e6
            print(f'  {npz_path.name}: {n_keys:>7,} keys, {n_new:>7,} new '
                  f'({size_mb:>7,.0f} MB on disk)')

    return n_unique, duplicates, sample_last_dim, mixed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', nargs='+', required=True,
                   help='Input NPZ files, or one or more directories '
                        'containing NPZ files.')
    p.add_argument('-o', '--output', required=True,
                   help='Output merged NPZ path.')
    p.add_argument('--no-compress', action='store_true',
                   help='Use uncompressed NPZ (zipfile ZIP_STORED) instead of '
                        'ZIP_DEFLATED. Faster to write and slightly lower peak '
                        'memory; larger on disk. Useful for very large '
                        'per-residue outputs where compression ratio is low.')
    args = p.parse_args()

    npz_files = list(iter_input_npzs(args.input))
    if not npz_files:
        sys.exit('ERROR: no NPZ files found')

    print(f'Stream-merging {len(npz_files)} NPZ files '
          f'(compressed={not args.no_compress}) ...')

    n_unique, duplicates, dim, mixed = stream_merge(
        npz_files, args.output, compress=not args.no_compress)

    if duplicates:
        print(f'\nDuplicates skipped: {duplicates:,}')
    print(f'\nTotal unique embeddings: {n_unique:,}')
    if dim is not None:
        if mixed:
            print(f'WARNING: mixed last-dim embeddings detected '
                  f'(first observed {dim})')
        else:
            print(f'Embedding dim (last axis): {dim}')

    verify = np.load(args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f'Verified: {len(verify.files):,} keys, {size_mb:,.0f} MB on disk')


if __name__ == '__main__':
    main()
