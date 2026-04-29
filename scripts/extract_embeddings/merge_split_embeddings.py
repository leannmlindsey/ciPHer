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


def _stream_one_split(npz_path, zout, state):
    """Stream every key from one input NPZ into an open output zip.

    `state` is a mutable dict carrying seen_keys / duplicates /
    sample_last_dim / mixed / n_unique across splits. Mutates in place.
    """
    data = np.load(npz_path)
    n_keys = len(data.files)
    n_new = 0
    for key in data.files:
        if key in state['seen_keys']:
            state['duplicates'] += 1
            continue
        state['seen_keys'].add(key)
        arr = data[key]

        if state['sample_last_dim'] is None and arr.ndim >= 1:
            state['sample_last_dim'] = int(arr.shape[-1])
        elif arr.ndim >= 1 and int(arr.shape[-1]) != state['sample_last_dim']:
            state['mixed'] = True

        # force_zip64 protects against the rare per-protein array that
        # exceeds 4 GB.
        member_name = f'{key}.npy'
        with zout.open(member_name, mode='w', force_zip64=True) as fp:
            np.lib.format.write_array(fp, arr, allow_pickle=False)

        n_new += 1
        state['n_unique'] += 1

        # Release this array back to the OS before the next one.
        del arr

    data.close()
    return n_keys, n_new


def stream_merge(npz_files, output_path, compress=True, delete_inputs=False):
    """Stream-copy every key from each input NPZ into a single output NPZ.

    Only one array is held in memory at a time. Returns (n_unique, n_duplicates,
    sample_last_dim, mixed).

    When `delete_inputs=True`:
      - Each split's unique arrays are streamed into the output.
      - The zipfile is closed-and-reopened-in-append-mode per split, so
        the output's central directory is committed to disk after each
        split completes.
      - The input split is unlinked only AFTER the output has committed
        that split's data. Peak disk stays near baseline (splits shrink
        as output grows). If the process crashes mid-merge, the output
        is still a valid partial NPZ with every fully-processed split;
        any already-deleted splits are gone but their data is safe in
        the output.

    When `delete_inputs=False`:
      - Existing behaviour: one zipfile open() for the whole merge,
        slightly faster, but all-or-nothing. Inputs are never touched.
    """
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    state = {
        'seen_keys': set(),
        'duplicates': 0,
        'sample_last_dim': None,
        'mixed': False,
        'n_unique': 0,
    }

    if delete_inputs:
        # Fresh start — remove any prior output so we don't append to it.
        if os.path.exists(output_path):
            os.unlink(output_path)
        for i, npz_path in enumerate(npz_files):
            mode = 'w' if i == 0 else 'a'
            with zipfile.ZipFile(output_path, mode=mode,
                                 compression=compression,
                                 allowZip64=True) as zout:
                size_mb_before = npz_path.stat().st_size / 1e6
                n_keys, n_new = _stream_one_split(npz_path, zout, state)
                print(f'  {npz_path.name}: {n_keys:>7,} keys, {n_new:>7,} new '
                      f'({size_mb_before:>7,.0f} MB on disk)')
            # zipfile.close() above has committed the central directory.
            # It's now safe to delete the input — the output contains
            # every key from it.
            npz_path.unlink()
            print(f'    (deleted {npz_path.name})')
    else:
        with zipfile.ZipFile(output_path, mode='w',
                             compression=compression,
                             allowZip64=True) as zout:
            for npz_path in npz_files:
                size_mb_before = npz_path.stat().st_size / 1e6
                n_keys, n_new = _stream_one_split(npz_path, zout, state)
                print(f'  {npz_path.name}: {n_keys:>7,} keys, {n_new:>7,} new '
                      f'({size_mb_before:>7,.0f} MB on disk)')

    return state['n_unique'], state['duplicates'], state['sample_last_dim'], state['mixed']


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
    p.add_argument('--delete-inputs', action='store_true',
                   help='Unlink each input NPZ after its arrays have been '
                        'committed to the output. Use when disk is tight — '
                        'peak disk stays near baseline because splits shrink '
                        'as output grows. The output is committed per-split '
                        '(close-and-reopen-append) so a mid-merge crash '
                        'leaves a valid partial NPZ with completed splits. '
                        'IRREVERSIBLE for deleted inputs if the process '
                        'dies after their data is committed but before '
                        'the next split completes.')
    args = p.parse_args()

    npz_files = list(iter_input_npzs(args.input))
    if not npz_files:
        sys.exit('ERROR: no NPZ files found')

    print(f'Stream-merging {len(npz_files)} NPZ files '
          f'(compressed={not args.no_compress}, '
          f'delete_inputs={args.delete_inputs}) ...')

    n_unique, duplicates, dim, mixed = stream_merge(
        npz_files, args.output,
        compress=not args.no_compress,
        delete_inputs=args.delete_inputs)

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
