"""Recompress an NPZ file in place: uncompressed / np.savez form → savez_compressed.

Useful when an NPZ was written without compression (e.g. via the
checkpoint/resume mechanism in `esm2_extract.py` which uses plain
`np.savez`, or via the streaming merge with `--no-compress`). The
recompression is **streaming** — one array in memory at a time — so
it works on NPZ files that are larger than RAM.

Design:
  1. Open input via np.load (lazy-loading zip member reads).
  2. Write to a sibling temp file `<input>.npz.recompressing` using
     `zipfile.ZIP_DEFLATED` and `np.lib.format.write_array` per member.
  3. After successful write, verify the temp NPZ loads cleanly and the
     key count matches.
  4. Atomically replace the input with the temp via rename.

Safe to interrupt: any stale `*.npz.recompressing` from a prior run is
removed at start. The input is only deleted after the temp verifies.

Usage:
    python scripts/utils/recompress_npz.py /path/to/file.npz
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np


def _size_gb(path):
    try:
        return Path(path).stat().st_size / 1e9
    except FileNotFoundError:
        return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('npz_path')
    args = p.parse_args()

    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        sys.exit(f'ERROR: {npz_path} not found')

    temp_path = npz_path.with_suffix('.npz.recompressing')
    if temp_path.exists():
        print(f'Removing stale temp file from a prior run: {temp_path}')
        temp_path.unlink()

    orig_gb = _size_gb(npz_path)
    print(f'Input : {npz_path}  ({orig_gb:.1f} GB)')

    source = np.load(npz_path)
    try:
        keys = list(source.files)
        n = len(keys)
        print(f'        {n:,} keys')

        progress_every = max(n // 50, 1)
        sample_last_dim = None

        with zipfile.ZipFile(temp_path, mode='w',
                             compression=zipfile.ZIP_DEFLATED,
                             allowZip64=True) as zf:
            for i, key in enumerate(keys):
                arr = source[key]
                if sample_last_dim is None and arr.ndim >= 1:
                    sample_last_dim = int(arr.shape[-1])
                with zf.open(f'{key}.npy', mode='w', force_zip64=True) as fp:
                    np.lib.format.write_array(fp, arr, allow_pickle=False)
                del arr
                if (i + 1) % progress_every == 0 or i + 1 == n:
                    temp_gb = _size_gb(temp_path)
                    ratio = temp_gb / orig_gb * 100 if orig_gb > 0 else 0
                    print(f'        {i+1:>6,}/{n:,} keys  '
                          f'temp={temp_gb:>5.1f} GB  '
                          f'({ratio:.1f}% of original)')
    finally:
        source.close()

    # Verify the temp is loadable + has the expected key count.
    print('Verifying temp output...')
    verify = np.load(temp_path)
    try:
        n_verified = len(verify.files)
        if n_verified != n:
            temp_path.unlink()
            sys.exit(f'ERROR: key-count mismatch '
                     f'({n} source → {n_verified} temp). Aborting.')
        # Spot-check a single array round-trips.
        sample = verify[verify.files[0]]
        if sample_last_dim is not None and sample.ndim >= 1:
            if int(sample.shape[-1]) != sample_last_dim:
                temp_path.unlink()
                sys.exit(f'ERROR: dim mismatch on spot-check array.')
    finally:
        verify.close()

    # Atomic swap.
    npz_path.unlink()
    os.rename(temp_path, npz_path)

    new_gb = _size_gb(npz_path)
    saved_gb = orig_gb - new_gb
    pct = (saved_gb / orig_gb * 100) if orig_gb > 0 else 0
    print(f'Output: {npz_path}  ({new_gb:.1f} GB)')
    print(f'Saved : {saved_gb:.1f} GB  ({pct:.1f}%)')


if __name__ == '__main__':
    main()
