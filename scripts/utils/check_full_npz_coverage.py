"""Check how much of each training-filter set is covered by an existing
per-residue NPZ file. Used to decide whether a partial full-extraction
can be reused or needs to be re-run.

Defaults match the Delta-AI layout and the canonical positive/high-conf
lists under data/training_data/metadata/. Override any path via flags
if running from a different checkout.

Usage (from the ciPHer repo root, on Delta or any machine with the
canonical files visible):
    python scripts/utils/check_full_npz_coverage.py
    python scripts/utils/check_full_npz_coverage.py \\
        --npz /work/hdd/bfzj/llindsey1/embeddings_full/candidates_embeddings_full_md5.npz
"""

import argparse
import hashlib
import os
import random
import statistics
import sys

import numpy as np


def pid_to_md5_from_fasta(fasta_path):
    pid_md5 = {}
    cur_pid = None
    cur_seq = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                if cur_pid:
                    pid_md5[cur_pid] = hashlib.md5(''.join(cur_seq).encode()).hexdigest()
                cur_pid = line[1:].strip().split()[0]
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_pid:
            pid_md5[cur_pid] = hashlib.md5(''.join(cur_seq).encode()).hexdigest()
    return pid_md5


def pids_to_md5s(list_path, pid_md5):
    if not os.path.exists(list_path):
        return None
    with open(list_path) as f:
        pids = {l.strip() for l in f if l.strip()}
    return {pid_md5[p] for p in pids if p in pid_md5}


def coverage_report(name, target_md5s, npz_keys):
    if target_md5s is None:
        print(f'{name}: list file not found, skipped')
        return
    n_total = len(target_md5s)
    n_in = len(npz_keys & target_md5s)
    n_out = n_total - n_in
    pct = 100 * n_in / n_total if n_total else 0
    print(f'{name}: {n_total:,} MD5s')
    print(f'    in NPZ: {n_in:,} ({pct:.1f}%)')
    print(f'    missing: {n_out:,}')


def sanity_report(npz_path, expected_dim=None, sample_size=200, seed=42):
    """Dim / NaN / shape sanity on a random sample of keys.

    Reads the NPZ lazily (np.load uses mmap under the hood), so even a
    100+ GB merged per-residue NPZ can be sampled cheaply. Pass
    `expected_dim` to hard-assert that every sampled array has
    `shape[-1] == expected_dim` (e.g. 1024 for ProtT5-XL, 1280 for
    ESM-2 650M).
    """
    print(f'=== Sanity report (sample of {sample_size} keys, seed={seed}) ===')
    npz = np.load(npz_path)
    keys = npz.files
    if not keys:
        print('  NPZ contains zero keys; nothing to check')
        return False
    rng = random.Random(seed)
    sample = rng.sample(keys, min(sample_size, len(keys)))

    dims: set = set()
    Ls: list = []
    bad_keys: list = []
    for k in sample:
        arr = npz[k]
        if arr.ndim >= 1:
            dims.add(int(arr.shape[-1]))
        if arr.ndim >= 2:
            Ls.append(int(arr.shape[0]))
        if np.isnan(arr).any() or np.isinf(arr).any():
            bad_keys.append(k)

    print(f'  unique last-dims:   {sorted(dims)}')
    if Ls:
        print(f'  L (first-dim):      min={min(Ls)}  max={max(Ls)}  '
              f'median={int(statistics.median(Ls))}')
    else:
        print('  (per-protein arrays are 1D; no L/first-dim to report)')
    print(f'  keys with NaN/Inf:  {len(bad_keys)}')
    if bad_keys:
        print(f'    examples: {bad_keys[:5]}')

    ok = True
    if expected_dim is not None:
        if dims != {expected_dim}:
            print(f'  FAIL: expected last-dim = {expected_dim}, got {sorted(dims)}')
            ok = False
        else:
            print(f'  PASS: last-dim = {expected_dim} on every sampled key')
    if bad_keys:
        ok = False
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz',
                   default='/work/hdd/bfzj/llindsey1/embeddings_full/'
                           'candidates_embeddings_full_md5.npz')
    p.add_argument('--fasta',
                   default='data/training_data/metadata/candidates.faa')
    p.add_argument('--positive-list',
                   default='data/training_data/metadata/pipeline_positive.list')
    p.add_argument('--tsp-K-list',
                   default='data/training_data/metadata/highconf_tsp_K.list')
    p.add_argument('--pos-K-list',
                   default='data/training_data/metadata/highconf_pipeline_positive_K.list')
    p.add_argument('--expected-dim', type=int, default=None,
                   help='If set, assert every sampled array has '
                        'shape[-1] == EXPECTED_DIM (e.g. 1024 for '
                        'ProtT5-XL, 1280 for ESM-2 650M).')
    p.add_argument('--sample-size', type=int, default=200,
                   help='How many keys to sample for the dim/NaN check '
                        '(default: 200).')
    args = p.parse_args()

    if not os.path.exists(args.npz):
        sys.exit(f'ERROR: NPZ not found: {args.npz}')
    if not os.path.exists(args.fasta):
        sys.exit(f'ERROR: FASTA not found: {args.fasta}')

    print(f'Loading NPZ keys from {args.npz}')
    npz_keys = set(np.load(args.npz).files)
    print(f'  {len(npz_keys):,} keys in NPZ\n')

    print(f'Computing pid -> md5 from {args.fasta}')
    pid_md5 = pid_to_md5_from_fasta(args.fasta)
    all_md5s = set(pid_md5.values())
    print(f'  {len(pid_md5):,} protein_ids, {len(all_md5s):,} unique MD5s\n')

    # Coverage against each candidate list
    print('=== Coverage report ===\n')
    coverage_report('candidates.faa (all)', all_md5s, npz_keys)
    print()
    coverage_report('pipeline_positive.list',
                    pids_to_md5s(args.positive_list, pid_md5), npz_keys)
    print()
    coverage_report('highconf_tsp_K.list',
                    pids_to_md5s(args.tsp_K_list, pid_md5), npz_keys)
    print()
    coverage_report('highconf_pipeline_positive_K.list',
                    pids_to_md5s(args.pos_K_list, pid_md5), npz_keys)

    print()
    ok = sanity_report(args.npz,
                       expected_dim=args.expected_dim,
                       sample_size=args.sample_size)
    if args.expected_dim is not None and not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
