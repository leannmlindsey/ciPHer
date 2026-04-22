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


if __name__ == '__main__':
    main()
