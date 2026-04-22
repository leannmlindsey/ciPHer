"""Audit a length-binned ESM-2/ProtT5 extraction: for every split_fasta/
FASTA, report whether the corresponding split_embeddings/ NPZ exists and
whether the protein counts match.

Defaults match the layout produced by
`klebsiella/scripts/data_prep/embed_by_length.sh` and its ProtT5
equivalent — split FASTAs under `<root>/split_fasta/`, split NPZs under
`<root>/split_embeddings/`, one merged NPZ at `<root>/candidates_...md5.npz`.

Usage:
    python scripts/utils/audit_split_extraction.py
    python scripts/utils/audit_split_extraction.py \\
        --root /work/hdd/bfzj/llindsey1/embeddings_full
"""

import argparse
import hashlib
import os
import sys

import numpy as np


def count_fasta(path):
    n = 0
    ids = []
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                n += 1
                if len(ids) < 3:
                    ids.append(line[1:].strip().split()[0])
    return n, ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',
                   default='/work/hdd/bfzj/llindsey1/embeddings_full')
    p.add_argument('--fasta-glob', default='candidates_maxlen')
    args = p.parse_args()

    fasta_dir = os.path.join(args.root, 'split_fasta')
    emb_dir = os.path.join(args.root, 'split_embeddings')

    if not os.path.isdir(fasta_dir):
        sys.exit(f'ERROR: {fasta_dir} does not exist')
    if not os.path.isdir(emb_dir):
        sys.exit(f'ERROR: {emb_dir} does not exist')

    # Collect all FASTA bins
    fastas = sorted([f for f in os.listdir(fasta_dir)
                     if f.startswith(args.fasta_glob) and f.endswith('.faa')])
    if not fastas:
        sys.exit(f'ERROR: no FASTAs matching {args.fasta_glob}* in {fasta_dir}')

    print(f'{"FASTA bin":<40} {"proteins":>10}  {"NPZ":<40} {"keys":>10}  status')
    print('-' * 120)

    fasta_total = 0
    npz_total = 0
    missing = []
    for f in fastas:
        fasta_path = os.path.join(fasta_dir, f)
        bin_label = f.replace('.faa', '')       # candidates_maxlen1024
        npz_name = f'{bin_label}_embeddings.npz'
        npz_path = os.path.join(emb_dir, npz_name)

        n_proteins, _ = count_fasta(fasta_path)
        fasta_total += n_proteins

        if os.path.exists(npz_path):
            try:
                n_keys = len(np.load(npz_path).files)
                status = 'OK' if n_keys == n_proteins else f'COUNT MISMATCH ({n_proteins - n_keys} missing)'
            except Exception as e:
                n_keys = 0
                status = f'CORRUPT ({type(e).__name__})'
            npz_total += n_keys
        else:
            n_keys = 0
            status = 'MISSING'
            missing.append(bin_label)

        print(f'{f:<40} {n_proteins:>10,}  {npz_name:<40} {n_keys:>10,}  {status}')

    print('-' * 120)
    print(f'{"TOTAL":<40} {fasta_total:>10,}  {"":<40} {npz_total:>10,}')
    print()
    print(f'Coverage: {npz_total:,} / {fasta_total:,} proteins ({100*npz_total/fasta_total:.1f}%)')

    # Check merged file
    merged_candidates = [f for f in os.listdir(args.root)
                         if f.endswith('.npz') and 'md5' in f]
    if merged_candidates:
        for name in merged_candidates:
            path = os.path.join(args.root, name)
            try:
                n_merged = len(np.load(path).files)
                print(f'Merged NPZ: {name} -> {n_merged:,} keys')
            except Exception as e:
                print(f'Merged NPZ {name}: FAILED to load ({type(e).__name__})')

    if missing:
        print()
        print('=== Missing split NPZs ===')
        for m in missing:
            print(f'  {m}_embeddings.npz  (FASTA: {m}.faa)')
        print()
        print('To re-run only the missing bins, submit the original extraction')
        print('script with those FASTAs as input, then re-merge.')


if __name__ == '__main__':
    main()
