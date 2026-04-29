"""Identify proteins in a FASTA whose MD5 is absent from an NPZ.

Use case: a length-binned extraction that ran to completion but hit
disk quota before the final save committed all proteins. The on-disk
NPZ contains the last-checkpointed subset; the FASTA is the full bin
input. This script finds which FASTA proteins aren't represented in
the NPZ and writes a small sub-FASTA of just those, ready to feed
back into the extractor for a gap fill.

MD5 is computed the same way the extractors do: hashlib.md5 over the
UTF-8-encoded amino-acid sequence string (no spaces, no header). NPZ
keys are assumed to be MD5 hex digests when `--key-by-md5` was used
during extraction.

Usage:
    python scripts/utils/find_missing_md5s.py \\
        --fasta <input.faa> \\
        --npz   <existing_output.npz> \\
        --out   <gap.faa>

    # Optional: also emit a text list of missing MD5s
    python scripts/utils/find_missing_md5s.py \\
        --fasta x.faa --npz x.npz --out gap.faa \\
        --md5-list-out gap_md5s.txt
"""

import argparse
import hashlib
import os
import sys

import numpy as np


def parse_fasta(path):
    """Yield (protein_id, sequence) tuples from a FASTA file."""
    pid = None
    seq_parts = []
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                if pid is not None:
                    yield pid, ''.join(seq_parts)
                pid = line[1:].strip().split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.strip())
    if pid is not None:
        yield pid, ''.join(seq_parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fasta', required=True,
                   help='Input FASTA (full bin, all proteins).')
    p.add_argument('--npz', required=True,
                   help='Existing NPZ, keyed by MD5, that is missing some '
                        'proteins from the FASTA.')
    p.add_argument('--out', required=True,
                   help='Output sub-FASTA of proteins whose MD5 is not a '
                        'key in the NPZ.')
    p.add_argument('--md5-list-out', default=None,
                   help='Optional: also write the missing MD5s as a '
                        'one-per-line text file at this path.')
    args = p.parse_args()

    if not os.path.exists(args.fasta):
        sys.exit(f'ERROR: FASTA not found: {args.fasta}')
    if not os.path.exists(args.npz):
        sys.exit(f'ERROR: NPZ not found: {args.npz}')

    # Load existing NPZ keys. np.load is lazy so this is cheap even for
    # a 200 GB file — we only touch the central directory.
    print(f'Reading NPZ keys from {args.npz} ...')
    with np.load(args.npz) as npz:
        present_md5s = set(npz.files)
    print(f'  {len(present_md5s):,} keys present')

    # Walk the FASTA, hash each sequence, compare.
    print(f'Scanning FASTA {args.fasta} ...')
    n_total = 0
    n_present = 0
    missing = []   # list of (pid, md5, seq)
    seen_md5s = set()
    for pid, seq in parse_fasta(args.fasta):
        n_total += 1
        md5 = hashlib.md5(seq.encode()).hexdigest()
        if md5 in present_md5s:
            n_present += 1
        else:
            if md5 in seen_md5s:
                # Sequence-level duplicate within the FASTA; only one copy
                # needs to be re-embedded. Skip subsequent instances.
                continue
            seen_md5s.add(md5)
            missing.append((pid, md5, seq))

    n_missing = len(missing)
    print(f'  FASTA total: {n_total:,} sequences')
    print(f'  in NPZ:      {n_present:,}')
    print(f'  missing:     {n_missing:,} unique MD5s')

    # Write sub-FASTA for the missing proteins.
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        for pid, _md5, seq in missing:
            f.write(f'>{pid}\n{seq}\n')
    print(f'Wrote {args.out} ({n_missing:,} sequences)')

    if args.md5_list_out:
        os.makedirs(os.path.dirname(args.md5_list_out) or '.', exist_ok=True)
        with open(args.md5_list_out, 'w') as f:
            for _pid, md5, _seq in missing:
                f.write(md5 + '\n')
        print(f'Wrote {args.md5_list_out} ({n_missing:,} MD5s)')


if __name__ == '__main__':
    main()
