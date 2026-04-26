"""Compare the OLD klebsiella `create_canonical_split.py` output to our
new `cipher.data.splits.create_canonical_split` output, given the SAME
upstream `training_data.npz` and `label_encoders.json`.

If the two splits differ on which MD5s land in train/val/test, we've
found the first concrete divergence between old and new pipelines that
could explain why our ciPHer port produces different trained models.

Usage:
    # After running scripts/analysis/reproduce_old_v3_training_local.sh,
    # the old canonical splits live at:
    #   <OLD_REPO>/output/local_test_v3/all_glycan_binders/splits.json
    # and the upstream training data at:
    #   <OLD_REPO>/output/local_test_v3/all_glycan_binders/training_data.npz
    #
    python scripts/analysis/diff_canonical_splits.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Make cipher importable from anywhere on this laptop
def _ensure_cipher_on_path():
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        candidate = parent / 'src' / 'cipher'
        if candidate.is_dir():
            src = str(parent / 'src')
            if src not in sys.path:
                sys.path.insert(0, src)
            return

_ensure_cipher_on_path()

import numpy as np

from cipher.data.splits import create_canonical_split


def load_old_splits_and_data(old_repo, run_dir='output/local_test_v3/all_glycan_binders'):
    base = Path(old_repo) / run_dir
    splits_path = base / 'splits.json'
    npz_path = base / 'training_data.npz'
    enc_path = base / 'label_encoders.json'
    for p in (splits_path, npz_path, enc_path):
        if not p.exists():
            sys.exit(f'ERROR: required file not found: {p}')
    with open(splits_path) as f:
        old_splits = json.load(f)
    npz = np.load(npz_path, allow_pickle=True)
    md5_list = npz['md5_list'].tolist()
    k_labels = npz['k_labels']
    o_labels = npz['o_labels']
    with open(enc_path) as f:
        enc = json.load(f)
    return old_splits, md5_list, k_labels, o_labels, enc['k_classes'], enc['o_classes']


def compare(old_splits, new_splits):
    print(f'{"split":<8} {"old size":>10} {"new size":>10} {"in both":>10} '
          f'{"only old":>10} {"only new":>10} {"jaccard":>8}')
    print('-' * 72)
    for s in ('train', 'val', 'test'):
        o = set(old_splits[s])
        n = set(new_splits[s])
        intersect = o & n
        only_o = o - n
        only_n = n - o
        union = o | n
        jacc = len(intersect) / max(len(union), 1)
        print(f'{s:<8} {len(o):>10} {len(n):>10} {len(intersect):>10} '
              f'{len(only_o):>10} {len(only_n):>10} {jacc:>8.4f}')

    # Bigger picture: do MD5s land in the SAME bucket in both pipelines?
    # Build md5 -> bucket maps and crosstab.
    md5_to_old = {}
    md5_to_new = {}
    for s in ('train', 'val', 'test'):
        for m in old_splits[s]:
            md5_to_old[m] = s
        for m in new_splits[s]:
            md5_to_new[m] = s

    all_md5s = set(md5_to_old) | set(md5_to_new)
    bucket_pairs = {}
    for m in all_md5s:
        o = md5_to_old.get(m, '_missing_')
        n = md5_to_new.get(m, '_missing_')
        bucket_pairs[(o, n)] = bucket_pairs.get((o, n), 0) + 1

    print()
    print('Crosstab (rows = old bucket, cols = new bucket):')
    buckets = ['train', 'val', 'test', '_missing_']
    print(f'{"":>10}' + ''.join(f'{b:>10}' for b in buckets))
    for o in buckets:
        row = [f'{bucket_pairs.get((o, n), 0):>10}' for n in buckets]
        print(f'{o:>10}' + ''.join(row))

    n_same = sum(c for (o, n), c in bucket_pairs.items() if o == n)
    n_total = sum(bucket_pairs.values())
    n_swapped = n_total - n_same
    print()
    print(f'  MD5s with matching bucket: {n_same:,} / {n_total:,} '
          f'({100 * n_same / n_total:.2f}%)')
    print(f'  MD5s in different buckets: {n_swapped:,}')

    return n_same == n_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--old-repo',
                   default='/Users/leannmlindsey/WORK/PHI_TSP/phi_tsp/klebsiella')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-ratio', type=float, default=0.7)
    p.add_argument('--val-ratio', type=float, default=0.15)
    args = p.parse_args()

    print(f'Loading old splits + upstream data from {args.old_repo} ...')
    old_splits, md5_list, k_labels, o_labels, k_classes, o_classes = \
        load_old_splits_and_data(args.old_repo)

    print(f'  MD5s in upstream data: {len(md5_list):,}')
    print(f'  K classes: {len(k_classes)}')
    print(f'  O classes: {len(o_classes)}')
    print(f'  Old splits: train={len(old_splits["train"])}, '
          f'val={len(old_splits["val"])}, test={len(old_splits["test"])}')

    print()
    print(f'Running cipher.data.splits.create_canonical_split (seed={args.seed}) ...')
    new_splits = create_canonical_split(
        md5_list=md5_list,
        k_labels=k_labels,
        o_labels=o_labels,
        k_classes=k_classes,
        o_classes=o_classes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f'  New splits: train={len(new_splits["train"])}, '
          f'val={len(new_splits["val"])}, test={len(new_splits["test"])}')

    print()
    print('=' * 72)
    print('SPLIT COMPARISON')
    print('=' * 72)
    matched = compare(old_splits, new_splits)
    print()
    if matched:
        print('VERDICT: splits match exactly. Bug is downstream of split logic.')
    else:
        print('VERDICT: splits DIFFER. This is one source of the model divergence.')
        print('         Look at create_canonical_split — RNG path, dict iteration')
        print('         order, cluster ordering, or the cluster-key normalization.')


if __name__ == '__main__':
    main()
