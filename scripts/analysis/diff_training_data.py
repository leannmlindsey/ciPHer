"""Compare the OLD klebsiella `training_data.npz` to the one OUR
ciPHer `prepare_training_data` would produce given the equivalent
config.

If md5_list, k_labels, o_labels, k_classes, or o_classes differ, our
pipeline is producing different training data than the old one — which
explains why the trained models diverge even when splits + arch match.

Inputs (assumed already present from
scripts/analysis/reproduce_old_v3_training_local.sh):
    /Users/leannmlindsey/WORK/PHI_TSP/phi_tsp/klebsiella/output/local_test_v3/
        all_glycan_binders/training_data.npz
        all_glycan_binders/label_encoders.json

OUR pipeline runs against:
    /Users/leannmlindsey/WORK/PHI_TSP/cipher/data/training_data/metadata/
        host_phage_protein_map.tsv
        glycan_binders_custom.tsv
        pipeline_positive.list

Usage:
    python scripts/analysis/diff_training_data.py
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Make cipher importable
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

from cipher.data.training import TrainingConfig, prepare_training_data


def load_old(old_npz, old_enc):
    npz = np.load(old_npz, allow_pickle=True)
    md5_list = npz['md5_list'].tolist()
    k_labels = npz['k_labels']
    o_labels = npz['o_labels']
    with open(old_enc) as f:
        enc = json.load(f)
    return md5_list, k_labels, o_labels, enc['k_classes'], enc['o_classes']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cipher-repo',
                   default='/Users/leannmlindsey/WORK/PHI_TSP/cipher')
    p.add_argument('--old-repo',
                   default='/Users/leannmlindsey/WORK/PHI_TSP/phi_tsp/klebsiella')
    args = p.parse_args()

    cipher = Path(args.cipher_repo)
    old = Path(args.old_repo)

    # Old training_data.npz from the bit-exact retrain
    old_npz = old / 'output/local_test_v3/all_glycan_binders/training_data.npz'
    old_enc = old / 'output/local_test_v3/all_glycan_binders/label_encoders.json'
    if not old_npz.exists() or not old_enc.exists():
        sys.exit(f'ERROR: missing old training data at {old_npz} or {old_enc}.\n'
                 f'Run scripts/analysis/reproduce_old_v3_training_local.sh first.')

    print(f'Loading OLD training_data from {old_npz} ...')
    o_md5, o_k, o_o, o_kc, o_oc = load_old(old_npz, old_enc)
    print(f'  OLD: {len(o_md5):,} MD5s, {len(o_kc)} K classes, {len(o_oc)} O classes')

    # Run OUR prepare_training_data with the equivalent config.
    # Old config_local_v3.yaml settings:
    #   protein_set: all_glycan_binders     (no tool filter — null tools)
    #   positive_list: pipeline_positive
    #   min_sources: 3
    #   max_k_types: 3
    #   max_o_types: 3
    #   single_label: true
    #   keep_null_classes: true              (default in our pipeline)
    cfg_kwargs = dict(
        positive_list_path=str(cipher / 'data/training_data/metadata/pipeline_positive.list'),
        min_sources=3,
        max_k_types=3,
        max_o_types=3,
        single_label=True,
        label_strategy='single_label',
    )
    print()
    print('Running OUR prepare_training_data with equivalent config ...')
    our_cfg = TrainingConfig(**cfg_kwargs)
    td = prepare_training_data(
        our_cfg,
        str(cipher / 'data/training_data/metadata/host_phage_protein_map.tsv'),
        str(cipher / 'data/training_data/metadata/glycan_binders_custom.tsv'),
        verbose=False,
    )
    n_md5, n_k, n_o, n_kc, n_oc = td.md5_list, td.k_labels, td.o_labels, td.k_classes, td.o_classes
    print(f'  NEW: {len(n_md5):,} MD5s, {len(n_kc)} K classes, {len(n_oc)} O classes')

    print()
    print('=' * 72)
    print('COMPARISON')
    print('=' * 72)

    # Class-list comparison
    o_kc_set = set(o_kc)
    n_kc_set = set(n_kc)
    o_oc_set = set(o_oc)
    n_oc_set = set(n_oc)

    print(f'K classes:  old={len(o_kc)}, new={len(n_kc)}, '
          f'in_both={len(o_kc_set & n_kc_set)}, only_old={len(o_kc_set - n_kc_set)}, '
          f'only_new={len(n_kc_set - o_kc_set)}')
    if o_kc_set != n_kc_set:
        print(f'  Only in OLD K: {sorted(o_kc_set - n_kc_set)[:10]}')
        print(f'  Only in NEW K: {sorted(n_kc_set - o_kc_set)[:10]}')

    print(f'O classes:  old={len(o_oc)}, new={len(n_oc)}, '
          f'in_both={len(o_oc_set & n_oc_set)}, only_old={len(o_oc_set - n_oc_set)}, '
          f'only_new={len(n_oc_set - o_oc_set)}')
    if o_oc_set != n_oc_set:
        print(f'  Only in OLD O: {sorted(o_oc_set - n_oc_set)[:10]}')
        print(f'  Only in NEW O: {sorted(n_oc_set - o_oc_set)[:10]}')

    # Class ordering
    if o_kc == n_kc:
        print('  K class ORDERING matches.')
    else:
        # find first diff
        for i, (a, b) in enumerate(zip(o_kc, n_kc)):
            if a != b:
                print(f'  K class ordering DIFFERS at index {i}: old="{a}", new="{b}"')
                break
        else:
            print(f'  K class ordering DIFFERS (length differs).')

    if o_oc == n_oc:
        print('  O class ORDERING matches.')
    else:
        for i, (a, b) in enumerate(zip(o_oc, n_oc)):
            if a != b:
                print(f'  O class ordering DIFFERS at index {i}: old="{a}", new="{b}"')
                break

    # MD5 list comparison
    o_md5_set = set(o_md5)
    n_md5_set = set(n_md5)
    print(f'MD5 set:    old={len(o_md5):,}, new={len(n_md5):,}, '
          f'in_both={len(o_md5_set & n_md5_set):,}, '
          f'only_old={len(o_md5_set - n_md5_set):,}, '
          f'only_new={len(n_md5_set - o_md5_set):,}')

    if o_md5 == n_md5:
        print('  MD5 list ORDERING matches.')
    else:
        if o_md5_set == n_md5_set:
            print('  MD5 sets match but ORDERING differs (sort vs insertion order)')
        else:
            print('  MD5 sets DIFFER — pipelines are filtering proteins differently.')

    # Label-vector comparison: only meaningful if both md5_list and class
    # lists match. Build aligned arrays via dict lookup on MD5.
    common = sorted(o_md5_set & n_md5_set)
    if o_kc_set == n_kc_set and o_oc_set == n_oc_set and common:
        # Re-index labels by MD5 for both old and new
        # We need the column ORDER to match too — re-arrange new labels
        # to match old's class ordering for a fair comparison.
        n_k_to_idx = {c: i for i, c in enumerate(n_kc)}
        n_o_to_idx = {c: i for i, c in enumerate(n_oc)}
        o_md5_to_idx = {m: i for i, m in enumerate(o_md5)}
        n_md5_to_idx = {m: i for i, m in enumerate(n_md5)}

        n_k_diff = 0
        n_o_diff = 0
        for md5 in common:
            o_kvec = o_k[o_md5_to_idx[md5]]
            o_ovec = o_o[o_md5_to_idx[md5]]
            n_kvec_raw = n_k[n_md5_to_idx[md5]]
            n_ovec_raw = n_o[n_md5_to_idx[md5]]
            # Re-order new vectors into old's class order
            n_kvec = np.array([n_kvec_raw[n_k_to_idx[c]] for c in o_kc])
            n_ovec = np.array([n_ovec_raw[n_o_to_idx[c]] for c in o_oc])
            if not np.array_equal(o_kvec, n_kvec):
                n_k_diff += 1
            if not np.array_equal(o_ovec, n_ovec):
                n_o_diff += 1
        print(f'Label vectors (over {len(common):,} common MD5s):')
        print(f'  K labels differ on: {n_k_diff:,} MD5s ({100*n_k_diff/len(common):.1f}%)')
        print(f'  O labels differ on: {n_o_diff:,} MD5s ({100*n_o_diff/len(common):.1f}%)')

        # Show one example of a label disagreement
        if n_k_diff or n_o_diff:
            for md5 in common:
                o_kvec = o_k[o_md5_to_idx[md5]]
                n_kvec_raw = n_k[n_md5_to_idx[md5]]
                n_kvec = np.array([n_kvec_raw[n_k_to_idx[c]] for c in o_kc])
                if not np.array_equal(o_kvec, n_kvec):
                    o_pos = [o_kc[i] for i in np.where(o_kvec > 0)[0]]
                    n_pos = [o_kc[i] for i in np.where(n_kvec > 0)[0]]
                    print(f'    Example K diff (md5={md5[:12]}...): '
                          f'old positives={o_pos}, new positives={n_pos}')
                    break

    print()
    print('=' * 72)
    if (o_md5 == n_md5
            and o_kc == n_kc and o_oc == n_oc
            and np.array_equal(o_k, n_k) and np.array_equal(o_o, n_o)):
        print('VERDICT: training data is BIT-EXACT identical. Bug is in training '
              'loop or model arch.')
    else:
        print('VERDICT: training data DIFFERS. Found the divergence point.')
        print('         Look at the specific differences above to localize the bug.')


if __name__ == '__main__':
    main()
