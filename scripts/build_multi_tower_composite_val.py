"""Build a composite validation NPZ for the multi_tower_attention_mlp model.

The canonical `per_head_strict_eval.py` loads ONE validation NPZ and
calls `predictor.predict_protein(emb_dict[md5])`. MultiTowerPredictor
expects a single CONCATENATED vector (Tower A || Tower B || Tower C),
so we pre-concat the three val NPZs into one composite NPZ before
running the standard eval.

Usage:
    python scripts/build_multi_tower_composite_val.py \\
        --val-emb-a path_a.npz \\
        --val-emb-b path_b.npz \\
        --val-emb-c path_c.npz \\
        --out path_composite.npz

The composite NPZ is keyed by MD5 (intersection of the three input
NPZs). Each value is a 1-D concatenated float32 vector of length
sum(input_dims). Reasonable: ~1500 val proteins x 170752-d float32
= ~1 GB on disk for ProtT5-XL seg8 (8192) + ESM-2 3B (2560) + kmer
aa20 k4 (160000).
"""

import argparse
import os
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--val-emb-a', required=True)
    p.add_argument('--val-emb-b', required=True)
    p.add_argument('--val-emb-c', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--force', action='store_true',
                   help='Overwrite existing output NPZ.')
    args = p.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f'Output already exists: {args.out}\n'
              f'Pass --force to overwrite.')
        return 0

    print(f'Loading three validation NPZs:')
    print(f'  A: {args.val_emb_a}')
    print(f'  B: {args.val_emb_b}')
    print(f'  C: {args.val_emb_c}')

    da = dict(np.load(args.val_emb_a))
    db = dict(np.load(args.val_emb_b))
    dc = dict(np.load(args.val_emb_c))

    print(f'  Loaded sizes: A={len(da)}  B={len(db)}  C={len(dc)}')

    common = set(da) & set(db) & set(dc)
    print(f'  Intersection: {len(common)} MD5s')
    if not common:
        sys.exit('ERROR: no common MD5s across the three NPZs')

    # Sniff dims
    first = next(iter(common))
    da_d = np.asarray(da[first]).reshape(-1).shape[0]
    db_d = np.asarray(db[first]).reshape(-1).shape[0]
    dc_d = np.asarray(dc[first]).reshape(-1).shape[0]
    total = da_d + db_d + dc_d
    print(f'  Dims: A={da_d}  B={db_d}  C={dc_d}  composite={total}')

    print(f'Building composite vectors...')
    composite = {}
    for md5 in common:
        va = np.asarray(da[md5], dtype=np.float32).reshape(-1)
        vb = np.asarray(db[md5], dtype=np.float32).reshape(-1)
        vc = np.asarray(dc[md5], dtype=np.float32).reshape(-1)
        if va.shape[0] != da_d or vb.shape[0] != db_d or vc.shape[0] != dc_d:
            sys.exit(
                f'ERROR: shape mismatch at {md5}: '
                f'got A={va.shape} B={vb.shape} C={vc.shape}')
        composite[md5] = np.concatenate([va, vb, vc])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.',
                exist_ok=True)
    print(f'Writing {args.out} ({len(composite)} keys, {total}-d each, '
          f'{len(composite) * total * 4 / 1e9:.2f} GB raw)')
    np.savez_compressed(args.out, **composite)
    print('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
