"""Compare layer-by-layer weights between OLD-codebase trained model
and OUR ciPHer-codebase trained model on this laptop. Both use the
same recipe, same hardware, same data, same splits — yet eval results
differ. The weights tell us how different the models actually are.

If the weights are identical: bug must be in evaluation (unlikely; we
already share the same predict path).

If weights are large-diff: training is converging to different minima.
Look at:
  - Random init (compare layer 0 weights between fresh inits)
  - DataLoader shuffle order (different batch sequences early on)
  - Some subtle BN/Dropout difference

If weights are small-diff: numerical accumulation. Possibly a tiny
arithmetic difference compounded over 200 epochs.

Usage:
    # First make sure you have:
    # 1. OLD model: <OLD_REPO>/output/local_test_v3/model_k_*_seed42_<TIMESTAMP>/best_model.pt
    #    (the freshly retrained one from reproduce_old_v3_training_local.sh)
    # 2. NEW model: <CIPHER_REPO>/experiments/attention_mlp/repro_old_v3_in_cipher_LAPTOP_*/model_k/best_model.pt
    #    (from run_cipher_repro_local.sh)
    python scripts/analysis/diff_trained_weights.py
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch


def find_latest(pattern_dir, pattern):
    """Return most recently modified file matching <pattern_dir>/<pattern>."""
    candidates = sorted(Path(pattern_dir).glob(pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--old-repo', default='/Users/leannmlindsey/WORK/PHI_TSP/phi_tsp/klebsiella')
    p.add_argument('--cipher-repo', default='/Users/leannmlindsey/WORK/PHI_TSP/cipher')
    p.add_argument('--head', choices=['k', 'o'], default='o',
                   help='Which head to compare (default: o, since that is the one that diverged most)')
    p.add_argument('--old-model-pt', default=None)
    p.add_argument('--new-model-pt', default=None)
    args = p.parse_args()

    if args.old_model_pt:
        old_pt = Path(args.old_model_pt)
    else:
        # Most recent freshly-retrained model_<head>_*_seed42_<TIMESTAMP>/best_model.pt
        old_dir = find_latest(
            Path(args.old_repo) / 'output/local_test_v3',
            f'model_{args.head}_*_seed42_*/'
        )
        if old_dir is None:
            raise SystemExit(f'No old model_{args.head}_*_seed42_*/ found under '
                             f'{args.old_repo}/output/local_test_v3/. '
                             f'Run reproduce_old_v3_training_local.sh first.')
        old_pt = old_dir / 'best_model.pt'

    if args.new_model_pt:
        new_pt = Path(args.new_model_pt)
    else:
        # Most recent ciPHer LAPTOP repro experiment dir
        new_exp = find_latest(
            Path(args.cipher_repo) / 'experiments/attention_mlp',
            'repro_old_v3_in_cipher_LAPTOP_*/'
        )
        if new_exp is None:
            raise SystemExit(f'No repro_old_v3_in_cipher_LAPTOP_*/ found under '
                             f'{args.cipher_repo}/experiments/attention_mlp/. '
                             f'Run run_cipher_repro_local.sh first.')
        new_pt = new_exp / f'model_{args.head}' / 'best_model.pt'

    print(f'OLD model checkpoint: {old_pt}')
    print(f'NEW model checkpoint: {new_pt}')
    print()

    if not old_pt.exists():
        raise SystemExit(f'OLD model checkpoint not found: {old_pt}')
    if not new_pt.exists():
        raise SystemExit(f'NEW model checkpoint not found: {new_pt}')

    old_sd = torch.load(old_pt, map_location='cpu', weights_only=True)
    new_sd = torch.load(new_pt, map_location='cpu', weights_only=True)

    old_keys = set(old_sd.keys())
    new_keys = set(new_sd.keys())

    print('Layer key comparison:')
    print(f'  in_both: {len(old_keys & new_keys)}')
    print(f'  only_old: {sorted(old_keys - new_keys)[:5]}')
    print(f'  only_new: {sorted(new_keys - old_keys)[:5]}')

    if old_keys != new_keys:
        print()
        print('VERDICT: layer keys differ — model architectures are NOT the same.')
        return

    print()
    print(f'{"Layer":<45} {"shape":<25} {"max|Δ|":>10} {"mean|Δ|":>10} {"old_mean":>10} {"new_mean":>10}')
    print('-' * 115)

    total_max = 0.0
    total_mean = 0.0
    n_layers = 0
    layers_identical = 0
    layers_close = 0  # within 1e-6
    layers_diff = 0   # noticeably different

    for k in sorted(old_keys):
        o = old_sd[k].float()
        n = new_sd[k].float()
        if o.shape != n.shape:
            print(f'{k[:44]:<45} {"SHAPE MISMATCH":<25}')
            continue
        diff = (o - n).abs()
        max_d = float(diff.max())
        mean_d = float(diff.mean())
        old_mu = float(o.mean())
        new_mu = float(n.mean())
        print(f'{k[:44]:<45} {str(tuple(o.shape)):<25} {max_d:>10.6f} {mean_d:>10.6f} '
              f'{old_mu:>10.4f} {new_mu:>10.4f}')

        total_max = max(total_max, max_d)
        total_mean += mean_d
        n_layers += 1

        if max_d < 1e-9:
            layers_identical += 1
        elif max_d < 1e-6:
            layers_close += 1
        else:
            layers_diff += 1

    print('-' * 115)
    print(f'OVERALL: max layer-wise |Δ| = {total_max:.6f}, '
          f'mean layer-wise |Δ| = {total_mean / max(n_layers, 1):.6f}')
    print(f'  Layers with max|Δ| < 1e-9 (identical):   {layers_identical} / {n_layers}')
    print(f'  Layers with max|Δ| < 1e-6 (numerical):   {layers_close} / {n_layers}')
    print(f'  Layers with max|Δ| ≥ 1e-6 (different):  {layers_diff} / {n_layers}')

    print()
    if layers_identical == n_layers:
        print('VERDICT: BIT-EXACT match. Models are identical. Difference is in eval.')
    elif layers_diff == 0:
        print('VERDICT: numerically near-identical. Tiny float differences only — '
              'training is essentially the same; prediction differences are noise.')
    elif layers_diff < n_layers * 0.3:
        print('VERDICT: localized divergence. Some layers identical, some diverged — '
              'inspect which layers and which weights specifically differ.')
    else:
        print('VERDICT: training trajectories materially diverged. Likely cause: '
              'different random initialization OR different DataLoader shuffle '
              'sequences. Drill into model __init__ RNG order vs DataLoader '
              'construction order.')


if __name__ == '__main__':
    main()
