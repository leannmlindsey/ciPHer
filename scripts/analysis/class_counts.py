#!/usr/bin/env python3
"""Print K/O class counts from each Light Attention experiment.

Diagnostic for the O-silence hypothesis: did match_sweep / posList_cl70 /
highconf drop O classes via `min_class_samples` filtering? Compares final
K/O class counts across experiments so we can see which profile retained
the most O signal.

Reads `label_encoders.json` from each run directory (written by
TrainingData.save() in src/cipher/data/training.py). Pure file I/O — no
model weights loaded, no cipher imports, no GPU.

Usage:
    python scripts/analysis/class_counts.py
    python scripts/analysis/class_counts.py experiments/light_attention/la_seg4_match_sweep/
    python scripts/analysis/class_counts.py path1 path2 path3
"""

import argparse
import glob
import json
import os
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('runs', nargs='*',
                   help='Experiment directories (default: all la_* under '
                        'experiments/light_attention/)')
    p.add_argument('--experiments-root',
                   default='experiments/light_attention',
                   help='Root for the default glob')
    args = p.parse_args()

    if args.runs:
        run_dirs = [d.rstrip('/') for d in args.runs]
    else:
        run_dirs = sorted(
            d.rstrip('/')
            for d in glob.glob(os.path.join(args.experiments_root, 'la_*'))
            if os.path.isdir(d)
        )

    if not run_dirs:
        print(f'No experiment directories found under {args.experiments_root}',
              file=sys.stderr)
        return 1

    rows = []
    for d in run_dirs:
        path = os.path.join(d, 'label_encoders.json')
        name = os.path.basename(d)
        if not os.path.exists(path):
            rows.append((name, None, None, None, None, None))
            continue
        with open(path) as f:
            r = json.load(f)
        k_classes = r.get('k_classes', [])
        o_classes = r.get('o_classes', [])
        strategy = r.get('strategy')
        tools = r.get('tools')
        rows.append((
            name, len(k_classes), len(o_classes),
            strategy,
            ','.join(tools) if tools else None,
            r.get('protein_set'),
        ))

    # Print table
    width = max(len(r[0]) for r in rows)
    width = max(width, 12)
    print(f'{"experiment":<{width}}  {"K":>5}  {"O":>5}  strategy                tools / protein_set')
    print('-' * (width + 75))
    for name, k, o, strategy, tools, protein_set in rows:
        if k is None:
            print(f'{name:<{width}}   (no label_encoders.json found)')
            continue
        filt = tools or protein_set or '(none)'
        print(f'{name:<{width}}  {k:>5}  {o:>5}  '
              f'{(strategy or "?"):<22}  {filt}')

    # Summary deltas — pick a reference row with the most O classes as the
    # "pre-filter" baseline and report per-experiment O drop relative to it.
    valid = [(n, k, o) for n, k, o, *_ in rows if k is not None]
    if len(valid) > 1:
        ref_name, ref_k, ref_o = max(valid, key=lambda r: r[2])
        print()
        print(f'Reference (most O classes): {ref_name}  (K={ref_k}, O={ref_o})')
        print(f'{"experiment":<{width}}  {"ΔK":>6}  {"ΔO":>6}')
        print('-' * (width + 20))
        for n, k, o in valid:
            print(f'{n:<{width}}  {k - ref_k:>+6}  {o - ref_o:>+6}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
