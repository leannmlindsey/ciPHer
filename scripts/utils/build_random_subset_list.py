"""Build a deterministic random subset of a positive-ID list file.

Size-matched control for the highconf_pipeline experiment. Samples
N IDs at random from an input list (seeded), writes them to an output
list one per line.

Usage:
    python scripts/utils/build_random_subset_list.py \\
        --in  data/training_data/metadata/pipeline_positive.list \\
        --out data/training_data/metadata/random12481_pipeline_positive_K.list \\
        --n   12481 \\
        --seed 42
"""

import argparse
import random
from pathlib import Path


def load_ids(path):
    ids = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s and not s.startswith('#'):
            ids.append(s)
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='inp', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    ids = load_ids(args.inp)
    if args.n > len(ids):
        raise SystemExit(f'ERROR: requested n={args.n} but input has only {len(ids)} IDs')

    rng = random.Random(args.seed)
    subset = rng.sample(ids, args.n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(subset) + '\n')

    print(f'Input IDs:   {len(ids):,}')
    print(f'Sampled:     {len(subset):,} (seed={args.seed})')
    print(f'Wrote:       {out}')


if __name__ == '__main__':
    main()
