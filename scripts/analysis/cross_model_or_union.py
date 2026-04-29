"""Cross-model hybrid OR HR@1 — take K head from one model and O head
from another, compute the OR-union HR@1 per dataset.

Mathematically equivalent to cipher's existing OR metric (which OR's
K-only and O-only hit indicators within ONE model) — only difference
is that here the K-only indicators come from one model's per-phage
TSV and the O-only indicators come from another's.

Inputs: two per-phage TSVs from `per_head_strict_eval.py --per-phage-out`.
Each TSV has columns:
  dataset, phage_id, k_only_rank, o_only_rank, merged_rank,
  k_hit@1, o_hit@1, merged_hit@1, or_hit@1

Output: stdout summary + optional joined TSV.

Usage:
  python scripts/analysis/cross_model_or_union.py \\
      --k-tsv  results/analysis/per_phage/per_phage_la_v3_uat_prott5_xl_seg8.tsv \\
      --o-tsv  results/analysis/per_phage/per_phage_sweep_prott5_mean_cl70.tsv \\
      --k-label 'la_v3_uat_prott5_xl_seg8' \\
      --o-label 'sweep_prott5_mean_cl70' \\
      --out-tsv results/analysis/hybrid_or_la_K_sweep_O.tsv
"""

import argparse
import csv
import os
from collections import defaultdict


def load_tsv(path):
    """Returns {(dataset, phage_id): {col: val}}."""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            out[(r['dataset'], r['phage_id'])] = r
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k-tsv', required=True,
                   help='per-phage TSV from model whose K head we want')
    p.add_argument('--o-tsv', required=True,
                   help='per-phage TSV from model whose O head we want')
    p.add_argument('--k-label', default='K_model')
    p.add_argument('--o-label', default='O_model')
    p.add_argument('--out-tsv', default=None,
                   help='Optional joined per-phage TSV with hybrid OR column')
    args = p.parse_args()

    k_data = load_tsv(args.k_tsv)
    o_data = load_tsv(args.o_tsv)

    common = sorted(set(k_data.keys()) & set(o_data.keys()))
    print(f'K-model TSV phages:    {len(k_data)}')
    print(f'O-model TSV phages:    {len(o_data)}')
    print(f'Joined (intersection): {len(common)}')
    print()

    # Per-dataset aggregation
    DSS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']
    per_ds = defaultdict(lambda: {'n':0,
                                   'k_only_hit':0,
                                   'o_only_hit':0,
                                   'orig_or_k':0,
                                   'orig_or_o':0,
                                   'hybrid_or_hit':0,
                                   'k_only':0, 'o_only':0,
                                   'k_and_o':0, 'neither':0})

    rows_out = []
    for key in common:
        ds, phage = key
        kr = k_data[key]
        orr = o_data[key]
        # K hit from k-model's K head; O hit from o-model's O head
        k_hit = int(kr['k_hit@1'] or 0)
        o_hit = int(orr['o_hit@1'] or 0)
        hybrid = int(k_hit or o_hit)
        # For comparison: each model's own OR
        k_orig_or = int(kr['or_hit@1'] or 0)
        o_orig_or = int(orr['or_hit@1'] or 0)

        per_ds[ds]['n'] += 1
        per_ds[ds]['k_only_hit'] += k_hit
        per_ds[ds]['o_only_hit'] += o_hit
        per_ds[ds]['hybrid_or_hit'] += hybrid
        per_ds[ds]['orig_or_k'] += k_orig_or
        per_ds[ds]['orig_or_o'] += o_orig_or
        # Cross-tab: which catches what
        if k_hit and o_hit: per_ds[ds]['k_and_o'] += 1
        elif k_hit: per_ds[ds]['k_only'] += 1
        elif o_hit: per_ds[ds]['o_only'] += 1
        else: per_ds[ds]['neither'] += 1

        rows_out.append({
            'dataset': ds, 'phage_id': phage,
            f'k_hit@1_{args.k_label}': k_hit,
            f'o_hit@1_{args.o_label}': o_hit,
            'hybrid_or_hit@1': hybrid,
            f'orig_or_hit@1_{args.k_label}': k_orig_or,
            f'orig_or_hit@1_{args.o_label}': o_orig_or,
        })

    # Print summary
    print(f'Hybrid OR HR@1: K from {args.k_label}, O from {args.o_label}')
    print('-' * 90)
    print(f'{"dataset":<16} {"n":>4}  {"K_hit":>6} {"O_hit":>6} '
          f'{"hybrid_OR":>10} {"K_OR_orig":>10} {"O_OR_orig":>10}')
    tot = {'n':0, 'hybrid':0, 'k_or':0, 'o_or':0}
    for ds in DSS:
        d = per_ds[ds]
        if d['n'] == 0:
            continue
        n = d['n']
        khit = d['k_only_hit']/n
        ohit = d['o_only_hit']/n
        hyb = d['hybrid_or_hit']/n
        kor = d['orig_or_k']/n
        oor = d['orig_or_o']/n
        print(f'{ds:<16} {n:>4}  {khit:>6.3f} {ohit:>6.3f} '
              f'{hyb:>10.3f} {kor:>10.3f} {oor:>10.3f}')
        tot['n'] += n
        tot['hybrid'] += d['hybrid_or_hit']
        tot['k_or'] += d['orig_or_k']
        tot['o_or'] += d['orig_or_o']
    print('-' * 90)
    if tot['n']:
        print(f'{"WEIGHTED":<16} {tot["n"]:>4}        '
              f'         {tot["hybrid"]/tot["n"]:>10.3f} '
              f'{tot["k_or"]/tot["n"]:>10.3f} {tot["o_or"]/tot["n"]:>10.3f}')

    # Cross-tab on PHL specifically (where the question matters most)
    phl = per_ds.get('PhageHostLearn', {})
    if phl.get('n'):
        n = phl['n']
        print()
        print(f'PhageHostLearn cross-tab (n={n}):')
        print(f'  {args.k_label} K catches AND {args.o_label} O catches: {phl["k_and_o"]:>3}  '
              f'({100*phl["k_and_o"]/n:.0f}%)')
        print(f'  {args.k_label} K only catches:                          {phl["k_only"]:>3}')
        print(f'  {args.o_label} O only catches:                          {phl["o_only"]:>3}')
        print(f'  Neither catches:                                        {phl["neither"]:>3}')
        print(f'  → hybrid OR @1 = {(phl["k_and_o"]+phl["k_only"]+phl["o_only"])/n:.3f}')

    # Optional: write joined TSV
    if args.out_tsv:
        os.makedirs(os.path.dirname(args.out_tsv) or '.', exist_ok=True)
        with open(args.out_tsv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, delimiter='\t',
                               fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f'\nWrote joined per-phage TSV: {args.out_tsv}')


if __name__ == '__main__':
    main()
