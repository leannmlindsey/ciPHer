"""Cross-model hybrid OR HR@k — take K head from one model and O head
from another, compute the OR-union HR@k=1..20 per dataset.

Mathematically equivalent to cipher's existing OR metric (which OR's
K-only and O-only rank indicators within ONE model) — only difference
is that here the K-only ranks come from one model's per-phage TSV
and the O-only ranks come from another's.

Per-phage hybrid rank = min(K_rank_from_model_A, O_rank_from_model_B)
HR@k_hybrid = fraction of phages with hybrid_rank ≤ k.

Outputs:
  - stdout: per-dataset summary + PHL cross-tab
  - --out-tsv: joined per-phage table
  - --out-curves-json: HR@k=1..20 curves per dataset (for plot scripts)

Usage:
  python scripts/analysis/cross_model_or_union.py \\
      --k-tsv  results/analysis/per_phage/per_phage_la_v3_uat_prott5_xl_seg8.tsv \\
      --o-tsv  results/analysis/per_phage/per_phage_sweep_prott5_mean_cl70.tsv \\
      --k-label 'la_v3_uat_prott5_xl_seg8' \\
      --o-label 'sweep_prott5_mean_cl70' \\
      --out-tsv results/analysis/hybrid_or_la_K_sweep_O.tsv \\
      --out-curves-json results/analysis/hybrid_or_la_K_sweep_O_curves.json
"""

import argparse
import csv
import json
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
    p.add_argument('--out-curves-json', default=None,
                   help='Optional JSON: per-dataset hybrid HR@k=1..20 curves '
                        '+ K-only and O-only curves (for plot scripts)')
    args = p.parse_args()

    k_data = load_tsv(args.k_tsv)
    o_data = load_tsv(args.o_tsv)

    common = sorted(set(k_data.keys()) & set(o_data.keys()))
    print(f'K-model TSV phages:    {len(k_data)}')
    print(f'O-model TSV phages:    {len(o_data)}')
    print(f'Joined (intersection): {len(common)}')
    print()

    # Per-dataset aggregation. We accumulate ranks so we can compute
    # full HR@k=1..20 curves, not just k=1.
    DSS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']
    per_ds_ranks = defaultdict(lambda: {'phages':[], 'k_ranks':[], 'o_ranks':[],
                                          'hybrid_ranks':[]})

    def parse_rank(v):
        try: return int(v)
        except (TypeError, ValueError):
            try: return int(float(v))
            except (TypeError, ValueError): return None

    rows_out = []
    for key in common:
        ds, phage = key
        kr = k_data[key]
        orr = o_data[key]
        k_rank = parse_rank(kr.get('k_only_rank'))   # K head rank from k-model
        o_rank = parse_rank(orr.get('o_only_rank'))  # O head rank from o-model
        # Hybrid rank: min over K and O ranks (None acts as +inf)
        valid = [r for r in (k_rank, o_rank) if r is not None]
        hybrid_rank = min(valid) if valid else None
        per_ds_ranks[ds]['phages'].append(phage)
        per_ds_ranks[ds]['k_ranks'].append(k_rank)
        per_ds_ranks[ds]['o_ranks'].append(o_rank)
        per_ds_ranks[ds]['hybrid_ranks'].append(hybrid_rank)

        # Per-row TSV output (hit@1 conveniences)
        k_hit1 = int(k_rank is not None and k_rank <= 1)
        o_hit1 = int(o_rank is not None and o_rank <= 1)
        h_hit1 = int(hybrid_rank is not None and hybrid_rank <= 1)
        rows_out.append({
            'dataset': ds, 'phage_id': phage,
            f'k_only_rank_{args.k_label}': k_rank if k_rank is not None else '',
            f'o_only_rank_{args.o_label}': o_rank if o_rank is not None else '',
            'hybrid_rank': hybrid_rank if hybrid_rank is not None else '',
            f'k_hit@1_{args.k_label}': k_hit1,
            f'o_hit@1_{args.o_label}': o_hit1,
            'hybrid_or_hit@1': h_hit1,
        })

    def hr_at_k(ranks, k):
        n = len(ranks)
        if n == 0: return None
        return sum(1 for r in ranks if r is not None and r <= k) / n

    # Per-dataset HR@k curves
    K_VALUES = list(range(1, 21))
    curves = {}
    for ds in DSS:
        d = per_ds_ranks.get(ds, {'phages':[]})
        n = len(d['phages'])
        if n == 0: continue
        curves[ds] = {
            'n': n,
            'k_only_hr@k':  {k: hr_at_k(d['k_ranks'], k) for k in K_VALUES},
            'o_only_hr@k':  {k: hr_at_k(d['o_ranks'], k) for k in K_VALUES},
            'hybrid_hr@k':  {k: hr_at_k(d['hybrid_ranks'], k) for k in K_VALUES},
        }
    # Phage-weighted overall (across all 5 datasets)
    all_k = []
    all_o = []
    all_h = []
    for ds in DSS:
        d = per_ds_ranks.get(ds, {})
        all_k.extend(d.get('k_ranks', []))
        all_o.extend(d.get('o_ranks', []))
        all_h.extend(d.get('hybrid_ranks', []))
    if all_h:
        curves['overall'] = {
            'n': len(all_h),
            'k_only_hr@k':  {k: hr_at_k(all_k, k) for k in K_VALUES},
            'o_only_hr@k':  {k: hr_at_k(all_o, k) for k in K_VALUES},
            'hybrid_hr@k':  {k: hr_at_k(all_h, k) for k in K_VALUES},
        }

    # Print HR@1 summary
    print(f'Hybrid OR HR@1: K from {args.k_label}, O from {args.o_label}')
    print('-' * 80)
    print(f'{"dataset":<16} {"n":>4}  {"K_hit@1":>8} {"O_hit@1":>8} {"hybrid@1":>10}')
    for ds in DSS + ['overall']:
        if ds not in curves: continue
        c = curves[ds]
        n = c['n']
        khit = c['k_only_hr@k'][1]
        ohit = c['o_only_hr@k'][1]
        hyb  = c['hybrid_hr@k'][1]
        label = ds if ds != 'overall' else 'WEIGHTED'
        print(f'{label:<16} {n:>4}  {khit:>8.3f} {ohit:>8.3f} {hyb:>10.3f}')

    # Cross-tab on PHL specifically
    phl = per_ds_ranks.get('PhageHostLearn', {'phages':[]})
    if phl['phages']:
        k_and_o = sum(1 for k, o in zip(phl['k_ranks'], phl['o_ranks'])
                      if k is not None and k <= 1 and o is not None and o <= 1)
        k_only = sum(1 for k, o in zip(phl['k_ranks'], phl['o_ranks'])
                     if (k is not None and k <= 1) and not (o is not None and o <= 1))
        o_only = sum(1 for k, o in zip(phl['k_ranks'], phl['o_ranks'])
                     if not (k is not None and k <= 1) and (o is not None and o <= 1))
        neither = len(phl['phages']) - k_and_o - k_only - o_only
        print()
        print(f'PhageHostLearn cross-tab (n={len(phl["phages"])}):')
        print(f'  K hits AND O hits:  {k_and_o:>3}')
        print(f'  K only:             {k_only:>3}')
        print(f'  O only:             {o_only:>3}')
        print(f'  neither:            {neither:>3}')
        print(f'  → hybrid OR @1 = {(k_and_o + k_only + o_only) / len(phl["phages"]):.3f}')

    # Optional curves JSON for plot scripts
    if args.out_curves_json:
        os.makedirs(os.path.dirname(args.out_curves_json) or '.', exist_ok=True)
        out = {
            'k_label': args.k_label,
            'o_label': args.o_label,
            'datasets': curves,
        }
        with open(args.out_curves_json, 'w') as fh:
            json.dump(out, fh, indent=2)
        print(f'\nWrote curves JSON: {args.out_curves_json}')

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
