"""Scan every (K-source, O-source) combination across the per-phage TSVs
on disk and report the best hybrid OR HR@1 — both PHL-only and overall.

Hybrid OR per-phage rank = min(K_rank_from_model_A, O_rank_from_model_B).
HR@1 = fraction of phages whose hybrid rank is ≤ 1.

By "every combination" we mean the cross-product of TSVs found under
results/analysis/per_phage/. Add more by submitting Delta jobs via
scripts/run_per_phage_extract.sh, then re-running this script.

Output: stdout table + optional --out-tsv leaderboard.

Usage:
    python scripts/analysis/find_best_hybrid_or.py
    python scripts/analysis/find_best_hybrid_or.py --top 20 --out-tsv \
        results/analysis/hybrid_or_leaderboard.tsv
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path


FIXED = {'CHEN': 3, 'GORODNICHIV': 3, 'UCSD': 11, 'PBIP': 103, 'PhageHostLearn': 100}
TOTAL = sum(FIXED.values())
PER_PHAGE_DIR = 'results/analysis/per_phage'


def load_tsv(path):
    """{(dataset, phage_id): row}"""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            out[(r['dataset'], r['phage_id'])] = r
    return out


def parse_rank(s, big=10**9):
    if s is None or not str(s).strip():
        return big
    try: return int(s)
    except (ValueError, TypeError): return big


def hybrid_hr_at_1(k_tsv, o_tsv):
    """{dataset: hr@1} for hybrid = min(K-rank from k_tsv, O-rank from o_tsv).
    Phages present in either TSV count; missing-from-other becomes the
    "use what you have" branch (rank from the side that has it).
    Returns (per_dataset_hits, total_hits_overall)."""
    by_ds = defaultdict(lambda: {'n': 0, 'hits': 0})
    keys = set(k_tsv.keys()) | set(o_tsv.keys())
    for (ds, ph) in keys:
        if ds not in FIXED: continue
        kr = parse_rank((k_tsv.get((ds, ph)) or {}).get('k_only_rank'))
        orr = parse_rank((o_tsv.get((ds, ph)) or {}).get('o_only_rank'))
        hyb = min(kr, orr)
        by_ds[ds]['n'] += 1
        if hyb <= 1: by_ds[ds]['hits'] += 1
    return by_ds


def fixed_hr(by_ds, dataset):
    return by_ds[dataset]['hits'] / FIXED[dataset]


def overall_fixed_hr(by_ds):
    return sum(by_ds[ds]['hits'] for ds in FIXED) / TOTAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-phage-dir', default=PER_PHAGE_DIR)
    ap.add_argument('--top', type=int, default=15)
    ap.add_argument('--out-tsv', default=None)
    ap.add_argument('--require', nargs='*', default=[],
                    help='if given, restrict K-source AND O-source to runs '
                         'whose name contains one of these substrings')
    args = ap.parse_args()

    paths = sorted(Path(args.per_phage_dir).glob('per_phage_*.tsv'))
    if args.require:
        paths = [p for p in paths
                 if any(s in p.name for s in args.require)]
    if not paths:
        raise SystemExit(f'No per-phage TSVs found under {args.per_phage_dir}')

    # Run name = filename minus per_phage_ prefix and .tsv suffix
    runs = {p.stem.replace('per_phage_', '', 1): load_tsv(str(p)) for p in paths}
    print(f'Loaded {len(runs)} per-phage TSVs:')
    for name in sorted(runs):
        n_phl = sum(1 for (ds, _) in runs[name] if ds == 'PhageHostLearn')
        print(f'  {name}  (PHL phages={n_phl})')

    # Compute hybrid for every (K_run, O_run) — includes "same run" (which
    # is just cipher's intra-model OR) for sanity.
    leaderboard = []
    for k_name, k_tsv in runs.items():
        for o_name, o_tsv in runs.items():
            by_ds = hybrid_hr_at_1(k_tsv, o_tsv)
            phl_hr  = fixed_hr(by_ds, 'PhageHostLearn')
            pbip_hr = fixed_hr(by_ds, 'PBIP')
            ov_hr   = overall_fixed_hr(by_ds)
            leaderboard.append({
                'k_source':       k_name,
                'o_source':       o_name,
                'phl_hits':       by_ds['PhageHostLearn']['hits'],
                'pbip_hits':      by_ds['PBIP']['hits'],
                'overall_hits':   sum(by_ds[ds]['hits'] for ds in FIXED),
                'phl_hr1':        phl_hr,
                'pbip_hr1':       pbip_hr,
                'overall_hr1':    ov_hr,
                'is_intra':       (k_name == o_name),
            })

    # Sort by overall first, then PHL
    leaderboard.sort(key=lambda r: (-r['overall_hr1'], -r['phl_hr1']))

    # ── Header ──
    print()
    print('═══ Hybrid OR HR@1 leaderboard (n_PHL=100, n_overall=220, fixed denom) ═══')
    print(f'  {"#":>3s} {"PHL HR@1":>9s} {"overall HR@1":>13s} {"K-source":<46s} {"O-source":<46s} note')
    seen = set()  # (k, o) deduped
    rank = 0
    for r in leaderboard:
        key = (r['k_source'], r['o_source'])
        if key in seen: continue
        seen.add(key)
        rank += 1
        if rank > args.top: break
        note = '(intra-model OR)' if r['is_intra'] else ''
        print(f'  {rank:>3d}    {r["phl_hr1"]:.3f}        {r["overall_hr1"]:.3f}     '
              f'{r["k_source"]:<46s} {r["o_source"]:<46s} {note}')

    # Show single-best K and single-best O within each direction for context
    print()
    best = leaderboard[0]
    print(f'═══ Top hybrid:  '
          f'PHL HR@1 = {best["phl_hr1"]:.3f}  '
          f'overall HR@1 = {best["overall_hr1"]:.3f} ═══')
    print(f'    K from: {best["k_source"]}')
    print(f'    O from: {best["o_source"]}')

    # PHL-only winner (might differ from overall)
    leaderboard_phl = sorted(leaderboard, key=lambda r: -r['phl_hr1'])
    best_phl = leaderboard_phl[0]
    if (best_phl['k_source'], best_phl['o_source']) != (best['k_source'], best['o_source']):
        print()
        print(f'═══ Top hybrid by PHL only:  '
              f'PHL HR@1 = {best_phl["phl_hr1"]:.3f}  '
              f'overall HR@1 = {best_phl["overall_hr1"]:.3f} ═══')
        print(f'    K from: {best_phl["k_source"]}')
        print(f'    O from: {best_phl["o_source"]}')

    if args.out_tsv:
        os.makedirs(os.path.dirname(args.out_tsv) or '.', exist_ok=True)
        with open(args.out_tsv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(leaderboard[0].keys()),
                                delimiter='\t')
            w.writeheader()
            for r in leaderboard: w.writerow(r)
        print(f'\nWrote leaderboard: {args.out_tsv}')


if __name__ == '__main__':
    main()
