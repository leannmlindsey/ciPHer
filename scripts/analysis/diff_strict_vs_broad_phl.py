"""Diff strict (8/8) vs broad (phold-only) PHL HR@1, per K-type and overall.

Closes the H5 (broader-PHL eval) test from agent 4: re-evaluate cipher
on the broader PHL set instead of the strict 8/8 tail. If broad HR@1
lifts ≥5 pp, the 8/8 filter was masking real signal.

Inputs are per-phage TSVs from per_head_strict_eval.py (one strict, one
broad) for each model. Joins by (dataset, phage_id) and looks up the
host K-type from the PHL interaction matrix.

Output: a markdown report with per-K HR@1 strict vs broad, plus the
overall PHL HR@1 deltas, per model.

Usage:
    python scripts/analysis/diff_strict_vs_broad_phl.py \\
        --models prott5:la_v3_uat_prott5_xl_seg8 \\
                 esm2:sweep_posList_esm2_3b_mean_cl70 \\
                 kmer:sweep_kmer_aa20_k4 \\
        --per-phage-dir results/analysis/per_phage \\
        --interaction-matrix data/validation_data/HOST_RANGE/PhageHostLearn/metadata/interaction_matrix.tsv \\
        --watch-K K30 K20 K10 K24 K39 K63 \\
        --out results/analysis/h5_strict_vs_broad_phl.md
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


HEAD_MODES = ['k_only', 'o_only', 'merged', 'or']  # which *_hit@1 cols to report


def load_per_phage(path):
    """Return list of dicts with one row per (dataset, phage_id)."""
    if not Path(path).exists():
        return None
    rows = []
    with open(path) as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            rows.append(row)
    return rows


def load_phage_to_K(im_path):
    """Return {phage_id: host_K} from PHL interaction matrix (positive pairs only)."""
    out = {}
    with open(im_path) as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            if row.get('label') in ('1', 1):
                ph = row.get('phage_id', '').strip()
                k = (row.get('host_K') or '').strip() or 'unknown'
                # If a phage has multiple positive hosts with different K-types,
                # keep all; we'll average/report per (phage,K) when computing.
                out.setdefault(ph, set()).add(k)
    return out


def hr1_for_subset(rows, head, phage_filter=None):
    """Compute HR@1 = mean(hit@1) over rows in PhageHostLearn dataset."""
    n = 0
    hits = 0
    for r in rows:
        if r.get('dataset') != 'PhageHostLearn':
            continue
        if phage_filter is not None and r.get('phage_id') not in phage_filter:
            continue
        v = r.get(f'{head}_hit@1', '').strip()
        if v == '':
            continue
        n += 1
        if v == '1':
            hits += 1
    return (hits / n if n else None), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True,
                    help='one or more "key:exp_name" pairs, e.g. prott5:la_v3_uat_prott5_xl_seg8')
    ap.add_argument('--per-phage-dir', required=True,
                    help='dir containing per_phage_<exp>.tsv (strict) and '
                         'per_phage_<exp>_broad_phl.tsv (broad) per model')
    ap.add_argument('--interaction-matrix', required=True,
                    help='PHL interaction_matrix.tsv for K-type lookup')
    ap.add_argument('--watch-K', nargs='+',
                    default=['K30', 'K20', 'K10', 'K24', 'K39', 'K63'],
                    help='K-types to highlight in the per-K table')
    ap.add_argument('--out', required=True, help='output markdown path')
    args = ap.parse_args()

    phage_to_K = load_phage_to_K(args.interaction_matrix)
    print(f'[load] {len(phage_to_K)} phages with positive K-host in PHL matrix')

    # Phages-by-K — for per-K HR@1
    K_to_phages = defaultdict(set)
    for ph, ks in phage_to_K.items():
        for k in ks:
            K_to_phages[k].add(ph)

    lines = []
    lines.append('# H5 — Strict (8/8) vs Broader (phold-only) PHL HR@1\n')
    lines.append('Per-phage TSV pairs joined to PHL interaction matrix; '
                 'HR@1 = mean(hit@1) over phages on the PhageHostLearn dataset.\n')

    for spec in args.models:
        if ':' not in spec:
            print(f'WARNING: malformed --models entry {spec!r} (expected key:exp_name); skipping')
            continue
        key, exp = spec.split(':', 1)
        strict_path = Path(args.per_phage_dir) / f'per_phage_{exp}.tsv'
        broad_path  = Path(args.per_phage_dir) / f'per_phage_{exp}_broad_phl.tsv'
        strict = load_per_phage(strict_path)
        broad  = load_per_phage(broad_path)
        if strict is None:
            print(f'WARNING: missing strict TSV for {key} ({strict_path})')
            continue
        if broad is None:
            print(f'WARNING: missing broad TSV for {key} ({broad_path})')
            continue

        lines.append(f'\n## Model: {key} ({exp})\n')

        # Headline overall
        lines.append('### Headline overall PHL HR@1\n')
        lines.append('| head_mode | strict | broad | Δ (broad - strict) | n_strict | n_broad |')
        lines.append('|---|---:|---:|---:|---:|---:|')
        for head in HEAD_MODES:
            s, ns = hr1_for_subset(strict, head)
            b, nb = hr1_for_subset(broad, head)
            d = (b - s) if (s is not None and b is not None) else None
            fmt = lambda x: f'{x:.3f}' if x is not None else '   --'
            fmt_d = lambda x: f'{x:+.3f}' if x is not None else '   --'
            lines.append(f'| {head} | {fmt(s)} | {fmt(b)} | {fmt_d(d)} | {ns} | {nb} |')

        # Per-K table — focused on watch list
        lines.append(f'\n### Per-K HR@1 (OR head_mode), focus on watch list\n')
        lines.append('| K-type | n_phages | strict OR | broad OR | Δ |')
        lines.append('|---|---:|---:|---:|---:|')
        # Watch-list first, then any K with ≥5 phages
        all_ks = sorted(K_to_phages.keys(), key=lambda k: (-len(K_to_phages[k]), k))
        ordered = [k for k in args.watch_K if k in K_to_phages] + \
                   [k for k in all_ks if k not in args.watch_K and len(K_to_phages[k]) >= 5]
        for k in ordered:
            ph_set = K_to_phages[k]
            s, ns = hr1_for_subset(strict, 'or', phage_filter=ph_set)
            b, nb = hr1_for_subset(broad, 'or', phage_filter=ph_set)
            if ns == 0 and nb == 0:
                continue
            d = (b - s) if (s is not None and b is not None) else None
            fmt = lambda x: f'{x:.3f}' if x is not None else '   --'
            fmt_d = lambda x: f'{x:+.3f}' if x is not None else '   --'
            star = ' ⬆' if (d is not None and d >= 0.05) else ''
            lines.append(f'| {k} | {len(ph_set)} | {fmt(s)} | {fmt(b)} | {fmt_d(d)}{star} |')

    lines.append('\n---\n')
    lines.append('**⬆ marker:** broad - strict ≥ +0.05 → 8/8 filter was '
                 'masking real signal on this K-type.\n')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[write] {args.out}')


if __name__ == '__main__':
    main()
