"""Per-K HR@1 comparison across models, stratified by PHL host K-type.

Consumes `per_pair_phl_predictions.csv` (one row per (phage, host) PHL
positive pair × model_id, with rank_k_only / rank_o_only / rank_merged)
and the PHL interaction matrix (for host_K), and emits a per-K table
comparing models side-by-side.

Two aggregations per (model, K) cell, both reported:
  * **any-hit** — for each phage with positive hosts of that K, did
                  ANY of those hosts land at rank 1? Headline metric
                  per agent 1's 2026-04-27-1600 broadcast.
  * **per-pair** — fraction of (phage, positive host) pairs of that K
                  with rank == 1. Stricter; useful for high-n K-types.

Filtering:
  * `--head` k_only / o_only / merged — which rank column to use
                                        (default k_only).
  * Pairs with blank rank in the CSV are treated as misses (not skipped).
  * Pairs whose phage was excluded from the strict denominator
    (no annotated RBPs) won't be in the CSV at all — handled implicitly.

Optional `--tropiseq_kl_list`: two-column file (or one column) of K/KL
identifiers to flag with `K_in_tropiseq_vocab` column. Per agent 4's
2026-04-28 ask: stratifying by TropiSEQ-vocab membership tells you
whether v4 lift comes from K-types TropiSEQ trained on or from
elsewhere.

Usage:
    python scripts/analysis/per_k_hr1_compare.py \\
        --per-pair-csv results/analysis/per_pair_phl_predictions.csv \\
        --interaction-matrix data/validation_data/HOST_RANGE/PhageHostLearn/metadata/interaction_matrix.tsv \\
        --head k_only \\
        --models la_v3_uat_prott5_xl_seg8_k_only la_highconf_prott5_xl_seg8_k_only

If --models is omitted, every model_id in the CSV is included.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict


def load_phl_host_k(interaction_matrix_path):
    """host_id -> host_K from PHL interaction matrix (positives only)."""
    host_k = {}
    with open(interaction_matrix_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            host_k[row['host_id']] = (row.get('host_K') or '').strip()
    return host_k


def load_per_pair(per_pair_csv, head, models_filter=None):
    """Load per-pair ranks from the CSV.

    Returns: {(model_id, phage_id, host_id): rank_or_None}
    Blank/missing rank => None (treated as miss).
    """
    rank_col = {'k_only': 'rank_k_only', 'o_only': 'rank_o_only',
                'merged': 'rank_merged'}[head]
    out = {}
    with open(per_pair_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row.get('model_id', '').strip()
            if models_filter and mid not in models_filter:
                continue
            phage = row.get('phage_id', '').strip()
            host = row.get('host_id', '').strip()
            v = (row.get(rank_col) or '').strip()
            try:
                rank = int(v) if v else None
            except ValueError:
                rank = None
            out[(mid, phage, host)] = rank
    return out


def load_tropiseq_vocab(path):
    if not path:
        return None
    vocab = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # accept first whitespace/comma/tab token
            tok = line.split()[0].split(',')[0].split('\t')[0]
            vocab.add(tok)
    return vocab


def aggregate(per_pair_ranks, host_k, models):
    """Per (model, K) -> {n_pairs, n_phages, hr1_pair, hr1_anyhit}."""
    # Group ranks: (model, K, phage) -> list[rank or None]
    bucket = defaultdict(list)
    for (mid, phage, host), rank in per_pair_ranks.items():
        k = host_k.get(host) or ''
        if not k or k.lower() == 'null':
            k = '<null>'
        bucket[(mid, k, phage)].append(rank)
    # Aggregate to (model, K)
    out = defaultdict(lambda: {'n_pairs': 0, 'n_phages': 0,
                               'pair_hits': 0, 'phage_hits': 0})
    for (mid, k, phage), ranks in bucket.items():
        rec = out[(mid, k)]
        rec['n_phages'] += 1
        rec['n_pairs'] += len(ranks)
        for r in ranks:
            if r is not None and r == 1:
                rec['pair_hits'] += 1
        # any-hit per phage: did ANY positive host of this phage with
        # this K land at rank 1?
        if any(r is not None and r == 1 for r in ranks):
            rec['phage_hits'] += 1
    # Compute rates
    table = {}
    for (mid, k), rec in out.items():
        n_pair = rec['n_pairs']
        n_phage = rec['n_phages']
        table[(mid, k)] = {
            'n_pairs': n_pair, 'n_phages': n_phage,
            'pair_hits': rec['pair_hits'],
            'phage_hits': rec['phage_hits'],
            'hr1_pair': rec['pair_hits'] / n_pair if n_pair else 0.0,
            'hr1_anyhit': rec['phage_hits'] / n_phage if n_phage else 0.0,
        }
    return table


def emit_table(table, models, tropiseq_vocab, out_csv, mode):
    """Write a per-K table with columns: K, K_in_tropiseq_vocab, then
    per-model {n_phages, hr1_anyhit, n_pairs, hr1_pair}.
    Sorted by descending total n_phages across models for that K."""
    all_ks = sorted({k for (_, k) in table}, key=lambda k: (
        -sum(table.get((m, k), {}).get('n_phages', 0) for m in models),
        k,
    ))
    fieldnames = ['K', 'K_in_tropiseq_vocab']
    for m in models:
        fieldnames += [f'{m}__n_phages', f'{m}__anyhit_HR1',
                       f'{m}__n_pairs', f'{m}__pair_HR1']

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        outf = open(out_csv, 'w', newline='')
        w = csv.DictWriter(outf, fieldnames=fieldnames)
        w.writeheader()
    else:
        outf = None

    # Pretty stdout
    print(f'\nPer-K HR@1 ({mode}) — sorted by total n_phages')
    name_w = max(8, max((len(m) for m in models), default=8))
    name_w = min(name_w, 36)
    short = {m: m[:name_w] for m in models}
    print(f'  {"K":<10} {"vocab":<6} ' + ' '.join(f'{short[m]:>{name_w}}' for m in models))
    print('  ' + '-' * (10 + 6 + (name_w + 1) * len(models) + len(models)))

    for k in all_ks:
        in_vocab = '?'
        if tropiseq_vocab is not None:
            in_vocab = 'yes' if k in tropiseq_vocab else 'no'
        per_m_rows = {}
        line = [f'{k:<10}', f'{in_vocab:<6}']
        for m in models:
            rec = table.get((m, k), {})
            n_p = rec.get('n_phages', 0)
            hr1a = rec.get('hr1_anyhit', None)
            n_pr = rec.get('n_pairs', 0)
            hr1p = rec.get('hr1_pair', None)
            per_m_rows[f'{m}__n_phages'] = n_p
            per_m_rows[f'{m}__anyhit_HR1'] = round(hr1a, 4) if hr1a is not None else ''
            per_m_rows[f'{m}__n_pairs'] = n_pr
            per_m_rows[f'{m}__pair_HR1'] = round(hr1p, 4) if hr1p is not None else ''
            cell = f'{hr1a:.2f} ({n_p:>2})' if (n_p > 0) else '   .   '
            line.append(f'{cell:>{name_w}}')
        print('  ' + ' '.join(line))

        if outf is not None:
            row = {'K': k, 'K_in_tropiseq_vocab': in_vocab}
            row.update(per_m_rows)
            w.writerow(row)

    if outf is not None:
        outf.close()
        print(f'\nWrote {out_csv}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--per-pair-csv', required=True,
                   help='per_pair_phl_predictions.csv (with model_id col)')
    p.add_argument('--interaction-matrix', required=True,
                   help='PHL interaction_matrix.tsv (host_id, host_K columns)')
    p.add_argument('--head', default='k_only',
                   choices=['k_only', 'o_only', 'merged'])
    p.add_argument('--models', nargs='+', default=None,
                   help='subset of model_ids to include (default: all)')
    p.add_argument('--tropiseq-kl-list', default=None,
                   help='optional file with TropiSEQ KL identifiers; flags '
                        'each K-type with K_in_tropiseq_vocab')
    p.add_argument('--out-csv', default=None,
                   help='write per-K table to this CSV (otherwise stdout only)')
    args = p.parse_args()

    host_k = load_phl_host_k(args.interaction_matrix)
    per_pair = load_per_pair(args.per_pair_csv, args.head,
                             models_filter=set(args.models) if args.models else None)

    # If --models not given, discover from CSV
    if args.models is None:
        models = sorted({mid for (mid, _, _) in per_pair})
    else:
        models = list(args.models)
    if not models:
        sys.exit('ERROR: no model_ids found in per-pair CSV (after filter)')

    print(f'Loaded {len(per_pair):,} per-pair rank rows over {len(models)} model(s):')
    for m in models:
        n = sum(1 for (mid, _, _) in per_pair if mid == m)
        print(f'  {m}: {n} rows')

    tropiseq_vocab = load_tropiseq_vocab(args.tropiseq_kl_list)
    if tropiseq_vocab:
        print(f'Loaded {len(tropiseq_vocab)} TropiSEQ-vocab K identifiers')

    table = aggregate(per_pair, host_k, models)
    emit_table(table, models, tropiseq_vocab, args.out_csv,
               mode=f'head={args.head}')


if __name__ == '__main__':
    main()
