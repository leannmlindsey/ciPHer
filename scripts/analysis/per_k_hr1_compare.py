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


def count_train_proteins_per_k(klist_path, hpp_map_path):
    """Per-K count of distinct training MD5s after applying a K-list filter.

    klist_path: protein-ID list (one per line) — the --positive_list_k
                used during training (e.g. v1 highconf, v3 UAT).
    hpp_map_path: host_phage_protein_map.tsv with columns
                  host_genome, K_type, O_type, phage_genome, protein_id,
                  is_tsp, md5.

    A training "protein" is a distinct (md5, K) pair, since the model
    sees one supervision signal per (sequence, host-K) tuple. Counting
    distinct MD5s per K matches the convention used in the
    2026-04-27-1115 v1-class-filter handoff.
    """
    with open(klist_path) as f:
        klist = {ln.strip() for ln in f if ln.strip()}
    md5s_by_k = defaultdict(set)
    with open(hpp_map_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        # Tolerate either header convention used historically:
        # ('host_assembly', 'host_K', 'host_O', 'phage_id', 'protein_id', 'is_tsp', 'protein_md5')
        # or the build-script's
        # ('host_genome',   'K_type', 'O_type', 'phage_genome', 'protein_id', 'is_tsp', 'md5').
        for row in reader:
            if row.get('protein_id', '') not in klist:
                continue
            k = (row.get('host_K') or row.get('K_type') or '').strip()
            if not k or k.lower() == 'null':
                k = '<null>'
            md5 = (row.get('protein_md5') or row.get('md5') or '').strip()
            if md5:
                md5s_by_k[k].add(md5)
    return {k: len(s) for k, s in md5s_by_k.items()}


def aggregate(per_pair_ranks, host_k, models):
    """Per (model, K) -> {n_pairs, n_phages, hr1_pair, hr1_anyhit, mrr_anyhit, mrr_pair}.

    Aggregation:
      - n_pairs: positive (phage, host) pairs of K
      - n_phages: distinct phages with ≥1 positive host of K
      - hr1_pair: pair-level HR@1 (fraction of pairs at rank 1)
      - hr1_anyhit: phage-level HR@1 (any positive host at rank 1)
      - mrr_pair: mean(1/rank) over all pairs (None → 0)
      - mrr_anyhit: per phage take 1/min(rank) across positives (or 0 if
                    none scored), then average across phages — the
                    standard retrieval MRR at the query (= phage) level.
    """
    bucket = defaultdict(list)
    for (mid, phage, host), rank in per_pair_ranks.items():
        k = host_k.get(host) or ''
        if not k or k.lower() == 'null':
            k = '<null>'
        bucket[(mid, k, phage)].append(rank)

    out = defaultdict(lambda: {'n_pairs': 0, 'n_phages': 0,
                               'pair_hits': 0, 'phage_hits': 0,
                               'rr_pair_sum': 0.0,
                               'rr_anyhit_sum': 0.0})
    for (mid, k, phage), ranks in bucket.items():
        rec = out[(mid, k)]
        rec['n_phages'] += 1
        rec['n_pairs'] += len(ranks)
        for r in ranks:
            if r is not None:
                rec['rr_pair_sum'] += 1.0 / r
                if r == 1:
                    rec['pair_hits'] += 1
        scored = [r for r in ranks if r is not None]
        if scored:
            best = min(scored)
            rec['rr_anyhit_sum'] += 1.0 / best
            if best == 1:
                rec['phage_hits'] += 1

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
            'mrr_pair': rec['rr_pair_sum'] / n_pair if n_pair else 0.0,
            'mrr_anyhit': rec['rr_anyhit_sum'] / n_phage if n_phage else 0.0,
        }
    return table


def emit_table(table, models, tropiseq_vocab, train_counts, out_csv, mode):
    """Emit a clean per-K table to stdout (column-aligned) and CSV.

    Columns per model: train_md5s (optional), n_phages, n_pairs, HR1_anyhit,
    HR1_pair, MRR_anyhit, MRR_pair.

    Sorted by descending total n_phages across models for that K."""
    all_ks = sorted({k for (_, k) in table}, key=lambda k: (
        -sum(table.get((m, k), {}).get('n_phages', 0) for m in models),
        k,
    ))

    base_fields = ['K', 'K_in_tropiseq_vocab']
    per_model_keys = []
    for m in models:
        keys = []
        if train_counts and m in train_counts:
            keys.append('train_md5s')
        keys += ['n_phages', 'n_pairs', 'HR1_anyhit', 'HR1_pair',
                 'MRR_anyhit', 'MRR_pair']
        per_model_keys.append(keys)

    fieldnames = list(base_fields)
    for m, keys in zip(models, per_model_keys):
        for k in keys:
            fieldnames.append(f'{m}__{k}')

    # Build all rows first (consistent for stdout and CSV)
    rows = []
    for k_lbl in all_ks:
        in_vocab = ''
        if tropiseq_vocab is not None:
            in_vocab = 'yes' if k_lbl in tropiseq_vocab else 'no'
        row = {'K': k_lbl, 'K_in_tropiseq_vocab': in_vocab}
        for m, keys in zip(models, per_model_keys):
            rec = table.get((m, k_lbl), {})
            tcount = (train_counts.get(m, {}).get(k_lbl, 0)
                      if train_counts and m in train_counts else None)
            for kk in keys:
                col = f'{m}__{kk}'
                if kk == 'train_md5s':
                    row[col] = tcount if tcount is not None else 0
                elif kk in ('n_phages', 'n_pairs'):
                    row[col] = rec.get(kk, 0)
                elif kk == 'HR1_anyhit':
                    row[col] = round(rec.get('hr1_anyhit', 0.0), 4)
                elif kk == 'HR1_pair':
                    row[col] = round(rec.get('hr1_pair', 0.0), 4)
                elif kk == 'MRR_anyhit':
                    row[col] = round(rec.get('mrr_anyhit', 0.0), 4)
                elif kk == 'MRR_pair':
                    row[col] = round(rec.get('mrr_pair', 0.0), 4)
        rows.append(row)

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f'Wrote {out_csv} ({len(rows)} K-types)', file=sys.stderr)

    # Stdout: aligned columns. Width per column = max(header len, max value len).
    widths = {}
    for col in fieldnames:
        w_col = len(col)
        for row in rows:
            v = row.get(col, '')
            w_col = max(w_col, len(str(v)))
        widths[col] = w_col

    print(f'\nPer-K HR@1 / MRR ({mode}) — sorted by total n_phages\n')
    print('  '.join(c.ljust(widths[c]) for c in fieldnames))
    print('  '.join('-' * widths[c] for c in fieldnames))
    for row in rows:
        print('  '.join(str(row.get(c, '')).ljust(widths[c]) for c in fieldnames))


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
    p.add_argument('--training-list', action='append', default=[],
                   metavar='MODEL_ID=PATH',
                   help='associate a model_id with a K-list file (e.g. '
                        'highconf_pipeline_positive_K.list, HC_K_UAT_multitop.list). '
                        'Repeat for multiple models. Triggers per-K training '
                        'protein counts to be added to the table.')
    p.add_argument('--association-map', default=None,
                   help='host_phage_protein_map.tsv (required if --training-list given)')
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

    train_counts = None
    if args.training_list:
        if not args.association_map:
            sys.exit('ERROR: --training-list requires --association-map '
                     '(host_phage_protein_map.tsv)')
        train_counts = {}
        for spec in args.training_list:
            if '=' not in spec:
                sys.exit(f'ERROR: --training-list expects MODEL_ID=PATH; got {spec!r}')
            mid, path = spec.split('=', 1)
            if mid not in models:
                print(f'WARNING: --training-list {mid!r} not in --models; skipping',
                      file=sys.stderr)
                continue
            print(f'  computing per-K training counts for {mid} from {path} ...',
                  file=sys.stderr)
            train_counts[mid] = count_train_proteins_per_k(path, args.association_map)
            print(f'    {len(train_counts[mid])} K-types covered, '
                  f'total distinct md5s: '
                  f'{sum(train_counts[mid].values())}', file=sys.stderr)

    tropiseq_vocab = load_tropiseq_vocab(args.tropiseq_kl_list)
    if tropiseq_vocab:
        print(f'Loaded {len(tropiseq_vocab)} TropiSEQ-vocab K identifiers')

    table = aggregate(per_pair, host_k, models)
    emit_table(table, models, tropiseq_vocab, train_counts, args.out_csv,
               mode=f'head={args.head}')


if __name__ == '__main__':
    main()
