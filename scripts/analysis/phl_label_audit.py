"""PHL label audit — diagnose why 85-90% of PHL proteins don't share a
K-type with their nearest training neighbour.

Three failure modes we want to separate:
  A. Legitimate novelty — the PHL protein's K-type is simply not in
     training. No amount of modelling fixes this without more data.
  B. Representation failure — the K-type IS in training, but the training
     proteins that carry it are not near this PHL protein in embedding
     space. Could be fixed by better pooling / different architecture.
  C. Label noise — a PHL protein's nearest training neighbour has a
     different K-type because the same sequence is annotated differently
     in training vs. PHL metadata (or the protein is genuinely
     polyspecific and annotation is arbitrary).

Uses the ESM-2 650M mean embeddings + existing analysis outputs — no GPU.

Outputs:
  results/analysis/phl_label_audit.txt      — summary + per-K-type breakdown
  results/analysis/phl_label_audit_failed.csv  — 1 row per failing PHL protein

Usage:
  python scripts/analysis/phl_label_audit.py
"""

import argparse
import csv
import os
from collections import Counter, defaultdict

import numpy as np

from cipher.data.embeddings import load_embeddings
from cipher.data.proteins import load_fasta_md5

NULL_LABELS = {'null', 'N/A', 'Unknown', '', 'NA', 'n/a', 'None'}


def load_training_md5_to_k(assoc_path):
    md5_k = defaultdict(set)
    md5_k_counts = defaultdict(Counter)  # md5 -> Counter of K observations
    with open(assoc_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            md5 = row.get('protein_md5', '').strip()
            k = row.get('host_K', '').strip()
            if not md5 or not k or k in NULL_LABELS:
                continue
            md5_k[md5].add(k)
            md5_k_counts[md5][k] += 1
    return dict(md5_k), dict(md5_k_counts)


def load_phl_protein_to_k(mapping_path, interaction_path):
    phage_k = defaultdict(set)
    with open(interaction_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row.get('label', '').strip() not in ('1', '1.0', 'True', 'true'):
                continue
            phage = row.get('phage_id', '').strip()
            k = row.get('host_K', '').strip()
            if not phage or not k or k in NULL_LABELS:
                continue
            phage_k[phage].add(k)
    protein_k = defaultdict(set)
    with open(mapping_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            phage = row.get('matrix_phage_name', '').strip()
            protein = row.get('protein_id', '').strip()
            if phage in phage_k:
                protein_k[protein] |= phage_k[phage]
    return dict(protein_k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-emb',
                   default='data/training_data/embeddings/esm2_650m_md5.npz')
    p.add_argument('--val-emb',
                   default='data/validation_data/embeddings/esm2_650m_md5.npz')
    p.add_argument('--val-fasta',
                   default='data/validation_data/metadata/validation_rbps_all.faa')
    p.add_argument('--assoc-map',
                   default='data/training_data/metadata/host_phage_protein_map.tsv')
    p.add_argument('--phl-mapping',
                   default='data/validation_data/HOST_RANGE/PhageHostLearn/'
                           'metadata/phage_protein_mapping.csv')
    p.add_argument('--phl-interactions',
                   default='data/validation_data/HOST_RANGE/PhageHostLearn/'
                           'metadata/interaction_matrix.tsv')
    p.add_argument('--topk', type=int, default=50)
    p.add_argument('--out-dir', default='results/analysis')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading training md5 -> K-types...')
    train_md5_k, train_md5_k_counts = load_training_md5_to_k(args.assoc_map)
    print(f'  {len(train_md5_k):,} training md5s with K labels')

    # Global training K-type coverage
    train_k_all = Counter()
    for md5, ks in train_md5_k.items():
        for k in ks:
            train_k_all[k] += 1

    print('Loading PHL protein -> K-types...')
    phl_k = load_phl_protein_to_k(args.phl_mapping, args.phl_interactions)
    print(f'  {len(phl_k):,} PHL proteins with K labels')

    # ===== AUDIT A: K-type coverage =====
    phl_k_all = Counter()
    for pid, ks in phl_k.items():
        for k in ks:
            phl_k_all[k] += 1

    k_novel = []        # in PHL, never in training
    k_thin = []         # in PHL, <5 training examples
    k_present = []      # well-represented in training
    for k, phl_n in phl_k_all.most_common():
        train_n = train_k_all.get(k, 0)
        if train_n == 0:
            k_novel.append((k, phl_n))
        elif train_n < 5:
            k_thin.append((k, phl_n, train_n))
        else:
            k_present.append((k, phl_n, train_n))

    # ===== AUDIT B: per-PHL-protein nearest-neighbor info =====
    print('\nLoading embeddings + computing nearest neighbours...')
    md5_filter = set(train_md5_k.keys())
    train = load_embeddings(args.train_emb, md5_filter=md5_filter)
    train_md5s = list(train.keys())
    train_vecs = np.stack([train[m] for m in train_md5s]).astype(np.float32)
    train_vecs /= np.linalg.norm(train_vecs, axis=1, keepdims=True)

    val = load_embeddings(args.val_emb)
    pid_md5 = load_fasta_md5(args.val_fasta)

    queries, q_pids, q_md5s, q_true_k = [], [], [], []
    for pid, true_k in phl_k.items():
        md5 = pid_md5.get(pid)
        if md5 and md5 in val:
            queries.append(val[md5])
            q_pids.append(pid)
            q_md5s.append(md5)
            q_true_k.append(true_k)
    queries = np.stack(queries).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    sims = queries @ train_vecs.T
    top_idx = np.argpartition(-sims, args.topk, axis=1)[:, :args.topk]
    for i in range(len(top_idx)):
        order = np.argsort(-sims[i, top_idx[i]])
        top_idx[i] = top_idx[i][order]
    top_sims = np.take_along_axis(sims, top_idx, axis=1)

    # Classify each PHL protein
    failed_rows = []    # no match in top-50
    bucket_counts = Counter()
    # Buckets:
    #   A_novel: failed + PHL K-type never in training
    #   A_thin: failed + K-type has <5 training proteins
    #   B_repr: failed + K-type well-represented but no near neighbour has it
    #   C_near_miss: matched at top-50 but not top-5 (deep search needed)
    #   ok_top1 / ok_top5 / ok_top10: matched nearby

    for i, pid in enumerate(q_pids):
        true_k = q_true_k[i]
        first_match = None
        top_neighbors = []
        for j in range(args.topk):
            nbr_md5 = train_md5s[top_idx[i, j]]
            nbr_k = train_md5_k.get(nbr_md5, set())
            top_neighbors.append((nbr_md5, nbr_k, float(top_sims[i, j])))
            if first_match is None and (true_k & nbr_k):
                first_match = j + 1

        if first_match is None:
            # Failed. Classify by why.
            phl_k_in_train = any(train_k_all.get(k, 0) > 0 for k in true_k)
            phl_k_thin = all(train_k_all.get(k, 0) < 5 for k in true_k)
            if not phl_k_in_train:
                bucket = 'A_novel'
            elif phl_k_thin:
                bucket = 'A_thin'
            else:
                bucket = 'B_repr'
            failed_rows.append({
                'protein_id': pid, 'md5': q_md5s[i],
                'true_k': ','.join(sorted(true_k)),
                'true_k_train_counts': ';'.join(
                    f'{k}:{train_k_all.get(k, 0)}' for k in sorted(true_k)),
                'nearest_sim': f'{top_neighbors[0][2]:.4f}',
                'top5_neighbors_k': ' | '.join(
                    ','.join(sorted(k)) if k else '(none)'
                    for _, k, _ in top_neighbors[:5]),
                'bucket': bucket,
            })
            bucket_counts[bucket] += 1
        elif first_match == 1:
            bucket_counts['ok_top1'] += 1
        elif first_match <= 5:
            bucket_counts['ok_top5'] += 1
        elif first_match <= 10:
            bucket_counts['ok_top10'] += 1
        else:
            bucket_counts['C_near_miss'] += 1

    # ===== AUDIT C: training label diversity =====
    multi_k = Counter()
    for md5, ks in train_md5_k.items():
        multi_k[len(ks)] += 1

    # ===== WRITE OUTPUTS =====
    summary_path = os.path.join(args.out_dir, 'phl_label_audit.txt')
    with open(summary_path, 'w') as f:
        def w(s=''): f.write(s + '\n')

        w('PHL LABEL AUDIT (ESM-2 650M mean)')
        w('=' * 72)
        w()
        w(f'PHL proteins with K labels:          {len(phl_k):,}')
        w(f'  resolved to embeddings:            {len(q_pids):,}')
        w(f'Training md5s with K labels:         {len(train_md5_k):,}')
        w(f'Distinct K-types in PHL:             {len(phl_k_all):,}')
        w(f'Distinct K-types in training:        {len(train_k_all):,}')
        w()

        w('=' * 72)
        w('AUDIT A — K-TYPE TRAINING COVERAGE')
        w('=' * 72)
        w(f'PHL K-types NEVER in training:       {len(k_novel)}')
        w(f'  (these PHL proteins cannot be correctly classified without more training data)')
        for k, n in k_novel:
            w(f'    {k:<20} — {n} PHL proteins carry this label')
        w()
        w(f'PHL K-types with <5 training examples:  {len(k_thin)}')
        for k, phl_n, train_n in k_thin:
            w(f'    {k:<20} — {phl_n} PHL proteins, {train_n} training proteins')
        w()
        w(f'PHL K-types well-represented (≥5 training proteins): {len(k_present)}')
        w(f'  Top 15 by training count:')
        for k, phl_n, train_n in sorted(k_present, key=lambda x: -x[2])[:15]:
            w(f'    {k:<20} — PHL: {phl_n:>4}, training: {train_n:>5}')
        w()

        w('=' * 72)
        w('AUDIT B — PER-PHL FAILURE BUCKETS')
        w('=' * 72)
        for b in ('ok_top1', 'ok_top5', 'ok_top10', 'C_near_miss', 'A_novel', 'A_thin', 'B_repr'):
            n = bucket_counts.get(b, 0)
            pct = 100 * n / len(q_pids) if q_pids else 0
            w(f'  {b:<15} {n:>4} / {len(q_pids)}  ({pct:5.1f}%)')
        w()
        w('Bucket key:')
        w('  ok_top1/5/10     = nearest matching neighbour found within top 1/5/10')
        w('  C_near_miss      = match found but only deep in top-50')
        w('  A_novel          = PHL K-type is NEVER in training  (unfixable w/o more data)')
        w('  A_thin           = PHL K-type has <5 training examples  (data-limited)')
        w('  B_repr           = PHL K-type well-represented but embeddings fail to bring it near '
          '(representation problem)')
        w()

        w('=' * 72)
        w('AUDIT C — TRAINING LABEL DIVERSITY (K-types per md5)')
        w('=' * 72)
        total_train = sum(multi_k.values())
        for n_k in sorted(multi_k):
            cnt = multi_k[n_k]
            pct = 100 * cnt / total_train
            w(f'  {n_k} K-type{"s" if n_k != 1 else " "}:     {cnt:>5} md5s ({pct:5.1f}%)')
        w()

        w('=' * 72)
        w('SAMPLE — 10 FAILING PHL PROTEINS (bucket B_repr)')
        w('=' * 72)
        b_repr = [r for r in failed_rows if r['bucket'] == 'B_repr']
        for r in b_repr[:10]:
            w(f"  {r['protein_id']}  true K: {r['true_k']} "
              f"(train counts: {r['true_k_train_counts']})")
            w(f"    nearest sim: {r['nearest_sim']}")
            w(f"    top-5 neighbour K-types: {r['top5_neighbors_k']}")
            w()

    # Failed rows CSV
    failed_csv = os.path.join(args.out_dir, 'phl_label_audit_failed.csv')
    if failed_rows:
        with open(failed_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(failed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(failed_rows)

    print(f'\nWrote {summary_path}')
    print(f'Wrote {failed_csv} ({len(failed_rows)} failing proteins)')

    print('\n=== BUCKET SUMMARY ===')
    for b in ('ok_top1', 'ok_top5', 'ok_top10', 'C_near_miss',
              'A_novel', 'A_thin', 'B_repr'):
        n = bucket_counts.get(b, 0)
        print(f'  {b:<15} {n:>4}  ({100*n/len(q_pids):5.1f}%)')


if __name__ == '__main__':
    main()
