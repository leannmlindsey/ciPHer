"""Assign proper cluster IDs to v4 additions using mmseqs2 best-hit
identity → existing cluster mapping.

Replaces the singleton-cluster shortcut in
`build_v4_companion_files.py`. For each new protein in v4 additions:
  1. Look up its best alignment hit in cipher's existing candidates.faa
     (from mmseqs2 easy-search output).
  2. For each cluster threshold T ∈ {30, 40, 50, 60, 70, 80, 85, 90, 95}:
     - If best-hit identity ≥ T% → inherit the best-hit's cluster ID at T.
     - Else → genuine singleton at T (no existing cluster matches at this
       identity).

Inputs:
  --mmseqs-results: mmseqs2 easy-search TSV (default outfmt: query,
                    target, fident, alnlen, mismatch, gapopen, qstart,
                    qend, tstart, tend, evalue, bits). Best hit per
                    query is selected by max fident, ties broken by bits.
  --existing-clusters: data/training_data/metadata/candidates_clusters.tsv
  --v4-additions: data/training_data/metadata/highconf_v4/candidates_v4_additions.faa
  --out: data/training_data/metadata/highconf_v4/candidates_clusters_v4.tsv
         (overwrites — replaces the singleton-trick file from earlier prep)

The output preserves all existing 143,240 rows verbatim and appends 1,067
new rows with proper cluster assignments.

Usage:
  python scripts/data_prep/assign_v4_clusters_from_mmseqs.py \\
      --mmseqs-results /work/hdd/.../v4_additions_vs_candidates.m8 \\
      --existing-clusters data/training_data/metadata/candidates_clusters.tsv \\
      --v4-additions data/training_data/metadata/highconf_v4/candidates_v4_additions.faa \\
      --out data/training_data/metadata/highconf_v4/candidates_clusters_v4.tsv
"""

import argparse
import csv
import json
import os
from collections import defaultdict


THRESHOLDS = [30, 40, 50, 60, 70, 80, 85, 90, 95]


def parse_fasta_ids(path):
    ids = []
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                ids.append(line[1:].split()[0])
    return ids


def load_existing_clusters(path):
    """Return {protein_id: {threshold_int: cluster_id_str}} and the max
    cluster index per threshold (for assigning new singletons)."""
    pid_to_clusters = {}
    max_per_t = {t: -1 for t in THRESHOLDS}
    with open(path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if not parts or len(parts) < 2:
                continue
            pid = parts[0]
            clusters = {}
            for col in parts[1:]:
                if not col.startswith('cl') or '_' not in col:
                    continue
                t_str, idx_str = col.split('_', 1)
                try:
                    t = int(t_str[2:])
                    idx = int(idx_str)
                except ValueError:
                    continue
                if t in max_per_t:
                    clusters[t] = col
                    if idx > max_per_t[t]:
                        max_per_t[t] = idx
            pid_to_clusters[pid] = clusters
    return pid_to_clusters, max_per_t


def load_best_hits(mmseqs_path):
    """Parse mmseqs2 easy-search output. Default columns:
       query, target, fident, alnlen, mismatch, gapopen, qstart, qend,
       tstart, tend, evalue, bits

    Returns {query: (best_target, best_fident, best_bits)}."""
    best = {}
    with open(mmseqs_path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 12:
                continue
            q, t = parts[0], parts[1]
            try:
                fident = float(parts[2])
                bits = float(parts[11])
            except ValueError:
                continue
            cur = best.get(q)
            if cur is None or fident > cur[1] or (fident == cur[1] and bits > cur[2]):
                best[q] = (t, fident, bits)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mmseqs-results', required=True,
                   help='mmseqs2 easy-search output (m8/blasttab format)')
    p.add_argument('--existing-clusters',
                   default='data/training_data/metadata/candidates_clusters.tsv')
    p.add_argument('--v4-additions',
                   default='data/training_data/metadata/highconf_v4/candidates_v4_additions.faa')
    p.add_argument('--out',
                   default='data/training_data/metadata/highconf_v4/candidates_clusters_v4.tsv')
    p.add_argument('--manifest',
                   default='data/training_data/metadata/highconf_v4/cluster_assignment_manifest.json')
    args = p.parse_args()

    print(f'Loading v4 addition IDs from {args.v4_additions}')
    new_ids = parse_fasta_ids(args.v4_additions)
    print(f'  → {len(new_ids):,} new protein IDs')

    print(f'Loading existing clusters from {args.existing_clusters}')
    pid_to_clusters, max_per_t = load_existing_clusters(args.existing_clusters)
    print(f'  → {len(pid_to_clusters):,} existing proteins, '
          f'max cluster IDs: '
          + ', '.join(f'cl{t}_{max_per_t[t]}' for t in THRESHOLDS))

    print(f'Loading mmseqs2 best-hit table from {args.mmseqs_results}')
    best_hits = load_best_hits(args.mmseqs_results)
    print(f'  → {len(best_hits):,} queries with at least one hit '
          f'(out of {len(new_ids):,} v4 additions)')

    # Stats: how many new proteins have hits at each threshold
    n_with_hit_at = {t: 0 for t in THRESHOLDS}
    n_orphan_at_threshold_30 = 0  # no hit at all (≥30% identity)
    for pid in new_ids:
        bh = best_hits.get(pid)
        if bh is None:
            n_orphan_at_threshold_30 += 1
            continue
        ident_pct = bh[1] * 100.0
        for t in THRESHOLDS:
            if ident_pct >= t:
                n_with_hit_at[t] += 1
    print('  hits per threshold:')
    for t in THRESHOLDS:
        n = n_with_hit_at[t]
        pct = 100.0 * n / len(new_ids) if new_ids else 0
        print(f'    cl{t}: {n:>5}/{len(new_ids)} ({pct:4.1f}%) get inherited cluster; '
              f'{len(new_ids) - n} singletons')
    print(f'  orphans (no hit at all, all 9 thresholds singleton): '
          f'{n_orphan_at_threshold_30}')

    # Assign clusters per new protein per threshold
    next_singleton_idx = {t: max_per_t[t] + 1 for t in THRESHOLDS}
    new_rows = []
    for pid in new_ids:
        row = [pid]
        bh = best_hits.get(pid)
        for t in THRESHOLDS:
            assigned = None
            if bh is not None:
                target, fident, _ = bh
                if fident * 100.0 >= t:
                    assigned = pid_to_clusters.get(target, {}).get(t)
            if assigned is None:
                assigned = f'cl{t}_{next_singleton_idx[t]}'
                next_singleton_idx[t] += 1
            row.append(assigned)
        new_rows.append(row)

    # Count assignment provenance per threshold
    n_inherited = {t: 0 for t in THRESHOLDS}
    n_new_singleton = {t: 0 for t in THRESHOLDS}
    for i, pid in enumerate(new_ids):
        for j, t in enumerate(THRESHOLDS):
            cluster_id = new_rows[i][j + 1]
            idx = int(cluster_id.split('_', 1)[1])
            if idx > max_per_t[t]:
                n_new_singleton[t] += 1
            else:
                n_inherited[t] += 1

    # Write output: existing rows verbatim + new rows
    print(f'Writing {args.out}')
    with open(args.out, 'w') as out, open(args.existing_clusters) as f:
        n_copied = 0
        for line in f:
            out.write(line)
            n_copied += 1
        for row in new_rows:
            out.write('\t'.join(row) + '\n')
    print(f'  existing rows copied: {n_copied:,}')
    print(f'  new rows added:       {len(new_rows):,}')
    print(f'  v4 total rows:        {n_copied + len(new_rows):,}')

    # Manifest
    manifest = {
        'method': 'mmseqs2 easy-search best-hit, threshold-aware cluster inheritance',
        'inputs': {
            'mmseqs_results': args.mmseqs_results,
            'existing_clusters': args.existing_clusters,
            'v4_additions': args.v4_additions,
            'n_v4_additions': len(new_ids),
            'n_existing_clusters_rows': n_copied,
        },
        'assignment_stats': {
            f'cl{t}': {
                'inherited_from_existing_cluster': n_inherited[t],
                'new_singleton': n_new_singleton[t],
            }
            for t in THRESHOLDS
        },
        'orphan_count': n_orphan_at_threshold_30,
        'output': {
            'path': args.out,
            'n_rows': n_copied + len(new_rows),
        },
    }
    with open(args.manifest, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nWrote manifest: {args.manifest}')

    print('\n=== Done. v4 clusters file now has proper assignments, not singletons. ===')


if __name__ == '__main__':
    main()
