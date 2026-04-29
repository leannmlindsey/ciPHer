"""Assign proper cluster IDs to v4 additions using mmseqs2 best-hits
+ union-find over BOTH v4-vs-cipher AND v4-vs-v4 hits.

For each new protein in v4 additions, at each threshold T ∈
{30, 40, 50, 60, 70, 80, 85, 90, 95}:
  1. If it has a cipher hit ≥T% → inherit the best cipher hit's
     cluster ID at T (top priority).
  2. Else if it has a v4-self hit ≥T% to another v4 addition → join
     that addition's connected component (via union-find).
  3. Else → genuine singleton at T (no homolog at this identity).

Connected components of v4 additions:
  - If a component contains members anchored to cipher clusters
    (case 1), all unanchored members in the component INHERIT the
    most-common anchor cluster (avoids merging distinct cipher
    clusters via v4-bridge edges).
  - If a component is unanchored (no cipher hits among its members),
    all members share a freshly-issued cluster ID.

Inputs:
  --mmseqs-cipher-results: v4_additions vs candidates.faa (m8 format)
  --mmseqs-self-results:   v4_additions vs v4_additions  (m8 format,
                           self-hits q==t are filtered out)
  --existing-clusters:     candidates_clusters.tsv
  --v4-additions:          candidates_v4_additions.faa

Output:
  --out:        candidates_clusters_v4.tsv (existing rows verbatim +
                1,067 new rows)
  --manifest:   cluster_assignment_manifest.json

Usage:
  python scripts/data_prep/assign_v4_clusters_from_mmseqs.py \\
      --mmseqs-cipher-results /tmp/v4_additions_vs_candidates.m8 \\
      --mmseqs-self-results /tmp/v4_self_search.m8 \\
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


def load_self_hits(mmseqs_path):
    """Parse mmseqs2 v4-self search results. Filter q==t.
    Returns list of (query, target, fident) tuples."""
    out = []
    with open(mmseqs_path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 12:
                continue
            q, t = parts[0], parts[1]
            if q == t:
                continue
            try:
                fident = float(parts[2])
            except ValueError:
                continue
            out.append((q, t, fident))
    return out


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def components(self):
        from collections import defaultdict
        d = defaultdict(list)
        for x in self.parent:
            d[self.find(x)].append(x)
        return d


def assign_at_threshold(new_ids, best_cipher_hits, self_hits,
                        pid_to_clusters, T, max_existing_idx):
    """Return {v4_pid: cluster_id_at_T} and (n_inherited, n_v4_shared, n_singleton)."""
    # Pass 1: anchor each v4 addition to a cipher cluster if best-cipher
    # hit identity ≥ T%
    anchor = {}
    for pid in new_ids:
        bh = best_cipher_hits.get(pid)
        if bh is None:
            continue
        target, fident, _ = bh
        if fident * 100.0 >= T:
            cluster = pid_to_clusters.get(target, {}).get(T)
            if cluster is not None:
                anchor[pid] = cluster

    # Pass 2: union-find on v4 additions, edges = self-hits at ≥T%
    uf = UnionFind(new_ids)
    for q, t, fident in self_hits:
        if q in uf.parent and t in uf.parent and fident * 100.0 >= T:
            uf.union(q, t)

    # Pass 3: assign cluster IDs per component
    next_idx = max_existing_idx + 1
    cluster_at_T = {}
    n_inherited = 0
    n_v4_shared = 0  # part of a multi-member v4-only component
    n_singleton = 0  # a single-member v4-only component
    for root, members in uf.components().items():
        # Collect all anchor clusters present in this component
        component_anchors = [anchor[m] for m in members if m in anchor]
        if component_anchors:
            # Inherit the most-common anchor (handles edge-case where
            # v4-self bridge connects different cipher clusters).
            from collections import Counter
            chosen = Counter(component_anchors).most_common(1)[0][0]
            for m in members:
                cluster_at_T[m] = chosen
                n_inherited += 1
        else:
            # No anchor: assign fresh shared cluster ID
            new_id = f'cl{T}_{next_idx}'
            next_idx += 1
            for m in members:
                cluster_at_T[m] = new_id
            if len(members) == 1:
                n_singleton += len(members)
            else:
                n_v4_shared += len(members)
    return cluster_at_T, (n_inherited, n_v4_shared, n_singleton)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mmseqs-cipher-results', required=True,
                   help='mmseqs2 easy-search v4_additions vs candidates.faa (m8)')
    p.add_argument('--mmseqs-self-results', required=True,
                   help='mmseqs2 easy-search v4_additions vs v4_additions (m8)')
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

    print(f'Loading mmseqs cipher-vs-v4 best hits from {args.mmseqs_cipher_results}')
    best_cipher_hits = load_best_hits(args.mmseqs_cipher_results)
    print(f'  → {len(best_cipher_hits):,} v4 queries with ≥1 cipher hit '
          f'(out of {len(new_ids):,})')

    print(f'Loading mmseqs v4-self hits from {args.mmseqs_self_results}')
    self_hits = load_self_hits(args.mmseqs_self_results)
    print(f'  → {len(self_hits):,} v4-vs-v4 hits (excluding self-pairs)')

    # Per-threshold assignment
    print('\nPer-threshold assignment (union-find: cipher anchor + v4-self edges):')
    print(f'{"thresh":<8} {"inherit":>10} {"v4-shared":>10} {"true-single":>12}')
    print('-' * 50)
    new_rows_per_t = {}
    stats_per_t = {}
    for T in THRESHOLDS:
        cluster_at_T, (n_inh, n_shared, n_single) = assign_at_threshold(
            new_ids, best_cipher_hits, self_hits, pid_to_clusters, T,
            max_per_t[T])
        new_rows_per_t[T] = cluster_at_T
        stats_per_t[T] = {
            'inherited_from_existing': n_inh,
            'in_v4_only_shared_cluster': n_shared,
            'true_singleton': n_single,
        }
        print(f'cl{T:<6} {n_inh:>10} {n_shared:>10} {n_single:>12}')

    # Build per-row output: protein_id, cl30, cl40, ..., cl95
    new_rows = []
    for pid in new_ids:
        row = [pid] + [new_rows_per_t[t][pid] for t in THRESHOLDS]
        new_rows.append(row)

    # Write output: existing rows verbatim + new rows
    print(f'\nWriting {args.out}')
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
        'method': ('mmseqs2 best-hit cipher anchor + union-find on '
                    'v4-self hits; cipher-anchored components inherit '
                    'most-frequent anchor cluster, unanchored components '
                    'share a fresh cluster ID, isolated v4 additions '
                    'become true singletons'),
        'inputs': {
            'mmseqs_cipher_results': args.mmseqs_cipher_results,
            'mmseqs_self_results': args.mmseqs_self_results,
            'existing_clusters': args.existing_clusters,
            'v4_additions': args.v4_additions,
            'n_v4_additions': len(new_ids),
            'n_existing_clusters_rows': n_copied,
            'n_cipher_hits': len(best_cipher_hits),
            'n_v4_self_hits': len(self_hits),
        },
        'assignment_stats': {
            f'cl{t}': stats_per_t[t]
            for t in THRESHOLDS
        },
        'output': {
            'path': args.out,
            'n_rows': n_copied + len(new_rows),
        },
    }
    with open(args.manifest, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nWrote manifest: {args.manifest}')

    print('\n=== Done. v4 clusters: cipher-anchored + v4-shared + true singletons. ===')


if __name__ == '__main__':
    main()
