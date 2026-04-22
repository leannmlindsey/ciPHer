"""Build two high-confidence positive lists from the analysis-repo f90 edges.

For each edges CSV, emits a protein-id list suitable as input to
`cipher-train --positive_list ...`. The list contains every candidate
protein whose 60%-identity cluster (cl60 column of
candidates_clusters.tsv) appears in the edges CSV. Edges CSVs live at:

    analyses/04_tsp_serotype_network/output/tsps/f90/edges_K.csv
    analyses/04_tsp_serotype_network/output/pipeline_positive/f90/edges_K.csv

in the sibling CLAUDE_PHI_DATA_ANALYSIS repository.

Usage:
    python scripts/utils/build_highconf_lists.py
"""

import argparse
import csv
import os


def load_cluster_ids(edges_csv, axis='K'):
    """Return the set of unique cluster_ids in an edges CSV."""
    ids = set()
    with open(edges_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('axis') and row['axis'] != axis:
                continue
            ids.add(row['cluster_id'].strip())
    return ids


def emit_positive_list(clusters_tsv, cluster_ids, out_path, cl_col_prefix='cl60_'):
    """Write one protein_id per line for every row whose cl60 is in cluster_ids."""
    n = 0
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(clusters_tsv) as f_in, open(out_path, 'w') as f_out:
        for line in f_in:
            parts = line.rstrip('\n').split('\t')
            if not parts or not parts[0]:
                continue
            pid = parts[0]
            cl60 = next((p for p in parts[1:] if p.startswith(cl_col_prefix)), None)
            if cl60 and cl60 in cluster_ids:
                f_out.write(pid + '\n')
                n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis-repo',
                   default='/Users/leannmlindsey/WORK/CLAUDE_PHI_DATA_ANALYSIS')
    p.add_argument('--clusters-tsv',
                   default='data/training_data/metadata/candidates_clusters.tsv')
    p.add_argument('--out-dir',
                   default='data/training_data/metadata')
    args = p.parse_args()

    edges_root = os.path.join(args.analysis_repo,
                              'analyses/04_tsp_serotype_network/output')
    specs = [
        ('tsps',              'highconf_tsp_K.list'),
        ('pipeline_positive', 'highconf_pipeline_positive_K.list'),
    ]

    for subdir, out_name in specs:
        edges_csv = os.path.join(edges_root, subdir, 'f90', 'edges_K.csv')
        out_path = os.path.join(args.out_dir, out_name)
        cluster_ids = load_cluster_ids(edges_csv, axis='K')
        n = emit_positive_list(args.clusters_tsv, cluster_ids, out_path)
        print(f'{subdir}: {len(cluster_ids):>4} high-confidence K clusters '
              f'-> {n:>6,} proteins  ->  {out_path}')


if __name__ == '__main__':
    main()
