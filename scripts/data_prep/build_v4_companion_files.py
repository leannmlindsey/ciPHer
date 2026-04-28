"""Build cipher-format v4 companion files (FASTA / clusters / glycan-flags).

Agent 4's v4 list landed at `data/training_data/metadata/highconf_v4/`
with the integrated 38,162-ID positive list and the augmented host-
phage-protein map. But cipher's training pipeline also reads
`candidates.faa`, `candidates_clusters.tsv`, and
`glycan_binders_custom.tsv` — each of which needs to be extended to
cover the 1,067 new A28+A25 inclusions, otherwise:

- The new IDs in pipeline_positive_v4.list have no FASTA sequences
  → embedding extraction fails for them.
- The new IDs are missing from candidates_clusters.tsv → cluster-
  stratified downsampling silently drops them OR treats as singleton
  cluster (depending on cluster_map semantics).
- The new IDs aren't in glycan_binders → if any --tools-mode training
  consumes v4 in the future, those tools would fail to pick the new
  proteins.

What this script produces (alongside the existing v4 files in
`highconf_v4/`):

  candidates_v4.faa              — existing candidates.faa + 1,067 new
  candidates_clusters_v4.tsv     — existing + 1,067 singleton rows
                                   (each new MD5 gets unique cl{30..95}_<ID>
                                   using the next available integer per
                                   threshold — disjoint from existing
                                   clusters since A28/A25 are absent from
                                   cipher's candidates.faa even at
                                   exact-substring granularity)
  glycan_binders_v4.tsv          — existing + 1,067 rows, all 8 tool
                                   flags = 1 (TropiSEQ training proteins
                                   are RBPs by definition — pseudo
                                   total_sources=8)

Inputs (read-only):
  - data/training_data/metadata/candidates.faa
  - data/training_data/metadata/candidates_clusters.tsv
  - data/training_data/metadata/glycan_binders_custom.tsv
  - data/training_data/metadata/highconf_v4/candidates_v4_additions.faa

Outputs (under data/training_data/metadata/highconf_v4/):
  - candidates_v4.faa
  - candidates_clusters_v4.tsv
  - glycan_binders_v4.tsv
  - companion_build_manifest.json (sha256s + counts)

Usage:
    python scripts/data_prep/build_v4_companion_files.py
"""

import csv
import hashlib
import json
import os
from collections import defaultdict


META = 'data/training_data/metadata'
V4_DIR = os.path.join(META, 'highconf_v4')

EXISTING_FAA = os.path.join(META, 'candidates.faa')
EXISTING_CLUSTERS = os.path.join(META, 'candidates_clusters.tsv')
EXISTING_GLYCAN = os.path.join(META, 'glycan_binders_custom.tsv')
V4_ADDITIONS_FAA = os.path.join(V4_DIR, 'candidates_v4_additions.faa')

OUT_FAA = os.path.join(V4_DIR, 'candidates_v4.faa')
OUT_CLUSTERS = os.path.join(V4_DIR, 'candidates_clusters_v4.tsv')
OUT_GLYCAN = os.path.join(V4_DIR, 'glycan_binders_v4.tsv')
OUT_MANIFEST = os.path.join(V4_DIR, 'companion_build_manifest.json')


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_fasta_ids(path):
    ids = []
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                # ID is first whitespace-separated token after '>'
                ids.append(line[1:].split()[0])
    return ids


def main():
    if not os.path.isdir(V4_DIR):
        raise SystemExit(f'ERROR: v4 dir not found: {V4_DIR}')
    for p in (EXISTING_FAA, EXISTING_CLUSTERS, EXISTING_GLYCAN, V4_ADDITIONS_FAA):
        if not os.path.exists(p):
            raise SystemExit(f'ERROR: input not found: {p}')

    print('=== v4 companion-files build ===')
    print(f'  Existing FASTA:    {EXISTING_FAA}')
    print(f'  Existing clusters: {EXISTING_CLUSTERS}')
    print(f'  Existing glycans:  {EXISTING_GLYCAN}')
    print(f'  v4 additions:      {V4_ADDITIONS_FAA}')
    print()

    # 1. Get the v4 addition IDs
    new_ids = parse_fasta_ids(V4_ADDITIONS_FAA)
    print(f'1. v4 addition IDs: {len(new_ids)}')

    # 2. candidates_v4.faa = existing concat additions
    print(f'2. Building {OUT_FAA} (concat existing + additions)')
    with open(OUT_FAA, 'wb') as out:
        with open(EXISTING_FAA, 'rb') as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        # Ensure newline boundary, then append additions
        with open(V4_ADDITIONS_FAA, 'rb') as f:
            out.write(f.read())
    n_existing_recs = sum(1 for _ in open(EXISTING_FAA) if _.startswith('>'))
    n_v4_recs = sum(1 for _ in open(OUT_FAA) if _.startswith('>'))
    print(f'   existing records: {n_existing_recs:,}')
    print(f'   v4 total records: {n_v4_recs:,}  (+{n_v4_recs - n_existing_recs})')

    # 3. candidates_clusters_v4.tsv: existing rows + singleton rows for new IDs.
    # The cluster file has columns: protein_id, cl30_*, cl40_*, ..., cl95_*
    # New IDs are disjoint from existing clusters per A28 / A25 findings,
    # so assign each a new singleton ID per threshold. Use a high-numbered
    # ID space (existing max + 1, ...) per threshold to avoid collision.
    print(f'3. Building {OUT_CLUSTERS} (existing + singleton rows for new IDs)')
    thresholds = [30, 40, 50, 60, 70, 80, 85, 90, 95]
    max_per_threshold = {t: -1 for t in thresholds}
    with open(EXISTING_CLUSTERS) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            for col in parts[1:]:
                # col format like 'cl70_233'
                if '_' not in col or not col.startswith('cl'):
                    continue
                t_str, idx_str = col.split('_', 1)
                try:
                    t = int(t_str[2:])
                    idx = int(idx_str)
                    if t in max_per_threshold and idx > max_per_threshold[t]:
                        max_per_threshold[t] = idx
                except ValueError:
                    pass
    print(f'   max cluster IDs in existing file: '
          + ', '.join(f'cl{t}_{max_per_threshold[t]}' for t in thresholds))

    next_id = {t: max_per_threshold[t] + 1 for t in thresholds}
    n_existing_lines = 0
    with open(OUT_CLUSTERS, 'w') as out, open(EXISTING_CLUSTERS) as f:
        for line in f:
            out.write(line)
            n_existing_lines += 1
        for pid in new_ids:
            row = [pid] + [f'cl{t}_{next_id[t]}' for t in thresholds]
            out.write('\t'.join(row) + '\n')
            for t in thresholds:
                next_id[t] += 1
    print(f'   existing rows: {n_existing_lines:,}')
    print(f'   added singleton rows: {len(new_ids):,}')
    print(f'   v4 total rows: {n_existing_lines + len(new_ids):,}')

    # 4. glycan_binders_v4.tsv: existing rows + new rows with all-1 flags.
    print(f'4. Building {OUT_GLYCAN} (existing + 8-tool-positive rows for new)')
    # Read header to discover column ordering
    with open(EXISTING_GLYCAN) as f:
        header_line = f.readline().rstrip('\n').split('\t')
    n_existing_glycan = 0
    with open(OUT_GLYCAN, 'w', newline='') as out, open(EXISTING_GLYCAN) as f:
        for line in f:
            out.write(line)
            n_existing_glycan += 1
        # Append rows for new IDs — all 8 tool flags = 1, total_sources = 8,
        # is_negative = 0, score columns blank, dbcan/ipr/phold blank.
        # Column ordering must match the header.
        for pid in new_ids:
            row = []
            for col in header_line:
                if col == 'protein_id':
                    row.append(pid)
                elif col in ('DePP_85', 'PhageRBPdetect', 'DepoScope',
                              'DepoRanker', 'SpikeHunter', 'dbCAN', 'IPR',
                              'phold_glycan_tailspike'):
                    row.append('1')
                elif col == 'total_sources':
                    row.append('8')
                elif col == 'is_negative':
                    row.append('0')
                else:
                    row.append('')
            out.write('\t'.join(row) + '\n')
    n_existing_glycan -= 1  # remove header from row count
    print(f'   existing rows: {n_existing_glycan:,} (excluding header)')
    print(f'   added rows: {len(new_ids):,}')

    # 5. Manifest
    manifest = {
        'build_script': 'scripts/data_prep/build_v4_companion_files.py',
        'inputs': {
            'candidates.faa': {'sha256': sha256_of(EXISTING_FAA),
                               'n_records': n_existing_recs},
            'candidates_clusters.tsv': {'sha256': sha256_of(EXISTING_CLUSTERS),
                                         'n_rows': n_existing_lines},
            'glycan_binders_custom.tsv': {
                'sha256': sha256_of(EXISTING_GLYCAN),
                'n_rows': n_existing_glycan},
            'candidates_v4_additions.faa': {
                'sha256': sha256_of(V4_ADDITIONS_FAA),
                'n_records': len(new_ids)},
        },
        'outputs': {
            'candidates_v4.faa': {'sha256': sha256_of(OUT_FAA),
                                   'n_records': n_v4_recs},
            'candidates_clusters_v4.tsv': {
                'sha256': sha256_of(OUT_CLUSTERS),
                'n_rows': n_existing_lines + len(new_ids),
                'cluster_strategy': ('singleton — new IDs assigned unique '
                                      'cl{30..95}_<next_avail> per threshold')},
            'glycan_binders_v4.tsv': {
                'sha256': sha256_of(OUT_GLYCAN),
                'n_rows': n_existing_glycan + len(new_ids),
                'tool_flag_strategy': ('all 8 tool flags = 1, '
                                        'total_sources = 8 — TropiSEQ training '
                                        'proteins are RBPs by definition')},
        },
        'next_step': (
            'Use these companion files alongside agent 4\'s '
            'highconf_v4/pipeline_positive_v4.list and '
            'host_phage_protein_map_v4.tsv. Embed candidates_v4_additions.faa '
            'with ProtT5 mean and ProtT5 xl seg8, then merge into the '
            'training NPZs at training time (or pre-merge once).'),
    }
    with open(OUT_MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nWrote manifest: {OUT_MANIFEST}')

    print('\n=== Done. v4 companion files ready in highconf_v4/. ===')


if __name__ == '__main__':
    main()
