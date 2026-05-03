"""Build the broader-PHL phage_protein_mapping.csv from the phold per-CDS
predictions TSV — for agent 4's H5 (broader-filter PHL eval) test.

The strict 8/8 PHL mapping has 256 proteins across 100 paper-aligned phages
(8 RBP detection tools all agreed). The broader (phold-only) set has the
same 100 phages but every phold-predicted protein in the FASTA — ~32x more
proteins per phage.

Output schema matches the existing strict mapping exactly:
    matrix_phage_name,protein_id

Usage:
    python scripts/analysis/build_broader_phl_mapping.py \\
        --fasta data/validation_data/PI_INFO/phagehostlearn_phold_aa.fasta \\
        --phold-tsv data/validation_data/PI_INFO/phagehostlearn_phold_per_cds_predictions.tsv \\
        --strict-mapping data/validation_data/HOST_RANGE/PhageHostLearn/metadata/phage_protein_mapping.csv \\
        --out data/validation_data/HOST_RANGE/PhageHostLearn/metadata/phage_protein_mapping_broad_phl.csv

The --strict-mapping arg is the source of truth for which phages to
include (the 100 paper-aligned phages we evaluate on). Any phage in the
phold TSV that's not in the strict mapping is dropped — the comparison
must be on the same phage denominator.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def parse_fasta_ids(fasta_path):
    """Return the set of FASTA record IDs (first whitespace-delimited token)."""
    ids = set()
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                ids.add(line[1:].split()[0])
    return ids


def parse_phold_tsv(tsv_path):
    """Return dict: cds_id -> contig_id."""
    out = {}
    with open(tsv_path) as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            cds_id = row.get('cds_id', '').strip()
            contig_id = row.get('contig_id', '').strip()
            if cds_id and contig_id:
                out[cds_id] = contig_id
    return out


def parse_strict_phages(strict_csv_path):
    """Return set of matrix_phage_name values from the strict mapping."""
    phages = set()
    with open(strict_csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            ph = row.get('matrix_phage_name', '').strip()
            if ph:
                phages.add(ph)
    return phages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fasta', required=True,
                    help='broader-PHL FASTA (phold-predicted proteins)')
    ap.add_argument('--phold-tsv', required=True,
                    help='per-CDS phold predictions TSV with contig_id + cds_id columns')
    ap.add_argument('--strict-mapping', required=True,
                    help='existing strict phage_protein_mapping.csv — source of truth '
                         'for the 100 paper-aligned phages we evaluate on')
    ap.add_argument('--out', required=True,
                    help='output broader phage_protein_mapping CSV path')
    args = ap.parse_args()

    print(f'[load] FASTA proteins from {args.fasta}')
    fasta_ids = parse_fasta_ids(args.fasta)
    print(f'  {len(fasta_ids):,} protein records in broader FASTA')

    print(f'[load] phold per-CDS TSV from {args.phold_tsv}')
    cds_to_contig = parse_phold_tsv(args.phold_tsv)
    print(f'  {len(cds_to_contig):,} CDS rows in phold TSV')

    print(f'[load] strict mapping phages from {args.strict_mapping}')
    strict_phages = parse_strict_phages(args.strict_mapping)
    print(f'  {len(strict_phages):,} phages in strict mapping (paper-aligned)')

    # Build broader mapping: every FASTA protein whose phold contig is one
    # of the strict-mapping phages
    rows = []
    n_no_phold_row = 0
    n_phage_outside_strict = 0
    by_phage = defaultdict(int)
    for cds_id in sorted(fasta_ids):
        contig_id = cds_to_contig.get(cds_id)
        if contig_id is None:
            n_no_phold_row += 1
            continue
        if contig_id not in strict_phages:
            n_phage_outside_strict += 1
            continue
        rows.append((contig_id, cds_id))
        by_phage[contig_id] += 1

    print()
    print(f'[stats] {n_no_phold_row:,} FASTA proteins with no phold TSV row '
          f'(likely indexing mismatch; investigate if non-trivial)')
    print(f'[stats] {n_phage_outside_strict:,} FASTA proteins on phages outside '
          f'the strict mapping (dropped — kept on same 100-phage denominator)')
    print(f'[stats] {len(rows):,} (phage, protein) rows in broader mapping')
    print(f'[stats] {len(by_phage):,} phages with at least one broader protein '
          f'(of {len(strict_phages)} strict phages)')
    if by_phage:
        counts = list(by_phage.values())
        print(f'[stats] proteins per phage: min={min(counts)} median={sorted(counts)[len(counts)//2]} '
              f'max={max(counts)} mean={sum(counts)/len(counts):.1f}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['matrix_phage_name', 'protein_id'])
        for ph, prot in sorted(rows):
            w.writerow([ph, prot])
    print(f'[write] {args.out}')


if __name__ == '__main__':
    main()
