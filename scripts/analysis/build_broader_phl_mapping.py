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


def parse_fasta_records(fasta_path):
    """Return list of (contig_id, cds_id) tuples from FASTA headers.

    The phold-derived broader-PHL FASTA uses ``>contig_id:cds_id`` headers
    (e.g. ``>A1a:DGCOCGBI_00001``). If a header has no colon, it falls back
    to (None, full_token) so the caller can look up the contig via the
    phold TSV.
    """
    out = []
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith('>'):
                continue
            tok = line[1:].split()[0]
            if ':' in tok:
                contig_id, cds_id = tok.split(':', 1)
                out.append((contig_id, cds_id))
            else:
                out.append((None, tok))
    return out


def parse_phold_tsv(tsv_path, function_filter=None):
    """Return dict: cds_id -> contig_id, optionally filtered by phold function.

    function_filter is a set of allowed `function` column values (e.g.
    {'tail'}). If None, all CDS rows are returned.
    """
    out = {}
    n_total = 0
    n_kept = 0
    with open(tsv_path) as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            n_total += 1
            cds_id = row.get('cds_id', '').strip()
            contig_id = row.get('contig_id', '').strip()
            func = row.get('function', '').strip()
            if function_filter is not None and func not in function_filter:
                continue
            if cds_id and contig_id:
                out[cds_id] = contig_id
                n_kept += 1
    if function_filter is not None:
        print(f'  filter: function in {sorted(function_filter)} -> '
              f'kept {n_kept:,} of {n_total:,} rows')
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
    ap.add_argument('--function-filter', default='tail',
                    help='comma-separated phold function values to keep '
                         '(default "tail" = the RBP-broad set; pass empty '
                         'string to keep all CDS regardless of function)')
    args = ap.parse_args()

    func_filter = None
    if args.function_filter.strip():
        func_filter = set(s.strip() for s in args.function_filter.split(','))

    print(f'[load] FASTA records from {args.fasta}')
    fasta_records = parse_fasta_records(args.fasta)
    print(f'  {len(fasta_records):,} protein records in broader FASTA')
    n_with_contig_in_header = sum(1 for c, _ in fasta_records if c is not None)
    print(f'  {n_with_contig_in_header:,} headers have ":contig:cds_id" form '
          f'(rest will fall back to phold TSV lookup)')

    print(f'[load] phold per-CDS TSV from {args.phold_tsv}')
    cds_to_contig = parse_phold_tsv(args.phold_tsv, function_filter=func_filter)
    print(f'  {len(cds_to_contig):,} CDS rows kept after function filter')

    print(f'[load] strict mapping phages from {args.strict_mapping}')
    strict_phages = parse_strict_phages(args.strict_mapping)
    print(f'  {len(strict_phages):,} phages in strict mapping')

    # Build broader mapping: every FASTA protein that
    #   (a) passes the phold function filter (kept in cds_to_contig), AND
    #   (b) lives on a phage that's in the strict mapping.
    #
    # protein_id MUST match the FASTA header token exactly so the eval
    # pipeline's pid->md5 lookup succeeds. Headers are "contig:cds" form,
    # so emit "contig:cds" as protein_id (not bare cds_id).
    rows = []
    n_no_contig = 0
    n_filtered_out = 0
    n_phage_outside_strict = 0
    by_phage = defaultdict(int)
    for contig_id, cds_id in fasta_records:
        # Apply phold function filter: cds_id must appear in (filtered) phold TSV
        if func_filter is not None and cds_id not in cds_to_contig:
            n_filtered_out += 1
            continue
        if contig_id is None:
            contig_id = cds_to_contig.get(cds_id)
            protein_id = cds_id
        else:
            protein_id = f'{contig_id}:{cds_id}'
        if contig_id is None:
            n_no_contig += 1
            continue
        if contig_id not in strict_phages:
            n_phage_outside_strict += 1
            continue
        rows.append((contig_id, protein_id))
        by_phage[contig_id] += 1

    print()
    print(f'[stats] {n_filtered_out:,} FASTA proteins dropped by phold function filter')
    print(f'[stats] {n_no_contig:,} FASTA proteins with no resolvable contig_id '
          f'(investigate if non-trivial)')
    print(f'[stats] {n_phage_outside_strict:,} FASTA proteins on phages outside '
          f'the strict mapping (dropped — kept on same phage denominator)')
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
