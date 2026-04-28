"""BLAST cipher training proteins against DpoTropi's depolymerase domains.

Builds a BLAST DB from DpoTropi's `domain_seq` column (~21k entries)
and queries cipher's training FASTA against it. Each cipher training
protein either has a depolymerase-domain hit (within DpoTropi's
representational space) or doesn't.

Asks: "of the 21k cipher training proteins, how many are recognizable
depolymerases by DpoTropi's standards?" High coverage = cipher and
DpoTropi are training on the same kind of protein. Low coverage =
cipher includes a lot of non-depolymerase RBPs (spike proteins,
tail-fiber lectins) that DpoTropi excludes.

Outputs:
  results/blast/cipher_train_vs_dpotropi_domains.tsv  (raw BLAST output, fmt 6)
  results/blast/cipher_train_hit_summary.csv          (per-protein: best hit + bitscore)
  prints summary

Skips re-build / re-blast if outputs already exist (idempotent).
"""

import csv
import hashlib
import os
import subprocess
import sys

import numpy as np


import argparse
DEFAULT_CIPHER_NPZ = ('/Users/leannmlindsey/WORK/PHI_TSP/cipher/experiments/'
                      'attention_mlp/sweep_prott5_mean_cl70/training_data.npz')
CIPHER_TRAIN_NPZ = DEFAULT_CIPHER_NPZ  # overridden in main()
# Cipher's training FASTA — proteins keyed by protein_id; we'll need to
# reconstruct sequences keyed by MD5 to match training_data.
# OLD klebsiella ships a FASTA with all candidate protein sequences:
CIPHER_TRAIN_FASTA = ('/Users/leannmlindsey/WORK/PHI_TSP/phi_tsp/'
                       'klebsiella/data/candidates.faa')
DPOTROPI_TSV = ('/Users/leannmlindsey/WORK/PHI_TSP/DpoTropiSearch_zenoto_data/'
                'TropiGATv2.final_df_v2.tsv')

BLAST_OUT_DIR = 'results/blast'
DOMAIN_FASTA = f'{BLAST_OUT_DIR}/dpotropi_domain_seq.fasta'
DOMAIN_DB = f'{BLAST_OUT_DIR}/dpotropi_domain_db'
QUERY_FASTA = f'{BLAST_OUT_DIR}/cipher_train_proteins.fasta'
BLAST_TSV = f'{BLAST_OUT_DIR}/cipher_train_vs_dpotropi_domains.tsv'
SUMMARY_CSV = f'{BLAST_OUT_DIR}/cipher_train_hit_summary.csv'

BLAST_FIELDS = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch',
                'gapopen', 'qstart', 'qend', 'sstart', 'send',
                'evalue', 'bitscore']

EVALUE = '1e-10'
BITSCORE_MIN = 75


def md5(s):
    return hashlib.md5(s.encode()).hexdigest() if s else ''


def parse_fasta_md5(path, want_md5s=None):
    """Yield (md5, sequence) pairs from a FASTA. Optionally filter to
    only MD5s in `want_md5s`."""
    cur_id = None
    cur_seq = []
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                if cur_id is not None:
                    seq = ''.join(cur_seq)
                    m = md5(seq)
                    if want_md5s is None or m in want_md5s:
                        yield m, seq
                cur_id = line[1:].strip().split()[0]
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_id is not None:
            seq = ''.join(cur_seq)
            m = md5(seq)
            if want_md5s is None or m in want_md5s:
                yield m, seq


def build_dpotropi_domain_fasta():
    if os.path.exists(DOMAIN_FASTA) and os.path.getsize(DOMAIN_FASTA) > 0:
        print(f'  reuse existing {DOMAIN_FASTA}')
        return
    print(f'  building {DOMAIN_FASTA} from {DPOTROPI_TSV}')
    n_written = 0
    seen = set()  # de-dup identical domain sequences
    with open(DPOTROPI_TSV) as fin, open(DOMAIN_FASTA, 'w') as fout:
        header = next(fin).rstrip('\n').split('\t')
        prot_col = header.index('Protein_name')
        dom_col = header.index('domain_seq')
        for line in fin:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= max(prot_col, dom_col):
                continue
            pid = parts[prot_col].strip()
            domain = parts[dom_col].strip()
            if not domain:
                continue
            m = md5(domain)
            if m in seen:
                continue
            seen.add(m)
            fout.write(f'>{pid}\n{domain}\n')
            n_written += 1
    print(f'  wrote {n_written} unique domain sequences')


def build_blast_db():
    if all(os.path.exists(f'{DOMAIN_DB}.{ext}') for ext in ('phr', 'pin', 'psq')):
        print(f'  reuse existing BLAST DB {DOMAIN_DB}')
        return
    print(f'  building BLAST DB {DOMAIN_DB}')
    subprocess.run(['makeblastdb', '-in', DOMAIN_FASTA,
                     '-dbtype', 'prot', '-out', DOMAIN_DB],
                    check=True, capture_output=True)


def write_cipher_query_fasta():
    if os.path.exists(QUERY_FASTA) and os.path.getsize(QUERY_FASTA) > 0:
        print(f'  reuse existing {QUERY_FASTA}')
        return
    print(f'  building cipher training query FASTA from {CIPHER_TRAIN_FASTA}')
    npz = np.load(CIPHER_TRAIN_NPZ, allow_pickle=True)
    cipher_md5_set = set(npz['md5_list'].tolist())
    n_written = 0
    n_seen = 0
    with open(QUERY_FASTA, 'w') as fout:
        for m, seq in parse_fasta_md5(CIPHER_TRAIN_FASTA, want_md5s=cipher_md5_set):
            n_seen += 1
            fout.write(f'>{m}\n{seq}\n')
            n_written += 1
    print(f'  cipher training MD5s: {len(cipher_md5_set):,}; wrote {n_written}')


def run_blastp():
    if os.path.exists(BLAST_TSV) and os.path.getsize(BLAST_TSV) > 0:
        print(f'  reuse existing BLAST output {BLAST_TSV}')
        return
    print(f'  running blastp …')
    cmd = ['blastp',
           '-query', QUERY_FASTA,
           '-db', DOMAIN_DB,
           '-outfmt', '6',
           '-evalue', EVALUE,
           '-num_threads', '4',
           '-max_target_seqs', '1',
           '-out', BLAST_TSV]
    subprocess.run(cmd, check=True)
    print(f'  wrote {BLAST_TSV}')


def summarize_hits():
    """Per cipher-MD5, find best hit (max bitscore)."""
    best = {}  # qseqid -> dict
    with open(BLAST_TSV) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            row = dict(zip(BLAST_FIELDS, parts))
            for fld in ('pident', 'length', 'mismatch', 'gapopen',
                        'qstart', 'qend', 'sstart', 'send'):
                row[fld] = int(float(row[fld])) if row[fld] else 0
            row['evalue'] = float(row['evalue'])
            row['bitscore'] = float(row['bitscore'])
            q = row['qseqid']
            if q not in best or row['bitscore'] > best[q]['bitscore']:
                best[q] = row

    npz = np.load(CIPHER_TRAIN_NPZ, allow_pickle=True)
    cipher_md5s = set(npz['md5_list'].tolist())

    n_total = len(cipher_md5s)
    n_with_hit = sum(1 for m in cipher_md5s if m in best)
    n_above_75 = sum(1 for m in cipher_md5s
                     if m in best and best[m]['bitscore'] >= BITSCORE_MIN)

    with open(SUMMARY_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cipher_md5', 'has_blast_hit',
                    'best_hit_sseqid', 'pident', 'length', 'evalue', 'bitscore'])
        for m in sorted(cipher_md5s):
            if m in best:
                r = best[m]
                w.writerow([m, 1, r['sseqid'], r['pident'],
                            r['length'], r['evalue'], r['bitscore']])
            else:
                w.writerow([m, 0, '', '', '', '', ''])
    print(f'\nWrote {SUMMARY_CSV}')

    print()
    print('=' * 70)
    print('SUMMARY: cipher training proteins vs DpoTropi domain DB')
    print('=' * 70)
    print(f'cipher training proteins:                      {n_total:,}')
    print(f'with ANY BLAST hit at e-value ≤ {EVALUE}:       {n_with_hit:,}  '
          f'({100*n_with_hit/n_total:.1f}%)')
    print(f'with hit AND bitscore ≥ {BITSCORE_MIN}:                   {n_above_75:,}  '
          f'({100*n_above_75/n_total:.1f}%)')
    print(f'no hit (potentially non-depolymerase RBPs):    {n_total - n_with_hit:,}  '
          f'({100*(n_total-n_with_hit)/n_total:.1f}%)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cipher-npz', default=DEFAULT_CIPHER_NPZ)
    p.add_argument('--tag', default='',
                   help='Optional suffix for output paths to keep multiple runs separate')
    args = p.parse_args()
    global CIPHER_TRAIN_NPZ, QUERY_FASTA, BLAST_TSV, SUMMARY_CSV
    CIPHER_TRAIN_NPZ = args.cipher_npz
    if args.tag:
        QUERY_FASTA = f'{BLAST_OUT_DIR}/cipher_train_proteins{args.tag}.fasta'
        BLAST_TSV = f'{BLAST_OUT_DIR}/cipher_train_vs_dpotropi_domains{args.tag}.tsv'
        SUMMARY_CSV = f'{BLAST_OUT_DIR}/cipher_train_hit_summary{args.tag}.csv'
    os.makedirs(BLAST_OUT_DIR, exist_ok=True)
    print(f'Cipher training NPZ: {CIPHER_TRAIN_NPZ}')
    print('Step 1: build DpoTropi domain FASTA')
    build_dpotropi_domain_fasta()
    print('\nStep 2: build BLAST DB')
    build_blast_db()
    print('\nStep 3: write cipher training query FASTA')
    write_cipher_query_fasta()
    print('\nStep 4: run blastp')
    run_blastp()
    print('\nStep 5: summarize')
    summarize_hits()


if __name__ == '__main__':
    main()
