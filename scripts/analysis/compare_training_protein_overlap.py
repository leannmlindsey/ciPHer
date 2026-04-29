"""Cross-corpus protein overlap by MD5: cipher training ∩ DpoTropi training.

Cipher training data MD5s are computed from the candidates FASTA and
shipped in training_data.npz under `md5_list`. DpoTropi ships actual
sequences in the training TSV columns `seq` and `domain_seq`. We
hash both columns and intersect with cipher's MD5 set.

Two overlap views:
  (a) full-sequence overlap (cipher's full RBP MD5 vs DpoTropi `seq`)
  (b) domain-sequence overlap (cipher's full RBP MD5 vs DpoTropi `domain_seq`)

(a) tests whether cipher and DpoTropi train on the same physical
proteins. (b) tests whether DpoTropi's structurally-extracted depolymerase
domain happens to MD5-match a cipher full RBP — usually impossible
because domains are subsets, but useful as a sanity check that
cipher's RBPs aren't already domain-trimmed somewhere.

Output:
  results/dpotropi_vs_cipher_protein_overlap.csv   (per-DpoTropi-row record
                                                     of whether seq / domain_seq
                                                     hit cipher's training set)
  printed summary

Usage:
  python scripts/analysis/compare_training_protein_overlap.py
"""

import csv
import hashlib
import os

import numpy as np


import argparse
DEFAULT_CIPHER_NPZ = ('/Users/leannmlindsey/WORK/PHI_TSP/cipher/experiments/'
                      'attention_mlp/sweep_prott5_mean_cl70/training_data.npz')
DPOTROPI_TSV = ('/Users/leannmlindsey/WORK/PHI_TSP/DpoTropiSearch_zenoto_data/'
                'TropiGATv2.final_df_v2.tsv')
OUT_CSV = 'results/dpotropi_vs_cipher_protein_overlap.csv'


def md5(s):
    return hashlib.md5(s.encode()).hexdigest() if s else ''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cipher-npz', default=DEFAULT_CIPHER_NPZ)
    p.add_argument('--out-csv', default=OUT_CSV)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    print(f'Loading cipher training MD5 set from {args.cipher_npz} ...')
    npz = np.load(args.cipher_npz, allow_pickle=True)
    cipher_md5 = set(npz['md5_list'].tolist())
    print(f'  cipher MD5s: {len(cipher_md5):,}')

    print('Streaming DpoTropi TSV (seq + domain_seq) ...')
    seq_hit = seq_miss = 0
    dom_hit = dom_miss = 0
    both_hit = 0
    seq_only_hit = 0
    dom_only_hit = 0
    n_rows = 0
    rows_out = []
    with open(DPOTROPI_TSV) as f:
        header = next(f).rstrip('\n').split('\t')
        seq_col = header.index('seq')
        dom_col = header.index('domain_seq')
        prot_col = header.index('Protein_name')
        kl_col = header.index('KL_type_LCA')
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= max(seq_col, dom_col):
                continue
            n_rows += 1
            seq_md5 = md5(parts[seq_col].strip())
            dom_md5 = md5(parts[dom_col].strip())
            seq_in = seq_md5 in cipher_md5
            dom_in = dom_md5 in cipher_md5
            if seq_in:    seq_hit += 1
            else:         seq_miss += 1
            if dom_in:    dom_hit += 1
            else:         dom_miss += 1
            if seq_in and dom_in: both_hit += 1
            if seq_in and not dom_in: seq_only_hit += 1
            if dom_in and not seq_in: dom_only_hit += 1
            rows_out.append({
                'protein_name': parts[prot_col],
                'kl_lca': parts[kl_col],
                'seq_md5': seq_md5,
                'dom_md5': dom_md5,
                'seq_in_cipher': seq_in,
                'dom_in_cipher': dom_in,
            })

    # Write a smaller summary CSV (only the matched rows; keeping all 21k
    # would be huge and most are empty). Plus a header counts row at top.
    matched_rows = [r for r in rows_out if r['seq_in_cipher'] or r['dom_in_cipher']]
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        for r in matched_rows:
            w.writerow(r)
    print(f'\nWrote {args.out_csv}  ({len(matched_rows)} matched rows)')

    print()
    print('=' * 70)
    print('OVERLAP SUMMARY')
    print('=' * 70)
    print(f'DpoTropi training rows scanned: {n_rows:,}')
    print(f'Cipher training MD5 universe:   {len(cipher_md5):,}')
    print()
    print(f'{"":<30} {"hits":>8}  {"% of dpotropi rows":>20}')
    print('-' * 60)
    print(f'{"DpoTropi seq → cipher MD5":<30} {seq_hit:>8,}  '
          f'{100*seq_hit/n_rows:>20.2f}')
    print(f'{"DpoTropi domain_seq → cipher MD5":<30} {dom_hit:>8,}  '
          f'{100*dom_hit/n_rows:>20.2f}')
    print(f'{"both (seq AND domain_seq)":<30} {both_hit:>8,}  '
          f'{100*both_hit/n_rows:>20.2f}')
    print(f'{"seq only (not domain_seq)":<30} {seq_only_hit:>8,}  '
          f'{100*seq_only_hit/n_rows:>20.2f}')
    print(f'{"domain_seq only (not seq)":<30} {dom_only_hit:>8,}  '
          f'{100*dom_only_hit/n_rows:>20.2f}')
    print()
    n_overlap = seq_hit + dom_only_hit  # any MD5 match
    print(f'Total DpoTropi rows with ≥1 MD5 match in cipher: {n_overlap:,} '
          f'({100*n_overlap/n_rows:.2f}%)')


if __name__ == '__main__':
    main()
