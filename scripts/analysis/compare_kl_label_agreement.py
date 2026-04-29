"""Label agreement on the cipher∩DpoTropi protein overlap.

For the ~3,331 proteins present in both training corpora (matched by
full-sequence MD5), compare:

  - Cipher's K labels (one-hot or multi-hot from k_labels matrix,
    canonicalised via K→KL).
  - DpoTropi's KL_type_LCA (single KL or pipe-separated multi-KL).

This directly probes the labeling-rule difference: cipher labels with
the *current host* K type; DpoTropi with the *infected ancestor's* K
via LCA on the host phylogeny. If they agree on most overlapping
proteins, the labeling rule barely matters; if they disagree, that's
the smoking gun for why TropiSEQ might be learning a different
function.

Output:
  results/dpotropi_vs_cipher_label_agreement.csv  — per-overlap-protein
                                                    cipher-K vs dpotropi-KL_LCA
  prints summary counts to stdout

Usage:
  python scripts/analysis/compare_kl_label_agreement.py
"""

import csv
import hashlib
import json
import os
from collections import Counter

import numpy as np


import argparse
DEFAULT_CIPHER_EXP = ('/Users/leannmlindsey/WORK/PHI_TSP/cipher/experiments/'
                      'attention_mlp/sweep_prott5_mean_cl70')
DPOTROPI_TSV = ('/Users/leannmlindsey/WORK/PHI_TSP/DpoTropiSearch_zenoto_data/'
                'TropiGATv2.final_df_v2.tsv')
OUT_CSV = 'results/dpotropi_vs_cipher_label_agreement.csv'


def md5(s):
    return hashlib.md5(s.encode()).hexdigest() if s else ''


def to_kl(name):
    if name in (None, '', 'null', 'N/A'):
        return None
    n = name.strip()
    if n.startswith('KL'):
        return n
    if n.startswith('K') and n[1:].isdigit():
        return 'KL' + n[1:]
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cipher-exp', default=DEFAULT_CIPHER_EXP)
    p.add_argument('--out-csv', default=OUT_CSV)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)

    print(f'Loading cipher training labels from {args.cipher_exp} ...')
    enc = json.load(open(os.path.join(args.cipher_exp, 'label_encoders.json')))
    k_classes = enc['k_classes']
    npz = np.load(os.path.join(args.cipher_exp, 'training_data.npz'), allow_pickle=True)
    md5_list = npz['md5_list'].tolist()
    k_labels = npz['k_labels']  # (n_proteins, 161)
    md5_to_idx = {m: i for i, m in enumerate(md5_list)}

    print(f'  cipher MD5s: {len(md5_to_idx):,}')

    print('Streaming DpoTropi TSV; matching by seq MD5 ...')
    n_overlap = 0
    rows = []
    with open(DPOTROPI_TSV) as f:
        header = next(f).rstrip('\n').split('\t')
        seq_col = header.index('seq')
        prot_col = header.index('Protein_name')
        kl_col = header.index('KL_type_LCA')
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= max(seq_col, kl_col):
                continue
            seq_md5 = md5(parts[seq_col].strip())
            if seq_md5 not in md5_to_idx:
                continue
            n_overlap += 1
            i = md5_to_idx[seq_md5]
            cipher_kls = {to_kl(c) for c, v in zip(k_classes, k_labels[i])
                          if v > 0 and to_kl(c) is not None}
            dpotropi_label = parts[kl_col].strip()
            dpotropi_kls = {to_kl(s.strip()) for s in dpotropi_label.split('|')
                            if to_kl(s.strip()) is not None}
            rows.append({
                'protein_name': parts[prot_col],
                'md5': seq_md5,
                'cipher_K': '|'.join(sorted(cipher_kls)) if cipher_kls else '',
                'dpotropi_KL_LCA': '|'.join(sorted(dpotropi_kls)) if dpotropi_kls else '',
                'cipher_n_K': len(cipher_kls),
                'dpotropi_n_KL': len(dpotropi_kls),
                'set_intersection': '|'.join(sorted(cipher_kls & dpotropi_kls)),
                'agree_any': bool(cipher_kls & dpotropi_kls),
                'cipher_only': '|'.join(sorted(cipher_kls - dpotropi_kls)),
                'dpotropi_only': '|'.join(sorted(dpotropi_kls - cipher_kls)),
            })

    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nWrote {args.out_csv}  ({len(rows)} overlap proteins)')

    # Summaries
    n = len(rows)
    n_agree = sum(r['agree_any'] for r in rows)
    n_dis = n - n_agree
    n_cipher_empty = sum(1 for r in rows if r['cipher_n_K'] == 0)
    n_dpotropi_empty = sum(1 for r in rows if r['dpotropi_n_KL'] == 0)
    print(f'\n{"":<40} {"count":>8}  {"%":>6}')
    print('-' * 56)
    print(f'{"agree on at least one KL":<40} {n_agree:>8,}  '
          f'{100*n_agree/n:>5.1f}')
    print(f'{"disagree (no KL in common)":<40} {n_dis:>8,}  '
          f'{100*n_dis/n:>5.1f}')
    print(f'{"cipher has empty K label":<40} {n_cipher_empty:>8,}  '
          f'{100*n_cipher_empty/n:>5.1f}')
    print(f'{"dpotropi has empty KL_LCA":<40} {n_dpotropi_empty:>8,}  '
          f'{100*n_dpotropi_empty/n:>5.1f}')

    # How concentrated are the disagreements?
    print(f'\nDisagreements — top patterns (cipher_K → dpotropi_KL_LCA):')
    dis_rows = [r for r in rows if not r['agree_any']
                and r['cipher_K'] and r['dpotropi_KL_LCA']]
    pat = Counter((r['cipher_K'], r['dpotropi_KL_LCA']) for r in dis_rows)
    print(f'  ({len(dis_rows)} proteins with non-empty disagreeing labels)')
    for (c, d), k in pat.most_common(15):
        print(f'    {c[:30]:<30} → {d[:30]:<30}  ({k}x)')

    # Distribution of "n_KL in DpoTropi label" — many disagree because LCA spans many KLs
    print('\nLabel multiplicity distribution:')
    print(f'  cipher  K labels per protein: '
          f'mean={np.mean([r["cipher_n_K"] for r in rows]):.2f}, '
          f'max={max(r["cipher_n_K"] for r in rows)}')
    print(f'  dpotropi KL labels per protein: '
          f'mean={np.mean([r["dpotropi_n_KL"] for r in rows]):.2f}, '
          f'max={max(r["dpotropi_n_KL"] for r in rows)}')


if __name__ == '__main__':
    main()
