"""Convert DpoTropiSearch's training TSV into cipher-format training inputs.

Inputs (from Zenodo unzipped):
    /Users/leannmlindsey/WORK/PHI_TSP/DpoTropiSearch_zenoto_data/TropiGATv2.final_df_v2.tsv

The TSV has 21,351 rows × 1289 cols:
    Phage, Protein_name, KL_type_LCA, Infected_ancestor, index, Dataset,
    seq, domain_seq, <1280 ESM-2 650M embedding values>

Outputs (cipher-style training inputs):
    data/training_data/dpotropi/
      dpotropi_train_proteins.faa           — FASTA, header = MD5(seq), seq = full RBP
      dpotropi_phage_protein_map.tsv        — host_genome, K_type, O_type,
                                              phage_genome, protein_id, is_tsp, md5
      dpotropi_glycan_binders.tsv           — protein_id, 8 tool flags (=1), total_sources=8
      dpotropi_positive.list                — protein_ids
      dpotropi_label_summary.json           — per-KL counts, label-multiplicity stats

Notes on labelling:
- Cipher's `K_type` column accepts a single K type per row. DpoTropi's
  `KL_type_LCA` may be pipe-separated (multi-KL ambiguity). Strategy:
    * `--label-strategy primary`: keep only the FIRST KL of multi-KL labels
      (treat as single-label primary KL).
    * `--label-strategy explode` (default): emit one row per KL in a
      multi-label assignment (e.g. "KLa|KLb" → 2 rows, same protein id,
      different K_type). Cipher's prepare_training_data handles
      multi-row-same-protein.
    * `--label-strategy drop_multi`: drop rows where label has more than 1
      KL — keep only unambiguous proteins.
- Cipher's O label set to "" (null) — DpoTropi doesn't ship O.
- Cipher's `is_tsp` set to 0 (no SpikeHunter info ported).
- Cipher's `host_genome` set to `Infected_ancestor` (DpoTropi's column).
- Cipher's `phage_genome` set to `Phage` (DpoTropi's column).
- KL canonicalisation: K1 → KL1, KL101 → KL101 (cipher's training pipeline
  will use whatever string we pass).

Usage:
    python scripts/data_prep/build_dpotropi_training_inputs.py
    python scripts/data_prep/build_dpotropi_training_inputs.py \\
        --label-strategy drop_multi
"""

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict, Counter

DEFAULT_TSV = ('/Users/leannmlindsey/WORK/PHI_TSP/DpoTropiSearch_zenoto_data/'
               'TropiGATv2.final_df_v2.tsv')
DEFAULT_OUT = 'data/training_data/dpotropi'

TOOL_COLS = ['DePP_85', 'PhageRBPdetect', 'DepoScope', 'DepoRanker',
             'SpikeHunter', 'dbCAN', 'IPR', 'phold_glycan_tailspike']


def md5(s):
    return hashlib.md5(s.encode()).hexdigest() if s else ''


def to_kl(k):
    if not k or k.strip() in ('', 'null', 'N/A', 'None'):
        return None
    s = k.strip()
    if s.startswith('KL'):
        return s
    if s.startswith('K') and s[1:].isdigit():
        return f'KL{s[1:]}'
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tsv', default=DEFAULT_TSV)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    p.add_argument('--label-strategy',
                   choices=['primary', 'explode', 'drop_multi'],
                   default='explode',
                   help='How to handle multi-KL_LCA labels (default: explode)')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_fasta = os.path.join(args.out_dir, 'dpotropi_train_proteins.faa')
    out_map = os.path.join(args.out_dir, 'dpotropi_phage_protein_map.tsv')
    out_gly = os.path.join(args.out_dir, 'dpotropi_glycan_binders.tsv')
    out_pos = os.path.join(args.out_dir, 'dpotropi_positive.list')
    out_summary = os.path.join(args.out_dir, 'dpotropi_label_summary.json')

    print(f'Reading {args.tsv}')
    print(f'Output to {args.out_dir}/')
    print(f'Label strategy: {args.label_strategy}')
    print()

    # First pass: dedup by MD5 (keep first occurrence's metadata + sequence).
    proteins = {}   # md5 -> {protein_id, phage, host, kls, seq}
    n_input_rows = 0
    n_skip_no_label = 0
    with open(args.tsv) as f:
        header = next(f).rstrip('\n').split('\t')
        idx = {col: header.index(col) for col in
               ('Phage', 'Protein_name', 'KL_type_LCA', 'Infected_ancestor', 'seq')}
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= max(idx.values()):
                continue
            n_input_rows += 1
            seq = parts[idx['seq']].strip()
            if not seq:
                continue
            m = md5(seq)
            label = parts[idx['KL_type_LCA']].strip()
            if not label:
                n_skip_no_label += 1
                continue
            kls = [to_kl(s.strip()) for s in label.split('|')]
            kls = [k for k in kls if k]
            if not kls:
                n_skip_no_label += 1
                continue
            if args.label_strategy == 'primary':
                kls = kls[:1]
            elif args.label_strategy == 'drop_multi':
                if len(kls) > 1:
                    continue
            # explode default — keep all
            if m not in proteins:
                proteins[m] = {
                    'protein_id': parts[idx['Protein_name']].strip(),
                    'phage':      parts[idx['Phage']].strip(),
                    'host':       parts[idx['Infected_ancestor']].strip(),
                    'kls':        set(kls),
                    'seq':        seq,
                }
            else:
                # multi-row dedup: union the KLs (some proteins appear in
                # multiple subsets like ppt + anubis with same/diff labels)
                proteins[m]['kls'].update(kls)

    print(f'Input rows scanned:                  {n_input_rows:,}')
    print(f'Rows skipped (no label):             {n_skip_no_label:,}')
    print(f'Unique proteins by MD5:              {len(proteins):,}')

    # Stats
    label_counts = Counter()
    multi_count = 0
    for d in proteins.values():
        if len(d['kls']) > 1:
            multi_count += 1
        for k in d['kls']:
            label_counts[k] += 1
    print(f'Proteins with multi-KL after dedup:  {multi_count:,}')
    print(f'Unique KL types in output:           {len(label_counts):,}')
    print(f'Top 10 KL counts: '
          + ', '.join(f'{k}={n}' for k, n in label_counts.most_common(10)))
    print()

    # Write FASTA (keyed by protein_id; cipher's training pipeline
    # computes MD5 itself so order/key mostly doesn't matter — but we use
    # protein_id to keep the output human-readable)
    print(f'Writing {out_fasta}')
    with open(out_fasta, 'w') as f:
        for m, d in proteins.items():
            f.write(f'>{d["protein_id"]}\n{d["seq"]}\n')

    # Write phage_protein_map (one row per (protein, KL) — cipher's
    # prepare_training_data unions multi-rows). Match cipher's column
    # names exactly.
    print(f'Writing {out_map}')
    with open(out_map, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['host_genome', 'K_type', 'O_type', 'phage_genome',
                    'protein_id', 'is_tsp', 'md5', 'matrix_phage_name'])
        for m, d in proteins.items():
            for kl in sorted(d['kls']):
                w.writerow([d['host'], kl, '', d['phage'],
                            d['protein_id'], 0, m, d['phage']])

    # Write glycan_binders (all 8 tool flags = 1, total_sources = 8 — we
    # don't have tool info from DpoTropi, so treat all as positives).
    print(f'Writing {out_gly}')
    with open(out_gly, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['protein_id'] + TOOL_COLS + ['total_sources'])
        for m, d in proteins.items():
            w.writerow([d['protein_id']] + [1] * len(TOOL_COLS) + [len(TOOL_COLS)])

    # Write positive list (just protein_ids)
    print(f'Writing {out_pos}')
    with open(out_pos, 'w') as f:
        for d in proteins.values():
            f.write(f'{d["protein_id"]}\n')

    # Summary JSON
    with open(out_summary, 'w') as f:
        json.dump({
            'n_input_rows': n_input_rows,
            'n_unique_proteins': len(proteins),
            'n_unique_kl_types': len(label_counts),
            'n_proteins_multi_kl': multi_count,
            'label_strategy': args.label_strategy,
            'kl_counts': dict(label_counts),
        }, f, indent=2)
    print(f'Writing {out_summary}')

    print()
    print(f'Done. Outputs in {args.out_dir}/')


if __name__ == '__main__':
    main()
