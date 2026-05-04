"""Extract K-head SE-attention weights conditioned on a target class
prediction (default: K1). For agent 5's interpretability question — what
4-mers does sweep_kmer_aa20_k4 actually attend to when predicting K1?

For each validation phage whose host has the target K-type AND where the
K-head correctly predicts that K-type at top-1 across at least one of the
phage's RBPs, we capture the per-feature SE-attention weights, average
across those K1-correct samples, map feature index → AA20 4-mer string,
and rank.

Output:
    1. <out-tsv>             — top-N attended 4-mers (rank, kmer, mean_attn, n)
    2. <out-tsv>.crosscheck  — specific values for the analysis-32 and
                                analysis-33 4-mers regardless of rank

Usage:
    python scripts/analysis/extract_kmer_attention_for_class.py \
        --experiment-dir experiments/attention_mlp/sweep_kmer_aa20_k4 \
        --val-fasta data/validation_data/metadata/validation_rbps_all.faa \
        --val-emb data/validation_data/embeddings/kmer_aa20_k4_md5.npz \
        --val-datasets-dir data/validation_data/HOST_RANGE \
        --target-class K1 \
        --top-n 300 \
        --out-tsv results/analysis/kmer_attention_K1.tsv
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


# Symbol enumerations per alphabet — must match
# klebsiella/scripts/data_prep/compute_kmer_features.py exactly.
# Order matters: this is the canonical lexicographic enumeration that
# defines the feature index → kmer string mapping.
ALPHABETS = {
    'aa20':    list('ACDEFGHIKLMNPQRSTVWY'),       # 20 standard AAs sorted
    'murphy8': list('ABCDEFGH'),                   # 8 chemistry classes
    'murphy10': list('ABCDEFGHIJ'),                # 10 chemistry classes
    'murphy4': list('ABCD'),                       # 4 chemistry classes
    'li10':    list('ABCDEFGHIJ'),                 # 10 li clusters
    'hydro3':  list('HNP'),                        # 3 hydrophobicity bins
}


def kmer_idx_to_str(idx, alphabet, k):
    """Map feature index → kmer string (canonical lexicographic enumeration)."""
    n = len(alphabet)
    chars = []
    for _ in range(k):
        chars.append(alphabet[idx % n])
        idx //= n
    return ''.join(reversed(chars))


def kmer_str_to_idx(kmer, alphabet):
    """Inverse — kmer string → feature index. Raises on invalid char."""
    n = len(alphabet)
    idx = 0
    for ch in kmer:
        idx = idx * n + alphabet.index(ch)
    return idx


def load_predictor(exp_dir, cipher_src=None):
    """Load the trained K-head AttentionMLP in eval mode."""
    if cipher_src and cipher_src not in sys.path:
        sys.path.insert(0, cipher_src)
    cipher_root = Path(exp_dir).resolve().parents[2]
    if str(cipher_root) not in sys.path:
        sys.path.insert(0, str(cipher_root))
    from models.attention_mlp.model import AttentionMLP

    model_dir = Path(exp_dir) / 'model_k'
    with open(model_dir / 'config.json') as f:
        cfg = json.load(f)
    model = AttentionMLP(
        input_dim=cfg['input_dim'],
        num_classes=cfg['num_classes'],
        hidden_dims=cfg.get('hidden_dims', [1280, 640, 320, 160]),
        se_dim=cfg.get('attention_dim', 640),
        dropout=cfg.get('dropout', 0.1),
    )
    state = torch.load(model_dir / 'best_model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with open(Path(exp_dir) / 'label_encoders.json') as f:
        le = json.load(f)
    k_classes = le.get('k_classes') or le['k']
    return model, k_classes


def load_fasta_md5(fasta_path):
    """{protein_id: md5_hex}."""
    import hashlib
    out = {}
    pid = None
    seq = []
    def emit():
        if pid is None: return
        s = ''.join(seq).upper()
        out[pid] = hashlib.md5(s.encode()).hexdigest()
    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                emit()
                pid = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        emit()
    return out


def collect_target_class_phages(datasets_dir, target_class):
    """Across all 5 validation datasets, find phages whose hosts have the
    target K-type. Returns list of (dataset, phage_id, [protein_id...]) tuples."""
    out = []
    for ds_dir in sorted(Path(datasets_dir).iterdir()):
        if not ds_dir.is_dir(): continue
        im = ds_dir / 'metadata' / 'interaction_matrix.tsv'
        ppm = ds_dir / 'metadata' / 'phage_protein_mapping.csv'
        if not im.exists() or not ppm.exists(): continue
        # Phage → proteins
        phage_proteins = defaultdict(list)
        with open(ppm) as f:
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    phage_proteins[parts[0]].append(parts[1])
        # Phages whose POSITIVE host has target_class
        target_phages = set()
        with open(im) as f:
            r = csv.DictReader(f, delimiter='\t')
            for row in r:
                if row.get('label') in ('1', 1, True):
                    k = (row.get('host_K') or '').strip()
                    if k == target_class:
                        target_phages.add(row['phage_id'])
        for ph in sorted(target_phages):
            prots = phage_proteins.get(ph, [])
            if prots:
                out.append((ds_dir.name, ph, prots))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment-dir', required=True,
                    help='cipher experiment dir with model_k/best_model.pt')
    ap.add_argument('--val-fasta', required=True)
    ap.add_argument('--val-emb', required=True,
                    help='kmer NPZ keyed by md5')
    ap.add_argument('--val-datasets-dir', required=True)
    ap.add_argument('--target-class', default='K1',
                    help='K-type to condition attention on (default K1)')
    ap.add_argument('--alphabet', default='aa20',
                    choices=list(ALPHABETS.keys()),
                    help='kmer alphabet — must match the trained model')
    ap.add_argument('--k', type=int, default=4,
                    help='kmer size — must match the trained model')
    ap.add_argument('--crosscheck-json', default=None,
                    help='optional JSON file mapping {source_label: [kmer, ...]} '
                         'for the crosscheck output. If omitted, no crosscheck file is written.')
    ap.add_argument('--top-n', type=int, default=300)
    ap.add_argument('--require-correct', action='store_true', default=True,
                    help='only use phages where the K-head ranks the target '
                         'class top-1 for at least one of the phage proteins')
    ap.add_argument('--include-all-target-phages', dest='require_correct',
                    action='store_false',
                    help='use all target-class phages, not just correct ones')
    ap.add_argument('--out-tsv', required=True)
    ap.add_argument('--cipher-src', default=None)
    args = ap.parse_args()

    alphabet = ALPHABETS[args.alphabet]
    expected_dim = len(alphabet) ** args.k

    print(f'[load] predictor from {args.experiment_dir}')
    model, k_classes = load_predictor(args.experiment_dir, args.cipher_src)
    if args.target_class not in k_classes:
        sys.exit(f'ERROR: target class {args.target_class!r} not in K-class vocab '
                 f'({len(k_classes)} classes; first few: {k_classes[:5]})')
    target_idx = k_classes.index(args.target_class)
    print(f'  K-classes: {len(k_classes)}; target {args.target_class} at index {target_idx}')

    print(f'[load] val FASTA → md5 from {args.val_fasta}')
    fasta_md5 = load_fasta_md5(args.val_fasta)
    print(f'  {len(fasta_md5)} validation proteins indexed')

    print(f'[load] val kmer embeddings from {args.val_emb}')
    emb = np.load(args.val_emb)
    n_emb = len(emb.files)
    sample = emb[emb.files[0]]
    feat_dim = sample.shape[-1]
    print(f'  {n_emb} embeddings, feature dim = {feat_dim}')
    if feat_dim != expected_dim:
        print(f'  WARNING: expected {expected_dim}-d {args.alphabet}_k{args.k} '
              f'features, got {feat_dim}')

    print(f'[load] target-class phages from {args.val_datasets_dir}')
    target_phages = collect_target_class_phages(args.val_datasets_dir, args.target_class)
    print(f'  {len(target_phages)} phages with positive {args.target_class} host across all 5 datasets')
    if not target_phages:
        sys.exit(f'No phages found with positive {args.target_class} host')

    # ── Run inference: per phage, average attention over its proteins
    #    that the K-head correctly identifies (top-1 = target_idx) ──
    print(f'[infer] running K-head + capturing attention')
    contributing_phages = []         # (dataset, phage_id, attention_vector_avg_over_correct_proteins)
    all_target_phages_attn = []      # fallback if not require_correct
    for ds, phage, prots in target_phages:
        attn_vecs_for_phage = []
        any_correct = False
        for pid in prots:
            md5 = fasta_md5.get(pid)
            if md5 is None or md5 not in emb:
                continue
            vec = emb[md5]
            x = torch.FloatTensor(vec).unsqueeze(0)   # (1, feat_dim)
            with torch.no_grad():
                logits, attention = model(x, return_attention=True)
            top1 = int(logits.argmax(dim=1).item())
            if top1 == target_idx:
                attn_vecs_for_phage.append(attention.squeeze(0).cpu().numpy())
                any_correct = True
            elif not args.require_correct:
                attn_vecs_for_phage.append(attention.squeeze(0).cpu().numpy())
        if attn_vecs_for_phage:
            avg = np.mean(np.stack(attn_vecs_for_phage), axis=0)
            if any_correct:
                contributing_phages.append((ds, phage, avg))
            all_target_phages_attn.append((ds, phage, avg))

    use_set = contributing_phages if args.require_correct else all_target_phages_attn
    print(f'  {len(contributing_phages)} phages with at least one correct top-1 K={args.target_class}')
    print(f'  {len(all_target_phages_attn)} phages with at least one ranked protein')
    if not use_set:
        sys.exit(f'No qualifying phages with the {args.target_class} prediction')
    print(f'  using {len(use_set)} phages for averaging '
          f'(--require-correct={args.require_correct})')

    # Average across phages
    mean_attn = np.mean(np.stack([a for _, _, a in use_set]), axis=0)
    print(f'  mean attention vector shape: {mean_attn.shape}')
    print(f'  range: [{mean_attn.min():.4f}, {mean_attn.max():.4f}]')

    # ── Output: top-N attended 4-mers ──
    n_phages = len(use_set)
    order = np.argsort(-mean_attn)
    os.makedirs(os.path.dirname(args.out_tsv) or '.', exist_ok=True)
    with open(args.out_tsv, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['rank', 'kmer', 'mean_attention',
                     f'n_phages_contributed_{args.target_class}'])
        for rank, idx in enumerate(order[:args.top_n], 1):
            kmer = kmer_idx_to_str(int(idx), alphabet, args.k)
            w.writerow([rank, kmer, f'{mean_attn[idx]:.6f}', n_phages])
    print(f'[write] top-{args.top_n}: {args.out_tsv}')

    # ── Crosscheck output: specific kmers regardless of rank ──
    if args.crosscheck_json:
        with open(args.crosscheck_json) as f:
            crosscheck = json.load(f)
        cc_path = args.out_tsv.replace('.tsv', '.crosscheck.tsv')
        if cc_path == args.out_tsv:
            cc_path = args.out_tsv + '.crosscheck'
        rank_lookup = {int(o): r for r, o in enumerate(order, 1)}
        with open(cc_path, 'w', newline='') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(['source', 'kmer', 'rank', 'mean_attention',
                         f'n_phages_contributed_{args.target_class}'])
            for source, lst in crosscheck.items():
                for kmer in lst:
                    idx = kmer_str_to_idx(kmer, alphabet)
                    w.writerow([source, kmer, rank_lookup[idx],
                                f'{mean_attn[idx]:.6f}', n_phages])
        print(f'[write] crosscheck: {cc_path}')


if __name__ == '__main__':
    main()
