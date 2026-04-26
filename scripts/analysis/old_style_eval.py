"""OLD-style class-ranking eval that auto-handles K-only / O-only / both heads.

Mirrors the scoring rule in OLD klebsiella's
`evaluate_validation_host_prediction.py`:

  For each positive (phage, host) pair:
    1. For each RBP of the phage, get K and/or O probability vectors.
    2. Build a "merged" ranking depending on what the model has:
         - both heads + host has both labels → K + O classes pooled, sorted
         - only K head, or host missing true_O → rank K classes only
         - only O head, or host missing true_K → rank O classes only
    3. Find rank of the first class matching true_K or true_O.
    4. Pair's score = MIN rank across all RBPs of the phage.
  HR@k = fraction of pairs where best rank ≤ k.

This matches OLD `--merge-strategy raw` byte-equivalently on both-head
experiments AND extends naturally to K-only or O-only experiments
(where there's nothing to "merge" with).

Why have this in cipher
-----------------------
OLD's script is hard-coded to require --k-model AND --o-model. We have
single-head experiments where pairing with an unrelated O head would
contaminate the score. This script reads cipher's predictor directly
and does the right thing per-experiment.

Usage
-----
    python scripts/analysis/old_style_eval.py <cipher_exp_dir>
    python scripts/analysis/old_style_eval.py <cipher_exp_dir> --datasets PhageHostLearn
    python scripts/analysis/old_style_eval.py <cipher_exp_dir> \\
        --val-embedding-file data/validation_data/embeddings/prott5_xl_md5.npz

Outputs JSON at <exp_dir>/results_old_style/old_style_eval.json with
per-dataset HR@K (merged / K-only / O-only) plus overall.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def _ensure_cipher_on_path():
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        candidate = parent / 'src' / 'cipher'
        if candidate.is_dir():
            src = str(parent / 'src')
            if src not in sys.path:
                sys.path.insert(0, src)
            return

_ensure_cipher_on_path()

from cipher.data.interactions import (
    load_interaction_pairs, load_phage_protein_mapping,
)
from cipher.evaluation.runner import load_predictor, load_validation_data


NULL_TOKENS = {'N/A', 'null', 'Unknown', '', None}


def _is_real(label):
    return label not in NULL_TOKENS


def _resolve_val_paths(experiment_dir, cli):
    cfg_path = os.path.join(experiment_dir, 'config.yaml')
    val_cfg = {}
    if os.path.exists(cfg_path):
        val_cfg = (yaml.safe_load(open(cfg_path)) or {}).get('validation') or {}
    return {
        'fasta': cli.val_fasta or val_cfg.get('val_fasta'),
        'emb': cli.val_embedding_file or val_cfg.get('val_embedding_file'),
        'emb2': cli.val_embedding_file_2 or val_cfg.get('val_embedding_file_2'),
        'ds_dir': cli.val_datasets_dir or val_cfg.get('val_datasets_dir'),
    }


def _precompute_probs(predictor, md5s, emb_dict):
    """Return (md5_to_idx, k_probs[N,Kc] or None, o_probs[N,Oc] or None,
              k_classes, o_classes)."""
    k_classes = list(predictor.k_classes) if predictor.k_classes else []
    o_classes = list(predictor.o_classes) if predictor.o_classes else []

    k_idx = {c: i for i, c in enumerate(k_classes)}
    o_idx = {c: i for i, c in enumerate(o_classes)}

    rows_md5 = []
    k_rows = [] if k_classes else None
    o_rows = [] if o_classes else None
    for m in md5s:
        if m not in emb_dict:
            continue
        out = predictor.predict_protein(emb_dict[m])
        rows_md5.append(m)
        if k_classes:
            kp = np.zeros(len(k_classes), dtype=np.float32)
            for c, p in out.get('k_probs', {}).items():
                if c in k_idx:
                    kp[k_idx[c]] = float(p)
            k_rows.append(kp)
        if o_classes:
            op = np.zeros(len(o_classes), dtype=np.float32)
            for c, p in out.get('o_probs', {}).items():
                if c in o_idx:
                    op[o_idx[c]] = float(p)
            o_rows.append(op)

    md5_to_idx = {m: i for i, m in enumerate(rows_md5)}
    k_arr = np.stack(k_rows) if k_rows else None
    o_arr = np.stack(o_rows) if o_rows else None
    return md5_to_idx, k_arr, o_arr, k_classes, o_classes


def _rank_in_merged(k_probs, o_probs, k_classes, o_classes, true_k, true_o):
    """Return rank of first K- or O-class match in the merged sort.

    Mirrors OLD `--merge-strategy raw`. If only one head's probs are
    provided (the other is None) OR only one true label is real, the
    merge degenerates to single-head ranking on the available side.
    """
    has_k_probs = k_probs is not None and k_classes
    has_o_probs = o_probs is not None and o_classes
    has_true_k = _is_real(true_k)
    has_true_o = _is_real(true_o)

    items = []
    if has_k_probs:
        for j, c in enumerate(k_classes):
            items.append((c, float(k_probs[j]), 'K'))
    if has_o_probs:
        for j, c in enumerate(o_classes):
            items.append((c, float(o_probs[j]), 'O'))
    if not items:
        return None

    items.sort(key=lambda x: -x[1])
    for r, (c, _p, _h) in enumerate(items, 1):
        if (has_true_k and c == true_k) or (has_true_o and c == true_o):
            return r
    return None


def _rank_in_head(probs, classes, target):
    """Rank of `target` within a single head's class sort. None if missing."""
    if probs is None or not classes or not _is_real(target) or target not in classes:
        return None
    sort = np.argsort(-probs)
    for r, ci in enumerate(sort, 1):
        if classes[ci] == target:
            return int(r)
    return None


def evaluate_dataset(predictor, dataset_dir, dataset_name,
                     emb_dict, pid_md5, max_k=20):
    pairs = load_interaction_pairs(dataset_dir)
    pm_path = os.path.join(dataset_dir, 'metadata', 'phage_protein_mapping.csv')
    phage_map = load_phage_protein_mapping(pm_path)

    interactions = defaultdict(dict)
    serotypes = {}
    for p in pairs:
        interactions[p['phage_id']][p['host_id']] = p['label']
        serotypes[p['host_id']] = {'K': p['host_K'], 'O': p['host_O']}

    needed = set()
    for proteins in phage_map.values():
        for pid in proteins:
            m = pid_md5.get(pid)
            if m is not None and m in emb_dict:
                needed.add(m)
    md5_to_idx, k_arr, o_arr, k_classes, o_classes = _precompute_probs(
        predictor, sorted(needed), emb_dict)

    n_pairs = 0
    n_skip_no_rbp = 0
    n_skip_no_emb = 0
    merged_ranks = []
    k_only_ranks = []
    o_only_ranks = []

    for phage in sorted(interactions):
        pos_hosts = [h for h, lbl in interactions[phage].items() if lbl == 1]
        if not pos_hosts:
            continue

        proteins = phage_map.get(phage, set())
        if not proteins:
            n_skip_no_rbp += len(pos_hosts)
            continue

        # Per-RBP probs
        rbp_probs = []
        for pid in proteins:
            md5 = pid_md5.get(pid)
            if md5 is None or md5 not in md5_to_idx:
                continue
            i = md5_to_idx[md5]
            kp = k_arr[i] if k_arr is not None else None
            op = o_arr[i] if o_arr is not None else None
            rbp_probs.append((pid, kp, op))

        if not rbp_probs:
            n_skip_no_emb += len(pos_hosts)
            continue

        for pos_h in pos_hosts:
            if pos_h not in serotypes:
                continue
            true_k = serotypes[pos_h]['K']
            true_o = serotypes[pos_h]['O']

            best_merged = None
            best_k = None
            best_o = None
            for _pid, kp, op in rbp_probs:
                r = _rank_in_merged(kp, op, k_classes, o_classes, true_k, true_o)
                if r is not None and (best_merged is None or r < best_merged):
                    best_merged = r
                rk = _rank_in_head(kp, k_classes, true_k)
                if rk is not None and (best_k is None or rk < best_k):
                    best_k = rk
                ro = _rank_in_head(op, o_classes, true_o)
                if ro is not None and (best_o is None or ro < best_o):
                    best_o = ro

            if best_merged is None and best_k is None and best_o is None:
                continue

            n_pairs += 1
            merged_ranks.append(best_merged if best_merged is not None else float('inf'))
            if best_k is not None:
                k_only_ranks.append(best_k)
            if best_o is not None:
                o_only_ranks.append(best_o)

    def hr_curve(ranks, max_k):
        return {k: float(np.mean([1 if r <= k else 0 for r in ranks])) if ranks else 0.0
                for k in range(1, max_k + 1)}

    return {
        'dataset': dataset_name,
        'n_pairs': n_pairs,
        'n_skipped_no_rbp': n_skip_no_rbp,
        'n_skipped_no_embedding': n_skip_no_emb,
        'has_k_head': bool(k_classes and k_arr is not None),
        'has_o_head': bool(o_classes and o_arr is not None),
        'n_k_classes': len(k_classes),
        'n_o_classes': len(o_classes),
        'hr_at_k': hr_curve(merged_ranks, max_k),
        'k_hr_at_k': hr_curve(k_only_ranks, max_k),
        'o_hr_at_k': hr_curve(o_only_ranks, max_k),
        'n_with_k': len(k_only_ranks),
        'n_with_o': len(o_only_ranks),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('experiment_dir')
    p.add_argument('--datasets', nargs='+', default=None)
    p.add_argument('--val-fasta', default=None)
    p.add_argument('--val-embedding-file', default=None)
    p.add_argument('--val-embedding-file-2', default=None)
    p.add_argument('--val-datasets-dir', default=None)
    p.add_argument('--out-dir', default=None,
                   help='Default: <experiment_dir>/results_old_style/')
    args = p.parse_args()

    if not os.path.isdir(args.experiment_dir):
        sys.exit(f'ERROR: {args.experiment_dir} is not a directory')
    paths = _resolve_val_paths(args.experiment_dir, args)
    for k in ('fasta', 'emb', 'ds_dir'):
        if not paths[k]:
            sys.exit(f'ERROR: missing validation path {k!r}; pass via CLI '
                     f'or put in config.yaml:validation')

    out_dir = args.out_dir or os.path.join(args.experiment_dir, 'results_old_style')
    os.makedirs(out_dir, exist_ok=True)

    print(f'Experiment: {args.experiment_dir}')
    print(f'Out dir:    {out_dir}')
    print(f'Embeddings: {paths["emb"]}')
    print('Loading predictor and validation data ...')
    predictor = load_predictor(args.experiment_dir)
    val = load_validation_data(paths['fasta'], paths['emb'], paths['ds_dir'],
                               val_embedding_file_2=paths['emb2'])

    print(f'Predictor: K classes={len(predictor.k_classes or [])}, '
          f'O classes={len(predictor.o_classes or [])}')

    datasets = args.datasets or val['available_datasets']
    results = {}
    for ds in datasets:
        ds_dir = os.path.join(paths['ds_dir'], ds)
        if not os.path.isdir(ds_dir):
            print(f'  Skipping {ds} (not found at {ds_dir})')
            continue
        print(f'  {ds} ...', end='', flush=True)
        r = evaluate_dataset(predictor, ds_dir, ds,
                             val['emb_dict'], val['pid_md5'])
        results[ds] = r
        print(f' n={r["n_pairs"]}, '
              f'merged HR@1={r["hr_at_k"][1]:.4f} | '
              f'K-only HR@1={r["k_hr_at_k"][1]:.4f} '
              f'(n={r["n_with_k"]}) | '
              f'O-only HR@1={r["o_hr_at_k"][1]:.4f} '
              f'(n={r["n_with_o"]})')

    # OVERALL: pool ranks across datasets
    overall_merged = []
    overall_k = []
    overall_o = []
    for r in results.values():
        # We don't have raw ranks here, so reconstruct from HR curve isn't trivial.
        # Easier: re-aggregate by dataset weight.
        pass
    # Compute weighted overall as per-pair sum / total
    total = sum(r['n_pairs'] for r in results.values())
    if total > 0:
        pooled = {}
        for k in range(1, 21):
            num_merged = sum(r['n_pairs'] * r['hr_at_k'][k] for r in results.values())
            num_k = sum(r['n_with_k'] * r['k_hr_at_k'][k] for r in results.values())
            num_o = sum(r['n_with_o'] * r['o_hr_at_k'][k] for r in results.values())
            den_k = sum(r['n_with_k'] for r in results.values())
            den_o = sum(r['n_with_o'] for r in results.values())
            pooled[k] = {
                'merged': num_merged / total,
                'k_only': num_k / max(den_k, 1),
                'o_only': num_o / max(den_o, 1),
            }
        results['OVERALL'] = {
            'n_pairs': total,
            'n_with_k': sum(r['n_with_k'] for r in results.values() if 'n_with_k' in r),
            'n_with_o': sum(r['n_with_o'] for r in results.values() if 'n_with_o' in r),
            'hr_at_k': {k: pooled[k]['merged'] for k in pooled},
            'k_hr_at_k': {k: pooled[k]['k_only'] for k in pooled},
            'o_hr_at_k': {k: pooled[k]['o_only'] for k in pooled},
        }

    out_json = os.path.join(out_dir, 'old_style_eval.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f'{"dataset":<18} {"n":>6} {"merged@1":>10} {"merged@5":>10} '
          f'{"K@1":>8} {"O@1":>8}')
    print('-' * 68)
    for ds, r in results.items():
        print(f'{ds:<18} {r["n_pairs"]:>6} '
              f'{r["hr_at_k"][1]:>10.4f} {r["hr_at_k"][5]:>10.4f} '
              f'{r["k_hr_at_k"][1]:>8.4f} {r["o_hr_at_k"][1]:>8.4f}')
    print(f'\nResults: {out_json}')


if __name__ == '__main__':
    main()
