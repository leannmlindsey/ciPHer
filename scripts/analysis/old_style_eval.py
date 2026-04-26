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


def _precompute_head_probs(predictor, classes, head_key, emb_dict, md5s):
    """Run predictor on each MD5's embedding, extract one head's prob vector.

    Returns dict {md5: ndarray[len(classes)]}. Skips MD5s missing from
    emb_dict. If `classes` is empty (head not present), returns {}.
    """
    if not classes:
        return {}
    idx = {c: i for i, c in enumerate(classes)}
    out = {}
    for m in md5s:
        if m not in emb_dict:
            continue
        preds = predictor.predict_protein(emb_dict[m])
        vec = np.zeros(len(classes), dtype=np.float32)
        for c, p in preds.get(head_key, {}).items():
            if c in idx:
                vec[idx[c]] = float(p)
        out[m] = vec
    return out


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


def evaluate_dataset(dataset_dir, dataset_name,
                     md5_to_kvec, md5_to_ovec,
                     k_classes, o_classes,
                     pid_md5, max_k=20):
    """Score one validation dataset.

    Strict denominator: every positive pair counts. Pairs whose phage has
    no RBP embeddings, or whose true label is out-of-vocab, are recorded
    as misses (rank = None). They do NOT get excluded — that would let
    embedding-input models off the hook for failures of coverage.

    Returns three families of HR@k curves:
      - hr_at_k                  merged class-ranking HR@k over ALL pairs
      - k_hr_at_k / o_hr_at_k    per-head HR@k over ALL pairs (strict)
      - or_hr_at_k               (K-rank ≤ k) OR (O-rank ≤ k) over ALL pairs
                                 — perfect-merge ceiling
    Plus the in-vocab variants (denominator = pairs with that head's true
    label in vocab AND an embedding for at least one RBP), kept for
    cross-paper comparison.

    Args:
        md5_to_kvec: dict {md5: K-prob ndarray} (empty if no K head)
        md5_to_ovec: dict {md5: O-prob ndarray} (empty if no O head)
        k_classes / o_classes: matching class lists (or []).
    """
    pairs = load_interaction_pairs(dataset_dir)
    pm_path = os.path.join(dataset_dir, 'metadata', 'phage_protein_mapping.csv')
    phage_map = load_phage_protein_mapping(pm_path)

    interactions = defaultdict(dict)
    serotypes = {}
    for p in pairs:
        interactions[p['phage_id']][p['host_id']] = p['label']
        serotypes[p['host_id']] = {'K': p['host_K'], 'O': p['host_O']}

    # One record per positive pair WHERE the phage has at least one RBP in
    # pid_md5 (i.e. an annotated RBP whose sequence is in the FASTA). Pairs
    # with zero annotated RBPs are excluded — there is literally no input
    # for an RBP-embedding model to operate on.
    #
    # Pairs WITH annotated RBPs but missing embeddings (NPZ coverage gap)
    # are KEPT and recorded as misses — that's a model coverage failure
    # the metric should reflect, not normalize away.
    per_pair = []  # list of dicts: {merged, k_rank, o_rank, has_data, k_in_vocab, o_in_vocab}

    for phage in sorted(interactions):
        pos_hosts = [h for h, lbl in interactions[phage].items() if lbl == 1]
        if not pos_hosts:
            continue

        proteins = phage_map.get(phage, set())
        annotated_md5s = [pid_md5[p] for p in proteins if p in pid_md5]
        # Phage has no annotated RBPs at all: skip (out of scope).
        if not annotated_md5s:
            continue

        # Per-RBP probs (None for either head if MD5 missing from that head's NPZ)
        rbp_probs = []
        for pid in proteins:
            md5 = pid_md5.get(pid)
            if md5 is None:
                continue
            kp = md5_to_kvec.get(md5)
            op = md5_to_ovec.get(md5)
            if kp is None and op is None:
                continue
            rbp_probs.append((pid, kp, op))

        for pos_h in pos_hosts:
            if pos_h not in serotypes:
                continue
            true_k = serotypes[pos_h]['K']
            true_o = serotypes[pos_h]['O']
            k_in_vocab = bool(k_classes) and _is_real(true_k) and true_k in k_classes
            o_in_vocab = bool(o_classes) and _is_real(true_o) and true_o in o_classes

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

            per_pair.append({
                'merged_rank': best_merged,
                'k_rank': best_k,
                'o_rank': best_o,
                'has_data': bool(rbp_probs),
                'k_in_vocab': k_in_vocab,
                'o_in_vocab': o_in_vocab,
            })

    n = len(per_pair)
    n_with_data = sum(1 for r in per_pair if r['has_data'])
    n_with_k = sum(1 for r in per_pair if r['k_in_vocab'] and r['has_data'])
    n_with_o = sum(1 for r in per_pair if r['o_in_vocab'] and r['has_data'])

    def hr_strict(field, k):
        if n == 0: return 0.0
        return sum(1 for r in per_pair
                   if r[field] is not None and r[field] <= k) / n

    def hr_or(k):
        if n == 0: return 0.0
        return sum(1 for r in per_pair if
                   (r['k_rank'] is not None and r['k_rank'] <= k) or
                   (r['o_rank'] is not None and r['o_rank'] <= k)) / n

    def hr_in_vocab(field, vocab_field, k):
        denom = sum(1 for r in per_pair if r[vocab_field] and r['has_data'])
        if denom == 0: return 0.0
        return sum(1 for r in per_pair
                   if r[vocab_field] and r['has_data']
                   and r[field] is not None and r[field] <= k) / denom

    return {
        'dataset': dataset_name,
        'n_pairs': n,
        'n_with_data': n_with_data,
        'n_with_k': n_with_k,
        'n_with_o': n_with_o,
        'has_k_head': bool(k_classes and md5_to_kvec),
        'has_o_head': bool(o_classes and md5_to_ovec),
        'n_k_classes': len(k_classes),
        'n_o_classes': len(o_classes),
        # Strict (over n_pairs): the right thing to compare across models.
        'hr_at_k': {k: hr_strict('merged_rank', k) for k in range(1, max_k + 1)},
        'k_hr_at_k': {k: hr_strict('k_rank', k) for k in range(1, max_k + 1)},
        'o_hr_at_k': {k: hr_strict('o_rank', k) for k in range(1, max_k + 1)},
        'or_hr_at_k': {k: hr_or(k) for k in range(1, max_k + 1)},
        # In-vocab (over n_with_k / n_with_o): for cross-paper compare with
        # OLD klebsiella's reported numbers.
        'k_hr_at_k_in_vocab': {k: hr_in_vocab('k_rank', 'k_in_vocab', k)
                                for k in range(1, max_k + 1)},
        'o_hr_at_k_in_vocab': {k: hr_in_vocab('o_rank', 'o_in_vocab', k)
                                for k in range(1, max_k + 1)},
    }


def main():
    p = argparse.ArgumentParser()
    # Single-experiment mode (positional)
    p.add_argument('experiment_dir', nargs='?', default=None,
                   help='Single-experiment mode: cipher experiment dir with model_k/ and/or model_o/. '
                        'Omit if using --k-experiment-dir / --o-experiment-dir.')
    # Dual-head mode
    p.add_argument('--k-experiment-dir', default=None,
                   help='Dual mode: experiment to source the K head from')
    p.add_argument('--o-experiment-dir', default=None,
                   help='Dual mode: experiment to source the O head from')
    p.add_argument('--k-val-embedding-file', default=None,
                   help='Dual mode: validation NPZ for K head (must match K embedding type)')
    p.add_argument('--o-val-embedding-file', default=None,
                   help='Dual mode: validation NPZ for O head (must match O embedding type)')
    # Common
    p.add_argument('--datasets', nargs='+', default=None)
    p.add_argument('--val-fasta', default=None)
    p.add_argument('--val-embedding-file', default=None)
    p.add_argument('--val-embedding-file-2', default=None)
    p.add_argument('--val-datasets-dir', default=None)
    p.add_argument('--out-dir', default=None,
                   help='Default: <experiment_dir>/results_old_style/  (single mode); '
                        'results/dual_head_old_style/<K>_x_<O>/  (dual mode)')
    args = p.parse_args()

    dual_mode = bool(args.k_experiment_dir or args.o_experiment_dir)
    if dual_mode and not (args.k_experiment_dir and args.o_experiment_dir):
        sys.exit('ERROR: dual-head mode requires BOTH --k-experiment-dir and --o-experiment-dir')
    if not dual_mode and not args.experiment_dir:
        sys.exit('ERROR: provide either <experiment_dir> (single mode) '
                 'or --k-experiment-dir + --o-experiment-dir (dual mode)')

    # Resolve val paths from one of the experiment configs (any one works
    # for fasta + datasets_dir; embeddings are picked separately below).
    base_for_paths = args.experiment_dir or args.k_experiment_dir
    paths = _resolve_val_paths(base_for_paths, args)
    if not paths['fasta'] or not paths['ds_dir']:
        sys.exit('ERROR: missing val_fasta or val_datasets_dir; pass via CLI '
                 'or put in config.yaml:validation of the experiment dir')

    if dual_mode:
        if not args.k_val_embedding_file or not args.o_val_embedding_file:
            sys.exit('ERROR: dual mode requires --k-val-embedding-file and --o-val-embedding-file')
        k_run = os.path.basename(args.k_experiment_dir.rstrip('/'))
        o_run = os.path.basename(args.o_experiment_dir.rstrip('/'))
        cipher_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cipher_root = os.path.dirname(cipher_root)  # scripts/analysis -> scripts -> cipher root
        out_dir = args.out_dir or os.path.join(
            cipher_root, 'results', 'dual_head_old_style', f'{k_run}_x_{o_run}')
    else:
        if not paths['emb']:
            sys.exit('ERROR: missing val_embedding_file; pass via CLI or put in config.yaml:validation')
        out_dir = args.out_dir or os.path.join(args.experiment_dir, 'results_old_style')
    os.makedirs(out_dir, exist_ok=True)

    print(f'Mode:       {"DUAL" if dual_mode else "SINGLE"}')
    print(f'Out dir:    {out_dir}')

    # Build prob dicts + class lists per mode
    if dual_mode:
        from cipher.evaluation.dual_head_predictor import _load_source_predictor
        from cipher.data.embeddings import load_embeddings
        from cipher.data.proteins import load_fasta_md5

        print(f'K source:   {args.k_experiment_dir}')
        print(f'O source:   {args.o_experiment_dir}')
        print(f'K val NPZ:  {args.k_val_embedding_file}')
        print(f'O val NPZ:  {args.o_val_embedding_file}')

        print('Loading K source predictor ...')
        k_pred = _load_source_predictor(args.k_experiment_dir)
        print('Loading O source predictor ...')
        o_pred = _load_source_predictor(args.o_experiment_dir)
        k_classes = list(k_pred.k_classes or [])
        o_classes = list(o_pred.o_classes or [])
        print(f'  K classes: {len(k_classes)}, O classes: {len(o_classes)}')

        print('Loading K val embeddings ...')
        k_emb = load_embeddings(args.k_val_embedding_file)
        print(f'  {len(k_emb)} K-side embeddings')
        print('Loading O val embeddings ...')
        o_emb = load_embeddings(args.o_val_embedding_file)
        print(f'  {len(o_emb)} O-side embeddings')

        pid_md5 = load_fasta_md5(paths['fasta'])
        all_md5s_needed = sorted(set(k_emb) | set(o_emb))

        print('Predicting K head over needed MD5s ...')
        md5_to_kvec = _precompute_head_probs(k_pred, k_classes, 'k_probs', k_emb, all_md5s_needed)
        print(f'  K probs for {len(md5_to_kvec)} MD5s')
        print('Predicting O head over needed MD5s ...')
        md5_to_ovec = _precompute_head_probs(o_pred, o_classes, 'o_probs', o_emb, all_md5s_needed)
        print(f'  O probs for {len(md5_to_ovec)} MD5s')
        n_with_both = sum(1 for m in md5_to_kvec if m in md5_to_ovec)
        print(f'  MD5s with both K + O probs: {n_with_both}')

        ds_dir_root = paths['ds_dir']
    else:
        print(f'Experiment: {args.experiment_dir}')
        print(f'Embeddings: {paths["emb"]}')
        print('Loading predictor and validation data ...')
        predictor = load_predictor(args.experiment_dir)
        val = load_validation_data(paths['fasta'], paths['emb'], paths['ds_dir'],
                                   val_embedding_file_2=paths['emb2'])
        k_classes = list(predictor.k_classes or [])
        o_classes = list(predictor.o_classes or [])
        print(f'Predictor: K classes={len(k_classes)}, O classes={len(o_classes)}')
        # Collect every MD5 used by any phage in the chosen datasets — but
        # we don't know which datasets yet, so just use every MD5 in emb_dict.
        all_md5s_needed = sorted(set(val['emb_dict']))
        md5_to_kvec = _precompute_head_probs(
            predictor, k_classes, 'k_probs', val['emb_dict'], all_md5s_needed)
        md5_to_ovec = _precompute_head_probs(
            predictor, o_classes, 'o_probs', val['emb_dict'], all_md5s_needed)
        pid_md5 = val['pid_md5']
        ds_dir_root = paths['ds_dir']

    # Discover available datasets (mirror runner.load_validation_data logic)
    from cipher.evaluation.runner import DATASETS as ALL_DATASETS
    available = [d for d in ALL_DATASETS if os.path.isdir(os.path.join(ds_dir_root, d))]
    datasets = args.datasets or available

    results = {}
    for ds in datasets:
        ds_dir = os.path.join(ds_dir_root, ds)
        if not os.path.isdir(ds_dir):
            print(f'  Skipping {ds} (not found at {ds_dir})')
            continue
        print(f'  {ds} ...', end='', flush=True)
        r = evaluate_dataset(ds_dir, ds, md5_to_kvec, md5_to_ovec,
                             k_classes, o_classes, pid_md5)
        results[ds] = r
        print(f' n={r["n_pairs"]} (data={r["n_with_data"]}), '
              f'merged HR@1={r["hr_at_k"][1]:.4f}  '
              f'K@1={r["k_hr_at_k"][1]:.4f}  '
              f'O@1={r["o_hr_at_k"][1]:.4f}  '
              f'OR@1={r["or_hr_at_k"][1]:.4f}')

    # OVERALL: pool by per-pair counts, weighted by n_pairs (strict).
    total = sum(r['n_pairs'] for r in results.values())
    if total > 0:
        def pool_strict(field, k):
            return sum(r['n_pairs'] * r[field][k] for r in results.values()) / total

        results['OVERALL'] = {
            'n_pairs': total,
            'n_with_data': sum(r['n_with_data'] for r in results.values()),
            'n_with_k': sum(r['n_with_k'] for r in results.values()),
            'n_with_o': sum(r['n_with_o'] for r in results.values()),
            'hr_at_k': {k: pool_strict('hr_at_k', k) for k in range(1, 21)},
            'k_hr_at_k': {k: pool_strict('k_hr_at_k', k) for k in range(1, 21)},
            'o_hr_at_k': {k: pool_strict('o_hr_at_k', k) for k in range(1, 21)},
            'or_hr_at_k': {k: pool_strict('or_hr_at_k', k) for k in range(1, 21)},
        }

    out_json = os.path.join(out_dir, 'old_style_eval.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f'{"dataset":<18} {"n":>6} {"merged@1":>10} {"K@1":>8} {"O@1":>8} '
          f'{"OR@1":>8}  {"OR@5":>8}')
    print('-' * 78)
    for ds, r in results.items():
        print(f'{ds:<18} {r["n_pairs"]:>6} '
              f'{r["hr_at_k"][1]:>10.4f} '
              f'{r["k_hr_at_k"][1]:>8.4f} {r["o_hr_at_k"][1]:>8.4f} '
              f'{r["or_hr_at_k"][1]:>8.4f}  {r["or_hr_at_k"][5]:>8.4f}')
    print()
    print('Strict denominator: every positive pair in scope, including pairs')
    print('whose phage has no RBP embeddings (those count as misses).')
    print('OR@k = ceiling under perfect K vs O merge (K-rank ≤ k OR O-rank ≤ k).')
    print(f'\nResults: {out_json}')


if __name__ == '__main__':
    main()
