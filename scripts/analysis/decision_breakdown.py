#!/usr/bin/env python3
"""Per-pair decision breakdown: which head (K or O) drove each rank-1 pair?

For each (phage, candidate_host) pair in a validation dataset we rank the
candidate, then classify the rank-1 pairs as:
  - TP (true positive):  pair has label=1, rank=1. Model correctly put a real
                          interaction partner at the top.
  - FP (false positive): pair has label=0, rank=1. Model incorrectly put a
                          non-interaction partner at the top (can happen via
                          score ties between positives and negatives).

FN and TN are not reported — focus is on where the model's rank-1 picks
come from.

For each TP and FP pair we also record which head (K or O) drove its score
(i.e. which head produced the larger z-scored probability on the winning
protein). This tells us:
  * of our TP rank-1 picks, what fraction were K-driven vs O-driven
    — i.e. which head is doing the "correct" work
  * of our FP rank-1 picks, what fraction were K-driven vs O-driven
    — i.e. which head is responsible for the errors

Also prints the bucketed rank distribution of positive pairs for context.

Usage:
    python scripts/analysis/decision_breakdown.py <run_dir> [--datasets ...]
"""

import argparse
import importlib.util
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from cipher.evaluation.runner import find_project_root, find_predict_module, load_validation_data
from cipher.evaluation.ranking import _ranks_with_ties
from cipher.data.interactions import load_interaction_pairs, load_phage_protein_mapping
from cipher.data.serotypes import is_null


PRIMARY_ORDER = ['PBIP', 'PhageHostLearn', 'UCSD', 'CHEN', 'GORODNICHIV']


def _load_predictor(run_dir):
    predict_path, model_dir = find_predict_module(run_dir)
    if predict_path is None:
        raise FileNotFoundError(f'No predict.py found for {run_dir}')
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    spec = importlib.util.spec_from_file_location('predict', predict_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_predictor(os.path.abspath(run_dir))


def _resolve_val_paths(run_dir):
    import yaml
    config_path = os.path.join(run_dir, 'config.yaml')
    val_cfg = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            full_cfg = yaml.safe_load(f) or {}
        val_cfg = full_cfg.get('validation', {})
    project_root = find_project_root(run_dir) or os.getcwd()
    def rp(key, default):
        v = val_cfg.get(key, default)
        return v if os.path.isabs(v) else os.path.join(project_root, v)
    return (
        rp('val_fasta',          'data/validation_data/metadata/validation_rbps_all.faa'),
        rp('val_embedding_file', 'data/validation_data/embeddings/esm2_650m_md5.npz'),
        rp('val_datasets_dir',   'data/validation_data/HOST_RANGE'),
    )


def _zscore(probs_dict, target_class):
    """Z-score of target_class's prob against the distribution of this head."""
    if not probs_dict or target_class not in probs_dict:
        return None
    target_p = probs_dict[target_class]
    vals = np.asarray(list(probs_dict.values()), dtype=np.float64)
    mu = vals.mean()
    sigma = vals.std()
    if sigma < 1e-12:
        return 0.0
    return float((target_p - mu) / sigma)


def _score_host_with_provenance(predictor, prot_ids, prot_embs, host_k, host_o):
    """Reimplementation of score_pair that tracks the winning head/protein.

    Returns (best_score, winning_head, winning_protein_id, k_val, o_val)
    or None if no protein can be scored.
    """
    has_k = host_k is not None and not is_null(host_k)
    has_o = host_o is not None and not is_null(host_o)
    if not has_k and not has_o:
        return None

    best = None  # (score, head, pid, k_val, o_val)

    for pid, emb in zip(prot_ids, prot_embs):
        preds = predictor.predict_protein(emb)
        k_probs = preds.get('k_probs', {})
        o_probs = preds.get('o_probs', {})

        k_val = _zscore(k_probs, host_k) if has_k else None
        o_val = _zscore(o_probs, host_o) if has_o else None

        options = []
        if k_val is not None:
            options.append(('K', k_val))
        if o_val is not None:
            options.append(('O', o_val))
        if not options:
            continue

        head, prot_score = max(options, key=lambda x: x[1])
        if best is None or prot_score > best[0]:
            best = (prot_score, head, pid, k_val, o_val)

    return best  # may be None


def _rank_bucket(rank):
    if rank == 1:
        return '1'
    if 2 <= rank <= 5:
        return '2-5'
    if 6 <= rank <= 20:
        return '6-20'
    return '>20'


def _analyze_dataset(predictor, ds_name, ds_dir, emb_dict, pid_md5):
    pairs = load_interaction_pairs(ds_dir)
    pm_path = os.path.join(ds_dir, 'metadata', 'phage_protein_mapping.csv')
    phage_protein_map = load_phage_protein_mapping(pm_path)

    interactions = defaultdict(dict)
    serotypes = {}
    for p in pairs:
        interactions[p['phage_id']][p['host_id']] = p['label']
        serotypes[p['host_id']] = {'K': p['host_K'], 'O': p['host_O']}

    # Per-pair counters at rank 1 (by winning head)
    tp_head = Counter()  # label=1, rank=1
    fp_head = Counter()  # label=0, rank=1
    # Positive-host rank distribution (for context; rank buckets × head)
    positive_by_bucket = defaultdict(Counter)

    n_pos_pairs = 0
    n_neg_pairs = 0
    n_phages_scored = 0

    for phage, host_labels in interactions.items():
        pos_hosts = {h for h, label in host_labels.items() if label == 1}
        if not pos_hosts:
            # Matches evaluate_rankings: phages without positives are skipped
            continue

        # Resolve protein IDs + embeddings for this phage
        proteins = phage_protein_map.get(phage, set())
        prot_items = []
        for pid in proteins:
            if pid not in pid_md5:
                continue
            md5 = pid_md5[pid]
            if md5 not in emb_dict:
                continue
            prot_items.append((pid, emb_dict[md5]))
        if not prot_items:
            continue
        prot_ids = [p[0] for p in prot_items]
        prot_embs = [p[1] for p in prot_items]

        # Score EVERY candidate (positive and negative), track provenance
        scored = []  # list of (host, label, score_tuple)
        for host, label in host_labels.items():
            if host not in serotypes:
                continue
            hk = serotypes[host]['K']
            ho = serotypes[host]['O']
            r = _score_host_with_provenance(predictor, prot_ids, prot_embs, hk, ho)
            if r is not None:
                scored.append((host, label, r))
        if not scored:
            continue
        n_phages_scored += 1

        # Rank dict with competition ties (positives and negatives both ranked)
        scored.sort(key=lambda x: -x[2][0])
        sorted_for_ties = [(h, s[0]) for h, _, s in scored]
        host_to_rank = _ranks_with_ties(sorted_for_ties, tie_method='competition')

        # Classify every pair: TP/FP for rank 1, buckets for positives
        for host, label, r in scored:
            head = r[1]
            rank = host_to_rank[host]
            if label == 1:
                n_pos_pairs += 1
                positive_by_bucket[_rank_bucket(rank)][head] += 1
                if rank == 1:
                    tp_head[head] += 1
            else:
                n_neg_pairs += 1
                if rank == 1:
                    fp_head[head] += 1

    return {
        'n_phages_scored': n_phages_scored,
        'n_pos_pairs': n_pos_pairs,
        'n_neg_pairs': n_neg_pairs,
        'tp_head': tp_head,
        'fp_head': fp_head,
        'positive_by_bucket': positive_by_bucket,
    }


def _pct(n, total):
    return f'{100 * n / total:.1f}%' if total else '—'


def _print_ds_report(ds_name, r):
    n = r['n_phages_scored']
    n_pos = r['n_pos_pairs']
    n_neg = r['n_neg_pairs']
    tp_total = sum(r['tp_head'].values())
    fp_total = sum(r['fp_head'].values())

    print(f'=== {ds_name} — {n} scored phages, {n_pos} pos pairs, {n_neg} neg pairs ===')
    if not n:
        print('  (no scorable phages)')
        print()
        return

    # TP@1 = positive (label=1) pair at rank 1
    tp_k = r['tp_head'].get('K', 0)
    tp_o = r['tp_head'].get('O', 0)
    tp_k_share = tp_k / tp_total if tp_total else 0.0
    tp_o_share = tp_o / tp_total if tp_total else 0.0
    print(f'  TP@1 (positive pair at rank 1): {tp_total}/{n_pos} = '
          f'{(tp_total / n_pos if n_pos else 0):.3f}   [= HR@1]')
    print(f'       K-driven: {tp_k:>4}  ({tp_k_share:.1%})')
    print(f'       O-driven: {tp_o:>4}  ({tp_o_share:.1%})')
    print()

    # FP@1 = negative (label=0) pair at rank 1 (via ties)
    fp_k = r['fp_head'].get('K', 0)
    fp_o = r['fp_head'].get('O', 0)
    fp_k_share = fp_k / fp_total if fp_total else 0.0
    fp_o_share = fp_o / fp_total if fp_total else 0.0
    print(f'  FP@1 (negative pair at rank 1): {fp_total}/{n_neg} = '
          f'{(fp_total / n_neg if n_neg else 0):.3f}')
    print(f'       K-driven: {fp_k:>4}  ({fp_k_share:.1%})')
    print(f'       O-driven: {fp_o:>4}  ({fp_o_share:.1%})')
    print()

    # Positive-pair rank distribution (keeps useful rank context)
    buckets = ['1', '2-5', '6-20', '>20']
    print(f'  Positive pair rank distribution — bucket × winning head:')
    print(f'    {"bucket":<8} {"K":>6} {"O":>6}  total')
    for b in buckets:
        c = r['positive_by_bucket'].get(b, Counter())
        k = c.get('K', 0)
        o = c.get('O', 0)
        print(f'    {b:<8} {k:>6} {o:>6}  {k + o:>5}')
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('run_dir')
    p.add_argument('--datasets', nargs='+', default=None)
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    val_fasta, val_emb_file, val_datasets_dir = _resolve_val_paths(run_dir)
    print(f'Run: {run_dir}')
    print(f'Val embeddings: {val_emb_file}')
    print()

    predictor = _load_predictor(run_dir)
    val_data = load_validation_data(val_fasta, val_emb_file, val_datasets_dir)
    datasets = args.datasets or val_data['available_datasets']
    ordered = [d for d in PRIMARY_ORDER if d in datasets]
    ordered += [d for d in datasets if d not in PRIMARY_ORDER]

    for ds in ordered:
        ds_dir = os.path.join(val_datasets_dir, ds)
        if not os.path.isdir(ds_dir):
            continue
        r = _analyze_dataset(predictor, ds, ds_dir,
                             val_data['emb_dict'], val_data['pid_md5'])
        _print_ds_report(ds, r)


if __name__ == '__main__':
    main()
