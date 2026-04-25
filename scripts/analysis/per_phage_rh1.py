#!/usr/bin/env python3
"""Per-phage rh@1 prediction CSV — for agent 5's per-phage / per-K analysis.

Produces one row per (phage, dataset) for PHL and PBIP (or any subset
specified via --datasets). Each row contains:

    dataset
    phage_id
    model_id
    k_true_hosts            ; -separated K-types of positive hosts for this phage
    n_positive_hosts
    k_top1_pred             K-type of the rank-1 predicted host
    rh1_hit                 1 if rank-1 host is a positive, else 0
    rank_of_best_positive   competition rank of the best-ranked positive host
    k_top5_preds            ; -separated K-types of ranks 1..5

Mirrors the ranking logic in `cipher.evaluation.ranking.rank_hosts` /
`evaluate_rankings` exactly; competition tie-breaking. Reuses the
trained model via `predict.py::get_predictor`. Honours `--head-mode`
(default `both`, but `k_only` is the recommended setting given the
2026-04-23 finding).

Usage:
    python scripts/analysis/per_phage_rh1.py <run_dir> -o results/analysis/per_phage_rh1_phl_pbip.csv
    python scripts/analysis/per_phage_rh1.py <run_dir> --head-mode k_only
    python scripts/analysis/per_phage_rh1.py <run_dir> --datasets PhageHostLearn PBIP CHEN UCSD
"""

import argparse
import csv
import importlib.util
import os
import sys
from collections import defaultdict
from pathlib import Path


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

from cipher.evaluation.runner import find_project_root, find_predict_module, load_validation_data
from cipher.evaluation.ranking import _ranks_with_ties
from cipher.data.interactions import load_interaction_pairs, load_phage_protein_mapping


def _load_predictor(run_dir):
    predict_path, model_dir = find_predict_module(run_dir)
    if predict_path is None:
        raise FileNotFoundError(f'No predict.py for {run_dir}')
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


def _per_phage_rh1_for_dataset(predictor, dataset_name, dataset_dir,
                               emb_dict, pid_md5, model_id):
    """Return list of dicts, one per phage, with the columns agent 5 asked for."""
    pairs = load_interaction_pairs(dataset_dir)
    pm_path = os.path.join(dataset_dir, 'metadata', 'phage_protein_mapping.csv')
    phage_protein_map = load_phage_protein_mapping(pm_path)

    interactions = defaultdict(dict)
    serotypes = {}
    for p in pairs:
        interactions[p['phage_id']][p['host_id']] = p['label']
        serotypes[p['host_id']] = {'K': p['host_K'], 'O': p['host_O']}

    rows = []
    for phage in sorted(interactions.keys()):
        host_labels = interactions[phage]
        pos_hosts = {h for h, label in host_labels.items() if label == 1}
        if not pos_hosts:
            continue

        proteins = phage_protein_map.get(phage, set())
        prot_md5s = [pid_md5[p] for p in proteins if p in pid_md5]
        prot_embs = [emb_dict[m] for m in prot_md5s if m in emb_dict]

        if not prot_embs:
            # Phage with no scorable proteins — record as-is with empty preds
            rows.append({
                'dataset': dataset_name,
                'phage_id': phage,
                'model_id': model_id,
                'k_true_hosts': ';'.join(sorted({serotypes.get(h, {}).get('K', '') for h in pos_hosts})),
                'n_positive_hosts': len(pos_hosts),
                'k_top1_pred': '',
                'rh1_hit': '',
                'rank_of_best_positive': '',
                'k_top5_preds': '',
            })
            continue

        # Score every candidate host
        host_scores = []
        for host in host_labels.keys():
            if host not in serotypes:
                continue
            hk = serotypes[host]['K']
            ho = serotypes[host]['O']
            score = predictor.score_pair(prot_embs, hk, ho)
            if score is not None:
                host_scores.append((host, score))

        if not host_scores:
            continue

        host_scores.sort(key=lambda x: -x[1])
        host_to_rank = _ranks_with_ties(host_scores, tie_method='competition')

        # Top-1 host + its K
        top_host, _ = host_scores[0]
        k_top1 = serotypes.get(top_host, {}).get('K', '')

        # Top-5 K-types (preserves the order returned by the sort)
        top5 = []
        for h, _ in host_scores[:5]:
            top5.append(serotypes.get(h, {}).get('K', ''))

        # rh1: did the top-1 host land in the positives?
        rh1_hit = 1 if top_host in pos_hosts else 0

        # Best (lowest) rank achieved by any positive host
        ranks_of_positives = [host_to_rank[h] for h in pos_hosts if h in host_to_rank]
        rank_of_best = min(ranks_of_positives) if ranks_of_positives else ''

        true_ks = sorted({serotypes.get(h, {}).get('K', '') for h in pos_hosts})

        rows.append({
            'dataset': dataset_name,
            'phage_id': phage,
            'model_id': model_id,
            'k_true_hosts': ';'.join(true_ks),
            'n_positive_hosts': len(pos_hosts),
            'k_top1_pred': k_top1,
            'rh1_hit': rh1_hit,
            'rank_of_best_positive': rank_of_best,
            'k_top5_preds': ';'.join(top5),
        })

    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('run_dir', help='Path to the trained experiment directory')
    p.add_argument('--datasets', nargs='+',
                   default=['PhageHostLearn', 'PBIP'],
                   help='Validation datasets to score (default: PHL + PBIP)')
    p.add_argument('--head-mode', choices=('both', 'k_only', 'o_only'),
                   default='both',
                   help="Eval mode for predictor.score_pair (default: both, "
                        "matching cipher-evaluate's default zscore combined). "
                        "'k_only' is recommended for PHL per "
                        "notes/findings/2026-04-23_head_eval_phage_breadth.md.")
    p.add_argument('--score-norm', choices=('zscore', 'raw'), default='zscore',
                   help='Score normalization to use (default: zscore)')
    p.add_argument('--model-id', default=None,
                   help='Override model_id in CSV (default: derived from run_dir basename + head-mode)')
    p.add_argument('-o', '--output',
                   default='results/analysis/per_phage_rh1_phl_pbip.csv',
                   help='Output CSV path')
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    val_fasta, val_emb_file, val_datasets_dir = _resolve_val_paths(run_dir)

    predictor = _load_predictor(run_dir)
    predictor.head_mode = args.head_mode
    predictor.score_normalization = args.score_norm

    val_data = load_validation_data(val_fasta, val_emb_file, val_datasets_dir)

    # Default model id: <run-name>_<head_mode>(+_raw)
    if args.model_id:
        model_id = args.model_id
    else:
        suffix = args.head_mode if args.head_mode != 'both' else 'combined'
        if args.score_norm == 'raw':
            suffix += '_raw'
        model_id = f'{os.path.basename(run_dir)}_{suffix}'

    print(f'run_dir:    {run_dir}')
    print(f'model_id:   {model_id}')
    print(f'head_mode:  {args.head_mode}')
    print(f'score_norm: {args.score_norm}')
    print(f'datasets:   {args.datasets}')
    print()

    rows = []
    for ds in args.datasets:
        ds_dir = os.path.join(val_datasets_dir, ds)
        if not os.path.isdir(ds_dir):
            print(f'  skip {ds} (no dataset dir)')
            continue
        ds_rows = _per_phage_rh1_for_dataset(
            predictor, ds, ds_dir, val_data['emb_dict'],
            val_data['pid_md5'], model_id)
        rows.extend(ds_rows)
        n_hits = sum(1 for r in ds_rows if r['rh1_hit'] == 1)
        print(f'  {ds}: {len(ds_rows)} phages, top-1 hit on {n_hits}'
              f' ({n_hits / len(ds_rows):.3f})' if ds_rows else f'  {ds}: 0 phages')

    # Write CSV
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not rows:
        print('No rows produced.', file=sys.stderr)
        return 1
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n{len(rows)} rows -> {out_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
