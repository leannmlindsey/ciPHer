"""Emit per-pair rank-5 predictions + per-protein K-probs for a validation dataset.

Produces two artifacts under `<experiment_dir>/results/analysis/`:

1. `<dataset>_pair_predictions.csv` — one row per (phage, positive_host) pair:
       phage_id, host_id, k_true, o_true,
       k_top1_pred, o_top1_pred, rank_of_true_host,
       k_top5_preds, o_top5_preds, top1_host_id

   Agent 5's K-type confusion matrix is one pd.crosstab away:
       pd.crosstab(df['k_true'], df['k_top1_pred'])

2. `<dataset>_protein_k_probs.npz` + `<dataset>_k_class_index.json` —
   per-validation-protein K-type probability vectors (indexed by the
   class ordering in `_k_class_index.json`, which mirrors the trained
   predictor's `.k_classes` list, which in turn matches the index order
   of the softmax output). Agent 5 uses this for representation-level
   confusion distinct from the pair-level confusion.

Re-uses `cipher.evaluation.runner.load_predictor` and
`cipher.evaluation.ranking.rank_hosts` so the ranking is byte-identical
to `python -m cipher.evaluation.runner`.

Usage:
    python scripts/analysis/export_pair_predictions.py \\
        experiments/attention_mlp/highconf_pipeline_K_prott5_mean
    # default dataset: PhageHostLearn

    python scripts/analysis/export_pair_predictions.py <exp_dir> \\
        --dataset PBIP --no-protein-probs
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


# Make the `cipher` package importable when run from a checkout without
# `pip install -e .` (common on HPC login nodes). Walks up from this
# file until it finds a `src/cipher/` directory, then prepends `src/`.
def _ensure_cipher_on_path():
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        candidate = parent / 'src' / 'cipher'
        if candidate.is_dir():
            src = str(parent / 'src')
            if src not in sys.path:
                sys.path.insert(0, src)
            return
    # Fall through: let the ImportError below surface with a helpful hint.

_ensure_cipher_on_path()

from cipher.evaluation.runner import load_predictor, load_validation_data
from cipher.evaluation.ranking import rank_hosts, _ranks_with_ties
from cipher.data.interactions import (
    load_interaction_pairs, load_phage_protein_mapping,
)


def export_pair_csv(predictor, emb_dict, pid_md5, dataset_name,
                    val_datasets_dir, out_csv):
    dataset_dir = os.path.join(val_datasets_dir, dataset_name)
    pairs = load_interaction_pairs(dataset_dir)
    pm_path = os.path.join(dataset_dir, 'metadata', 'phage_protein_mapping.csv')
    phage_protein_map = load_phage_protein_mapping(pm_path)

    interactions = defaultdict(dict)
    serotypes = {}
    for p in pairs:
        interactions[p['phage_id']][p['host_id']] = p['label']
        serotypes[p['host_id']] = {'K': p['host_K'], 'O': p['host_O']}

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    n_written = 0
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'phage_id', 'host_id',
            'k_true', 'o_true',
            'k_top1_pred', 'o_top1_pred',
            'rank_of_true_host',
            'k_top5_preds', 'o_top5_preds',
            'top1_host_id',
        ])
        for phage in sorted(interactions.keys()):
            pos_hosts = [h for h, label in interactions[phage].items()
                         if label == 1]
            if not pos_hosts:
                continue
            candidates = list(interactions[phage].keys())
            proteins = phage_protein_map.get(phage, set())
            ranked = rank_hosts(predictor, proteins, candidates, serotypes,
                                emb_dict, pid_md5)
            if not ranked:
                continue

            host_to_rank = _ranks_with_ties(ranked, tie_method='competition')
            top5_hosts = [h for h, _ in ranked[:5]]
            k_top5 = [str(serotypes[h]['K']) for h in top5_hosts if h in serotypes]
            o_top5 = [str(serotypes[h]['O']) for h in top5_hosts if h in serotypes]
            top1_host = ranked[0][0]
            k_top1 = serotypes[top1_host]['K']
            o_top1 = serotypes[top1_host]['O']

            for pos_h in pos_hosts:
                if pos_h not in host_to_rank or pos_h not in serotypes:
                    continue
                w.writerow([
                    phage, pos_h,
                    serotypes[pos_h]['K'], serotypes[pos_h]['O'],
                    k_top1, o_top1,
                    host_to_rank[pos_h],
                    ';'.join(k_top5),
                    ';'.join(o_top5),
                    top1_host,
                ])
                n_written += 1

    print(f'  Wrote {n_written} rows to {out_csv}')


def export_protein_probs(predictor, emb_dict, pid_md5, dataset_name,
                         val_datasets_dir, out_npz, out_class_index):
    """Per-validation-protein K-type probability vectors.

    Keyed by MD5 so agent 5 can join to any protein-level metadata.
    Only writes proteins that appear in phages in the target dataset.
    """
    dataset_dir = os.path.join(val_datasets_dir, dataset_name)
    pm_path = os.path.join(dataset_dir, 'metadata', 'phage_protein_mapping.csv')
    phage_protein_map = load_phage_protein_mapping(pm_path)

    # Collect the MD5s of every protein in phages of this dataset
    md5s_needed = set()
    for proteins in phage_protein_map.values():
        for p in proteins:
            if p in pid_md5:
                md5s_needed.add(pid_md5[p])

    k_class_list = list(predictor.k_classes)
    class_to_idx = {c: i for i, c in enumerate(k_class_list)}
    n_classes = len(k_class_list)

    probs_out = {}
    for md5 in md5s_needed:
        if md5 not in emb_dict:
            continue
        result = predictor.predict_protein(emb_dict[md5])
        k_probs = result.get('k_probs', {})
        vec = np.zeros(n_classes, dtype=np.float32)
        for c, p in k_probs.items():
            if c in class_to_idx:
                vec[class_to_idx[c]] = float(p)
        probs_out[md5] = vec

    os.makedirs(os.path.dirname(out_npz) or '.', exist_ok=True)
    np.savez_compressed(out_npz, **probs_out)
    with open(out_class_index, 'w') as f:
        json.dump(k_class_list, f, indent=2)
    print(f'  Wrote {len(probs_out)} protein K-prob vectors '
          f'(dim={n_classes}) to {out_npz}')
    print(f'  Class index: {out_class_index}')


def _resolve_val_paths(experiment_dir, cli):
    """Pick up validation paths from the CLI first, else the run's config.yaml."""
    cfg_path = os.path.join(experiment_dir, 'config.yaml')
    val_cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            val_cfg = yaml.safe_load(f).get('validation', {}) or {}
    return {
        'fasta': cli.val_fasta or val_cfg.get('val_fasta'),
        'emb': cli.val_embedding_file or val_cfg.get('val_embedding_file'),
        'emb2': cli.val_embedding_file_2 or val_cfg.get('val_embedding_file_2'),
        'ds_dir': cli.val_datasets_dir or val_cfg.get('val_datasets_dir'),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('experiment_dir',
                   help='Path to a trained experiment directory (config.yaml + predict.py)')
    p.add_argument('--dataset', default='PhageHostLearn',
                   help='Which validation dataset to export (default: PhageHostLearn)')
    p.add_argument('--out-dir', default=None,
                   help='Output directory (default: <experiment_dir>/results/analysis/)')
    p.add_argument('--val-fasta', default=None)
    p.add_argument('--val-embedding-file', default=None)
    p.add_argument('--val-embedding-file-2', default=None)
    p.add_argument('--val-datasets-dir', default=None)
    p.add_argument('--no-protein-probs', action='store_true',
                   help='Skip the per-protein K-prob NPZ (primary CSV only).')
    args = p.parse_args()

    if not os.path.isdir(args.experiment_dir):
        sys.exit(f'ERROR: {args.experiment_dir} is not a directory')

    paths = _resolve_val_paths(args.experiment_dir, args)
    for key, val in paths.items():
        if not val:
            sys.exit(f'ERROR: missing validation path {key!r}; '
                     f'pass via CLI or put in config.yaml:validation')

    out_dir = args.out_dir or os.path.join(
        args.experiment_dir, 'results', 'analysis')

    print(f'Experiment: {args.experiment_dir}')
    print(f'Dataset:    {args.dataset}')
    print(f'Out dir:    {out_dir}')

    predictor = load_predictor(args.experiment_dir)
    val_data = load_validation_data(
        paths['fasta'], paths['emb'], paths['ds_dir'],
        val_embedding_file_2=paths['emb2'])

    out_csv = os.path.join(out_dir, f'{args.dataset}_pair_predictions.csv')
    export_pair_csv(predictor,
                    val_data['emb_dict'], val_data['pid_md5'],
                    args.dataset, paths['ds_dir'], out_csv)

    if not args.no_protein_probs:
        out_npz = os.path.join(out_dir, f'{args.dataset}_protein_k_probs.npz')
        out_class = os.path.join(out_dir, f'{args.dataset}_k_class_index.json')
        export_protein_probs(predictor,
                             val_data['emb_dict'], val_data['pid_md5'],
                             args.dataset, paths['ds_dir'],
                             out_npz, out_class)

    print('Done.')


if __name__ == '__main__':
    main()
