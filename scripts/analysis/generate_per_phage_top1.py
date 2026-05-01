"""Per-phage top-1 K predictions from a cipher experiment.

Runs cipher's standard predictor over every validation phage's
annotated RBPs, takes the max-probability K per phage (top-1), and
emits a TSV that mirrors agent 6's 4-way per_phage_diagnostics
schema. This is the per-phage output needed to ask "which K-types
does cipher catch that Tropi misses, and vice versa".

Inputs (standard cipher validation pipeline):
  - experiment dir (sweep_prott5_mean_cl70 by default)
  - val FASTA (RBPs)
  - val embedding NPZ (md5-keyed)
  - val datasets dir (HOST_RANGE/<DS>/metadata/phage_protein_mapping.csv +
                                 metadata/interaction_pairs.csv)

Output:
  <out>/per_phage_predictions_<run_name>.tsv with columns:
    dataset, phage_id, n_proteins, positive_K_types,
    cp_top1_set, cp_hit@1, cp_top5_set, cp_hit@5

Usage (Delta):
    python scripts/analysis/generate_per_phage_top1.py \\
        experiments/attention_mlp/sweep_prott5_mean_cl70 \\
        --val-embedding-file /work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz \\
        --out-dir results/analysis/per_phage_top1
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

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
from cipher.evaluation.runner import (
    find_predict_module, load_validation_data,
)


DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']


def _load_predictor(run_dir):
    predict_path, model_dir = find_predict_module(run_dir)
    if predict_path is None:
        raise FileNotFoundError(f'No predict.py for {run_dir}')
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    spec = importlib.util.spec_from_file_location('predict', predict_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_predictor(os.path.abspath(run_dir))


def _resolve_val_paths(run_dir, cli):
    cfg = {}
    cfg_path = os.path.join(run_dir, 'config.yaml')
    if os.path.exists(cfg_path):
        cfg = (yaml.safe_load(open(cfg_path)) or {}).get('validation') or {}
    return {
        'fasta': cli.val_fasta or cfg.get('val_fasta'),
        'emb':   cli.val_embedding_file or cfg.get('val_embedding_file'),
        'ds_dir': cli.val_datasets_dir or cfg.get('val_datasets_dir'),
    }


def predict_phage_top_k(predictor, proteins, emb_dict, pid_md5, k=5):
    """For one phage: for each annotated RBP, get K probabilities; take
    max over proteins per K class; return top-k K-types by max-prob."""
    per_class_max = defaultdict(float)
    n_proteins = 0
    for pid in proteins:
        md5 = pid_md5.get(pid)
        if md5 is None:
            continue
        emb = emb_dict.get(md5)
        if emb is None:
            continue
        n_proteins += 1
        out = predictor.predict_protein(emb)
        for k_type, p in out.get('k_probs', {}).items():
            if p > per_class_max[k_type]:
                per_class_max[k_type] = p
    if not per_class_max:
        return [], 0
    # Sort by max-prob desc, take top-k
    ranked = sorted(per_class_max.items(), key=lambda x: -x[1])[:k]
    return [k for k, _ in ranked], n_proteins


def main():
    p = argparse.ArgumentParser()
    p.add_argument('experiment_dir')
    p.add_argument('--val-fasta', default=None)
    p.add_argument('--val-embedding-file', default=None)
    p.add_argument('--val-datasets-dir', default=None)
    p.add_argument('--out-dir', default=None,
                   help='default: <experiment>/results/per_phage_top1.tsv')
    args = p.parse_args()

    paths = _resolve_val_paths(args.experiment_dir, args)
    for k in ('fasta', 'emb', 'ds_dir'):
        if not paths[k]:
            sys.exit(f'ERROR: missing val path {k!r}')

    run_name = os.path.basename(args.experiment_dir.rstrip('/'))
    out_dir = args.out_dir or os.path.join(args.experiment_dir, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_tsv = os.path.join(out_dir, f'per_phage_top1_{run_name}.tsv')

    print(f'Run: {run_name}')
    print(f'Out: {out_tsv}')

    predictor = _load_predictor(args.experiment_dir)
    val = load_validation_data(paths['fasta'], paths['emb'], paths['ds_dir'])

    rows_out = []
    for ds in DATASETS:
        ds_dir = os.path.join(paths['ds_dir'], ds)
        if not os.path.isdir(ds_dir):
            print(f'  Skipping {ds} (not found)')
            continue
        pairs = load_interaction_pairs(ds_dir)
        pm_path = os.path.join(ds_dir, 'metadata', 'phage_protein_mapping.csv')
        phage_protein_map = load_phage_protein_mapping(pm_path)

        # Map phage → set of positive K-types from the interaction pairs
        phage_pos_K = defaultdict(set)
        for pp in pairs:
            if pp['label'] == 1 and pp['host_K']:
                phage_pos_K[pp['phage_id']].add(pp['host_K'])

        n_phages = 0
        n_hit1 = 0
        n_hit5 = 0
        for phage_id in sorted(phage_pos_K):
            pos_Ks = phage_pos_K[phage_id]
            proteins = phage_protein_map.get(phage_id, set())
            if not proteins:
                continue
            top_k, n_prot = predict_phage_top_k(
                predictor, proteins, val['emb_dict'], val['pid_md5'], k=5)
            if n_prot == 0:
                continue
            n_phages += 1
            top1 = top_k[:1]
            top5 = top_k[:5]
            hit1 = int(any(t in pos_Ks for t in top1))
            hit5 = int(any(t in pos_Ks for t in top5))
            n_hit1 += hit1
            n_hit5 += hit5
            rows_out.append({
                'dataset': ds,
                'phage_id': phage_id,
                'n_proteins': n_prot,
                'positive_K_types': ';'.join(sorted(pos_Ks)),
                'cp_top1_set': ';'.join(top1),
                'cp_hit@1': hit1,
                'cp_top5_set': ';'.join(top5),
                'cp_hit@5': hit5,
            })
        if n_phages > 0:
            print(f'  {ds:<16} n={n_phages:>4}  HR@1={n_hit1/n_phages:.3f}  '
                  f'HR@5={n_hit5/n_phages:.3f}')

    with open(out_tsv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'dataset', 'phage_id', 'n_proteins', 'positive_K_types',
            'cp_top1_set', 'cp_hit@1', 'cp_top5_set', 'cp_hit@5'])
        w.writeheader()
        w.writerows(rows_out)

    print(f'\nWrote {len(rows_out)} rows to {out_tsv}')


if __name__ == '__main__':
    main()
