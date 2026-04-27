"""Per-head host-rank HR@k under strict denominator, for ONE cipher experiment.

For each validation dataset, runs cipher's standard host-ranking
(`cipher.evaluation.ranking.rank_hosts`) under THREE head-modes —
masking the predictor's `predict_protein()` to suppress one head's
output. Computes HR@k=1..20 per mode plus the OR-combine ceiling
(K-rank ≤ k OR O-rank ≤ k), all over the STRICT denominator:

  strict denominator = positive pairs whose phage has at least one
  annotated RBP in the validation FASTA. Pairs whose phage has no
  annotated RBPs are excluded (model has no input). Pairs where
  embedding NPZ has no vector for any annotated RBP are KEPT and
  counted as misses (the embedding extractor failed — that's a model
  coverage failure to surface, not normalize away).

Output: <experiment>/results/per_head_strict_eval.json with:
  {
    "<dataset>": {
      "n_strict": int,           # positive pairs with annotated RBPs
      "n_with_embedded_rbp": int,# subset where ≥1 RBP has embedding
      "k_only": {"hr_at_k": {1: ..., 2: ..., ..., 20: ...}},
      "o_only": {"hr_at_k": {...}},
      "merged": {"hr_at_k": {...}},
      "or":     {"hr_at_k": {...}}
    },
    ...
  }

This is the host-rank analog of `old_style_eval.py` (which is class-rank).

Usage:
    python scripts/analysis/per_head_strict_eval.py <experiment_dir>
    python scripts/analysis/per_head_strict_eval.py <experiment_dir> \\
        --val-embedding-file data/validation_data/embeddings/prott5_xl_md5.npz
"""

import argparse
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
from cipher.evaluation.ranking import rank_hosts, _ranks_with_ties


MAX_K = 20
DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']
MODES = ('k_only', 'o_only', 'merged')


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


def _mask_head(orig, which):
    if which == 'merged':
        return orig
    def wrapped(emb):
        out = orig(emb)
        if which == 'k_only':
            out['o_probs'] = {}
        elif which == 'o_only':
            out['k_probs'] = {}
        return out
    return wrapped


def _resolve_val_paths(run_dir, cli):
    cfg = {}
    cfg_path = os.path.join(run_dir, 'config.yaml')
    if os.path.exists(cfg_path):
        cfg = (yaml.safe_load(open(cfg_path)) or {}).get('validation') or {}
    return {
        'fasta': cli.val_fasta or cfg.get('val_fasta'),
        'emb': cli.val_embedding_file or cfg.get('val_embedding_file'),
        'emb2': cli.val_embedding_file_2 or cfg.get('val_embedding_file_2'),
        'ds_dir': cli.val_datasets_dir or cfg.get('val_datasets_dir'),
    }


def evaluate_dataset_per_head(predictor, original_predict, dataset_dir,
                              emb_dict, pid_md5, max_k=MAX_K):
    """Run all three head-modes on one dataset; return strict per-head HR@k."""
    pairs = load_interaction_pairs(dataset_dir)
    pm_path = os.path.join(dataset_dir, 'metadata', 'phage_protein_mapping.csv')
    phage_protein_map = load_phage_protein_mapping(pm_path)

    interactions = defaultdict(dict)
    serotypes = {}
    for p in pairs:
        interactions[p['phage_id']][p['host_id']] = p['label']
        serotypes[p['host_id']] = {'K': p['host_K'], 'O': p['host_O']}

    # Strict denominator: positive pairs whose phage has at least one
    # annotated RBP (md5 present in pid_md5). No-embedding pairs are
    # KEPT (counted as misses), not excluded.
    strict_pairs = []
    n_with_embedded_rbp = 0
    for phage in sorted(interactions):
        pos_hosts = [h for h, lbl in interactions[phage].items() if lbl == 1]
        if not pos_hosts:
            continue
        proteins = phage_protein_map.get(phage, set())
        annotated_md5s = [pid_md5.get(p) for p in proteins
                          if p in pid_md5]
        if not annotated_md5s:
            continue  # phage has zero annotated RBPs → out of scope
        any_emb = any(m in emb_dict for m in annotated_md5s)
        for h in pos_hosts:
            if h not in serotypes:
                continue
            strict_pairs.append((phage, h))
            if any_emb:
                pass
        if any_emb:
            n_with_embedded_rbp += sum(1 for h in pos_hosts if h in serotypes)

    n_strict = len(strict_pairs)

    # Run each mode, collect per-pair ranks
    mode_ranks = {m: {} for m in MODES}  # {mode: {(phage, host): rank or None}}
    for mode in MODES:
        predictor.predict_protein = _mask_head(original_predict, mode)
        per_phage_ranks = {}  # cache: phage -> {host: rank}
        for phage, host in strict_pairs:
            if phage not in per_phage_ranks:
                proteins = phage_protein_map.get(phage, set())
                candidates = list(interactions[phage].keys())
                ranked = rank_hosts(predictor, proteins, candidates,
                                    serotypes, emb_dict, pid_md5)
                if ranked:
                    per_phage_ranks[phage] = _ranks_with_ties(
                        ranked, tie_method='competition')
                else:
                    per_phage_ranks[phage] = {}
            mode_ranks[mode][(phage, host)] = per_phage_ranks[phage].get(host)

    # Compute strict HR@k per mode + OR ceiling
    out = {
        'n_strict': n_strict,
        'n_with_embedded_rbp': n_with_embedded_rbp,
    }
    for mode in MODES:
        out[mode] = {
            'hr_at_k': {
                k: sum(1 for r in mode_ranks[mode].values()
                       if r is not None and r <= k) / max(n_strict, 1)
                for k in range(1, max_k + 1)
            }
        }
    # OR: K_rank ≤ k OR O_rank ≤ k
    out['or'] = {
        'hr_at_k': {
            k: sum(1 for pair in mode_ranks['k_only']
                   if (mode_ranks['k_only'].get(pair) is not None
                       and mode_ranks['k_only'][pair] <= k)
                   or (mode_ranks['o_only'].get(pair) is not None
                       and mode_ranks['o_only'][pair] <= k)) / max(n_strict, 1)
            for k in range(1, max_k + 1)
        }
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('experiment_dir')
    p.add_argument('--datasets', nargs='+', default=None)
    p.add_argument('--val-fasta', default=None)
    p.add_argument('--val-embedding-file', default=None)
    p.add_argument('--val-embedding-file-2', default=None)
    p.add_argument('--val-datasets-dir', default=None)
    p.add_argument('--out-json', default=None,
                   help='default: <experiment>/results/per_head_strict_eval.json')
    args = p.parse_args()

    if not os.path.isdir(args.experiment_dir):
        sys.exit(f'ERROR: not a directory: {args.experiment_dir}')
    paths = _resolve_val_paths(args.experiment_dir, args)
    for k in ('fasta', 'emb', 'ds_dir'):
        if not paths[k]:
            sys.exit(f'ERROR: missing val path {k!r}; pass via CLI or '
                     'put in config.yaml:validation')

    out_json = args.out_json or os.path.join(
        args.experiment_dir, 'results', 'per_head_strict_eval.json')
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    print(f'Experiment: {args.experiment_dir}')
    print(f'Out JSON:   {out_json}')
    print(f'Val NPZ:    {paths["emb"]}')

    predictor = _load_predictor(args.experiment_dir)
    original_predict = predictor.predict_protein
    val = load_validation_data(paths['fasta'], paths['emb'], paths['ds_dir'],
                               val_embedding_file_2=paths['emb2'])

    datasets = args.datasets or val['available_datasets']
    out = {}
    for ds in datasets:
        ds_dir = os.path.join(paths['ds_dir'], ds)
        if not os.path.isdir(ds_dir):
            print(f'  Skipping {ds} (not found)')
            continue
        print(f'  {ds} ...', end='', flush=True)
        out[ds] = evaluate_dataset_per_head(
            predictor, original_predict, ds_dir,
            val['emb_dict'], val['pid_md5'])
        out[ds]['dataset'] = ds
        # Add a 'best_strict' field per dataset = max(K@1, O@1, merged@1)
        best = max(
            out[ds]['k_only']['hr_at_k'][1],
            out[ds]['o_only']['hr_at_k'][1],
            out[ds]['merged']['hr_at_k'][1])
        out[ds]['best_strict_HR1'] = best
        out[ds]['or_HR1'] = out[ds]['or']['hr_at_k'][1]
        n = out[ds]['n_strict']
        print(f' n_strict={n}, '
              f'K@1={out[ds]["k_only"]["hr_at_k"][1]:.3f}  '
              f'O@1={out[ds]["o_only"]["hr_at_k"][1]:.3f}  '
              f'merged@1={out[ds]["merged"]["hr_at_k"][1]:.3f}  '
              f'OR@1={out[ds]["or"]["hr_at_k"][1]:.3f}  '
              f'best={best:.3f}')

    predictor.predict_protein = original_predict  # restore

    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {out_json}')


if __name__ == '__main__':
    main()
