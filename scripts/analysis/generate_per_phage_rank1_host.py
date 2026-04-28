"""Per-phage rank-1 host K-type — using cipher's actual rank_hosts
logic (not raw top-1 K class).

This is the corrected version of generate_per_phage_top1.py. The
prior script took raw `argmax(K_probs)` as cipher's prediction; that
disagreed with cipher's ranking metric (HR@1) by 4-14× depending on
dataset, because cipher's ranker applies z-score normalization within
each protein's K-distribution and aggregates as max-over-proteins of
z-score(P(host_K)). The right metric to compare to Tropi's discrete
top-1 K is: "what K-type does cipher's rank-1 host have?".

Runs in K-only mode (O head masked, mirroring `per_head_strict_eval.py`)
so the comparison to Tropi (K-class only) is apples-to-apples.

Output: per_phage_rank1_<run_name>.tsv with columns
    dataset, phage_id, n_proteins, positive_K_types,
    cp_top1_host, cp_top1_host_K, cp_hit@1, cp_top1_host_score

Usage (Delta):
    python scripts/analysis/generate_per_phage_rank1_host.py \\
        experiments/attention_mlp/sweep_prott5_mean_cl70 \\
        --val-embedding-file /work/hdd/bfzj/llindsey1/validation_embeddings_prott5/validation_embeddings_md5.npz \\
        --out-dir results/analysis/per_phage_rank1
"""

import argparse
import csv
import importlib.util
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
from cipher.evaluation.ranking import rank_hosts


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


def _mask_o_head(orig):
    """Wrap predict_protein to suppress O head — K-only mode, mirroring
    per_head_strict_eval.py. Returns wrapped function."""
    def wrapped(emb):
        out = orig(emb)
        out['o_probs'] = {}
        return out
    return wrapped


def _resolve_val_paths(run_dir, cli):
    cfg = {}
    cfg_path = os.path.join(run_dir, 'config.yaml')
    if os.path.exists(cfg_path):
        cfg = (yaml.safe_load(open(cfg_path)) or {}).get('validation') or {}
    return {
        'fasta':  cli.val_fasta or cfg.get('val_fasta'),
        'emb':    cli.val_embedding_file or cfg.get('val_embedding_file'),
        'ds_dir': cli.val_datasets_dir or cfg.get('val_datasets_dir'),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('experiment_dir')
    p.add_argument('--val-fasta', default=None)
    p.add_argument('--val-embedding-file', default=None)
    p.add_argument('--val-datasets-dir', default=None)
    p.add_argument('--out-dir', default=None,
                   help='default: <experiment>/results/per_phage_rank1.tsv')
    args = p.parse_args()

    paths = _resolve_val_paths(args.experiment_dir, args)
    for k in ('fasta', 'emb', 'ds_dir'):
        if not paths[k]:
            sys.exit(f'ERROR: missing val path {k!r}')

    run_name = os.path.basename(args.experiment_dir.rstrip('/'))
    out_dir = args.out_dir or os.path.join(args.experiment_dir, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_tsv = os.path.join(out_dir, f'per_phage_rank1_{run_name}.tsv')

    print(f'Run: {run_name}')
    print(f'Out: {out_tsv}')

    predictor = _load_predictor(args.experiment_dir)
    original_predict = predictor.predict_protein
    # K-only mode: mask O head (same as per_head_strict_eval k_only mode)
    predictor.predict_protein = _mask_o_head(original_predict)

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

        # Build interaction structure: {phage: {host: label}}
        interactions = defaultdict(dict)
        serotypes = {}
        for pp in pairs:
            interactions[pp['phage_id']][pp['host_id']] = pp['label']
            serotypes[pp['host_id']] = {'K': pp['host_K'], 'O': pp['host_O']}

        # Phages with at least one positive AND at least one annotated RBP
        # (matching strict denominator from per_head_strict_eval.py)
        n_phages = 0
        n_hit1 = 0
        for phage in sorted(interactions):
            pos_hosts = [h for h, lbl in interactions[phage].items() if lbl == 1]
            if not pos_hosts:
                continue
            proteins = phage_protein_map.get(phage, set())
            annotated_md5s = [val['pid_md5'].get(p) for p in proteins
                              if p in val['pid_md5']]
            if not annotated_md5s:
                continue
            # Candidate hosts = all hosts with interaction record (matches
            # cipher's rank_hosts called from per_head_strict_eval)
            candidates = list(interactions[phage].keys())

            ranked = rank_hosts(predictor, proteins, candidates,
                                serotypes, val['emb_dict'], val['pid_md5'])
            if not ranked:
                continue
            n_phages += 1
            top_host, top_score = ranked[0]
            top_host_K = serotypes[top_host]['K'] if top_host in serotypes else ''

            # Positive K-type set
            pos_Ks = sorted({serotypes[h]['K'] for h in pos_hosts
                            if h in serotypes and serotypes[h]['K']})

            # Hit@1: was the top-1 host one of the positives?
            # (matches the harvest's any-hit metric)
            hit1 = int(top_host in pos_hosts)
            n_hit1 += hit1

            rows_out.append({
                'dataset': ds,
                'phage_id': phage,
                'n_proteins': len(annotated_md5s),
                'positive_K_types': ';'.join(pos_Ks),
                'cp_top1_host': top_host,
                'cp_top1_host_K': top_host_K,
                'cp_hit@1': hit1,
                'cp_top1_host_score': f'{top_score:.4f}',
            })
        if n_phages > 0:
            print(f'  {ds:<16} n={n_phages:>4}  HR@1={n_hit1/n_phages:.3f}')

    predictor.predict_protein = original_predict  # restore

    with open(out_tsv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, delimiter='\t', fieldnames=[
            'dataset', 'phage_id', 'n_proteins', 'positive_K_types',
            'cp_top1_host', 'cp_top1_host_K', 'cp_hit@1', 'cp_top1_host_score'])
        w.writeheader()
        w.writerows(rows_out)
    print(f'\nWrote {len(rows_out)} rows to {out_tsv}')


if __name__ == '__main__':
    main()
