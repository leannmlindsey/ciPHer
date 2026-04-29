"""Per-pair PHL host-ranking predictions for ONE model, in three modes.

For each PHL positive (phage, host) pair, computes the competition rank
of the TRUE host under three inference modes:
  - K-only (mask O probs to {})
  - O-only (mask K probs to {})
  - merged (default cipher predictor — zscore + competition tie-handling)

Appends rows to results/analysis/per_pair_phl_predictions.csv with a
model_id label so multiple models can be accumulated by re-invoking
this script with different (--model-id, experiment_dir, val NPZ).

Built for agent 5's homology-bucket analysis (request:
notes/inbox/agent1/2026-04-26-0949-from-agent5-need-per-pair-predictions.md).

Reuses the masking trick from scripts/analysis/eval_per_head.py.

Usage:
    # Laptop repro (ESM-2 mean):
    python scripts/analysis/dump_per_pair_phl_predictions.py \\
        experiments/attention_mlp/repro_old_v3_in_cipher_LAPTOP_20260425_235817 \\
        --model-id laptop_repro

    # ProtT5 highconf K (need to point at the right val NPZ):
    python scripts/analysis/dump_per_pair_phl_predictions.py \\
        experiments/attention_mlp/highconf_pipeline_K_prott5_mean \\
        --model-id highconf_pipeline_K_prott5_mean \\
        --val-embedding-file data/validation_data/embeddings/prott5_xl_md5.npz

    # Delta best-K seg4 (run on Delta with matching NPZ):
    python scripts/analysis/dump_per_pair_phl_predictions.py \\
        experiments/attention_mlp/sweep_posList_esm2_650m_seg4_cl70 \\
        --model-id seg4_cl70 \\
        --val-embedding-file /work/hdd/.../validation_embeddings_segments4_md5.npz
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

import yaml

from cipher.evaluation.runner import (
    find_predict_module, load_validation_data,
)
from cipher.evaluation.ranking import rank_hosts, _ranks_with_ties
from cipher.data.interactions import (
    load_interaction_pairs, load_phage_protein_mapping,
)


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
    val_cfg = {}
    cfg_path = os.path.join(run_dir, 'config.yaml')
    if os.path.exists(cfg_path):
        val_cfg = (yaml.safe_load(open(cfg_path)) or {}).get('validation') or {}
    return {
        'fasta': cli.val_fasta or val_cfg.get('val_fasta')
                 or os.path.join(_repo_root(), 'data/validation_data/metadata/validation_rbps_all.faa'),
        'emb': cli.val_embedding_file or val_cfg.get('val_embedding_file'),
        'ds_dir': cli.val_datasets_dir or val_cfg.get('val_datasets_dir')
                  or os.path.join(_repo_root(), 'data/validation_data/HOST_RANGE'),
    }


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def per_pair_ranks_for_mode(predictor, original_predict, mode,
                            interactions, serotypes, phage_map,
                            emb_dict, pid_md5):
    """Returns dict {(phage, host) -> rank} for PHL positive pairs under `mode`."""
    predictor.predict_protein = _mask_head(original_predict, mode)
    out = {}
    for phage in sorted(interactions):
        pos_hosts = [h for h, lbl in interactions[phage].items() if lbl == 1]
        if not pos_hosts:
            continue
        candidates = list(interactions[phage].keys())
        proteins = phage_map.get(phage, set())
        ranked = rank_hosts(predictor, proteins, candidates, serotypes,
                            emb_dict, pid_md5)
        if not ranked:
            for h in pos_hosts:
                out[(phage, h)] = None
            continue
        h_to_rank = _ranks_with_ties(ranked, tie_method='competition')
        for h in pos_hosts:
            out[(phage, h)] = h_to_rank.get(h)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('experiment_dir')
    p.add_argument('--model-id', required=True,
                   help='Short label written to the model_id column')
    p.add_argument('--out-csv', default=None,
                   help='Default: <repo>/results/analysis/per_pair_phl_predictions.csv')
    p.add_argument('--val-fasta', default=None)
    p.add_argument('--val-embedding-file', default=None)
    p.add_argument('--val-datasets-dir', default=None)
    p.add_argument('--overwrite', action='store_true',
                   help='Truncate the CSV before writing (default: append)')
    args = p.parse_args()

    if not os.path.isdir(args.experiment_dir):
        sys.exit(f'ERROR: not a directory: {args.experiment_dir}')

    paths = _resolve_val_paths(args.experiment_dir, args)
    if not paths['emb']:
        sys.exit('ERROR: missing val_embedding_file (CLI or config.yaml)')
    out_csv = args.out_csv or os.path.join(
        _repo_root(), 'results', 'analysis', 'per_pair_phl_predictions.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    print(f'Experiment: {args.experiment_dir}')
    print(f'model_id:   {args.model_id}')
    print(f'val NPZ:    {paths["emb"]}')
    print(f'out CSV:    {out_csv}')

    predictor = _load_predictor(args.experiment_dir)
    original_predict = predictor.predict_protein
    val = load_validation_data(paths['fasta'], paths['emb'], paths['ds_dir'])

    ds_dir = os.path.join(paths['ds_dir'], 'PhageHostLearn')
    if not os.path.isdir(ds_dir):
        sys.exit(f'ERROR: PHL dir not found: {ds_dir}')

    pairs = load_interaction_pairs(ds_dir)
    phage_map = load_phage_protein_mapping(
        os.path.join(ds_dir, 'metadata', 'phage_protein_mapping.csv'))
    interactions = defaultdict(dict)
    serotypes = {}
    for p_ in pairs:
        interactions[p_['phage_id']][p_['host_id']] = p_['label']
        serotypes[p_['host_id']] = {'K': p_['host_K'], 'O': p_['host_O']}

    print('Computing per-pair ranks for K-only ...')
    rk = per_pair_ranks_for_mode(predictor, original_predict, 'k_only',
                                  interactions, serotypes, phage_map,
                                  val['emb_dict'], val['pid_md5'])
    print(f'  {len(rk)} pairs')
    print('Computing per-pair ranks for O-only ...')
    ro = per_pair_ranks_for_mode(predictor, original_predict, 'o_only',
                                  interactions, serotypes, phage_map,
                                  val['emb_dict'], val['pid_md5'])
    print(f'  {len(ro)} pairs')
    print('Computing per-pair ranks for merged ...')
    rm = per_pair_ranks_for_mode(predictor, original_predict, 'merged',
                                  interactions, serotypes, phage_map,
                                  val['emb_dict'], val['pid_md5'])
    print(f'  {len(rm)} pairs')

    predictor.predict_protein = original_predict  # restore

    keys = sorted(set(rk) | set(ro) | set(rm))
    print(f'Writing {len(keys)} rows for model_id={args.model_id!r} ...')

    write_header = args.overwrite or not os.path.exists(out_csv) \
                   or os.path.getsize(out_csv) == 0
    mode = 'w' if args.overwrite else 'a'
    with open(out_csv, mode, newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['phage_id', 'host_id', 'k_true', 'o_true',
                        'model_id', 'rank_k_only', 'rank_o_only', 'rank_merged'])
        for (ph, h) in keys:
            sero = serotypes.get(h, {})
            w.writerow([
                ph, h,
                sero.get('K', ''), sero.get('O', ''),
                args.model_id,
                rk.get((ph, h)) if rk.get((ph, h)) is not None else '',
                ro.get((ph, h)) if ro.get((ph, h)) is not None else '',
                rm.get((ph, h)) if rm.get((ph, h)) is not None else '',
            ])

    # Summary stats
    n = len(keys)
    def hr1(d):
        hits = sum(1 for v in d.values() if v == 1)
        return hits / n if n else 0.0
    print(f'\nPHL host-ranking HR@1 for {args.model_id}:')
    print(f'  K-only: {hr1(rk):.4f}  ({sum(1 for v in rk.values() if v == 1)}/{n})')
    print(f'  O-only: {hr1(ro):.4f}  ({sum(1 for v in ro.values() if v == 1)}/{n})')
    print(f'  merged: {hr1(rm):.4f}  ({sum(1 for v in rm.values() if v == 1)}/{n})')


if __name__ == '__main__':
    main()
