#!/usr/bin/env python3
"""Re-evaluate a trained ciPHer experiment using only K, only O, or both heads.

Reuses the model's Predictor via get_predictor, then monkey-patches
predict_protein() to blank out the unwanted head's probability dict before
calling score_pair(). No retraining, no change to shared code.

Usage:
    python scripts/analysis/eval_per_head.py <run_dir>
    python scripts/analysis/eval_per_head.py experiments/light_attention/la_seg4_match_sweep/
"""

import argparse
import importlib.util
import os
import sys

from cipher.data.embeddings import load_embeddings
from cipher.data.proteins import load_fasta_md5
from cipher.evaluation.runner import (
    find_project_root, find_predict_module,
    load_validation_data, resolve_data_dir,
)
from cipher.evaluation.ranking import evaluate_rankings


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


def _mask_head(original_fn, which):
    """Wrap a saved copy of predict_protein to blank out one head's probs.

    Takes the ORIGINAL predict_protein as an explicit argument so that stacking
    (reassigning predictor.predict_protein across loop iterations) doesn't
    cause successive wrappers to nest on top of each other.

    which: 'k_only'  -> blank o_probs
           'o_only'  -> blank k_probs
           'both'    -> passthrough (no mask)
    """
    if which == 'both':
        return original_fn
    def wrapped(embedding):
        out = original_fn(embedding)
        if which == 'k_only':
            out['o_probs'] = {}
        elif which == 'o_only':
            out['k_probs'] = {}
        return out
    return wrapped


def _resolve_val_paths(run_dir):
    """Pull validation paths from the run's config.yaml, fallback to defaults."""
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
        rp('val_fasta',           'data/validation_data/metadata/validation_rbps_all.faa'),
        rp('val_embedding_file',  'data/validation_data/embeddings/esm2_650m_md5.npz'),
        rp('val_datasets_dir',    'data/validation_data/HOST_RANGE'),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('run_dir', help='Path to the experiment run directory')
    p.add_argument('--datasets', nargs='+', default=None,
                   help='Datasets to evaluate (default: all available)')
    p.add_argument('--k', type=int, default=1, help='HR@k to report (default: 1)')
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    val_fasta, val_emb_file, val_datasets_dir = _resolve_val_paths(run_dir)

    print(f'Run: {run_dir}')
    print(f'Val FASTA:      {val_fasta}')
    print(f'Val embeddings: {val_emb_file}')
    print(f'Val datasets:   {val_datasets_dir}')
    print()

    predictor = _load_predictor(run_dir)
    # Capture the true original predict_protein once; every loop iteration
    # wraps this (not the currently-installed wrapper) to avoid nested masking.
    original_predict = predictor.predict_protein

    val_data = load_validation_data(val_fasta, val_emb_file, val_datasets_dir)
    datasets = args.datasets or val_data['available_datasets']
    ordered = [d for d in PRIMARY_ORDER if d in datasets]
    ordered += [d for d in datasets if d not in PRIMARY_ORDER]

    # Run three passes; collect HR@k for rank_hosts and rank_phages.
    results = {}  # results[dataset][mode] = (host_hr, phage_hr, n_pairs_host, n_pairs_phage)
    for mode in ('both', 'k_only', 'o_only'):
        predictor.predict_protein = _mask_head(original_predict, mode)
        for ds in ordered:
            ds_dir = os.path.join(val_datasets_dir, ds)
            if not os.path.isdir(ds_dir):
                continue
            r = evaluate_rankings(
                predictor, ds, ds_dir, val_data['emb_dict'],
                val_data['pid_md5'], max_k=args.k, tie_method='competition')
            host_hr = r['rank_hosts']['hr_at_k'].get(args.k)
            phage_hr = r['rank_phages']['hr_at_k'].get(args.k)
            results.setdefault(ds, {})[mode] = (
                host_hr, phage_hr,
                r['rank_hosts']['n_pairs'], r['rank_phages']['n_pairs'])

    # Restore original predict_protein
    predictor.predict_protein = original_predict

    # Print tables
    for metric_idx, metric_name in ((0, 'host HR@%d' % args.k),
                                     (1, 'phage HR@%d' % args.k)):
        print(f'=== {metric_name} (given {"phage" if metric_idx == 0 else "host"}) ===')
        print(f'{"Dataset":<16} {"both":>8} {"K only":>8} {"O only":>8}  '
              f'{"dK (both-O)":>12} {"dO (both-K)":>12}')
        print('-' * 78)
        for ds in ordered:
            if ds not in results:
                continue
            both = results[ds]['both'][metric_idx]
            k_only = results[ds]['k_only'][metric_idx]
            o_only = results[ds]['o_only'][metric_idx]
            if both is None:
                continue
            # Signed contributions: combined vs single-head
            dK = both - o_only if o_only is not None else None
            dO = both - k_only if k_only is not None else None
            print(f'{ds:<16} {both:>8.3f} {k_only:>8.3f} {o_only:>8.3f}  '
                  f'{(dK if dK is not None else 0):>12.3f} '
                  f'{(dO if dO is not None else 0):>12.3f}')
        print()

    print('Interpretation:')
    print('  K only  — what you get ignoring O entirely (pure K-head ranking)')
    print('  O only  — what you get ignoring K entirely')
    print('  dK      — how much the K head added on top of O alone (both - O only)')
    print('  dO      — how much the O head added on top of K alone (both - K only)')


if __name__ == '__main__':
    main()
