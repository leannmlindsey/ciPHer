"""Per-head host-rank HR@k under strict denominator, for ONE cipher experiment.

For each validation dataset, runs cipher's standard host-ranking
(`cipher.evaluation.ranking.rank_hosts`) under THREE head-modes —
masking the predictor's `predict_protein()` to suppress one head's
output. Computes HR@k=1..20 per mode plus the OR-combine ceiling
(K-rank ≤ k OR O-rank ≤ k), all over the STRICT denominator:

  strict denominator (project policy, 2026-04-30): every positive
  (phage, host) pair from the interaction matrix counts. Phages with no
  annotated RBPs are KEPT in the denominator and counted as misses.
  Phages with annotated RBPs but no embeddings are KEPT and counted as
  misses (embedding-extraction failure surfaced, not normalized away).
  The denominator is the same for every model evaluated against the
  same dataset — n_strict_phage equals the count of distinct phages
  with ≥1 positive interaction in the matrix.

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
from cipher.evaluation.ranking import rank_hosts, rank_phages, _ranks_with_ties


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
    """Run all three head-modes on one dataset; return strict HR@k per mode.

    Emits TWO HR@k families per mode:
      - hr_at_k_any_hit: phage-level any-hit-in-top-k (headline; matches
                         PhageHostLearn / DpoTropiSearch convention).
                         Per phage with positives, take the BEST rank
                         across its positive hosts; HR@k = fraction of
                         phages whose best-positive rank ≤ k.
      - hr_at_k_pair:    per-pair (legacy / stricter). For every (phage,
                         positive host) pair, did that specific host
                         land at rank ≤ k.
    Same denominator structure for both — both over strict denom.
    Reverse direction (rank phages given host) emitted under
    `<mode>_phage`/`<mode>_host` only as host-level any-hit.
    """
    pairs = load_interaction_pairs(dataset_dir)
    pm_path = os.path.join(dataset_dir, 'metadata', 'phage_protein_mapping.csv')
    phage_protein_map = load_phage_protein_mapping(pm_path)

    interactions = defaultdict(dict)
    serotypes = {}
    for p in pairs:
        interactions[p['phage_id']][p['host_id']] = p['label']
        serotypes[p['host_id']] = {'K': p['host_K'], 'O': p['host_O']}

    # Fixed-denominator policy: every (phage, host) positive pair from
    # the interaction matrix counts. A phage with zero annotated RBPs is
    # KEPT in the denominator — its rank is None → counted as a miss in
    # every HR@k. (This is the project-policy strict denominator; older
    # versions of this script skipped no-RBP phages and produced
    # inflated HR@k by ~10–28 % depending on dataset.)
    strict_pairs = []                           # (phage, host) pair-level
    phages_with_strict_pos = set()              # phages with ≥1 strict positive host
    for phage in sorted(interactions):
        pos_hosts = [h for h, lbl in interactions[phage].items() if lbl == 1]
        if not pos_hosts:
            continue
        for h in pos_hosts:
            if h in serotypes:
                strict_pairs.append((phage, h))
                phages_with_strict_pos.add(phage)

    n_strict_pair = len(strict_pairs)
    n_strict_phage = len(phages_with_strict_pos)

    # Symmetric for reverse direction (rank phages given host)
    host_to_phages = defaultdict(list)
    for phage, host in strict_pairs:
        host_to_phages[host].append(phage)
    hosts_with_strict_pos = set(host_to_phages.keys())
    n_strict_host = len(hosts_with_strict_pos)

    # ===== Forward: rank hosts given phage =====
    mode_pair_ranks = {m: {} for m in MODES}
    mode_phage_anyhit = {m: {} for m in MODES}  # {mode: {phage: best_rank_or_None}}
    for mode in MODES:
        predictor.predict_protein = _mask_head(original_predict, mode)
        per_phage_h2r = {}
        for phage, host in strict_pairs:
            if phage not in per_phage_h2r:
                proteins = phage_protein_map.get(phage, set())
                candidates = list(interactions[phage].keys())
                ranked = rank_hosts(predictor, proteins, candidates,
                                    serotypes, emb_dict, pid_md5)
                per_phage_h2r[phage] = (
                    _ranks_with_ties(ranked, tie_method='competition')
                    if ranked else {})
            r = per_phage_h2r[phage].get(host)
            mode_pair_ranks[mode][(phage, host)] = r
        # any-hit per phage (min over its strict positives)
        for phage in phages_with_strict_pos:
            ranks_for_phage = [
                mode_pair_ranks[mode][(p, h)]
                for (p, h) in strict_pairs
                if p == phage and mode_pair_ranks[mode].get((p, h)) is not None
            ]
            mode_phage_anyhit[mode][phage] = (min(ranks_for_phage)
                                              if ranks_for_phage else None)

    # ===== Reverse: rank phages given host =====
    mode_host_anyhit = {m: {} for m in MODES}
    # Build phage-protein-map and serotypes already loaded.
    for mode in MODES:
        predictor.predict_protein = _mask_head(original_predict, mode)
        for host in hosts_with_strict_pos:
            pos_phages_for_host = host_to_phages[host]
            # Candidate set: all phages that have any interaction record
            # with this host (not just positives — symmetric to rank_hosts).
            candidates = [phage for phage in interactions
                          if host in interactions[phage]]
            ranked = rank_phages(predictor, host, candidates,
                                 phage_protein_map, serotypes,
                                 emb_dict, pid_md5)
            if not ranked:
                mode_host_anyhit[mode][host] = None
                continue
            phage_to_rank = _ranks_with_ties(ranked, tie_method='competition')
            ranks = [phage_to_rank[p] for p in pos_phages_for_host
                     if p in phage_to_rank]
            mode_host_anyhit[mode][host] = min(ranks) if ranks else None

    # Compute HR@k per mode (any-hit + per-pair) + OR ceilings
    def _hr_pair(mode, k):
        return sum(1 for r in mode_pair_ranks[mode].values()
                   if r is not None and r <= k) / max(n_strict_pair, 1)
    def _hr_phage_anyhit(mode, k):
        return sum(1 for r in mode_phage_anyhit[mode].values()
                   if r is not None and r <= k) / max(n_strict_phage, 1)
    def _hr_host_anyhit(mode, k):
        return sum(1 for r in mode_host_anyhit[mode].values()
                   if r is not None and r <= k) / max(n_strict_host, 1)
    def _hr_or_pair(k):
        return sum(1 for pair in mode_pair_ranks['k_only']
                   if (mode_pair_ranks['k_only'].get(pair) is not None
                       and mode_pair_ranks['k_only'][pair] <= k)
                   or (mode_pair_ranks['o_only'].get(pair) is not None
                       and mode_pair_ranks['o_only'][pair] <= k)
                   ) / max(n_strict_pair, 1)
    def _hr_or_phage_anyhit(k):
        return sum(1 for phage in phages_with_strict_pos
                   if (mode_phage_anyhit['k_only'].get(phage) is not None
                       and mode_phage_anyhit['k_only'][phage] <= k)
                   or (mode_phage_anyhit['o_only'].get(phage) is not None
                       and mode_phage_anyhit['o_only'][phage] <= k)
                   ) / max(n_strict_phage, 1)
    def _hr_or_host_anyhit(k):
        # Host→phage OR ceiling: per host, did K-only OR O-only land
        # at least one positive phage at rank ≤ k?
        return sum(1 for host in hosts_with_strict_pos
                   if (mode_host_anyhit['k_only'].get(host) is not None
                       and mode_host_anyhit['k_only'][host] <= k)
                   or (mode_host_anyhit['o_only'].get(host) is not None
                       and mode_host_anyhit['o_only'][host] <= k)
                   ) / max(n_strict_host, 1)

    out = {
        'n_strict_pair': n_strict_pair,
        'n_strict_phage': n_strict_phage,
        'n_strict_host': n_strict_host,
        # legacy alias for backward-compat with old harvest code:
        'n_strict': n_strict_pair,
    }
    for mode in MODES:
        out[mode] = {
            'hr_at_k_any_hit':       {k: _hr_phage_anyhit(mode, k) for k in range(1, max_k + 1)},
            'hr_at_k_pair':          {k: _hr_pair(mode, k) for k in range(1, max_k + 1)},
            'hr_at_k_phage_any_hit': {k: _hr_host_anyhit(mode, k) for k in range(1, max_k + 1)},
            # alias for backward-compat
            'hr_at_k': {k: _hr_pair(mode, k) for k in range(1, max_k + 1)},
        }
    out['or'] = {
        'hr_at_k_any_hit':       {k: _hr_or_phage_anyhit(k) for k in range(1, max_k + 1)},
        'hr_at_k_pair':          {k: _hr_or_pair(k) for k in range(1, max_k + 1)},
        'hr_at_k':               {k: _hr_or_pair(k) for k in range(1, max_k + 1)},
        # Host→phage OR ceiling — same union logic as phage→host OR
        # but on the reverse direction.
        'hr_at_k_phage_any_hit': {k: _hr_or_host_anyhit(k) for k in range(1, max_k + 1)},
    }

    # Per-phage rank data — for cross-model OR-union experiments.
    # Each {mode: {phage_id: best_rank_or_None}}.
    per_phage_ranks = {
        mode: dict(mode_phage_anyhit[mode]) for mode in MODES
    }
    return out, per_phage_ranks


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
    p.add_argument('--per-phage-out', default=None,
                   help='Optional TSV: dataset, phage_id, k_only_rank, '
                        'o_only_rank, merged_rank, or_hit@1 — for cross-'
                        'model OR-union experiments. None values dumped as ""')
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
    per_phage_all = {}  # {dataset: {mode: {phage: rank}}}
    for ds in datasets:
        ds_dir = os.path.join(paths['ds_dir'], ds)
        if not os.path.isdir(ds_dir):
            print(f'  Skipping {ds} (not found)')
            continue
        print(f'  {ds} ...', end='', flush=True)
        out[ds], per_phage_all[ds] = evaluate_dataset_per_head(
            predictor, original_predict, ds_dir,
            val['emb_dict'], val['pid_md5'])
        out[ds]['dataset'] = ds
        # Headline (any-hit, phage-level) — best over the three head-modes.
        best_any = max(
            out[ds]['k_only']['hr_at_k_any_hit'][1],
            out[ds]['o_only']['hr_at_k_any_hit'][1],
            out[ds]['merged']['hr_at_k_any_hit'][1])
        out[ds]['best_anyhit_HR1'] = best_any
        out[ds]['or_anyhit_HR1'] = out[ds]['or']['hr_at_k_any_hit'][1]
        # Legacy per-pair (kept for backward-compat with old plots):
        best_pair = max(
            out[ds]['k_only']['hr_at_k_pair'][1],
            out[ds]['o_only']['hr_at_k_pair'][1],
            out[ds]['merged']['hr_at_k_pair'][1])
        out[ds]['best_strict_HR1'] = best_pair
        out[ds]['or_HR1'] = out[ds]['or']['hr_at_k_pair'][1]

        n_phage = out[ds]['n_strict_phage']
        n_pair = out[ds]['n_strict_pair']
        print(f' n_phage={n_phage} n_pair={n_pair}')
        print(f'    any-hit (PHL-style headline): '
              f'K@1={out[ds]["k_only"]["hr_at_k_any_hit"][1]:.3f}  '
              f'O@1={out[ds]["o_only"]["hr_at_k_any_hit"][1]:.3f}  '
              f'merged@1={out[ds]["merged"]["hr_at_k_any_hit"][1]:.3f}  '
              f'OR@1={out[ds]["or"]["hr_at_k_any_hit"][1]:.3f}  '
              f'best={best_any:.3f}')
        print(f'    per-pair (legacy):           '
              f'K@1={out[ds]["k_only"]["hr_at_k_pair"][1]:.3f}  '
              f'O@1={out[ds]["o_only"]["hr_at_k_pair"][1]:.3f}  '
              f'merged@1={out[ds]["merged"]["hr_at_k_pair"][1]:.3f}  '
              f'OR@1={out[ds]["or"]["hr_at_k_pair"][1]:.3f}  '
              f'best={best_pair:.3f}')

    predictor.predict_protein = original_predict  # restore

    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {out_json}')

    # Optional: per-phage hit indicator TSV (for cross-model OR-union)
    if args.per_phage_out:
        import csv as _csv
        out_tsv = args.per_phage_out
        os.makedirs(os.path.dirname(out_tsv) or '.', exist_ok=True)
        with open(out_tsv, 'w', newline='') as fh:
            w = _csv.writer(fh, delimiter='\t')
            w.writerow(['dataset', 'phage_id',
                        'k_only_rank', 'o_only_rank', 'merged_rank',
                        'k_hit@1', 'o_hit@1', 'merged_hit@1', 'or_hit@1'])
            for ds, modes in per_phage_all.items():
                # Union of phages seen across modes (should be the same set)
                phages = set()
                for m in MODES:
                    phages.update(modes.get(m, {}).keys())
                for phage in sorted(phages):
                    k_r = modes.get('k_only', {}).get(phage)
                    o_r = modes.get('o_only', {}).get(phage)
                    m_r = modes.get('merged', {}).get(phage)
                    k_hit = int(k_r is not None and k_r <= 1)
                    o_hit = int(o_r is not None and o_r <= 1)
                    m_hit = int(m_r is not None and m_r <= 1)
                    or_hit = int(k_hit or o_hit)
                    w.writerow([ds, phage,
                                '' if k_r is None else k_r,
                                '' if o_r is None else o_r,
                                '' if m_r is None else m_r,
                                k_hit, o_hit, m_hit, or_hit])
        print(f'Wrote per-phage TSV: {out_tsv}')


if __name__ == '__main__':
    main()
