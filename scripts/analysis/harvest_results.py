"""Harvest every experiment's metadata + results into one wide CSV.

Scans one or more experiments roots for runs with results/evaluation.json
and produces results/experiment_log.csv — one row per run, sorted by
PHL+PBIP combined HR@1 (descending). The CSV is committed to git so the
results log is permanent and auditable.

When working with multiple parallel worktrees (one per model architecture),
pass each worktree's experiments directory via --experiments-dir so the
CSV aggregates results across all of them.

Idempotent: rerunning rebuilds the CSV from scratch. To reproduce any row,
check out its git_commit and run with the saved config.yaml.

Usage:
    # Local experiments only (default):
    python scripts/analysis/harvest_results.py

    # Aggregate across any number of worktree experiments directories:
    python scripts/analysis/harvest_results.py --experiments-dirs \\
        experiments \\
        /u/llindsey1/llindsey/PHI_TSP/cipher-light-attention/experiments \\
        /u/llindsey1/llindsey/PHI_TSP/cipher-light-attention-binary/experiments

    # With shell globbing to pick up every sibling worktree automatically:
    python scripts/analysis/harvest_results.py --experiments-dirs \\
        experiments ../cipher-*/experiments

    # Labels can be supplied explicitly with label=path:
    python scripts/analysis/harvest_results.py --experiments-dirs \\
        main=experiments la=../cipher-light-attention/experiments

    # Filter to a specific model across all worktrees:
    python scripts/analysis/harvest_results.py --model light_attention --experiments-dirs ...
"""

import argparse
import csv
import json
import os
from glob import glob

DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']


def _safe_load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _num(v):
    return v if isinstance(v, (int, float)) else None


def extract_row(exp_dir, source_label=''):
    name = os.path.basename(exp_dir.rstrip('/'))
    model = os.path.basename(os.path.dirname(exp_dir.rstrip('/')))
    exp = _safe_load(os.path.join(exp_dir, 'experiment.json'))
    ev = _safe_load(os.path.join(exp_dir, 'results', 'evaluation.json'))

    cfg = exp.get('config', {})
    prov = exp.get('provenance', {})
    data = cfg.get('data', {})
    val = cfg.get('validation', {})
    train = cfg.get('training', {})
    exper = cfg.get('experiment', {})

    row = {
        'run_name': name,
        'model': model,
        'source': source_label,       # worktree / experiments-root label
        'exp_dir': exp_dir,           # absolute or relative path to the run
        'timestamp': exp.get('timestamp', ''),
        'git_commit': prov.get('git_commit', ''),
        'git_dirty': prov.get('git_dirty', ''),
        'host': prov.get('host', ''),
        'slurm_job_id': prov.get('slurm_job_id', ''),
        'user': prov.get('user', ''),
        'cli_argv': prov.get('cli_argv', ''),
        'embedding_type': data.get('embedding_type', ''),
        'embedding_file': data.get('embedding_file', ''),
        'embedding_type_2': data.get('embedding_type_2', ''),
        'embedding_file_2': data.get('embedding_file_2', ''),
        'association_map': data.get('association_map', ''),
        'glycan_binders': data.get('glycan_binders', ''),
        'val_embedding_file': val.get('val_embedding_file', ''),
        'val_embedding_file_2': val.get('val_embedding_file_2', ''),
        'tools': ','.join(exper.get('tools') or []),
        'exclude_tools': ','.join(exper.get('exclude_tools') or []),
        'label_strategy': exper.get('label_strategy', ''),
        'min_sources': exper.get('min_sources', ''),
        'min_class_samples': exper.get('min_class_samples', ''),
        'max_samples_per_k': exper.get('max_samples_per_k', ''),
        'max_samples_per_o': exper.get('max_samples_per_o', ''),
        'lr': train.get('learning_rate', ''),
        'batch_size': train.get('batch_size', ''),
        'seed': train.get('seed', ''),
        'patience': train.get('patience', ''),
        'n_md5s': exp.get('n_md5s', ''),
        'n_k_classes': exp.get('n_k_classes', ''),
        'n_o_classes': exp.get('n_o_classes', ''),
    }

    # Per-dataset metrics — cipher's NEW eval (host-rank/phage-rank, merged,
    # over the lenient denominator that cipher's runner uses by default).
    for ds in DATASETS:
        r = ev.get(ds, {})
        rh = r.get('rank_hosts', {}).get('hr_at_k', {})
        rp = r.get('rank_phages', {}).get('hr_at_k', {})
        for k in (1, 5, 10):
            row[f'{ds}_rh{k}'] = rh.get(str(k), '')
            row[f'{ds}_rp{k}'] = rp.get(str(k), '')
        row[f'{ds}_rh_mrr'] = r.get('rank_hosts', {}).get('mrr', '')
        row[f'{ds}_rp_mrr'] = r.get('rank_phages', {}).get('mrr', '')

    # Per-head strict-denominator HR@k from results/per_head_strict_eval.json.
    # We emit the FULL k=1..20 curve for the headline metric (best-of-three
    # any-hit, both directions) so plot scripts can read the harvest CSV
    # alone — no per-experiment JSON lookup needed.
    #
    # Metric families:
    #   any-hit  (PhageHostLearn-comparable):
    #     phage→host: for each phage with positives, did ≥1 positive host
    #                 land at rank ≤ k? (cipher primary use case)
    #     host→phage: for each host with positives, did ≥1 positive phage
    #                 land at rank ≤ k? (PHL-tool comparison direction)
    #   per-pair (legacy, stricter): per (phage, positive host) pair, did
    #                                THIS host land at rank ≤ k?
    ph = _safe_load(os.path.join(exp_dir, 'results', 'per_head_strict_eval.json'))
    K_VALUES = list(range(1, 21))
    for ds in DATASETS:
        d = ph.get(ds, {})

        # Per-mode HR@1 only (saving CSV width)
        for mode_key, col_key in (('k_only', 'K'), ('o_only', 'O'),
                                    ('merged', 'merged'), ('or', 'OR')):
            v = d.get(mode_key, {}).get('hr_at_k_any_hit', {}).get('1')
            if v is None:
                v = d.get(mode_key, {}).get('hr_at_k_any_hit', {}).get(1, '')
            row[f'{ds}_{col_key}_anyhit_HR1'] = v if v != '' else ''
            v = d.get(mode_key, {}).get('hr_at_k_pair', {}).get('1')
            if v is None:
                v = d.get(mode_key, {}).get('hr_at_k_pair', {}).get(1, '')
            row[f'{ds}_{col_key}_pair_HR1'] = v if v != '' else ''

        # Best-of-three (K, O, merged) per direction — the leaderboard metric
        row[f'{ds}_best_anyhit_HR1'] = d.get('best_anyhit_HR1', '')
        row[f'{ds}_best_pair_HR1'] = d.get('best_strict_HR1', '')

        # FULL HR@k=1..20 curves for the headline (best-of-three) per direction.
        # Compute from per_head JSON: max over (k_only, o_only, merged) at each k.
        def _max_curve(family):
            out = []
            for k in K_VALUES:
                vs = []
                for mode_key in ('k_only', 'o_only', 'merged'):
                    v = (d.get(mode_key, {}).get(family, {}).get(str(k))
                         or d.get(mode_key, {}).get(family, {}).get(k))
                    if v is not None:
                        try: vs.append(float(v))
                        except (TypeError, ValueError): pass
                out.append(max(vs) if vs else None)
            return out
        # phage→host any-hit (cipher primary direction)
        ph2h = _max_curve('hr_at_k_any_hit')
        for k, v in zip(K_VALUES, ph2h):
            row[f'{ds}_best_phage2host_anyhit_HR{k}'] = (v if v is not None else '')
        # host→phage any-hit (PHL comparison direction)
        h2p = _max_curve('hr_at_k_phage_any_hit')
        for k, v in zip(K_VALUES, h2p):
            row[f'{ds}_best_host2phage_anyhit_HR{k}'] = (v if v is not None else '')
        # OR ceiling curve (phage→host direction only, headline)
        or_curve = []
        for k in K_VALUES:
            v = (d.get('or', {}).get('hr_at_k_any_hit', {}).get(str(k))
                 or d.get('or', {}).get('hr_at_k_any_hit', {}).get(k))
            or_curve.append(v if v is not None else None)
        for k, v in zip(K_VALUES, or_curve):
            row[f'{ds}_OR_phage2host_anyhit_HR{k}'] = (v if v is not None else '')

        # Backward-compatible aliases
        row[f'{ds}_K_strict_HR1'] = row[f'{ds}_K_anyhit_HR1']
        row[f'{ds}_O_strict_HR1'] = row[f'{ds}_O_anyhit_HR1']
        row[f'{ds}_merged_strict_HR1'] = row[f'{ds}_merged_anyhit_HR1']
        row[f'{ds}_OR_strict_HR1'] = row[f'{ds}_OR_anyhit_HR1']
        row[f'{ds}_best_strict_HR1'] = row[f'{ds}_best_anyhit_HR1']

        # Counts
        row[f'{ds}_n_strict_phage'] = d.get('n_strict_phage', '')
        row[f'{ds}_n_strict_pair'] = d.get('n_strict_pair', '')
        row[f'{ds}_n_strict_host'] = d.get('n_strict_host', '')
        row[f'{ds}_n_strict'] = d.get('n_strict_pair', d.get('n_strict', ''))

    # Derived scores
    phl_pbip_vals = [
        _num(row['PhageHostLearn_rh1']), _num(row['PhageHostLearn_rp1']),
        _num(row['PBIP_rh1']), _num(row['PBIP_rp1']),
    ]
    if all(v is not None for v in phl_pbip_vals):
        row['phl_pbip_combined_hr1'] = sum(phl_pbip_vals) / 4
    else:
        row['phl_pbip_combined_hr1'] = ''

    all_hr1 = [_num(row[f'{ds}_rh1']) for ds in DATASETS] + \
              [_num(row[f'{ds}_rp1']) for ds in DATASETS]
    if all(v is not None for v in all_hr1):
        row['five_ds_mean_hr1'] = sum(all_hr1) / len(all_hr1)
    else:
        row['five_ds_mean_hr1'] = ''

    # Strict-denominator composites (only computable when per-head JSON exists)
    # any-hit family — primary leaderboard metric
    phl_any = _num(row.get('PhageHostLearn_best_anyhit_HR1'))
    pbip_any = _num(row.get('PBIP_best_anyhit_HR1'))
    if phl_any is not None and pbip_any is not None:
        row['phl_pbip_best_anyhit_HR1'] = (phl_any + pbip_any) / 2
    else:
        row['phl_pbip_best_anyhit_HR1'] = ''

    phl_or_any = _num(row.get('PhageHostLearn_OR_anyhit_HR1'))
    pbip_or_any = _num(row.get('PBIP_OR_anyhit_HR1'))
    if phl_or_any is not None and pbip_or_any is not None:
        row['phl_pbip_OR_anyhit_HR1'] = (phl_or_any + pbip_or_any) / 2
    else:
        row['phl_pbip_OR_anyhit_HR1'] = ''

    # Backward-compat alias: phl_pbip_best_strict_HR1 now means any-hit too.
    row['phl_pbip_best_strict_HR1'] = row['phl_pbip_best_anyhit_HR1']
    row['phl_pbip_OR_strict_HR1'] = row['phl_pbip_OR_anyhit_HR1']

    # per-pair composite (legacy)
    phl_pair = _num(row.get('PhageHostLearn_best_pair_HR1'))
    pbip_pair = _num(row.get('PBIP_best_pair_HR1'))
    if phl_pair is not None and pbip_pair is not None:
        row['phl_pbip_best_pair_HR1'] = (phl_pair + pbip_pair) / 2
    else:
        row['phl_pbip_best_pair_HR1'] = ''

    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='*',
                   help='Model dir glob (default: all models)')
    p.add_argument('--out', default='results/experiment_log.csv')
    p.add_argument('--experiments-dirs', nargs='+', default=None,
                   help='Space-separated list of experiments root '
                        'directories to aggregate over. Each entry is '
                        'either a bare path or <label>=<path>; bare '
                        'paths auto-derive a label from the parent '
                        'directory. Shell globs (e.g. ../cipher-*/experiments) '
                        'are expanded before this script sees them. '
                        'Defaults to the local ./experiments only.')
    args = p.parse_args()

    # Parse entries as optional "label=path" pairs.
    roots = []
    entries = args.experiments_dirs if args.experiments_dirs else ['experiments']
    for entry in entries:
        if '=' in entry:
            label, path = entry.split('=', 1)
        else:
            path = entry
            # Derive label from the parent directory name (e.g.
            # "../cipher-light-attention/experiments" -> "cipher-light-attention").
            parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
            label = parent or 'local'
        roots.append((label, os.path.expanduser(path)))

    all_rows = []
    for label, root in roots:
        pattern = os.path.join(root, args.model, '*', 'results',
                               'evaluation.json')
        eval_paths = sorted(glob(pattern))
        exp_dirs = [os.path.dirname(os.path.dirname(p)) for p in eval_paths]
        if not exp_dirs:
            print(f'  [{label}] no evaluated experiments under {root}/{args.model}/')
            continue
        print(f'  [{label}] {len(exp_dirs)} runs under {root}/')
        for d in exp_dirs:
            all_rows.append(extract_row(d, source_label=label))

    if not all_rows:
        print('No evaluated experiments found in any supplied root.')
        return

    # Sort priority: any-hit best-head composite > legacy combined.
    have_anyhit = any(isinstance(r.get('phl_pbip_best_anyhit_HR1'), (int, float))
                      for r in all_rows)
    sort_field = ('phl_pbip_best_anyhit_HR1' if have_anyhit
                  else 'phl_pbip_combined_hr1')
    print(f'Sorting by: {sort_field}')

    def sort_key(r):
        v = r.get(sort_field, '')
        return -v if isinstance(v, (int, float)) else 1.0
    all_rows.sort(key=sort_key)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'Wrote {args.out} ({len(all_rows)} rows across '
          f'{len(roots)} experiments root{"s" if len(roots) != 1 else ""})')


if __name__ == '__main__':
    main()
