"""Harvest every experiment's metadata + results into one wide CSV.

Scans experiments/*/*/ for runs with results/evaluation.json and produces
results/experiment_log.csv — one row per run, sorted by PHL+PBIP combined
HR@1 (descending). The CSV is committed to git so the results log is
permanent and auditable.

Idempotent: rerunning rebuilds the CSV from scratch. To reproduce any row,
check out its git_commit and run with the saved config.yaml.

Usage:
    python scripts/harvest_results.py
    python scripts/harvest_results.py --model attention_mlp --out results/experiment_log.csv
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


def extract_row(exp_dir):
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
        'timestamp': exp.get('timestamp', ''),
        'git_commit': prov.get('git_commit', ''),
        'git_dirty': prov.get('git_dirty', ''),
        'host': prov.get('host', ''),
        'slurm_job_id': prov.get('slurm_job_id', ''),
        'user': prov.get('user', ''),
        'cli_argv': prov.get('cli_argv', ''),
        'embedding_type': data.get('embedding_type', ''),
        'embedding_file': data.get('embedding_file', ''),
        'association_map': data.get('association_map', ''),
        'glycan_binders': data.get('glycan_binders', ''),
        'val_embedding_file': val.get('val_embedding_file', ''),
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

    # Per-dataset metrics
    for ds in DATASETS:
        r = ev.get(ds, {})
        rh = r.get('rank_hosts', {}).get('hr_at_k', {})
        rp = r.get('rank_phages', {}).get('hr_at_k', {})
        for k in (1, 5, 10):
            row[f'{ds}_rh{k}'] = rh.get(str(k), '')
            row[f'{ds}_rp{k}'] = rp.get(str(k), '')
        row[f'{ds}_rh_mrr'] = r.get('rank_hosts', {}).get('mrr', '')
        row[f'{ds}_rp_mrr'] = r.get('rank_phages', {}).get('mrr', '')

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

    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='*',
                   help='Model dir glob (default: all models)')
    p.add_argument('--out', default='results/experiment_log.csv')
    args = p.parse_args()

    pattern = f'experiments/{args.model}/*/results/evaluation.json'
    eval_paths = sorted(glob(pattern))
    exp_dirs = [os.path.dirname(os.path.dirname(p)) for p in eval_paths]

    if not exp_dirs:
        print(f'No evaluated experiments found under experiments/{args.model}/')
        return

    rows = [extract_row(d) for d in exp_dirs]

    def sort_key(r):
        v = r.get('phl_pbip_combined_hr1', '')
        return -v if isinstance(v, (int, float)) else 1.0
    rows.sort(key=sort_key)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {args.out} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
