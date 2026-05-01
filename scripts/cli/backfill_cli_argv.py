"""Backfill `provenance.cli_argv` into an experiment.json that pre-dates
the provenance refactor. Constructs a synthetic cli_argv from the
existing config.yaml so submit_training_variant.py can use the run as
a base for variant submissions.

Usage:
    python scripts/cli/backfill_cli_argv.py \
        experiments/attention_mlp/sweep_kmer_aa20_k4

    # Or batch:
    python scripts/cli/backfill_cli_argv.py \
        experiments/attention_mlp/sweep_kmer_aa20_k4 \
        experiments/attention_mlp/sweep_kmer_murphy8_k5 \
        experiments/attention_mlp/sweep_esm2_3b_mean

Idempotent: skips runs that already have cli_argv.
"""

import json
import sys
from pathlib import Path

import yaml


def build_cli_argv(config, run_name):
    """Construct cipher.cli.train_runner argv from a merged config.yaml dict."""
    a = ['python', '-m', 'cipher.cli.train_runner', '--model', 'attention_mlp']

    # data block
    d = config.get('data', {})
    a += ['--association_map', d['association_map'],
          '--glycan_binders', d['glycan_binders'],
          '--embedding_type', d['embedding_type'],
          '--embedding_file', d['embedding_file']]
    if d.get('embedding_file_2'):
        a += ['--embedding_type_2', d.get('embedding_type_2', ''),
              '--embedding_file_2', d['embedding_file_2']]

    # validation
    v = config.get('validation', {})
    a += ['--val_fasta', v['val_fasta'],
          '--val_datasets_dir', v['val_datasets_dir'],
          '--val_embedding_file', v['val_embedding_file']]
    if v.get('val_embedding_file_2'):
        a += ['--val_embedding_file_2', v['val_embedding_file_2']]

    # training
    t = config.get('training', {})
    if 'learning_rate' in t: a += ['--lr', str(t['learning_rate'])]
    if 'batch_size' in t:    a += ['--batch_size', str(t['batch_size'])]
    if 'epochs' in t:        a += ['--epochs', str(t['epochs'])]
    if 'patience' in t:      a += ['--patience', str(t['patience'])]
    if 'seed' in t:          a += ['--seed', str(t['seed'])]

    # experiment block — filter, label strategy, caps
    e = config.get('experiment', {})
    if e.get('tools'):
        a += ['--tools', ','.join(e['tools'])]
    if e.get('exclude_tools'):
        a += ['--exclude_tools', ','.join(e['exclude_tools'])]
    if e.get('positive_list_path'):
        a += ['--positive_list', e['positive_list_path']]
    if e.get('positive_list_k'):
        a += ['--positive_list_k', e['positive_list_k']]
    if e.get('positive_list_o'):
        a += ['--positive_list_o', e['positive_list_o']]
    if e.get('label_strategy'):
        a += ['--label_strategy', e['label_strategy']]
    if e.get('min_class_samples') is not None:
        a += ['--min_class_samples', str(e['min_class_samples'])]
    if e.get('min_sources') is not None:
        a += ['--min_sources', str(e['min_sources'])]
    if e.get('max_samples_per_k') is not None:
        a += ['--max_samples_per_k', str(e['max_samples_per_k'])]
    if e.get('max_samples_per_o') is not None:
        a += ['--max_samples_per_o', str(e['max_samples_per_o'])]
    if e.get('cluster_file_path'):
        a += ['--cluster_file', e['cluster_file_path']]
    if e.get('cluster_threshold') is not None:
        a += ['--cluster_threshold', str(e['cluster_threshold'])]

    a += ['--name', run_name]
    return a


def backfill_one(exp_dir):
    exp_dir = Path(exp_dir).resolve()
    ej_path = exp_dir / 'experiment.json'
    cfg_path = exp_dir / 'config.yaml'
    if not ej_path.exists():
        print(f'  SKIP {exp_dir}: no experiment.json'); return
    if not cfg_path.exists():
        print(f'  SKIP {exp_dir}: no config.yaml'); return

    with open(ej_path) as fh:
        ej = json.load(fh)
    prov = ej.setdefault('provenance', {})
    if prov.get('cli_argv'):
        print(f'  SKIP {exp_dir.name}: cli_argv already present')
        return

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    run_name = exp_dir.name
    cli = build_cli_argv(cfg, run_name)
    prov['cli_argv'] = cli
    if 'host' not in prov: prov['host'] = ''
    if 'slurm_job_id' not in prov: prov['slurm_job_id'] = ''
    if 'user' not in prov: prov['user'] = ''
    if 'git_commit' not in prov: prov['git_commit'] = ''
    if 'git_dirty' not in prov: prov['git_dirty'] = False
    if 'timestamp' not in prov: prov['timestamp'] = ej.get('timestamp', '')

    with open(ej_path, 'w') as fh:
        json.dump(ej, fh, indent=2)
    print(f'  PATCHED {exp_dir.name}: {len(cli)}-token cli_argv added')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: backfill_cli_argv.py <exp_dir> [<exp_dir> ...]', file=sys.stderr)
        sys.exit(1)
    for d in sys.argv[1:]:
        backfill_one(d)
