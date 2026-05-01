"""Submit a SLURM training job that re-uses an existing experiment's
cli_argv with one or more flag overrides — for cap-sweep, HP-sweep,
and similar focused ablations.

Reads the base experiment's `experiment.json` to recover its cli_argv,
substitutes the requested flags (replacing existing flag values or
appending if new), updates --name to the variant name, and writes a
SLURM sbatch script that runs train_runner + standard eval steps
(default eval + per_head_strict_eval).

Output:
  - SBATCH script under <CIPHER_DIR>/logs/<NAME>.sbatch
  - Optionally submitted (set --dry-run to skip submission)
  - Prints the SLURM job ID on submission

Usage:
  python scripts/cli/submit_training_variant.py \\
      --base-exp experiments/attention_mlp/sweep_prott5_mean_cl70 \\
      --name sweep_prott5_mean_cl70_cap500 \\
      --override max_samples_per_k=500 \\
      --cipher-dir /projects/bfzj/llindsey1/PHI_TSP/ciPHer

Multiple overrides:
  --override max_samples_per_k=500 dropout=0.2

Notes:
  - Flags are matched by their canonical name (with leading --) AND by
    the underscore form (`max_samples_per_k`). Both are stripped from
    the existing argv before appending the new value.
  - Some flags must reference different worktrees (e.g. light_attention
    runs in cipher-light-attention/). The script honors the model
    type from experiment.json's config.model name and cd's
    accordingly.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


def _strip_flag(argv, flag):
    """Remove --flag and its single value from argv list. Idempotent."""
    flag_long = f'--{flag}'
    out = []
    skip = 0
    for tok in argv:
        if skip:
            skip -= 1
            continue
        if tok == flag_long:
            skip = 1
            continue
        out.append(tok)
    return out


def _apply_overrides(argv, overrides):
    """Replace each `--flag value` in argv with the new value (or append
    if absent). overrides is a dict of {flag_name_no_dashes: new_value}."""
    for flag, value in overrides.items():
        argv = _strip_flag(argv, flag)
        argv.extend([f'--{flag}', str(value)])
    return argv


def _replace_name(argv, new_name):
    """Replace --name's value with new_name."""
    argv = _strip_flag(argv, 'name')
    argv.extend(['--name', new_name])
    return argv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-exp', required=True,
                   help='Path to base experiment directory (relative or absolute)')
    p.add_argument('--name', required=True,
                   help='New experiment --name for the variant')
    p.add_argument('--override', nargs='+', default=[],
                   help='flag=value pairs to override, e.g. max_samples_per_k=500 dropout=0.2')
    p.add_argument('--cipher-dir',
                   default='/projects/bfzj/llindsey1/PHI_TSP/ciPHer',
                   help='Cipher repo root on Delta')
    p.add_argument('--account', default='bfzj-dtai-gh')
    p.add_argument('--partition', default='ghx4')
    p.add_argument('--conda-env', default='esmfold2')
    p.add_argument('--time', default='4:00:00')
    p.add_argument('--mem', default='64G')
    p.add_argument('--cpus', default='8')
    p.add_argument('--dry-run', action='store_true',
                   help='Render sbatch but do not submit')
    args = p.parse_args()

    overrides = {}
    for ov in args.override:
        if '=' not in ov:
            sys.exit(f'ERROR: --override entries must be flag=value, got: {ov}')
        flag, val = ov.split('=', 1)
        overrides[flag] = val

    # Locate base experiment.json (either local path or absolute)
    base_exp = args.base_exp.rstrip('/')
    if not os.path.isabs(base_exp):
        base_exp_abs = os.path.join(args.cipher_dir, base_exp)
    else:
        base_exp_abs = base_exp
    if not os.path.exists(base_exp_abs):
        # Try cipher-light-attention worktree
        la_path = os.path.join(args.cipher_dir, '..', 'cipher-light-attention',
                                base_exp.replace('experiments/', 'experiments/'))
        if os.path.exists(la_path):
            base_exp_abs = la_path
        else:
            sys.exit(f'ERROR: base experiment not found at {base_exp_abs} or LA worktree')

    exp_json_path = os.path.join(base_exp_abs, 'experiment.json')
    if not os.path.exists(exp_json_path):
        sys.exit(f'ERROR: experiment.json not found at {exp_json_path}')

    with open(exp_json_path) as f:
        meta = json.load(f)
    cli_argv_str = meta.get('provenance', {}).get('cli_argv', '')
    if not cli_argv_str:
        sys.exit(f'ERROR: no cli_argv in {exp_json_path}/provenance')

    # cli_argv is a single string; tokenize
    argv = shlex.split(cli_argv_str)
    if not argv:
        sys.exit(f'ERROR: empty cli_argv after tokenization')

    # First token is the script path; rest are arguments
    script = argv[0]
    args_only = argv[1:]

    args_only = _apply_overrides(args_only, overrides)
    args_only = _replace_name(args_only, args.name)

    # Determine model type for working-dir routing
    model_name = meta.get('config', {}).get('model', {}).get('name') \
                 or meta.get('config', {}).get('model_name', '') \
                 or os.path.basename(os.path.dirname(base_exp_abs))
    is_la = 'light_attention' in model_name.lower()

    # Validation embedding file (for the eval steps)
    val_emb = meta.get('config', {}).get('validation', {}).get('val_embedding_file', '')
    if not val_emb:
        # Try to infer from cli args
        for i, tok in enumerate(args_only):
            if tok == '--val_embedding_file' and i + 1 < len(args_only):
                val_emb = args_only[i + 1]
                break

    # Build SBATCH script
    logs_dir = os.path.join(args.cipher_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    sbatch_path = os.path.join(logs_dir, f'variant_{args.name}.sbatch')
    log_path = os.path.join(logs_dir, f'variant_{args.name}_%j.log')

    cmd = ' \\\n    '.join(['python ' + script] + args_only)
    if is_la:
        cd_line = f'LA_DIR="$(cd {args.cipher_dir}/../cipher-light-attention && pwd)"\ncd "$LA_DIR"\nexport PYTHONPATH="$LA_DIR/src:{args.cipher_dir}/src:${{PYTHONPATH:-}}"'
        worktree_dir = '$LA_DIR'
    else:
        cd_line = f'cd {args.cipher_dir}\nexport PYTHONPATH={args.cipher_dir}/src:${{PYTHONPATH:-}}'
        worktree_dir = args.cipher_dir

    exp_dir_after = f'{worktree_dir}/experiments/{model_name}/{args.name}'

    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=v_{args.name[:24]}
#SBATCH --account={args.account}
#SBATCH --partition={args.partition}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={args.mem}
#SBATCH --time={args.time}
#SBATCH --output={log_path}
#SBATCH --error={log_path}

set -euo pipefail
source $(conda info --base)/etc/profile.d/conda.sh
conda activate {args.conda_env}
{cd_line}

echo "=== Variant: {args.name} ==="
echo "Base experiment: {base_exp_abs}"
echo "Overrides: {' '.join(args.override)}"
echo ""

{cmd}

EXP_DIR="{exp_dir_after}"
echo ""
echo "=== Eval: default (rank_hosts/rank_phages) ==="
python -m cipher.evaluation.runner "$EXP_DIR" \\
    --val-embedding-file "{val_emb}"

echo ""
echo "=== Eval: per-head strict (any-hit + per-pair, all 5 datasets) ==="
python {args.cipher_dir}/scripts/analysis/per_head_strict_eval.py "$EXP_DIR" \\
    --val-embedding-file "{val_emb}"
"""
    with open(sbatch_path, 'w') as f:
        f.write(sbatch_content)
    os.chmod(sbatch_path, 0o755)

    print(f'Variant: {args.name}')
    print(f'  Base:      {base_exp_abs}')
    print(f'  Overrides: {overrides}')
    print(f'  SBATCH:    {sbatch_path}')

    if args.dry_run:
        print(f'  [DRY RUN] Not submitted. Run: sbatch {sbatch_path}')
        return

    result = subprocess.run(['sbatch', sbatch_path], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f'sbatch failed: {result.stderr}')
    job_id = result.stdout.strip().split()[-1]
    print(f'  Submitted: JOB {job_id}')


if __name__ == '__main__':
    main()
