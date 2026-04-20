"""CLI runner for cipher-evaluate.

Loads a Predictor from an experiment's predict.py, runs evaluation
across all validation datasets, prints a summary table, and saves
results as JSON.

Usage:
    cipher-evaluate experiments/attention_mlp/runs/v4_downsample/
    cipher-evaluate experiments/attention_mlp/runs/v4_downsample/ --datasets CHEN PBIP
    cipher-evaluate experiments/attention_mlp/runs/v4_downsample/ -o results.json
"""

import argparse
import importlib.util
import json
import os
import sys
import time

from cipher.data.embeddings import load_embeddings, load_embeddings_concat
from cipher.data.proteins import load_fasta_md5
from cipher.evaluation.ranking import evaluate_rankings


DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']

EMBEDDING_FILES = {
    'esm2_650m': 'esm2_650m_md5.npz',
    'esm2_3b': 'esm2_3b_md5.npz',
    'kmer3_8000d': 'kmer3_8000d_md5.npz',
    'kmer3': 'kmer3_8000d_md5.npz',
}


def find_project_root(start_path):
    """Walk up from start_path to find the repo root (contains setup.py)."""
    path = os.path.abspath(start_path)
    for _ in range(10):
        if os.path.exists(os.path.join(path, 'setup.py')):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return None


def find_predict_module(run_dir):
    """Find predict.py for an experiment.

    Search order:
    1. experiment.json -> config.model -> models/{model}/predict.py
    2. Walk up from run_dir looking for predict.py

    Returns:
        (predict_path, model_dir) or (None, None)
    """
    run_dir = os.path.abspath(run_dir)
    project_root = find_project_root(run_dir)

    # Try experiment.json first
    exp_json = os.path.join(run_dir, 'experiment.json')
    if os.path.exists(exp_json) and project_root:
        with open(exp_json) as f:
            meta = json.load(f)
        model_name = meta.get('model')
        if model_name:
            model_dir = os.path.join(project_root, 'models', model_name)
            candidate = os.path.join(model_dir, 'predict.py')
            if os.path.exists(candidate):
                return candidate, model_dir

    # Try inferring model name from path: experiments/{model_name}/...
    if project_root:
        exp_dir = os.path.join(project_root, 'experiments')
        rel = os.path.relpath(run_dir, exp_dir)
        model_name = rel.split(os.sep)[0] if os.sep in rel else rel
        model_dir = os.path.join(project_root, 'models', model_name)
        candidate = os.path.join(model_dir, 'predict.py')
        if os.path.exists(candidate):
            return candidate, model_dir

    # Fallback: walk up from run_dir
    path = run_dir
    for _ in range(5):
        candidate = os.path.join(path, 'predict.py')
        if os.path.exists(candidate):
            return candidate, path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent

    return None, None


def load_predictor(run_dir):
    """Load a Predictor instance from an experiment.

    Finds predict.py via experiment.json model name or by walking up
    from run_dir, imports it, and calls get_predictor(run_dir).

    Args:
        run_dir: path to the experiment directory

    Returns:
        Predictor instance
    """
    predict_path, model_dir = find_predict_module(run_dir)
    if predict_path is None:
        raise FileNotFoundError(
            f'No predict.py found for {run_dir}. '
            f'Check that models/{{model_name}}/predict.py exists.')

    # Add model dir to sys.path so predict.py can import model.py
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    spec = importlib.util.spec_from_file_location('predict', predict_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'get_predictor'):
        raise RuntimeError(
            f'{predict_path} must define get_predictor(run_dir) -> Predictor')

    return module.get_predictor(os.path.abspath(run_dir))


def resolve_data_dir(explicit_dir, run_dir):
    """Find the data/ directory."""
    if explicit_dir:
        return os.path.abspath(explicit_dir)

    # Try CIPHER_DATA_DIR env var
    env_dir = os.environ.get('CIPHER_DATA_DIR')
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    # Try relative to project root
    root = find_project_root(run_dir)
    if root:
        candidate = os.path.join(root, 'data')
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        'Cannot find data directory. Use --data-dir or set CIPHER_DATA_DIR.')


def embedding_filename(embedding_type):
    """Map embedding_type string to NPZ filename."""
    if embedding_type in EMBEDDING_FILES:
        return EMBEDDING_FILES[embedding_type]
    if embedding_type.endswith('.npz'):
        return embedding_type
    return f'{embedding_type}_md5.npz'


def load_validation_data(val_fasta, val_embedding_file, val_datasets_dir,
                         val_embedding_file_2=None):
    """Load all shared validation data.

    Args:
        val_fasta: path to validation FASTA (for pid->md5 mapping)
        val_embedding_file: path to validation embedding NPZ
        val_datasets_dir: path to directory containing dataset subdirs
            (e.g., .../HOST_RANGE/ with CHEN/, PBIP/, etc.)
        val_embedding_file_2: optional path to a second validation embedding
            NPZ. When set, features are concatenated per MD5 to match a
            combined-feature training run.

    Returns:
        dict with 'emb_dict', 'pid_md5', 'hr_dir', 'available_datasets'
    """
    # Protein ID -> MD5 mapping
    if not os.path.exists(val_fasta):
        raise FileNotFoundError(f'Validation FASTA not found: {val_fasta}')
    pid_md5 = load_fasta_md5(val_fasta)

    # Embeddings — optionally concatenated from two NPZs
    if not os.path.exists(val_embedding_file):
        raise FileNotFoundError(
            f'Validation embedding file not found: {val_embedding_file}')
    if val_embedding_file_2:
        if not os.path.exists(val_embedding_file_2):
            raise FileNotFoundError(
                f'Second validation embedding file not found: '
                f'{val_embedding_file_2}')
        needed = set(pid_md5.values())
        emb_dict = load_embeddings_concat(
            val_embedding_file, val_embedding_file_2, md5_filter=needed)
    else:
        emb_dict = load_embeddings(val_embedding_file)

    # Available datasets
    hr_dir = val_datasets_dir
    available = [d for d in DATASETS if os.path.isdir(os.path.join(hr_dir, d))]

    return {
        'emb_dict': emb_dict,
        'pid_md5': pid_md5,
        'hr_dir': hr_dir,
        'available_datasets': available,
    }


def run_evaluation(predictor, val_data, datasets=None, max_k=20,
                   tie_method='competition', verbose=True):
    """Run evaluation across datasets.

    Args:
        predictor: Predictor instance
        val_data: dict from load_validation_data()
        datasets: list of dataset names (default: all available)
        max_k: maximum k for HR@k
        tie_method: 'competition' (default) or 'arbitrary' — see evaluate_rankings
        verbose: print progress

    Returns:
        dict: {dataset_name: evaluation_results}
    """
    emb_dict = val_data['emb_dict']
    pid_md5 = val_data['pid_md5']
    hr_dir = val_data['hr_dir']

    if datasets is None:
        datasets = val_data['available_datasets']

    results = {}
    for ds_name in datasets:
        ds_dir = os.path.join(hr_dir, ds_name)
        if not os.path.isdir(ds_dir):
            if verbose:
                print(f'  Skipping {ds_name} (not found)')
            continue

        if verbose:
            print(f'  {ds_name}...', end='', flush=True)
        t0 = time.time()

        ds_results = evaluate_rankings(
            predictor, ds_name, ds_dir, emb_dict, pid_md5, max_k=max_k,
            tie_method=tie_method)

        elapsed = time.time() - t0
        if verbose:
            rh = ds_results['rank_hosts']
            rp = ds_results['rank_phages']
            print(f' {elapsed:.1f}s  '
                  f'hosts: HR@1={rh["hr_at_k"].get(1, 0):.3f}  '
                  f'phages: HR@1={rp["hr_at_k"].get(1, 0):.3f}  '
                  f'({rh["n_pairs"]}+{rp["n_pairs"]} pairs)')

        results[ds_name] = ds_results

    return results


def print_summary(results):
    """Print a formatted summary table."""
    print('\n' + '=' * 78)
    print('RANK HOSTS GIVEN PHAGE')
    print('=' * 78)
    print(f'{"Dataset":<16} {"HR@1":>6} {"HR@5":>6} {"HR@10":>7} '
          f'{"HR@20":>7} {"MRR":>6} {"Pairs":>6}')
    print('-' * 78)

    hr1_vals = []
    for ds, r in results.items():
        rh = r['rank_hosts']
        hr = rh['hr_at_k']
        print(f'{ds:<16} {hr.get(1,0):>6.3f} {hr.get(5,0):>6.3f} '
              f'{hr.get(10,0):>7.3f} {hr.get(20,0):>7.3f} '
              f'{rh["mrr"]:>6.3f} {rh["n_pairs"]:>6}')
        hr1_vals.append(hr.get(1, 0))

    if hr1_vals:
        print('-' * 78)
        print(f'{"Mean":<16} {sum(hr1_vals)/len(hr1_vals):>6.3f}')

    print('\n' + '=' * 78)
    print('RANK PHAGES GIVEN HOST')
    print('=' * 78)
    print(f'{"Dataset":<16} {"HR@1":>6} {"HR@5":>6} {"HR@10":>7} '
          f'{"HR@20":>7} {"MRR":>6} {"Pairs":>6}')
    print('-' * 78)

    hr1_vals = []
    for ds, r in results.items():
        rp = r['rank_phages']
        hr = rp['hr_at_k']
        print(f'{ds:<16} {hr.get(1,0):>6.3f} {hr.get(5,0):>6.3f} '
              f'{hr.get(10,0):>7.3f} {hr.get(20,0):>7.3f} '
              f'{rp["mrr"]:>6.3f} {rp["n_pairs"]:>6}')
        hr1_vals.append(hr.get(1, 0))

    if hr1_vals:
        print('-' * 78)
        print(f'{"Mean":<16} {sum(hr1_vals)/len(hr1_vals):>6.3f}')
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a ciPHer experiment against validation datasets.')
    parser.add_argument(
        'run_dir',
        help='Path to the run directory (must have predict.py in or above it)')
    parser.add_argument(
        '--val-fasta',
        help='Path to validation protein FASTA (for pid->md5 mapping). '
             'Overrides config.yaml validation.val_fasta')
    parser.add_argument(
        '--val-embedding-file',
        help='Path to validation embedding NPZ file. '
             'Overrides config.yaml validation.val_embedding_file')
    parser.add_argument(
        '--val-embedding-file-2',
        help='Optional path to a second validation embedding NPZ to '
             'concatenate with --val-embedding-file. Must match the '
             'combined-feature setup used at training time.')
    parser.add_argument(
        '--val-datasets-dir',
        help='Path to directory containing validation dataset subdirs '
             '(CHEN/, PBIP/, PhageHostLearn/, etc.). '
             'Overrides config.yaml validation.val_datasets_dir')
    parser.add_argument(
        '--datasets', nargs='+',
        help=f'Datasets to evaluate (default: all). Options: {", ".join(DATASETS)}')
    parser.add_argument(
        '--max-k', type=int, default=20,
        help='Maximum k for HR@k (default: 20)')
    parser.add_argument(
        '--tie-method', choices=['competition', 'arbitrary'],
        default='competition',
        help='How to handle hosts/phages with identical scores. '
             '"competition" (default) gives all tied items the same rank, '
             'reflecting that the model cannot discriminate within a serotype. '
             '"arbitrary" uses positional ranking (legacy behavior).')
    parser.add_argument(
        '--score-norm', choices=['zscore', 'raw'], default='zscore',
        help='How to combine K and O probabilities. '
             '"zscore" (default) z-scores each head per-protein, accounting '
             'for the K head having ~7x more classes than O. '
             '"raw" compares raw probabilities (legacy; biased toward O).')
    parser.add_argument(
        '--output', '-o',
        help='Save results JSON to this path (default: {run_dir}/results/evaluation.json)')
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress progress output')
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    verbose = not args.quiet

    if verbose:
        print(f'Loading predictor from {run_dir}')
    predictor = load_predictor(run_dir)
    predictor.score_normalization = args.score_norm

    if verbose:
        print(f'  Embedding type: {predictor.embedding_type}')

    # Resolve validation paths: CLI args > config.yaml > defaults
    import yaml
    config_path = os.path.join(run_dir, 'config.yaml')
    val_cfg = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            full_cfg = yaml.safe_load(f) or {}
        val_cfg = full_cfg.get('validation', {})

    # Find project root for resolving relative paths
    project_root = find_project_root(run_dir) or os.getcwd()

    def resolve_path(cli_val, config_key, default):
        """Pick CLI > config > default, and resolve relative paths."""
        val = cli_val or val_cfg.get(config_key, default)
        if val and not os.path.isabs(val):
            val = os.path.join(project_root, val)
        return val

    val_fasta = resolve_path(
        args.val_fasta, 'val_fasta',
        'data/validation_data/metadata/validation_rbps_all.faa')
    val_embedding_file = resolve_path(
        args.val_embedding_file, 'val_embedding_file',
        'data/validation_data/embeddings/esm2_650m_md5.npz')
    val_embedding_file_2 = resolve_path(
        args.val_embedding_file_2, 'val_embedding_file_2', None)
    val_datasets_dir = resolve_path(
        args.val_datasets_dir, 'val_datasets_dir',
        'data/validation_data/HOST_RANGE')

    if verbose:
        print(f'Validation paths:')
        print(f'  FASTA:      {val_fasta}')
        print(f'  Embeddings: {val_embedding_file}')
        if val_embedding_file_2:
            print(f'  + second:   {val_embedding_file_2}')
        print(f'  Datasets:   {val_datasets_dir}')

    val_data = load_validation_data(val_fasta, val_embedding_file, val_datasets_dir,
                                    val_embedding_file_2=val_embedding_file_2)

    if verbose:
        print(f'Evaluating on: {", ".join(args.datasets or val_data["available_datasets"])}')
    results = run_evaluation(
        predictor, val_data, datasets=args.datasets, max_k=args.max_k,
        tie_method=args.tie_method, verbose=verbose)

    if verbose:
        print_summary(results)

    # Save results
    output_path = args.output
    if output_path is None:
        results_dir = os.path.join(run_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, 'evaluation.json')

    # Convert int keys to strings for JSON serialization
    serializable = {'_meta': {
        'tie_method': args.tie_method,
        'score_norm': args.score_norm,
        'max_k': args.max_k,
    }}
    for ds_name, ds_results in results.items():
        serializable[ds_name] = {}
        for mode in ('rank_hosts', 'rank_phages'):
            r = dict(ds_results[mode])
            r['hr_at_k'] = {str(k): v for k, v in r['hr_at_k'].items()}
            serializable[ds_name][mode] = r

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    if verbose:
        print(f'Results saved to {output_path}')


if __name__ == '__main__':
    main()
