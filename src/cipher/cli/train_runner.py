"""CLI runner for cipher-train.

Loads a model's base config, overrides with CLI args, auto-generates an
experiment directory name, and calls the model's train() function.

Usage:
    cipher-train --model attention_mlp
    cipher-train --model attention_mlp --lr 1e-4 --batch_size 256 --protein_set tsp_only
    cipher-train --model attention_mlp --max_samples_per_k 200 --max_samples_per_o 2000
    cipher-train --model attention_mlp --name my_custom_run
"""

import argparse
import importlib.util
import os
import sys
import time

import yaml


def find_project_root(start_path=None):
    """Walk up to find the repo root (contains setup.py)."""
    path = os.path.abspath(start_path or os.getcwd())
    for _ in range(10):
        if os.path.exists(os.path.join(path, 'setup.py')):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return os.getcwd()


def load_base_config(model_name, project_root):
    """Load base_config.yaml for a model."""
    config_path = os.path.join(project_root, 'models', model_name, 'base_config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f'No base_config.yaml found for model {model_name!r} at {config_path}')
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_overrides(config, args):
    """Apply CLI arg overrides to the config dict."""
    # Training overrides
    if args.lr is not None:
        config.setdefault('training', {})['learning_rate'] = args.lr
    if args.batch_size is not None:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.epochs is not None:
        config.setdefault('training', {})['epochs'] = args.epochs
    if args.patience is not None:
        config.setdefault('training', {})['patience'] = args.patience
    if args.seed is not None:
        config.setdefault('training', {})['seed'] = args.seed
    if args.no_pos_weight:
        config.setdefault('training', {})['use_pos_weight'] = False

    # Experiment/data overrides
    if args.protein_set is not None:
        # Deprecated but still supported; clear tools/exclude_tools so they don't conflict
        config.setdefault('experiment', {})['protein_set'] = args.protein_set
        config['experiment'].pop('tools', None)
        config['experiment'].pop('exclude_tools', None)
    if args.tools is not None:
        tools = [t.strip() for t in args.tools.split(',') if t.strip()]
        config.setdefault('experiment', {})['tools'] = tools
        # Clear legacy protein_set so it doesn't override
        config['experiment'].pop('protein_set', None)
    if args.exclude_tools is not None:
        excl = [t.strip() for t in args.exclude_tools.split(',') if t.strip()]
        config.setdefault('experiment', {})['exclude_tools'] = excl
        config['experiment'].pop('protein_set', None)
    if args.min_sources is not None:
        config.setdefault('experiment', {})['min_sources'] = args.min_sources
    if args.max_k_types is not None:
        config.setdefault('experiment', {})['max_k_types'] = args.max_k_types
    if args.max_o_types is not None:
        config.setdefault('experiment', {})['max_o_types'] = args.max_o_types
    if args.max_samples_per_k is not None:
        config.setdefault('experiment', {})['max_samples_per_k'] = args.max_samples_per_k
    if args.max_samples_per_o is not None:
        config.setdefault('experiment', {})['max_samples_per_o'] = args.max_samples_per_o
    if args.label_strategy is not None:
        config.setdefault('experiment', {})['label_strategy'] = args.label_strategy
        config['experiment']['single_label'] = (args.label_strategy == 'single_label')
    if args.min_label_count is not None:
        config.setdefault('experiment', {})['min_label_count'] = args.min_label_count
    if args.min_label_fraction is not None:
        config.setdefault('experiment', {})['min_label_fraction'] = args.min_label_fraction
    if args.min_class_samples is not None:
        config.setdefault('experiment', {})['min_class_samples'] = args.min_class_samples

    # Model overrides
    if args.hidden_dims is not None:
        dims = [int(d) for d in args.hidden_dims.split(',')]
        config.setdefault('model', {})['hidden_dims'] = dims
    if args.attention_dim is not None:
        config.setdefault('model', {})['attention_dim'] = args.attention_dim
    if args.dropout is not None:
        config.setdefault('model', {})['dropout'] = args.dropout

    # Embedding overrides
    if args.embedding_type is not None:
        config.setdefault('data', {})['embedding_type'] = args.embedding_type
    if args.embedding_file is not None:
        config.setdefault('data', {})['embedding_file'] = args.embedding_file

    return config


def _tools_name_component(exp):
    """Build a short string describing the tool filter for the run name.

    Examples:
        no tools, no exclude                   -> 'allTools'
        tools=['SpikeHunter']                  -> 'SpikeHunter'
        tools=['DepoScope','DepoRanker']       -> 'DepoRanker-DepoScope' (sorted)
        exclude=['SpikeHunter']                -> 'noSpikeHunter'
        tools=['DepoScope'], exclude=['SpikeHunter'] -> 'DepoScope_noSpikeHunter'
        protein_set='tsp_only' (legacy)        -> 'tsp_only'
    """
    # If legacy protein_set is still set (not translated), use it
    ps = exp.get('protein_set')
    tools = exp.get('tools')
    excl = exp.get('exclude_tools')

    if ps and not tools and not excl:
        return ps

    parts = []
    if tools:
        parts.append('-'.join(sorted(tools)))
    if excl:
        parts.append('no' + '-'.join(sorted(excl)))

    if not parts:
        return 'allTools'
    return '_'.join(parts)


def generate_run_name(config):
    """Auto-generate a run name from key config parameters."""
    exp = config.get('experiment', {})
    train = config.get('training', {})

    parts = [
        _tools_name_component(exp),
    ]

    # Add non-default parameters to the name
    lr = train.get('learning_rate', 1e-5)
    parts.append(f'lr{lr:.0e}'.replace('+', ''))

    bs = train.get('batch_size', 64)
    parts.append(f'bs{bs}')

    seed = train.get('seed', 42)
    parts.append(f'seed{seed}')

    # Add downsampling if present
    max_k = exp.get('max_samples_per_k')
    max_o = exp.get('max_samples_per_o')
    if max_k:
        parts.append(f'dsK{max_k}')
    if max_o:
        parts.append(f'dsO{max_o}')

    # Add label strategy if non-default (default='single_label' in our base)
    strategy = exp.get('label_strategy')
    if strategy and strategy != 'single_label':
        # Short aliases for readability
        aliases = {
            'multi_label': 'ml',
            'multi_label_threshold': 'mlT',
            'weighted_soft': 'wSoft',
            'weighted_multi_label': 'wml',
        }
        parts.append(aliases.get(strategy, strategy))
        if strategy == 'multi_label_threshold':
            mc = exp.get('min_label_count', 2)
            mf = exp.get('min_label_fraction', 0.1)
            parts.append(f'mc{mc}f{mf:g}')

    # Timestamp
    parts.append(time.strftime('%Y%m%d_%H%M%S'))

    return '_'.join(parts)


def load_train_module(model_name, project_root):
    """Dynamically import a model's train.py."""
    model_dir = os.path.join(project_root, 'models', model_name)
    train_path = os.path.join(model_dir, 'train.py')

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f'No train.py found for model {model_name!r} at {train_path}')

    # Add model dir to sys.path so train.py can import model.py
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    spec = importlib.util.spec_from_file_location('train', train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'train'):
        raise RuntimeError(
            f'{train_path} must define train(experiment_dir, config)')

    return module


def _available_models(project_root):
    """List models that have a base_config.yaml."""
    models_dir = os.path.join(project_root, 'models')
    if not os.path.isdir(models_dir):
        return []
    found = []
    for name in sorted(os.listdir(models_dir)):
        if os.path.exists(os.path.join(models_dir, name, 'base_config.yaml')):
            found.append(name)
    return found


def _default_hint(config, section, key, fallback='<from base_config>'):
    """Format a default value from base config for help text."""
    val = config.get(section, {}).get(key)
    if val is None:
        return fallback
    if isinstance(val, list):
        return ','.join(str(v) for v in val)
    return str(val)


def _get_default_config_for_help():
    """Try to load a base config for showing defaults in --help.

    Looks for --model in sys.argv. If present, loads that model's base config.
    If not present, tries the only available model (if unique).
    Returns (config, model_name) or (None, None).
    """
    project_root = find_project_root()

    # Look for --model in argv
    model_name = None
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == '--model' and i + 1 < len(argv):
            model_name = argv[i + 1]
            break
        if arg.startswith('--model='):
            model_name = arg.split('=', 1)[1]
            break

    if model_name is None:
        available = _available_models(project_root)
        if len(available) == 1:
            model_name = available[0]

    if model_name is None:
        return {}, None

    try:
        return load_base_config(model_name, project_root), model_name
    except FileNotFoundError:
        return {}, None


def main():
    # Load base config (if model is known) to show defaults in --help
    base_cfg, base_model = _get_default_config_for_help()
    d_train = base_cfg.get('training', {})
    d_exp = base_cfg.get('experiment', {})
    d_model = base_cfg.get('model', {})
    d_data = base_cfg.get('data', {})

    def fmt(val, fallback='<base_config>'):
        if val is None:
            return fallback
        if isinstance(val, list):
            return ','.join(str(v) for v in val)
        return str(val)

    if base_model:
        defaults_note = f'(defaults shown from models/{base_model}/base_config.yaml)'
    else:
        defaults_note = '(pass --model first to see defaults; otherwise defaults come from base_config.yaml)'

    parser = argparse.ArgumentParser(
        description=f'Train a ciPHer model. {defaults_note}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cipher-train --model attention_mlp
  cipher-train --model attention_mlp --lr 1e-4 --protein_set tsp_only
  cipher-train --model attention_mlp --max_samples_per_k 200 --max_samples_per_o 2000
  cipher-train --model attention_mlp --name my_experiment
""")

    # Required
    parser.add_argument('--model', required=True,
                        help='Model name (directory in models/)')

    # Training params
    parser.add_argument('--lr', type=float,
                        help=f'Learning rate (default: {fmt(d_train.get("learning_rate"))})')
    parser.add_argument('--batch_size', type=int,
                        help=f'Batch size (default: {fmt(d_train.get("batch_size"))})')
    parser.add_argument('--epochs', type=int,
                        help=f'Max epochs (default: {fmt(d_train.get("epochs"))})')
    parser.add_argument('--patience', type=int,
                        help=f'Early stopping patience (default: {fmt(d_train.get("patience"))})')
    parser.add_argument('--seed', type=int,
                        help=f'Random seed (default: {fmt(d_train.get("seed"))})')
    parser.add_argument('--no_pos_weight', action='store_true',
                        help='Disable automatic per-class pos_weight for BCE loss '
                             '(only affects multi-label strategies; default: enabled)')

    # Data/experiment params
    tool_list = 'DePP_85, PhageRBPdetect, DepoScope, DepoRanker, SpikeHunter, dbCAN, IPR, phold_glycan_tailspike'
    parser.add_argument('--tools',
                        help=f'Comma-separated tool names; keep proteins flagged by ANY. '
                             f'Valid: {tool_list}. '
                             f'Default: no filter (all 8 tools)')
    parser.add_argument('--exclude_tools',
                        help=f'Comma-separated tool names; drop proteins flagged by ANY. '
                             f'Default: no exclusion')
    parser.add_argument('--protein_set',
                        help='DEPRECATED: use --tools/--exclude_tools. '
                             'Values: all_glycan_binders, tsp_only, rbp_only')
    parser.add_argument('--min_sources', type=int,
                        help=f'Min tool sources for filtering; set to 1 to disable '
                             f'(default: {fmt(d_exp.get("min_sources"))})')
    parser.add_argument('--max_k_types', type=int,
                        help=f'Max distinct K-types per protein (default: {fmt(d_exp.get("max_k_types"))})')
    parser.add_argument('--max_o_types', type=int,
                        help=f'Max distinct O-types per protein (default: {fmt(d_exp.get("max_o_types"))})')
    parser.add_argument('--max_samples_per_k', type=int,
                        help=f'Downsample cap per K-type class (default: {fmt(d_exp.get("max_samples_per_k"), "no downsampling")})')
    parser.add_argument('--max_samples_per_o', type=int,
                        help=f'Downsample cap per O-type class (default: {fmt(d_exp.get("max_samples_per_o"), "no downsampling")})')
    parser.add_argument('--label_strategy',
                        help=f'Label strategy: single_label, multi_label, '
                             f'multi_label_threshold, weighted_soft, weighted_multi_label '
                             f'(default: {fmt(d_exp.get("label_strategy"))})')
    parser.add_argument('--min_label_count', type=int,
                        help=f'For multi_label_threshold: min count per class to label as positive '
                             f'(default: {fmt(d_exp.get("min_label_count"), 1)})')
    parser.add_argument('--min_label_fraction', type=float,
                        help=f'For multi_label_threshold: min fraction of observations per class '
                             f'(default: {fmt(d_exp.get("min_label_fraction"), 0.1)})')
    parser.add_argument('--min_class_samples', type=int,
                        help=f'Drop K and O classes with < N training samples '
                             f'(default: {fmt(d_exp.get("min_class_samples"), "no filter")}; '
                             f'try 25)')

    # Model params
    parser.add_argument('--hidden_dims',
                        help=f'Hidden dims comma-separated (default: {fmt(d_model.get("hidden_dims"))})')
    parser.add_argument('--attention_dim', type=int,
                        help=f'SE attention bottleneck dim (default: {fmt(d_model.get("attention_dim"))})')
    parser.add_argument('--dropout', type=float,
                        help=f'Dropout rate (default: {fmt(d_model.get("dropout"))})')

    # Embedding
    parser.add_argument('--embedding_type',
                        help=f'Embedding type label: esm2_650m, kmer_murphy8_k5, etc. '
                             f'(default: {fmt(d_data.get("embedding_type"))})')
    parser.add_argument('--embedding_file',
                        help=f'Path to training embedding NPZ file. Overrides the '
                             f'default path derived from embedding_type. '
                             f'(default: {fmt(d_data.get("embedding_file"))})')

    # Naming
    parser.add_argument('--name',
                        help='Custom run name (default: auto-generated from params + timestamp)')

    args = parser.parse_args()

    project_root = find_project_root()

    # Load base config and apply overrides
    config = load_base_config(args.model, project_root)
    config = apply_overrides(config, args)

    # Generate experiment directory
    run_name = args.name or generate_run_name(config)
    experiment_dir = os.path.join(
        project_root, 'experiments', args.model, run_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Save merged config
    config_path = os.path.join(experiment_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f'Experiment: {experiment_dir}')
    print(f'Config saved to: {config_path}')

    # Load and run model's train function
    train_module = load_train_module(args.model, project_root)
    train_module.train(experiment_dir, config)


if __name__ == '__main__':
    main()
