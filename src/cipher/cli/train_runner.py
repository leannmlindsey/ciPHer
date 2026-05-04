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
    if args.positive_list is not None:
        # Mutex with tool-based filters — clear any inherited defaults
        config.setdefault('experiment', {})['positive_list_path'] = args.positive_list
        config['experiment'].pop('tools', None)
        config['experiment'].pop('exclude_tools', None)
        config['experiment'].pop('protein_set', None)
        # Also clear per-head lists from base config — single-list mode wins.
        config['experiment'].pop('positive_list_k_path', None)
        config['experiment'].pop('positive_list_o_path', None)
    if args.positive_list_k is not None or args.positive_list_o is not None:
        # Per-head v2 mode. Clear the legacy single-list + tool filters.
        exp = config.setdefault('experiment', {})
        if args.positive_list_k is not None:
            exp['positive_list_k_path'] = args.positive_list_k
        if args.positive_list_o is not None:
            exp['positive_list_o_path'] = args.positive_list_o
        exp.pop('positive_list_path', None)
        exp.pop('tools', None)
        exp.pop('exclude_tools', None)
        exp.pop('protein_set', None)
    if args.heads is not None:
        config.setdefault('training', {})['heads'] = args.heads
    if args.split_style is not None:
        config.setdefault('training', {})['split_style'] = args.split_style
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
    if args.cluster_file is not None:
        config.setdefault('experiment', {})['cluster_file_path'] = args.cluster_file
    if args.cluster_threshold is not None:
        config.setdefault('experiment', {})['cluster_threshold'] = args.cluster_threshold

    # Model overrides
    if args.hidden_dims is not None:
        dims = [int(d) for d in args.hidden_dims.split(',')]
        config.setdefault('model', {})['hidden_dims'] = dims
    if args.attention_dim is not None:
        config.setdefault('model', {})['attention_dim'] = args.attention_dim
    if args.dropout is not None:
        config.setdefault('model', {})['dropout'] = args.dropout

    # Contrastive-encoder-specific overrides (ignored by other models)
    if args.lambda_k is not None:
        config.setdefault('training', {})['lambda_k'] = args.lambda_k
    if args.lambda_o is not None:
        config.setdefault('training', {})['lambda_o'] = args.lambda_o
    if args.arcface_margin is not None:
        config.setdefault('arcface', {})['margin'] = args.arcface_margin
    if args.arcface_scale is not None:
        config.setdefault('arcface', {})['scale'] = args.arcface_scale
    if args.sampler_hard_negative_mining is not None:
        config.setdefault('sampler', {})['hard_negative_mining'] = args.sampler_hard_negative_mining
    if args.sampler_hard_negative_start_epoch is not None:
        config.setdefault('sampler', {})['hard_negative_start_epoch'] = args.sampler_hard_negative_start_epoch

    # Embedding overrides
    if args.embedding_type is not None:
        config.setdefault('data', {})['embedding_type'] = args.embedding_type
    if args.embedding_file is not None:
        config.setdefault('data', {})['embedding_file'] = args.embedding_file
    if args.embedding_type_2 is not None:
        config.setdefault('data', {})['embedding_type_2'] = args.embedding_type_2
    if args.embedding_file_2 is not None:
        config.setdefault('data', {})['embedding_file_2'] = args.embedding_file_2
    if args.val_embedding_file_2 is not None:
        config.setdefault('validation', {})['val_embedding_file_2'] = args.val_embedding_file_2

    # Data path overrides
    if args.association_map is not None:
        config.setdefault('data', {})['association_map'] = args.association_map
    if args.glycan_binders is not None:
        config.setdefault('data', {})['glycan_binders'] = args.glycan_binders

    # Validation path overrides (saved in config.yaml for cipher-evaluate)
    if args.val_fasta is not None:
        config.setdefault('validation', {})['val_fasta'] = args.val_fasta
    if args.val_embedding_file is not None:
        config.setdefault('validation', {})['val_embedding_file'] = args.val_embedding_file
    if args.val_datasets_dir is not None:
        config.setdefault('validation', {})['val_datasets_dir'] = args.val_datasets_dir

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
        positive_list_path set                 -> 'posList'
    """
    cluster_tag = None
    if exp.get('cluster_file_path'):
        t = exp.get('cluster_threshold') or 70
        cluster_tag = f'cl{t}'

    def _with_cluster(base):
        return f'{base}_{cluster_tag}' if cluster_tag else base

    if exp.get('positive_list_path'):
        return _with_cluster('posList')

    # If legacy protein_set is still set (not translated), use it
    ps = exp.get('protein_set')
    tools = exp.get('tools')
    excl = exp.get('exclude_tools')

    if ps and not tools and not excl:
        return _with_cluster(ps)

    parts = []
    if tools:
        parts.append('-'.join(sorted(tools)))
    if excl:
        parts.append('no' + '-'.join(sorted(excl)))

    if not parts:
        return _with_cluster('allTools')
    return _with_cluster('_'.join(parts))


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
    parser.add_argument('--positive_list_k',
                        help='v2 per-head: positive list for the K head. '
                             'When combined with --positive_list_o, training '
                             'samples are the UNION of both lists; each '
                             'sample contributes only to the head-loss for '
                             'the list it appears in (label-level masking). '
                             'Mutex with --positive_list, --tools, '
                             '--exclude_tools, --protein_set.')
    parser.add_argument('--positive_list_o',
                        help='v2 per-head: positive list for the O head. '
                             'See --positive_list_k.')
    parser.add_argument('--heads', choices=('both', 'k', 'o'),
                        default=None,
                        help='Which head(s) to train: both (default), k, or o. '
                             'attention_mlp skips the other head\'s training '
                             'loop; contrastive_encoder forces the other '
                             'lambda to 0. Orthogonal to the positive-list '
                             'flags — you can set per-head lists and still '
                             'train only one head.')
    parser.add_argument('--split-style', choices=('independent', 'canonical'),
                        default=None,
                        help='How to split into train/val/test for the K and '
                             'O heads. "independent" (default): K and O get '
                             'separate stratified splits with seeds (s, s+1). '
                             '"canonical" (matches the old klebsiella '
                             'pipeline): one shared split keyed on primary K '
                             '(with O-fallback when K is null/N/A); both heads '
                             'see the same train/val/test partition.')
    parser.add_argument('--positive_list',
                        help='Path to a positive-list file (one protein ID per line). '
                             'Filters candidates to these IDs only. Mutually '
                             'exclusive with --tools/--exclude_tools/--protein_set.')
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
    parser.add_argument('--cluster_file',
                        help='Path to candidates_clusters.tsv. When set, '
                             'per-class downsampling uses round-robin across '
                             'clusters instead of random sampling (default: off).')
    parser.add_argument('--cluster_threshold', type=int,
                        choices=[30, 40, 50, 60, 70, 80, 85, 90, 95],
                        help=f'Identity threshold column in cluster_file '
                             f'(default: {fmt(d_exp.get("cluster_threshold"), 70)})')

    # Model params
    parser.add_argument('--hidden_dims',
                        help=f'Hidden dims comma-separated (default: {fmt(d_model.get("hidden_dims"))})')
    parser.add_argument('--attention_dim', type=int,
                        help=f'SE attention bottleneck dim (default: {fmt(d_model.get("attention_dim"))})')
    parser.add_argument('--dropout', type=float,
                        help=f'Dropout rate (default: {fmt(d_model.get("dropout"))})')

    # Contrastive-encoder-specific (ignored by other models)
    d_train = base_cfg.get('training', {})
    d_arc = base_cfg.get('arcface', {})
    d_samp = base_cfg.get('sampler', {})
    parser.add_argument('--lambda_k', type=float,
                        help=f'Contrastive encoder: K-head loss weight. Set 0 to disable. '
                             f'(default: {fmt(d_train.get("lambda_k"))})')
    parser.add_argument('--lambda_o', type=float,
                        help=f'Contrastive encoder: O-head loss weight. Set 0 to disable. '
                             f'(default: {fmt(d_train.get("lambda_o"))})')
    parser.add_argument('--arcface_margin', type=float,
                        help=f'Contrastive encoder: ArcFace additive angular margin '
                             f'(default: {fmt(d_arc.get("margin"))})')
    parser.add_argument('--arcface_scale', type=float,
                        help=f'Contrastive encoder: ArcFace scale '
                             f'(default: {fmt(d_arc.get("scale"))})')
    parser.add_argument('--sampler_hard_negative_mining',
                        type=lambda s: s.lower() in ('1', 'true', 'yes'),
                        help='Contrastive encoder: enable prototype-based hard-negative '
                             'class sampling in PK sampler (true/false).')
    parser.add_argument('--sampler_hard_negative_start_epoch', type=int,
                        help=f'Contrastive encoder: epoch to switch from uniform to '
                             f'hard-negative class sampling. '
                             f'(default: {fmt(d_samp.get("hard_negative_start_epoch"))})')

    # Embedding
    parser.add_argument('--embedding_type',
                        help=f'Embedding type label: esm2_650m, kmer_murphy8_k5, etc. '
                             f'(default: {fmt(d_data.get("embedding_type"))})')
    parser.add_argument('--embedding_file',
                        help=f'Path to training embedding NPZ file. Overrides the '
                             f'default path derived from embedding_type. '
                             f'(default: {fmt(d_data.get("embedding_file"))})')
    parser.add_argument('--embedding_type_2',
                        help='Optional second embedding type label (e.g. '
                             'kmer_aa20_k4). When set along with '
                             '--embedding_file_2, features become the '
                             'concatenation of the two embeddings per MD5.')
    parser.add_argument('--embedding_file_2',
                        help='Path to a second training embedding NPZ to '
                             'concatenate with --embedding_file (pLM+kmer '
                             'combo experiments).')
    parser.add_argument('--val_embedding_file_2',
                        help='Path to the matching second validation embedding '
                             'NPZ. Required if --embedding_file_2 is set and '
                             'evaluation will be run from the saved config.')
    parser.add_argument('--association_map',
                        help=f'Path to host_phage_protein_map.tsv '
                             f'(default: {fmt(d_data.get("association_map"))})')
    parser.add_argument('--glycan_binders',
                        help=f'Path to glycan_binders_custom.tsv '
                             f'(default: {fmt(d_data.get("glycan_binders"))})')

    # Validation paths (saved in config.yaml for cipher-evaluate to use)
    d_val = base_cfg.get('validation', {})
    parser.add_argument('--val_fasta',
                        help=f'Path to validation protein FASTA '
                             f'(default: {fmt(d_val.get("val_fasta"))})')
    parser.add_argument('--val_embedding_file',
                        help=f'Path to validation embedding NPZ '
                             f'(default: {fmt(d_val.get("val_embedding_file"))})')
    parser.add_argument('--val_datasets_dir',
                        help=f'Path to validation datasets directory '
                             f'(default: {fmt(d_val.get("val_datasets_dir"))})')

    # Naming
    parser.add_argument('--name',
                        help='Custom run name (default: auto-generated from params + timestamp)')

    args = parser.parse_args()

    # Enforce mutual exclusion early (clearer error than at config-load time)
    if args.positive_list and (args.tools or args.exclude_tools or args.protein_set):
        parser.error(
            '--positive_list is mutually exclusive with --tools, '
            '--exclude_tools, and --protein_set.')
    if args.positive_list and (args.positive_list_k or args.positive_list_o):
        parser.error(
            '--positive_list is mutually exclusive with '
            '--positive_list_k / --positive_list_o. Use one mode or the other.')
    if (args.positive_list_k or args.positive_list_o) and (
            args.tools or args.exclude_tools or args.protein_set):
        parser.error(
            '--positive_list_k / --positive_list_o are mutually exclusive '
            'with --tools, --exclude_tools, and --protein_set.')

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

    # Auto-run per_head_strict_eval after training. This produces the
    # `<exp>/results/per_head_strict_eval.json` file the harvest reads
    # for the headline any-hit columns. Mandatory — no opt-out, so every
    # cipher-train invocation produces a comparable headline JSON.
    # Skipped only when validation paths aren't configured (no input
    # surface for evaluation).
    _auto_run_strict_eval(experiment_dir, config, project_root)


def _auto_run_strict_eval(experiment_dir, config, project_root):
    """Invoke scripts/analysis/per_head_strict_eval.py on a freshly-trained run.

    Subprocess invocation (rather than in-process import) keeps the eval's
    side effects — sys.path mutations, CUDA context lifecycle, predict.py
    import — isolated from the training process. Same pattern the SLURM
    launchers used to do explicitly; now baked in so every cipher-train
    call produces the headline JSON without a launcher needing to chain.
    """
    import subprocess
    val_cfg = config.get('validation', {})
    val_emb = val_cfg.get('val_embedding_file')
    val_fasta = val_cfg.get('val_fasta')
    val_ds = val_cfg.get('val_datasets_dir')
    if not (val_emb and val_fasta and val_ds):
        print('\n[strict-eval] Skipping — validation paths not configured.')
        return
    if not os.path.exists(val_emb):
        print(f'\n[strict-eval] Skipping — val_embedding_file not found: {val_emb}')
        return

    script = os.path.join(project_root, 'scripts', 'analysis',
                          'per_head_strict_eval.py')
    if not os.path.exists(script):
        print(f'\n[strict-eval] Skipping — script not found at {script}')
        return

    cmd = [
        sys.executable, '-u', script, experiment_dir,
        '--val-embedding-file', val_emb,
        '--val-fasta', val_fasta,
        '--val-datasets-dir', val_ds,
    ]
    val_emb_2 = val_cfg.get('val_embedding_file_2')
    if val_emb_2:
        cmd += ['--val-embedding-file-2', val_emb_2]

    print('\n=== Auto-running per_head_strict_eval ===')
    print('  ' + ' '.join(cmd))
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        print(f'[strict-eval] WARNING: per_head_strict_eval exited {rc}; '
              f'training itself succeeded — eval can be re-run manually.')


if __name__ == '__main__':
    main()
