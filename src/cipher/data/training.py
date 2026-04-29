"""Training data preparation pipeline.

Prepares filtered, labeled training data from the protein index and
association table. Each experiment configures its own filtering via
TrainingConfig (protein set, min_sources, max_k/o_types, downsampling).

Heavy data (embeddings) is NOT loaded here. After prepare_training_data()
returns, the experiment's train.py loads embeddings filtered to the
surviving MD5 set.

Data flow:
    1. Load association table (host_phage_protein_map.tsv)
       → links protein_id/md5 to host K/O serotypes
    2. Load protein index (glycan_binders_custom.tsv)
       → has tool flags (SpikeHunter, total_sources, etc.) for filtering
    3. Filter proteins by config
    4. Build per-MD5 serotype association counts
    5. Apply specificity filters, single-label reduction, downsampling
    6. Build label vectors
    → Returns TrainingData(md5_list, k_labels, o_labels, ...)
"""

import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np


NULL_LABELS = {'null', 'N/A', 'Unknown', '', 'NA', 'n/a'}

# The 8 tool flag columns in glycan_binders_custom.tsv
TOOL_COLUMNS = [
    'DePP_85', 'PhageRBPdetect', 'DepoScope', 'DepoRanker',
    'SpikeHunter', 'dbCAN', 'IPR', 'phold_glycan_tailspike',
]

# Map legacy protein_set values to tool-based filters
_PROTEIN_SET_ALIASES = {
    'all_glycan_binders': {'tools': None, 'exclude_tools': None},
    'tsp_only': {'tools': ['SpikeHunter'], 'exclude_tools': None},
    'rbp_only': {'tools': None, 'exclude_tools': ['SpikeHunter']},
}


def _validate_tools(names, field_name='tools'):
    """Validate a list of tool names against TOOL_COLUMNS."""
    if names is None:
        return None
    if isinstance(names, str):
        # Allow comma-separated string for convenience
        names = [n.strip() for n in names.split(',') if n.strip()]
    invalid = [n for n in names if n not in TOOL_COLUMNS]
    if invalid:
        raise ValueError(
            f"Unknown tool{'s' if len(invalid) > 1 else ''} in --{field_name}: {invalid}. "
            f"Valid tools: {TOOL_COLUMNS}")
    return list(names) if names else None


@dataclass
class TrainingConfig:
    """All configuration knobs for the filtering/labeling pipeline."""
    tools: list = None          # None/empty = no tool filter (keep all glycan binders)
    exclude_tools: list = None  # None/empty = no exclusion
    protein_set: str = None     # DEPRECATED: use tools/exclude_tools
    # If set, intersect candidate proteins with IDs in this file and skip
    # the tool filter entirely. Mutually exclusive with tools/exclude_tools.
    positive_list_path: str = None
    # Per-head positive lists (v2 highconf design, 2026-04-22).
    # When set, the training sample pool is the UNION of both lists, and
    # each sample's K (resp. O) labels are zeroed for samples not in the
    # K (resp. O) list — so only proteins in the K list contribute to K
    # head loss, and only proteins in the O list contribute to O head loss.
    # Mutually exclusive with positive_list_path and with tools/exclude_tools.
    # Back-compat: if only positive_list_path is set, both heads use it
    # (identical to pre-v2 behaviour).
    positive_list_k_path: str = None
    positive_list_o_path: str = None
    # Optional cluster-stratified downsampling. cluster_file_path points at
    # a multi-threshold TSV (protein_id + cl30_/cl40_/.../cl95_ columns);
    # cluster_threshold picks which identity column to use. When set, the
    # max_samples_per_k/o caps are filled by round-robin across clusters
    # instead of random sampling, maximizing sequence diversity per class.
    cluster_file_path: str = None
    cluster_threshold: int = 70
    min_sources: int = 1
    max_k_types: int = None
    max_o_types: int = None
    single_label: bool = False
    max_samples_per_k: int = None
    max_samples_per_o: int = None
    max_k_observations: int = None
    drop_null_k: bool = False
    drop_null_o: bool = False
    label_strategy: str = 'multi_label'
    # Thresholds used only when label_strategy='multi_label_threshold':
    # a K-type is a positive label for a protein only if
    # count[K] >= min_label_count AND count[K]/total >= min_label_fraction
    # Defaults: keep all single observations (count=1), but drop K-types
    # that are less than 10% of a protein's total observations (likely noise).
    min_label_count: int = 1
    min_label_fraction: float = 0.1
    # Drop K and O classes with fewer than this many training samples
    # (counted AFTER all other filters and downsampling). None = no filter.
    # Empirical heuristic: classes with < 25 samples are typically not
    # learnable from sequence alone.
    min_class_samples: int = None
    downsample_seed: int = 42

    def __post_init__(self):
        # Backward-compat: translate legacy protein_set to tools/exclude_tools
        if self.protein_set and self.tools is None and self.exclude_tools is None:
            if self.protein_set not in _PROTEIN_SET_ALIASES:
                raise ValueError(
                    f"Unknown protein_set: {self.protein_set!r}. "
                    f"Valid: {list(_PROTEIN_SET_ALIASES.keys())}. "
                    f"Prefer --tools / --exclude_tools for new experiments.")
            warnings.warn(
                f"protein_set={self.protein_set!r} is deprecated. "
                f"Use --tools / --exclude_tools instead.",
                DeprecationWarning, stacklevel=3)
            alias = _PROTEIN_SET_ALIASES[self.protein_set]
            self.tools = alias['tools']
            self.exclude_tools = alias['exclude_tools']

        # Validate tool names
        self.tools = _validate_tools(self.tools, 'tools')
        self.exclude_tools = _validate_tools(self.exclude_tools, 'exclude_tools')

        # Enforce mutual exclusion: positive_list_path vs tool-based filters
        if self.positive_list_path and (self.tools or self.exclude_tools):
            raise ValueError(
                'positive_list_path is mutually exclusive with tools/'
                'exclude_tools. Use one or the other, not both.')

        # Per-head positive lists are mutex with the single-list field
        # and with tool-based filters.
        per_head = self.positive_list_k_path or self.positive_list_o_path
        if per_head and self.positive_list_path:
            raise ValueError(
                'positive_list_path is mutually exclusive with '
                'positive_list_k_path / positive_list_o_path. '
                'Use one mode or the other.')
        if per_head and (self.tools or self.exclude_tools):
            raise ValueError(
                'positive_list_k_path / positive_list_o_path are mutually '
                'exclusive with tools/exclude_tools.')

    @classmethod
    def from_dict(cls, d):
        """Create from a flat dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    @classmethod
    def from_yaml(cls, path):
        """Load from a YAML config file.

        Reads the 'experiment' section and merges with 'labels.strategy'.
        """
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        d = dict(raw.get('experiment', {}))
        strategy = raw.get('labels', {}).get('strategy')
        if strategy:
            d['label_strategy'] = strategy
        if d.get('single_label_only'):
            d['single_label'] = True
            d['label_strategy'] = 'single_label'
        d.pop('single_label_only', None)
        return cls.from_dict(d)


@dataclass
class TrainingData:
    """Output of the preparation pipeline."""
    md5_list: list
    k_labels: np.ndarray
    o_labels: np.ndarray
    k_classes: list
    o_classes: list
    is_tsp: np.ndarray
    config: TrainingConfig

    def save(self, output_dir):
        """Save training_data.npz and label_encoders.json."""
        import json
        import os
        os.makedirs(output_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join(output_dir, 'training_data.npz'),
            md5_list=np.array(self.md5_list, dtype='U32'),
            k_labels=self.k_labels,
            o_labels=self.o_labels,
            is_tsp=self.is_tsp,
        )

        with open(os.path.join(output_dir, 'label_encoders.json'), 'w') as f:
            json.dump({
                'k_classes': self.k_classes,
                'o_classes': self.o_classes,
                'strategy': self.config.label_strategy,
                'tools': self.config.tools,
                'exclude_tools': self.config.exclude_tools,
                # legacy field kept for backward compat with older readers
                'protein_set': self.config.protein_set,
            }, f, indent=2)

    @classmethod
    def load(cls, output_dir):
        """Load from previously saved directory."""
        import json
        import os

        data = np.load(os.path.join(output_dir, 'training_data.npz'))
        with open(os.path.join(output_dir, 'label_encoders.json')) as f:
            encoders = json.load(f)

        # Build config preferring new fields; fall back to legacy protein_set
        cfg_kwargs = {
            'label_strategy': encoders.get('strategy', 'multi_label'),
        }
        if 'tools' in encoders or 'exclude_tools' in encoders:
            cfg_kwargs['tools'] = encoders.get('tools')
            cfg_kwargs['exclude_tools'] = encoders.get('exclude_tools')
        elif encoders.get('protein_set'):
            cfg_kwargs['protein_set'] = encoders['protein_set']

        return cls(
            md5_list=data['md5_list'].tolist(),
            k_labels=data['k_labels'],
            o_labels=data['o_labels'],
            k_classes=encoders['k_classes'],
            o_classes=encoders['o_classes'],
            is_tsp=data['is_tsp'],
            config=TrainingConfig(**cfg_kwargs),
        )


def prepare_training_data(config, association_map_path, glycan_binders_path,
                          ref_k_classes=None, ref_o_classes=None, verbose=True):
    """Run the full filtering + labeling pipeline.

    Args:
        config: TrainingConfig instance
        association_map_path: path to host_phage_protein_map.tsv
            Columns: host_assembly, host_K, host_O, phage_id, protein_id,
                     is_tsp, protein_md5
        glycan_binders_path: path to glycan_binders_custom.tsv
            Columns: protein_id, total_sources, SpikeHunter, ...
        ref_k_classes: optional list of K classes (for matching a trained model)
        ref_o_classes: optional list of O classes
        verbose: print progress

    Returns:
        TrainingData
    """
    from cipher.data.interactions import load_training_map
    from cipher.data.proteins import load_glycan_binders

    def log(msg):
        if verbose:
            print(msg)

    # Step 1: Load data
    log('Loading association table...')
    rows = load_training_map(association_map_path)
    log(f'  {len(rows)} rows')

    log('Loading protein index...')
    glycan_dict = load_glycan_binders(glycan_binders_path)
    log(f'  {len(glycan_dict)} proteins')

    # Step 2: Get initial protein set from association table
    all_protein_ids = {r['protein_id'] for r in rows}
    log(f'  {len(all_protein_ids)} unique proteins in association table')

    # Step 3: Filter proteins. Three mutex modes:
    #   (a) positive_list_path          -> single list for both heads
    #   (b) positive_list_{k,o}_path    -> union-of-both-lists filter; each
    #                                      head later zeroes labels for
    #                                      samples not in its own list
    #   (c) tools / exclude_tools       -> tool-based filter
    # k_positive_set / o_positive_set track per-head membership for the
    # later masking step; in (a) both point at positive_set, in (c) both
    # stay None (no masking needed, all samples contribute to both heads).
    positive_set = None
    k_positive_set = None
    o_positive_set = None
    from cipher.data.proteins import load_positive_list
    if config.positive_list_path:
        positive_set = load_positive_list(config.positive_list_path)
        k_positive_set = positive_set
        o_positive_set = positive_set
        log(f'Loaded positive list: {len(positive_set)} IDs from '
            f'{config.positive_list_path}')
    elif config.positive_list_k_path or config.positive_list_o_path:
        if config.positive_list_k_path:
            k_positive_set = load_positive_list(config.positive_list_k_path)
            log(f'Loaded K positive list: {len(k_positive_set)} IDs from '
                f'{config.positive_list_k_path}')
        if config.positive_list_o_path:
            o_positive_set = load_positive_list(config.positive_list_o_path)
            log(f'Loaded O positive list: {len(o_positive_set)} IDs from '
                f'{config.positive_list_o_path}')
        positive_set = (k_positive_set or set()) | (o_positive_set or set())
        log(f'Per-head union positive set: {len(positive_set)} IDs')
    filtered_ids = _filter_proteins(all_protein_ids, glycan_dict, config, log,
                                    positive_set=positive_set)

    # Step 4: Build per-MD5 serotype associations
    md5_k_counts, md5_o_counts, md5_is_tsp = _build_md5_associations(
        rows, filtered_ids)
    log(f'  {len(md5_k_counts)} unique MD5s with associations')

    # Step 5: Apply specificity filters
    _filter_non_specific(md5_k_counts, md5_o_counts, md5_is_tsp, config, log)

    # Step 6: Single-label reduction
    if config.single_label:
        _apply_single_label(md5_k_counts, md5_o_counts, log)

    # Step 7: Downsampling
    if config.max_samples_per_k or config.max_samples_per_o:
        cluster_map = None
        if config.cluster_file_path:
            cluster_map = _build_md5_cluster_map(
                config.cluster_file_path, config.cluster_threshold, rows, log)
        _downsample(md5_k_counts, md5_o_counts, md5_is_tsp, config, log,
                    cluster_map=cluster_map)
    else:
        log('  Downsampling: disabled (no max_samples_per_k/o set)')

    # Step 8: Optional stricter K-only filters
    _filter_k_strict(md5_k_counts, md5_o_counts, md5_is_tsp, config, log)

    # Step 9: Build label vectors.
    # Label-format routing:
    #   single_label           -> multi_label (one-hot)
    #   multi_label            -> multi_label (multi-hot)
    #   multi_label_threshold  -> multi_label with count/fraction threshold
    #   weighted_soft          -> weighted (fractional, used with KL loss)
    #   weighted_multi_label   -> weighted (fractional, used with BCE loss)
    strategy = config.label_strategy
    label_format = strategy
    if strategy in ('single_label', 'multi_label'):
        label_format = 'multi_label'
    elif strategy == 'weighted_multi_label':
        label_format = 'weighted_soft'
    elif strategy not in ('multi_label_threshold', 'weighted_soft'):
        raise ValueError(
            f'Unknown label_strategy: {strategy!r}. Valid: single_label, '
            f'multi_label, multi_label_threshold, weighted_soft, '
            f'weighted_multi_label')

    md5_list, k_labels, o_labels, k_classes, o_classes = _build_label_vectors(
        md5_k_counts, md5_o_counts, label_format,
        ref_k_classes, ref_o_classes,
        min_count=config.min_label_count,
        min_fraction=config.min_label_fraction)

    # Per-head label masking (v2 highconf mode). If the K list and O list
    # differ, zero out K labels for samples not in the K list, and the
    # same for O. This ensures only K-clean proteins contribute to K head
    # training and only O-clean proteins contribute to O head training,
    # while still letting the shared encoder (e.g. contrastive_encoder)
    # see the union of both sets.
    # In back-compat mode (positive_list_path set), k_positive_set IS
    # o_positive_set IS positive_set, so this is a no-op: every surviving
    # sample is in both sets.
    if (k_positive_set is not None
            and o_positive_set is not None
            and k_positive_set is not o_positive_set):
        md5_to_pids = defaultdict(set)
        for r in rows:
            if r['protein_md5'] in md5_k_counts or r['protein_md5'] in md5_o_counts:
                md5_to_pids[r['protein_md5']].add(r['protein_id'])
        n_k_masked = 0
        n_o_masked = 0
        for i, md5 in enumerate(md5_list):
            pids = md5_to_pids.get(md5, set())
            if not (pids & k_positive_set):
                k_labels[i] = 0
                n_k_masked += 1
            if not (pids & o_positive_set):
                o_labels[i] = 0
                n_o_masked += 1
        log(f'  Per-head masking: {n_k_masked} MD5s zeroed for K head, '
            f'{n_o_masked} for O head')

    # For threshold strategy, some proteins may have all-zero labels after
    # thresholding. Drop them — training on "no labels" is misleading.
    if strategy == 'multi_label_threshold':
        before = len(md5_list)
        k_row_sum = k_labels.sum(axis=1)
        o_row_sum = o_labels.sum(axis=1)
        keep_mask = (k_row_sum > 0) | (o_row_sum > 0)
        if not keep_mask.all():
            md5_list = [m for m, k in zip(md5_list, keep_mask) if k]
            k_labels = k_labels[keep_mask]
            o_labels = o_labels[keep_mask]
            log(f'  multi_label_threshold drop zero-label: {before} -> {len(md5_list)}')

    # Drop K and O classes with insufficient training samples.
    # Counts are based on "any positive label" (binary labels > 0).
    # K and O are filtered independently since they're independent heads.
    dropped_k = []
    dropped_o = []
    if config.min_class_samples and config.min_class_samples > 0:
        thr = config.min_class_samples

        # K classes
        k_counts = (k_labels > 0).sum(axis=0)  # samples per K class
        keep_k = k_counts >= thr
        dropped_k = [c for c, k in zip(k_classes, keep_k) if not k]
        if dropped_k:
            k_labels = k_labels[:, keep_k]
            k_classes = [c for c, k in zip(k_classes, keep_k) if k]
            log(f'  min_class_samples >= {thr} (K): dropped {len(dropped_k)} classes, '
                f'kept {len(k_classes)}')

        # O classes
        o_counts = (o_labels > 0).sum(axis=0)
        keep_o = o_counts >= thr
        dropped_o = [c for c, k in zip(o_classes, keep_o) if not k]
        if dropped_o:
            o_labels = o_labels[:, keep_o]
            o_classes = [c for c, k in zip(o_classes, keep_o) if k]
            log(f'  min_class_samples >= {thr} (O): dropped {len(dropped_o)} classes, '
                f'kept {len(o_classes)}')

    # Build is_tsp array
    is_tsp_array = np.array(
        [1 if md5_is_tsp.get(md5, False) else 0 for md5 in md5_list],
        dtype=np.int8)

    log(f'\nFinal dataset: {len(md5_list)} MD5s, '
        f'{len(k_classes)} K classes, {len(o_classes)} O classes')
    log(f'  TSPs: {is_tsp_array.sum()}, '
        f'non-TSP: {len(md5_list) - is_tsp_array.sum()}')

    td = TrainingData(
        md5_list=md5_list,
        k_labels=k_labels,
        o_labels=o_labels,
        k_classes=k_classes,
        o_classes=o_classes,
        is_tsp=is_tsp_array,
        config=config,
    )
    td.dropped_k_classes = dropped_k
    td.dropped_o_classes = dropped_o
    return td


# ============================================================
# Internal helpers
# ============================================================

def _filter_proteins(all_protein_ids, glycan_dict, config, log, positive_set=None):
    """Filter proteins by positive_list OR by tools (mutually exclusive) and min_sources.

    Filter logic:
        1. Keep only proteins present in glycan_dict
        2. min_sources: keep if total_sources >= min_sources (applies either branch)
        3. If positive_set is given: keep only proteins in that set, and skip
           the tool filter entirely.
        4. Else, use tool-based filtering:
           - tools: if set, keep proteins flagged by AT LEAST ONE listed tool
           - exclude_tools: if set, drop proteins flagged by ANY listed tool
    """
    min_sources = config.min_sources

    # Start with proteins that have glycan binder data
    filtered = {pid for pid in all_protein_ids if pid in glycan_dict}
    if len(filtered) < len(all_protein_ids):
        log(f'  in glycan_binders: {len(all_protein_ids)} -> {len(filtered)}')

    # Filter by min_sources (applies regardless of primary filter)
    if min_sources > 1:
        before = len(filtered)
        filtered = {pid for pid in filtered
                    if int(glycan_dict[pid].get('total_sources', 0)) >= min_sources}
        log(f'  min_sources >= {min_sources}: {before} -> {len(filtered)}')

    # Primary filter: positive_list OR tools (mutex enforced by TrainingConfig)
    if positive_set is not None:
        before = len(filtered)
        filtered = {pid for pid in filtered if pid in positive_set}
        log(f'  positive_list filter: {before} -> {len(filtered)}')
        return filtered

    tools = config.tools
    exclude_tools = config.exclude_tools

    if tools:
        before = len(filtered)
        filtered = {pid for pid in filtered
                    if any(int(glycan_dict[pid].get(t, 0)) == 1 for t in tools)}
        log(f'  tools={tools}: {before} -> {len(filtered)}')

    if exclude_tools:
        before = len(filtered)
        filtered = {pid for pid in filtered
                    if not any(int(glycan_dict[pid].get(t, 0)) == 1 for t in exclude_tools)}
        log(f'  exclude_tools={exclude_tools}: {before} -> {len(filtered)}')

    if not tools and not exclude_tools:
        log(f'  no tool filter (allTools): {len(filtered)} proteins')

    return filtered


def _build_md5_associations(rows, filtered_ids):
    """Build per-MD5 serotype association counts from association table rows.

    Args:
        rows: list of dicts from load_training_map()
        filtered_ids: set of protein_ids that passed filtering

    Returns:
        md5_k_counts: {md5: Counter({k_type: count})}
        md5_o_counts: {md5: Counter({o_type: count})}
        md5_is_tsp: {md5: bool}
    """
    md5_k_counts = defaultdict(Counter)
    md5_o_counts = defaultdict(Counter)
    md5_is_tsp = {}

    for row in rows:
        pid = row['protein_id']
        if pid not in filtered_ids:
            continue

        md5 = row['protein_md5']
        md5_k_counts[md5][row['host_K']] += 1
        md5_o_counts[md5][row['host_O']] += 1
        md5_is_tsp[md5] = row['is_tsp'] == 1

    return md5_k_counts, md5_o_counts, md5_is_tsp


def _remove_md5(md5, md5_k_counts, md5_o_counts, md5_is_tsp):
    """Remove an MD5 from all tracking dicts."""
    md5_k_counts.pop(md5, None)
    md5_o_counts.pop(md5, None)
    md5_is_tsp.pop(md5, None)


def _filter_non_specific(md5_k_counts, md5_o_counts, md5_is_tsp, config, log):
    """Remove MD5s that map to too many distinct K or O types."""
    if config.max_k_types:
        before = len(md5_k_counts)
        to_drop = {md5 for md5, counts in md5_k_counts.items()
                   if len(counts) > config.max_k_types}
        for md5 in to_drop:
            _remove_md5(md5, md5_k_counts, md5_o_counts, md5_is_tsp)
        log(f'  max_k_types <= {config.max_k_types}: {before} -> {len(md5_k_counts)} '
            f'(removed {len(to_drop)})')

    if config.max_o_types:
        before = len(md5_o_counts)
        to_drop = {md5 for md5, counts in md5_o_counts.items()
                   if len(counts) > config.max_o_types}
        for md5 in to_drop:
            _remove_md5(md5, md5_k_counts, md5_o_counts, md5_is_tsp)
        log(f'  max_o_types <= {config.max_o_types}: {before} -> {len(md5_o_counts)} '
            f'(removed {len(to_drop)})')


def _apply_single_label(md5_k_counts, md5_o_counts, log):
    """Majority-vote reduction: keep only most frequent K and O per MD5."""
    n_reduced_k = 0
    n_reduced_o = 0

    for md5 in list(md5_k_counts.keys()):
        if len(md5_k_counts[md5]) > 1:
            top_k = md5_k_counts[md5].most_common(1)[0][0]
            md5_k_counts[md5] = Counter({top_k: md5_k_counts[md5][top_k]})
            n_reduced_k += 1

        o_counts = md5_o_counts.get(md5, Counter())
        if len(o_counts) > 1:
            top_o = o_counts.most_common(1)[0][0]
            md5_o_counts[md5] = Counter({top_o: o_counts[top_o]})
            n_reduced_o += 1

    log(f'  Single-label reduction: {n_reduced_k} multi-K -> single-K, '
        f'{n_reduced_o} multi-O -> single-O (kept all {len(md5_k_counts)} MD5s)')


def _downsample(md5_k_counts, md5_o_counts, md5_is_tsp, config, log,
                cluster_map=None):
    """Cap per-class sample counts. K and O passes are independent.

    Each MD5 is assigned to its primary (most frequent) K-type and O-type.
    When a class exceeds the cap:
      - if cluster_map is given, round-robin across clusters to max diversity
      - otherwise, random uniform sample.
    Works for both single-label and multi-label strategies.
    """
    rng = np.random.default_rng(config.downsample_seed)
    keep = set(md5_k_counts.keys())
    before = len(keep)

    def sample(md5s, cap):
        if cluster_map is not None:
            return _cluster_stratified_sample(md5s, cap, cluster_map, rng)
        return rng.choice(md5s, size=cap, replace=False).tolist()

    if config.max_samples_per_k:
        # Assign each MD5 to its primary (most frequent) K-type
        k_buckets = defaultdict(list)
        for md5 in keep:
            primary_k = md5_k_counts[md5].most_common(1)[0][0]
            k_buckets[primary_k].append(md5)

        new_keep = set()
        n_capped = 0
        for primary_k, md5s in sorted(k_buckets.items()):
            if len(md5s) > config.max_samples_per_k:
                chosen = sample(md5s, config.max_samples_per_k)
                new_keep.update(chosen)
                log(f'    K downsample {primary_k}: {len(md5s)} -> {config.max_samples_per_k}')
                n_capped += 1
            else:
                new_keep.update(md5s)
        keep = new_keep
        mode = 'cluster-stratified' if cluster_map is not None else 'random'
        log(f'  Downsample K (max={config.max_samples_per_k}, {mode}): '
            f'{before} -> {len(keep)} ({n_capped} classes capped)')

    if config.max_samples_per_o:
        before_o = len(keep)
        # Assign each MD5 to its primary (most frequent) O-type
        o_buckets = defaultdict(list)
        for md5 in keep:
            if md5 not in md5_o_counts or not md5_o_counts[md5]:
                continue
            primary_o = md5_o_counts[md5].most_common(1)[0][0]
            o_buckets[primary_o].append(md5)

        new_keep = set()
        n_capped = 0
        for primary_o, md5s in sorted(o_buckets.items()):
            if len(md5s) > config.max_samples_per_o:
                chosen = sample(md5s, config.max_samples_per_o)
                new_keep.update(chosen)
                log(f'    O downsample {primary_o}: {len(md5s)} -> {config.max_samples_per_o}')
                n_capped += 1
            else:
                new_keep.update(md5s)
        keep = new_keep
        mode = 'cluster-stratified' if cluster_map is not None else 'random'
        log(f'  Downsample O (max={config.max_samples_per_o}, {mode}): '
            f'{before_o} -> {len(keep)} ({n_capped} classes capped)')

    # Remove MD5s not in keep
    removed = 0
    for md5 in list(md5_k_counts.keys()):
        if md5 not in keep:
            _remove_md5(md5, md5_k_counts, md5_o_counts, md5_is_tsp)
            removed += 1

    log(f'  After downsampling: {len(md5_k_counts)} MD5s (removed {removed})')


def _cluster_stratified_sample(md5s, cap, cluster_map, rng):
    """Round-robin sample up to `cap` MD5s, maximizing cluster diversity.

    md5s without a cluster assignment are placed in singleton pseudo-clusters
    (keyed by the md5 itself) so they still get sampled but don't collide
    with one another.
    """
    clusters = defaultdict(list)
    for md5 in md5s:
        cid = cluster_map.get(md5) or f'_solo_{md5}'
        clusters[cid].append(md5)

    cluster_ids = list(clusters.keys())
    rng.shuffle(cluster_ids)
    for cid in cluster_ids:
        rng.shuffle(clusters[cid])

    selected = []
    cursor = {cid: 0 for cid in cluster_ids}
    while len(selected) < cap:
        advanced = False
        for cid in cluster_ids:
            if len(selected) >= cap:
                break
            i = cursor[cid]
            if i < len(clusters[cid]):
                selected.append(clusters[cid][i])
                cursor[cid] = i + 1
                advanced = True
        if not advanced:
            break
    return selected


def _build_md5_cluster_map(cluster_path, threshold, rows, log):
    """Build {md5: cluster_id} at the given identity threshold.

    The cluster file has one row per protein_id with columns cl30_X, cl40_X,
    ..., cl95_X (no header). We join via protein_id -> md5 from `rows`.
    """
    target_prefix = f'cl{threshold}_'
    pid_to_md5 = {r['protein_id']: r.get('protein_md5')
                  for r in rows if r.get('protein_md5')}

    md5_to_cluster = {}
    missing_col = 0
    matched_pids = 0
    with open(cluster_path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if not parts:
                continue
            pid = parts[0]
            md5 = pid_to_md5.get(pid)
            if not md5:
                continue
            matched_pids += 1
            cid = next((p for p in parts[1:] if p.startswith(target_prefix)), None)
            if cid is None:
                missing_col += 1
                continue
            md5_to_cluster[md5] = cid

    log(f'  cluster map: {len(md5_to_cluster):,} MD5s assigned to '
        f'cl{threshold} clusters (matched {matched_pids:,} protein_ids)')
    if missing_col:
        log(f'  WARNING: {missing_col} rows lacked a cl{threshold}_ column')
    return md5_to_cluster


def _filter_k_strict(md5_k_counts, md5_o_counts, md5_is_tsp, config, log):
    """Optional stricter K-only filters (max_k_observations, drop_null_k)."""
    if config.max_k_observations:
        before = len(md5_k_counts)
        to_drop = {md5 for md5 in md5_k_counts
                   if len(md5_k_counts[md5]) > config.max_k_observations}
        for md5 in to_drop:
            _remove_md5(md5, md5_k_counts, md5_o_counts, md5_is_tsp)
        log(f'  max_k_observations <= {config.max_k_observations}: '
            f'{before} -> {len(md5_k_counts)} (removed {len(to_drop)})')

    if config.drop_null_k:
        before = len(md5_k_counts)
        to_drop = {md5 for md5, counts in md5_k_counts.items()
                   if any(k in NULL_LABELS for k in counts)}
        for md5 in to_drop:
            _remove_md5(md5, md5_k_counts, md5_o_counts, md5_is_tsp)
        log(f'  drop_null_k: {before} -> {len(md5_k_counts)} '
            f'(removed {len(to_drop)})')


def _build_label_vectors(md5_k_counts, md5_o_counts, strategy,
                         ref_k_classes=None, ref_o_classes=None,
                         min_count=1, min_fraction=0.0):
    """Build numpy label arrays from per-MD5 serotype counts.

    Args:
        md5_k_counts: {md5: Counter({k_type: count})}
        md5_o_counts: {md5: Counter({o_type: count})}
        strategy: 'multi_label', 'multi_label_threshold', or 'weighted_soft'
        ref_k_classes: optional fixed class list
        ref_o_classes: optional fixed class list
        min_count: for multi_label_threshold, minimum count per class
        min_fraction: for multi_label_threshold, minimum fraction per class

    Returns:
        md5_list, k_labels, o_labels, k_classes, o_classes
    """
    # Determine class lists
    if ref_k_classes is not None:
        k_classes = list(ref_k_classes)
    else:
        all_k = set()
        for counts in md5_k_counts.values():
            all_k.update(counts.keys())
        k_classes = sorted(all_k)

    if ref_o_classes is not None:
        o_classes = list(ref_o_classes)
    else:
        all_o = set()
        for counts in md5_o_counts.values():
            all_o.update(counts.keys())
        o_classes = sorted(all_o)

    k_to_idx = {k: i for i, k in enumerate(k_classes)}
    o_to_idx = {o: i for i, o in enumerate(o_classes)}

    md5_list = sorted(md5_k_counts.keys())
    n = len(md5_list)

    k_labels = np.zeros((n, len(k_classes)), dtype=np.float32)
    o_labels = np.zeros((n, len(o_classes)), dtype=np.float32)

    for i, md5 in enumerate(md5_list):
        k_counts = md5_k_counts[md5]
        o_counts = md5_o_counts.get(md5, Counter())

        if strategy == 'multi_label':
            for k_type in k_counts:
                if k_type in k_to_idx:
                    k_labels[i, k_to_idx[k_type]] = 1.0
            for o_type in o_counts:
                if o_type in o_to_idx:
                    o_labels[i, o_to_idx[o_type]] = 1.0

        elif strategy == 'multi_label_threshold':
            k_total = sum(k_counts.values())
            for k_type, count in k_counts.items():
                if k_type not in k_to_idx:
                    continue
                frac = count / k_total if k_total > 0 else 0.0
                if count >= min_count and frac >= min_fraction:
                    k_labels[i, k_to_idx[k_type]] = 1.0

            o_total = sum(o_counts.values())
            for o_type, count in o_counts.items():
                if o_type not in o_to_idx:
                    continue
                frac = count / o_total if o_total > 0 else 0.0
                if count >= min_count and frac >= min_fraction:
                    o_labels[i, o_to_idx[o_type]] = 1.0

        elif strategy == 'weighted_soft':
            k_total = sum(k_counts.values())
            for k_type, count in k_counts.items():
                if k_type in k_to_idx:
                    k_labels[i, k_to_idx[k_type]] = count / k_total

            o_total = sum(o_counts.values())
            if o_total > 0:
                for o_type, count in o_counts.items():
                    if o_type in o_to_idx:
                        o_labels[i, o_to_idx[o_type]] = count / o_total

        else:
            raise ValueError(f'Unknown label strategy: {strategy}')

    return md5_list, k_labels, o_labels, k_classes, o_classes
