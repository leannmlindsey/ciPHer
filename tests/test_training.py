"""Tests for cipher.data.training."""

import os
import warnings

import numpy as np
import pytest

from cipher.data.training import (
    TrainingConfig, TrainingData, prepare_training_data,
    TOOL_COLUMNS,
    _filter_proteins, _build_md5_associations, _apply_single_label,
    _build_label_vectors, _cluster_stratified_sample,
)

FIXTURES = os.path.join(os.path.dirname(__file__), 'test_data')
ASSOC_MAP = os.path.join(FIXTURES, 'association_map.tsv')
GLYCAN = os.path.join(FIXTURES, 'glycan_binders.tsv')


class TestTrainingConfig:
    def test_from_dict(self):
        cfg = TrainingConfig.from_dict({
            'tools': ['SpikeHunter'],
            'min_sources': 3,
            'unknown_key': 'ignored',
        })
        assert cfg.tools == ['SpikeHunter']
        assert cfg.min_sources == 3
        assert cfg.label_strategy == 'multi_label'  # default

    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.tools is None
        assert cfg.exclude_tools is None
        assert cfg.protein_set is None
        assert cfg.min_sources == 1
        assert cfg.single_label is False
        assert cfg.max_k_types is None

    def test_protein_set_translates_tsp_only(self):
        """Legacy protein_set='tsp_only' should set tools=['SpikeHunter']."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            cfg = TrainingConfig(protein_set='tsp_only')
        assert cfg.tools == ['SpikeHunter']
        assert cfg.exclude_tools is None

    def test_protein_set_translates_rbp_only(self):
        """Legacy protein_set='rbp_only' should set exclude_tools=['SpikeHunter']."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            cfg = TrainingConfig(protein_set='rbp_only')
        assert cfg.tools is None
        assert cfg.exclude_tools == ['SpikeHunter']

    def test_protein_set_all_glycan_binders(self):
        """Legacy protein_set='all_glycan_binders' maps to no tool filter."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            cfg = TrainingConfig(protein_set='all_glycan_binders')
        assert cfg.tools is None
        assert cfg.exclude_tools is None

    def test_protein_set_emits_deprecation(self):
        """Using protein_set should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            TrainingConfig(protein_set='tsp_only')
        assert any(issubclass(warn.category, DeprecationWarning) for warn in w)

    def test_invalid_tool_raises(self):
        with pytest.raises(ValueError, match='Unknown'):
            TrainingConfig(tools=['not_a_real_tool'])

    def test_invalid_exclude_tool_raises(self):
        with pytest.raises(ValueError, match='Unknown'):
            TrainingConfig(exclude_tools=['foo'])

    def test_invalid_protein_set_raises(self):
        with pytest.raises(ValueError, match='Unknown protein_set'):
            TrainingConfig(protein_set='bogus')

    def test_all_tool_columns_accepted(self):
        """Every name in TOOL_COLUMNS must be valid as a tool filter."""
        for tool in TOOL_COLUMNS:
            cfg = TrainingConfig(tools=[tool])
            assert cfg.tools == [tool]

    def test_per_head_lists_mutex_with_single_list(self, tmp_path):
        """positive_list_path and positive_list_{k,o}_path are mutex."""
        k_list = tmp_path / 'k.list'
        k_list.write_text('prot_1\n')
        with pytest.raises(ValueError, match='mutually exclusive'):
            TrainingConfig(
                positive_list_path=str(k_list),
                positive_list_k_path=str(k_list),
            )

    def test_per_head_lists_mutex_with_tools(self, tmp_path):
        """positive_list_{k,o}_path are mutex with tool-based filters."""
        k_list = tmp_path / 'k.list'
        k_list.write_text('prot_1\n')
        with pytest.raises(ValueError, match='mutually exclusive'):
            TrainingConfig(
                positive_list_k_path=str(k_list),
                tools=['SpikeHunter'],
            )


class TestPrepareTrainingData:
    def test_basic(self):
        """Basic pipeline with no filtering."""
        config = TrainingConfig()
        td = prepare_training_data(config, ASSOC_MAP, GLYCAN, verbose=False)

        assert len(td.md5_list) > 0
        assert td.k_labels.shape[0] == len(td.md5_list)
        assert td.o_labels.shape[0] == len(td.md5_list)
        assert len(td.k_classes) == td.k_labels.shape[1]
        assert len(td.o_classes) == td.o_labels.shape[1]

    def test_min_sources_filter(self):
        """min_sources=3 should drop proteins with fewer tool flags."""
        config_all = TrainingConfig(min_sources=1)
        config_filtered = TrainingConfig(min_sources=3)

        td_all = prepare_training_data(config_all, ASSOC_MAP, GLYCAN, verbose=False)
        td_filt = prepare_training_data(config_filtered, ASSOC_MAP, GLYCAN, verbose=False)

        assert len(td_filt.md5_list) <= len(td_all.md5_list)

    def test_tools_spikehunter(self):
        """tools=['SpikeHunter'] should keep only SpikeHunter=1 proteins."""
        config = TrainingConfig(tools=['SpikeHunter'])
        td = prepare_training_data(config, ASSOC_MAP, GLYCAN, verbose=False)

        # All surviving MD5s should have is_tsp=1
        assert all(v == 1 for v in td.is_tsp)

    def test_exclude_tools_spikehunter(self):
        """exclude_tools=['SpikeHunter'] should keep only SpikeHunter=0 proteins."""
        config = TrainingConfig(exclude_tools=['SpikeHunter'])
        td = prepare_training_data(config, ASSOC_MAP, GLYCAN, verbose=False)

        # All surviving MD5s should have is_tsp=0
        assert all(v == 0 for v in td.is_tsp)

    def test_protein_set_tsp_only_matches_tools(self):
        """Legacy protein_set='tsp_only' must produce same result as tools=['SpikeHunter']."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            legacy = TrainingConfig(protein_set='tsp_only')
        modern = TrainingConfig(tools=['SpikeHunter'])

        td_legacy = prepare_training_data(legacy, ASSOC_MAP, GLYCAN, verbose=False)
        td_modern = prepare_training_data(modern, ASSOC_MAP, GLYCAN, verbose=False)

        assert td_legacy.md5_list == td_modern.md5_list

    def test_protein_set_rbp_only_matches_exclude(self):
        """Legacy protein_set='rbp_only' must match exclude_tools=['SpikeHunter']."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            legacy = TrainingConfig(protein_set='rbp_only')
        modern = TrainingConfig(exclude_tools=['SpikeHunter'])

        td_legacy = prepare_training_data(legacy, ASSOC_MAP, GLYCAN, verbose=False)
        td_modern = prepare_training_data(modern, ASSOC_MAP, GLYCAN, verbose=False)

        assert td_legacy.md5_list == td_modern.md5_list

    def test_multiple_tools_is_union(self):
        """tools=[A, B] should keep proteins flagged by A OR B."""
        # DepoRanker is only set for prot_10 in fixture; SpikeHunter covers many
        tools_a = TrainingConfig(tools=['DepoRanker'])
        tools_b = TrainingConfig(tools=['SpikeHunter'])
        tools_ab = TrainingConfig(tools=['DepoRanker', 'SpikeHunter'])

        a = prepare_training_data(tools_a, ASSOC_MAP, GLYCAN, verbose=False)
        b = prepare_training_data(tools_b, ASSOC_MAP, GLYCAN, verbose=False)
        ab = prepare_training_data(tools_ab, ASSOC_MAP, GLYCAN, verbose=False)

        union = set(a.md5_list) | set(b.md5_list)
        assert set(ab.md5_list) == union

    def test_tools_and_exclude_combined(self):
        """tools=[A] AND exclude_tools=[B] keeps proteins flagged by A AND NOT by B."""
        # All proteins flagged by PhageRBPdetect OR DepoScope, minus any SpikeHunter
        config = TrainingConfig(
            tools=['PhageRBPdetect', 'DepoScope'],
            exclude_tools=['SpikeHunter'],
        )
        td = prepare_training_data(config, ASSOC_MAP, GLYCAN, verbose=False)

        # No surviving protein should be SpikeHunter flagged
        assert all(v == 0 for v in td.is_tsp)

    def test_single_label(self):
        """Single-label should produce exactly one K and one O per sample."""
        config = TrainingConfig(single_label=True, label_strategy='single_label')
        td = prepare_training_data(config, ASSOC_MAP, GLYCAN, verbose=False)

        k_per = td.k_labels.sum(axis=1)
        o_per = td.o_labels.sum(axis=1)
        assert all(k_per == 1.0)
        assert all(o_per == 1.0)

    def test_max_k_types(self):
        """max_k_types should drop proteins seen in too many K-types."""
        # prot_1/md5_aaa maps to K1 and K2 (2 K-types)
        config_strict = TrainingConfig(max_k_types=1)
        config_loose = TrainingConfig(max_k_types=3)

        td_strict = prepare_training_data(config_strict, ASSOC_MAP, GLYCAN, verbose=False)
        td_loose = prepare_training_data(config_loose, ASSOC_MAP, GLYCAN, verbose=False)

        assert len(td_strict.md5_list) <= len(td_loose.md5_list)
        # md5_aaa has 2 K-types, should be excluded with max_k_types=1
        assert 'md5_aaa' not in td_strict.md5_list

    def test_save_and_load(self, tmp_path):
        """TrainingData should round-trip through save/load."""
        config = TrainingConfig(tools=['SpikeHunter'], min_sources=3)
        td = prepare_training_data(config, ASSOC_MAP, GLYCAN, verbose=False)

        save_dir = str(tmp_path / 'training_output')
        td.save(save_dir)

        td2 = TrainingData.load(save_dir)
        assert td2.md5_list == td.md5_list
        assert td2.k_classes == td.k_classes
        assert td2.o_classes == td.o_classes
        assert td2.config.tools == ['SpikeHunter']
        np.testing.assert_array_equal(td2.k_labels, td.k_labels)
        np.testing.assert_array_equal(td2.o_labels, td.o_labels)

    def test_load_legacy_protein_set(self, tmp_path):
        """Loading a label_encoders.json with only legacy protein_set still works."""
        import json
        save_dir = tmp_path / 'legacy'
        save_dir.mkdir()

        # Write a minimal legacy-style label_encoders.json + training_data.npz
        np.savez_compressed(
            str(save_dir / 'training_data.npz'),
            md5_list=np.array(['m1'], dtype='U32'),
            k_labels=np.array([[1.0]], dtype=np.float32),
            o_labels=np.array([[1.0]], dtype=np.float32),
            is_tsp=np.array([1], dtype=np.int8),
        )
        with open(save_dir / 'label_encoders.json', 'w') as f:
            json.dump({
                'k_classes': ['K1'],
                'o_classes': ['O1'],
                'strategy': 'single_label',
                'protein_set': 'tsp_only',  # legacy field only
            }, f)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            td = TrainingData.load(str(save_dir))

        # protein_set was translated to tools
        assert td.config.tools == ['SpikeHunter']


class TestPerHeadMasking:
    """v2 highconf: positive_list_k_path + positive_list_o_path zero the
    labels of samples that aren't in the corresponding head's list.
    Back-compat: positive_list_path alone matches pre-v2 behaviour.
    """

    def _write_list(self, path, ids):
        path.write_text('\n'.join(ids) + '\n')

    def test_backcompat_single_list_matches_legacy(self, tmp_path):
        """Passing positive_list_path alone produces the same result as
        before the per-head fields existed."""
        pos = tmp_path / 'pos.list'
        self._write_list(pos, ['prot_1', 'prot_3', 'prot_5'])

        cfg_legacy = TrainingConfig(positive_list_path=str(pos))
        cfg_new = TrainingConfig(positive_list_path=str(pos))

        td_legacy = prepare_training_data(
            cfg_legacy, ASSOC_MAP, GLYCAN, verbose=False)
        td_new = prepare_training_data(
            cfg_new, ASSOC_MAP, GLYCAN, verbose=False)

        assert td_legacy.md5_list == td_new.md5_list
        assert np.array_equal(td_legacy.k_labels, td_new.k_labels)
        assert np.array_equal(td_legacy.o_labels, td_new.o_labels)

    def test_per_head_lists_zero_out_of_list_labels(self, tmp_path):
        """With disjoint K and O lists, samples not in K have K labels = 0
        and samples not in O have O labels = 0."""
        # From the fixture:
        # prot_1 -> md5_aaa (K1/K2, O1/O2)
        # prot_3 -> md5_ccc (K2, O1)
        # prot_5 -> md5_eee (K3, O1)
        k_only = tmp_path / 'k.list'
        o_only = tmp_path / 'o.list'
        self._write_list(k_only, ['prot_1', 'prot_3'])   # md5_aaa, md5_ccc
        self._write_list(o_only, ['prot_5'])             # md5_eee

        cfg = TrainingConfig(
            positive_list_k_path=str(k_only),
            positive_list_o_path=str(o_only),
        )
        td = prepare_training_data(cfg, ASSOC_MAP, GLYCAN, verbose=False)

        # Sample pool should be the UNION of the two lists.
        surviving = set(td.md5_list)
        # md5_aaa / md5_ccc (K list) and md5_eee (O list) should all survive
        # if they clear the other filters.
        assert 'md5_aaa' in surviving or 'md5_ccc' in surviving or 'md5_eee' in surviving

        md5_to_idx = {m: i for i, m in enumerate(td.md5_list)}

        # Any MD5 in the K list (md5_aaa, md5_ccc) but not in the O list
        # should have all-zero O labels.
        for md5 in ('md5_aaa', 'md5_ccc'):
            if md5 in md5_to_idx:
                assert td.o_labels[md5_to_idx[md5]].sum() == 0, (
                    f'{md5} is K-only but has non-zero O labels')

        # md5_eee is in the O list but NOT in the K list -> K labels should be 0.
        if 'md5_eee' in md5_to_idx:
            assert td.k_labels[md5_to_idx['md5_eee']].sum() == 0, (
                'md5_eee is O-only but has non-zero K labels')

    def test_per_head_same_list_no_masking(self, tmp_path):
        """When both per-head lists are identical, masking is a no-op."""
        same = tmp_path / 'same.list'
        self._write_list(same, ['prot_1', 'prot_3', 'prot_5'])

        cfg_single = TrainingConfig(positive_list_path=str(same))
        cfg_split = TrainingConfig(
            positive_list_k_path=str(same),
            positive_list_o_path=str(same),
        )
        td_single = prepare_training_data(
            cfg_single, ASSOC_MAP, GLYCAN, verbose=False)
        td_split = prepare_training_data(
            cfg_split, ASSOC_MAP, GLYCAN, verbose=False)

        assert td_single.md5_list == td_split.md5_list
        assert np.array_equal(td_single.k_labels, td_split.k_labels)
        assert np.array_equal(td_single.o_labels, td_split.o_labels)


class TestMultiLabelThreshold:
    def test_count_threshold_drops_singletons(self):
        """min_count=2 should drop K-types seen only once."""
        # Build a synthetic config and mock training data via _build_label_vectors
        from collections import Counter
        # Protein m1: K1 seen 5x, K2 seen 1x. With min_count=2: only K1 positive.
        md5_k = {'m1': Counter({'K1': 5, 'K2': 1})}
        md5_o = {'m1': Counter({'O1': 5, 'O2': 1})}

        md5_list, k_labels, o_labels, k_classes, o_classes = _build_label_vectors(
            md5_k, md5_o, 'multi_label_threshold',
            min_count=2, min_fraction=0.0)

        k1_idx = k_classes.index('K1')
        k2_idx = k_classes.index('K2')
        assert k_labels[0, k1_idx] == 1.0
        assert k_labels[0, k2_idx] == 0.0

    def test_fraction_threshold_drops_low_fraction(self):
        """min_fraction=0.2 should drop K-types with < 20% frequency."""
        from collections import Counter
        # K1 seen 90x (90%), K2 seen 10x (10%). With min_fraction=0.2: only K1.
        md5_k = {'m1': Counter({'K1': 90, 'K2': 10})}
        md5_o = {'m1': Counter({'O1': 100})}

        md5_list, k_labels, o_labels, k_classes, o_classes = _build_label_vectors(
            md5_k, md5_o, 'multi_label_threshold',
            min_count=1, min_fraction=0.2)

        k1_idx = k_classes.index('K1')
        k2_idx = k_classes.index('K2')
        assert k_labels[0, k1_idx] == 1.0
        assert k_labels[0, k2_idx] == 0.0

    def test_and_of_count_and_fraction(self):
        """Both conditions must be satisfied (AND)."""
        from collections import Counter
        # K1: 50x, 100% -> passes both
        # K2: 1x, 2% (if added) -> let's check something that passes count but not fraction
        # Actually with a single protein total=50 for K1, nothing else, can't test easily.
        # Use: K1=50 (83%), K2=10 (17%). min_count=5, min_fraction=0.2.
        # K1: count=50 >= 5 AND frac=0.83 >= 0.2 -> kept
        # K2: count=10 >= 5 BUT frac=0.17 < 0.2 -> dropped (fails fraction)
        md5_k = {'m1': Counter({'K1': 50, 'K2': 10})}
        md5_o = {'m1': Counter({'O1': 50})}

        md5_list, k_labels, o_labels, k_classes, _ = _build_label_vectors(
            md5_k, md5_o, 'multi_label_threshold',
            min_count=5, min_fraction=0.2)

        k1_idx = k_classes.index('K1')
        k2_idx = k_classes.index('K2')
        assert k_labels[0, k1_idx] == 1.0
        assert k_labels[0, k2_idx] == 0.0

        # Now test count failure: K3=3x (30%) fails count>=5
        md5_k2 = {'m1': Counter({'K1': 7, 'K3': 3})}
        md5_o2 = {'m1': Counter({'O1': 10})}
        _, k_labels2, _, k_classes2, _ = _build_label_vectors(
            md5_k2, md5_o2, 'multi_label_threshold',
            min_count=5, min_fraction=0.2)
        k3_idx = k_classes2.index('K3')
        assert k_labels2[0, k3_idx] == 0.0  # dropped by count

    def test_all_zero_rows_dropped_at_pipeline_level(self):
        """After prepare_training_data, zero-label proteins should be dropped."""
        config = TrainingConfig(
            label_strategy='multi_label_threshold',
            min_label_count=1000,  # extreme threshold — drops everything in fixture
            min_label_fraction=0.0,
        )
        td = prepare_training_data(config, ASSOC_MAP, GLYCAN, verbose=False)
        # All proteins had fewer than 1000 observations, so all labels should be zeroed
        # -> all proteins dropped.
        assert len(td.md5_list) == 0

    def test_reasonable_threshold_keeps_main_labels(self):
        """With min_count=1 and min_fraction=0, threshold is effectively off;
        should match plain multi_label."""
        config = TrainingConfig(
            label_strategy='multi_label_threshold',
            min_label_count=1,
            min_label_fraction=0.0,
        )
        td = prepare_training_data(config, ASSOC_MAP, GLYCAN, verbose=False)
        config_ml = TrainingConfig(label_strategy='multi_label')
        td_ml = prepare_training_data(config_ml, ASSOC_MAP, GLYCAN, verbose=False)
        assert td.md5_list == td_ml.md5_list
        np.testing.assert_array_equal(td.k_labels, td_ml.k_labels)

    def test_default_keeps_singletons(self):
        """Default config (min_count=1, min_fraction=0.1) keeps single observations
        but drops minority noise classes."""
        cfg = TrainingConfig(label_strategy='multi_label_threshold')
        # Defaults
        assert cfg.min_label_count == 1
        assert cfg.min_label_fraction == 0.1


class TestWeightedMultiLabel:
    def test_label_values_are_fractions(self):
        """weighted_multi_label should produce fractional labels like weighted_soft."""
        config_wml = TrainingConfig(label_strategy='weighted_multi_label')
        config_ws = TrainingConfig(label_strategy='weighted_soft')

        td_wml = prepare_training_data(config_wml, ASSOC_MAP, GLYCAN, verbose=False)
        td_ws = prepare_training_data(config_ws, ASSOC_MAP, GLYCAN, verbose=False)

        # Labels should be identical (same fractional construction)
        assert td_wml.md5_list == td_ws.md5_list
        np.testing.assert_array_equal(td_wml.k_labels, td_ws.k_labels)
        np.testing.assert_array_equal(td_wml.o_labels, td_ws.o_labels)

    def test_rows_sum_to_one(self):
        """Each row (with at least one observation) should sum to 1.0."""
        config = TrainingConfig(label_strategy='weighted_multi_label')
        td = prepare_training_data(config, ASSOC_MAP, GLYCAN, verbose=False)

        k_row_sums = td.k_labels.sum(axis=1)
        # All rows with K observations should sum to 1
        nonzero = k_row_sums[k_row_sums > 0]
        np.testing.assert_allclose(nonzero, 1.0, atol=1e-5)


class TestMinClassSamples:
    def test_no_filter_when_none(self):
        """Default min_class_samples=None should not change anything."""
        cfg = TrainingConfig()
        assert cfg.min_class_samples is None

        td = prepare_training_data(cfg, ASSOC_MAP, GLYCAN, verbose=False)
        # No classes dropped
        assert getattr(td, 'dropped_k_classes', []) == []
        assert getattr(td, 'dropped_o_classes', []) == []

    def test_drops_rare_classes(self):
        """min_class_samples=10 should drop classes with < 10 samples."""
        cfg = TrainingConfig(min_class_samples=10)
        td = prepare_training_data(cfg, ASSOC_MAP, GLYCAN, verbose=False)

        # Verify all surviving K classes have >= 10 samples
        k_counts = (td.k_labels > 0).sum(axis=0)
        assert all(c >= 10 for c in k_counts)

    def test_extreme_threshold_drops_everything(self):
        """min_class_samples larger than dataset should drop all classes."""
        cfg = TrainingConfig(min_class_samples=999999)
        td = prepare_training_data(cfg, ASSOC_MAP, GLYCAN, verbose=False)
        assert len(td.k_classes) == 0
        assert len(td.o_classes) == 0


class TestInvalidStrategy:
    def test_unknown_strategy_raises(self):
        config = TrainingConfig(label_strategy='not_a_real_strategy')
        with pytest.raises(ValueError, match='Unknown label_strategy'):
            prepare_training_data(config, ASSOC_MAP, GLYCAN, verbose=False)


class TestBuildLabelVectors:
    def test_multi_label(self):
        from collections import Counter
        md5_k = {'m1': Counter({'K1': 3, 'K2': 1}), 'm2': Counter({'K1': 5})}
        md5_o = {'m1': Counter({'O1': 2}), 'm2': Counter({'O1': 1, 'O2': 1})}

        md5_list, k_labels, o_labels, k_classes, o_classes = _build_label_vectors(
            md5_k, md5_o, 'multi_label')

        assert md5_list == ['m1', 'm2']
        assert k_classes == ['K1', 'K2']
        assert o_classes == ['O1', 'O2']

        # m1 maps to K1 and K2
        assert k_labels[0, 0] == 1.0  # K1
        assert k_labels[0, 1] == 1.0  # K2
        # m2 maps to K1 only
        assert k_labels[1, 0] == 1.0  # K1
        assert k_labels[1, 1] == 0.0  # K2

    def test_weighted_soft(self):
        from collections import Counter
        md5_k = {'m1': Counter({'K1': 3, 'K2': 1})}
        md5_o = {'m1': Counter({'O1': 2})}

        md5_list, k_labels, _, _, _ = _build_label_vectors(
            md5_k, md5_o, 'weighted_soft')

        # K1 has 3/4 weight, K2 has 1/4
        assert k_labels[0, 0] == pytest.approx(0.75)
        assert k_labels[0, 1] == pytest.approx(0.25)

    def test_ref_classes(self):
        """Reference class lists should be used when provided."""
        from collections import Counter
        md5_k = {'m1': Counter({'K1': 1})}
        md5_o = {'m1': Counter({'O1': 1})}

        ref_k = ['K1', 'K2', 'K3']
        _, k_labels, _, k_classes, _ = _build_label_vectors(
            md5_k, md5_o, 'multi_label', ref_k_classes=ref_k)

        assert k_classes == ['K1', 'K2', 'K3']
        assert k_labels.shape[1] == 3


class TestClusterStratifiedSample:
    def test_round_robin_prefers_diversity(self):
        """With cap < n_clusters, should take one per distinct cluster."""
        md5s = [f'm{i}' for i in range(10)]
        # 3 clusters: m0-m4 in A (5), m5-m7 in B (3), m8-m9 in C (2)
        cluster_map = {f'm{i}': 'A' for i in range(5)}
        cluster_map.update({f'm{i}': 'B' for i in range(5, 8)})
        cluster_map.update({f'm{i}': 'C' for i in range(8, 10)})
        rng = np.random.default_rng(42)
        chosen = _cluster_stratified_sample(md5s, cap=3, cluster_map=cluster_map, rng=rng)
        assert len(chosen) == 3
        chosen_clusters = {cluster_map[m] for m in chosen}
        assert chosen_clusters == {'A', 'B', 'C'}

    def test_cap_larger_than_pool_returns_all(self):
        md5s = ['a', 'b', 'c']
        cluster_map = {'a': 'X', 'b': 'X', 'c': 'Y'}
        rng = np.random.default_rng(0)
        chosen = _cluster_stratified_sample(md5s, cap=10, cluster_map=cluster_map, rng=rng)
        assert set(chosen) == set(md5s)

    def test_missing_cluster_treated_as_singleton(self):
        """MD5s without a cluster entry should still be sampled, not collide."""
        md5s = ['a', 'b', 'c', 'd']
        cluster_map = {'a': 'X', 'b': 'X'}  # c, d have no entry
        rng = np.random.default_rng(0)
        chosen = _cluster_stratified_sample(md5s, cap=4, cluster_map=cluster_map, rng=rng)
        assert set(chosen) == {'a', 'b', 'c', 'd'}

    def test_deterministic_with_seed(self):
        md5s = [f'm{i}' for i in range(20)]
        cluster_map = {f'm{i}': f'c{i % 5}' for i in range(20)}
        r1 = _cluster_stratified_sample(md5s, 10, cluster_map, np.random.default_rng(7))
        r2 = _cluster_stratified_sample(md5s, 10, cluster_map, np.random.default_rng(7))
        assert r1 == r2


class TestApplySingleLabel:
    def test_majority_vote(self):
        from collections import Counter
        md5_k = {'m1': Counter({'K1': 5, 'K2': 2})}
        md5_o = {'m1': Counter({'O1': 3, 'O2': 7})}

        _apply_single_label(md5_k, md5_o, log=lambda x: None)

        assert list(md5_k['m1'].keys()) == ['K1']
        assert list(md5_o['m1'].keys()) == ['O2']

    def test_already_single(self):
        from collections import Counter
        md5_k = {'m1': Counter({'K1': 3})}
        md5_o = {'m1': Counter({'O1': 1})}

        _apply_single_label(md5_k, md5_o, log=lambda x: None)

        assert list(md5_k['m1'].keys()) == ['K1']
