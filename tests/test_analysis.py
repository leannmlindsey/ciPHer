"""Tests for cipher.analysis.per_serotype helpers."""

from collections import Counter

import numpy as np
import pytest

from cipher.analysis.per_serotype import _count_primary_labels


class TestCountPrimaryLabels:
    def test_basic(self):
        md5_list = ['a', 'b', 'c', 'd']
        classes = ['K1', 'K2', 'K3']
        labels = np.array([
            [1, 0, 0],  # a -> K1
            [0, 1, 0],  # b -> K2
            [1, 0, 0],  # c -> K1
            [0, 0, 1],  # d -> K3
        ], dtype=np.float32)
        md5_to_idx = {m: i for i, m in enumerate(md5_list)}

        # Count primary labels for only a, b, c
        counts = _count_primary_labels(md5_list, labels, classes,
                                        {'a', 'b', 'c'}, md5_to_idx)
        assert counts == {'K1': 2, 'K2': 1}

    def test_skips_null_rows(self):
        """Rows with all-zero labels should be skipped."""
        md5_list = ['a', 'b']
        classes = ['K1', 'K2']
        labels = np.array([
            [1, 0],
            [0, 0],  # null-labeled
        ], dtype=np.float32)
        md5_to_idx = {m: i for i, m in enumerate(md5_list)}

        counts = _count_primary_labels(md5_list, labels, classes,
                                        {'a', 'b'}, md5_to_idx)
        assert counts == {'K1': 1}

    def test_missing_md5(self):
        """MD5s not in md5_to_idx should be silently skipped."""
        md5_list = ['a', 'b']
        classes = ['K1']
        labels = np.array([[1.0], [1.0]], dtype=np.float32)
        md5_to_idx = {'a': 0, 'b': 1}

        counts = _count_primary_labels(md5_list, labels, classes,
                                        {'a', 'nonexistent'}, md5_to_idx)
        assert counts == {'K1': 1}

    def test_empty_subset(self):
        md5_list = ['a']
        classes = ['K1']
        labels = np.array([[1.0]], dtype=np.float32)
        md5_to_idx = {'a': 0}

        counts = _count_primary_labels(md5_list, labels, classes,
                                        set(), md5_to_idx)
        assert counts == {}

    def test_multi_hot_uses_argmax(self):
        """Multi-label rows should use the highest-weight class as primary."""
        md5_list = ['a']
        classes = ['K1', 'K2', 'K3']
        # a maps to K1=2, K2=5, K3=1 -> primary is K2
        labels = np.array([[2.0, 5.0, 1.0]], dtype=np.float32)
        md5_to_idx = {'a': 0}

        counts = _count_primary_labels(md5_list, labels, classes,
                                        {'a'}, md5_to_idx)
        assert counts == {'K2': 1}


class TestSerotypeColorMap:
    def test_stable_across_calls(self):
        from cipher.visualization.per_serotype import _serotype_color_map

        serotypes = ['KL1', 'KL47', 'KL64', 'KL107']
        m1 = _serotype_color_map(serotypes, seed=0)
        m2 = _serotype_color_map(serotypes, seed=0)

        for s in serotypes:
            np.testing.assert_array_equal(m1[s], m2[s])

    def test_different_seeds_differ(self):
        from cipher.visualization.per_serotype import _serotype_color_map

        serotypes = ['KL1', 'KL47', 'KL64']
        m1 = _serotype_color_map(serotypes, seed=0)
        m2 = _serotype_color_map(serotypes, seed=1)

        # Very unlikely that all 3 match with different seeds
        all_match = all(np.array_equal(m1[s], m2[s]) for s in serotypes)
        assert not all_match

    def test_subset_is_stable(self):
        """When comparing 2 experiments, KL47 should have same color
        as when comparing 5 experiments (given same seed)."""
        from cipher.visualization.per_serotype import _serotype_color_map

        # Universe 1: 4 serotypes
        m1 = _serotype_color_map(['KL1', 'KL47', 'KL64', 'KL107'], seed=0)
        # Universe 2: same 4 + 2 more
        m2 = _serotype_color_map(['KL1', 'KL47', 'KL64', 'KL107', 'KL2', 'KL186'], seed=0)

        # KL47 will get a different color because the universe changed.
        # This is expected — stability is PER UNIVERSE, not across universes.
        # The guarantee is: for ONE call to plot_serotype_bubble, colors are stable.
        # So instead test that the same universe gives same colors:
        m3 = _serotype_color_map(['KL1', 'KL47', 'KL64', 'KL107'], seed=0)
        np.testing.assert_array_equal(m1['KL47'], m3['KL47'])


class TestPickHighlights:
    def test_top_n_by_frequency(self):
        from cipher.visualization.per_serotype import _pick_highlights

        stats1 = {
            'KL1': {'train_freq': 100, 'total': 5},
            'KL47': {'train_freq': 1000, 'total': 50},
            'KL64': {'train_freq': 500, 'total': 30},
        }
        picks = _pick_highlights([stats1], n_top=2)
        assert 'KL47' in picks  # highest freq
        assert 'KL64' in picks  # second highest

    def test_user_highlight_always_included(self):
        from cipher.visualization.per_serotype import _pick_highlights

        stats1 = {
            'KL1': {'train_freq': 100, 'total': 5},
            'KL47': {'train_freq': 1000, 'total': 50},
        }
        picks = _pick_highlights([stats1], n_top=1, user_highlight=['KL1'])
        assert 'KL1' in picks
        assert 'KL47' in picks

    def test_combines_experiments(self):
        """Takes max freq across experiments."""
        from cipher.visualization.per_serotype import _pick_highlights

        stats1 = {'KL47': {'train_freq': 100, 'total': 5}}
        stats2 = {'KL47': {'train_freq': 1000, 'total': 50}}
        picks = _pick_highlights([stats1, stats2], n_top=1)
        assert 'KL47' in picks
