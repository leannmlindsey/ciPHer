"""Tests for models.contrastive_encoder.sampler.PKClusterSampler."""

import os
import sys
from collections import Counter

import pytest

# Add models/ to path so we can import contrastive_encoder.
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from contrastive_encoder.sampler import (
    PKClusterSampler, batch_class_distribution,
)


def _make_dataset(classes_and_sizes):
    """Build class_labels + cluster_ids for `{class: (n_samples, n_clusters)}`.

    Within each class, samples are assigned round-robin to n_clusters clusters.
    Returns (class_labels, cluster_ids) as parallel lists.
    """
    class_labels, cluster_ids = [], []
    for c, (n, ncl) in classes_and_sizes.items():
        for i in range(n):
            class_labels.append(c)
            cluster_ids.append(f'{c}_c{i % ncl}')
    return class_labels, cluster_ids


class TestBasicBatchShape:
    def test_batch_size_and_class_count(self):
        """Every batch has exactly P*K samples from exactly P distinct classes."""
        labels, clusters = _make_dataset({
            'K1': (100, 10),
            'K2': (100, 10),
            'K3': (100, 10),
            'K4': (100, 10),
        })
        sampler = PKClusterSampler(labels, clusters, P=3, K=4,
                                   num_batches_per_epoch=5, seed=1)
        for batch in sampler:
            assert len(batch) == 12
            counts = batch_class_distribution(batch, labels)
            assert len(counts) == 3, f'expected 3 classes, got {counts}'
            assert all(v == 4 for v in counts.values()), counts

    def test_all_indices_in_range(self):
        labels, clusters = _make_dataset({'A': (50, 5), 'B': (50, 5)})
        sampler = PKClusterSampler(labels, clusters, P=2, K=3,
                                   num_batches_per_epoch=3, seed=0)
        for batch in sampler:
            for idx in batch:
                assert 0 <= idx < len(labels)


class TestClassBalance:
    def test_rare_class_represented_equally(self):
        """With PK sampling, rare class should appear as often as common ones."""
        # K1 has 1000 samples, K2 has 20. Without balancing, K1 dominates.
        labels, clusters = _make_dataset({
            'K1': (1000, 50),
            'K2': (20, 10),
            'K3': (50, 10),
            'K4': (50, 10),
            'K5': (50, 10),
        })
        sampler = PKClusterSampler(labels, clusters, P=3, K=4,
                                   num_batches_per_epoch=100, seed=42)
        # Each batch picks 3 classes uniformly. Over 100 batches we expect
        # each of the 5 usable classes in ~60 batches.
        class_batch_count = Counter()
        for batch in sampler:
            for c in set(labels[i] for i in batch):
                class_batch_count[c] += 1
        expected = 100 * 3 / 5  # 60
        # Tolerance: +/- 20% of expected
        for c, cnt in class_batch_count.items():
            assert abs(cnt - expected) < 0.2 * expected, \
                f'class {c} appeared in {cnt} batches, expected ~{expected}'


class TestClusterStratification:
    def test_K_samples_span_K_clusters_when_possible(self):
        """With K clusters available, K samples should each come from a distinct cluster."""
        labels, clusters = _make_dataset({
            'K1': (40, 10),  # 10 clusters, 4 samples each
            'K2': (40, 10),
            'K3': (40, 10),
        })
        sampler = PKClusterSampler(labels, clusters, P=2, K=8,
                                   num_batches_per_epoch=10, seed=7)
        for batch in sampler:
            # Group batch by class
            by_class = {}
            for idx in batch:
                by_class.setdefault(labels[idx], []).append(clusters[idx])
            for c, cl_list in by_class.items():
                # K=8 and n_clusters=10 -> all 8 should be distinct clusters
                assert len(set(cl_list)) == 8, \
                    f'class {c}: {cl_list} has duplicate clusters'

    def test_class_with_few_clusters_cycles(self):
        """Class with only 2 clusters and K=4 should pick 2 per cluster."""
        labels, clusters = _make_dataset({
            'K1': (20, 2),   # 2 clusters, 10 samples each
            'K2': (20, 2),
            'K3': (20, 2),
        })
        sampler = PKClusterSampler(labels, clusters, P=2, K=4,
                                   num_batches_per_epoch=5, seed=0)
        for batch in sampler:
            by_class = {}
            for idx in batch:
                by_class.setdefault(labels[idx], []).append(clusters[idx])
            for c, cl_list in by_class.items():
                cluster_counts = Counter(cl_list)
                # Round-robin: should be as balanced as possible across 2 clusters.
                assert max(cluster_counts.values()) - min(cluster_counts.values()) <= 1


class TestUsableClassFilter:
    def test_classes_with_fewer_than_K_samples_excluded(self):
        labels, clusters = _make_dataset({
            'A': (100, 10),
            'B': (3, 3),      # < K=4, should be excluded
            'C': (100, 10),
            'D': (100, 10),
        })
        sampler = PKClusterSampler(labels, clusters, P=3, K=4,
                                   num_batches_per_epoch=20, seed=0)
        for batch in sampler:
            seen_classes = set(labels[i] for i in batch)
            assert 'B' not in seen_classes

    def test_strict_mode_requires_K_distinct_clusters(self):
        labels, clusters = _make_dataset({
            'A': (100, 10),
            'B': (100, 2),    # plenty of samples but only 2 clusters
            'C': (100, 10),
            'D': (100, 10),
        })
        # K=4; strict=True means B is excluded because only 2 clusters
        sampler = PKClusterSampler(labels, clusters, P=3, K=4,
                                   num_batches_per_epoch=20, seed=0,
                                   strict=True)
        for batch in sampler:
            assert 'B' not in set(labels[i] for i in batch)

    def test_raises_when_not_enough_usable_classes(self):
        labels, clusters = _make_dataset({'A': (100, 10), 'B': (100, 10)})
        with pytest.raises(ValueError, match='usable classes'):
            PKClusterSampler(labels, clusters, P=5, K=4, seed=0)


class TestDeterminism:
    def test_same_seed_same_epoch_same_batches(self):
        labels, clusters = _make_dataset({
            'A': (50, 5), 'B': (50, 5), 'C': (50, 5), 'D': (50, 5),
        })
        s1 = PKClusterSampler(labels, clusters, P=2, K=3,
                              num_batches_per_epoch=5, seed=42)
        s2 = PKClusterSampler(labels, clusters, P=2, K=3,
                              num_batches_per_epoch=5, seed=42)
        s1.set_epoch(7)
        s2.set_epoch(7)
        assert list(s1) == list(s2)

    def test_different_epoch_different_batches(self):
        labels, clusters = _make_dataset({
            'A': (50, 5), 'B': (50, 5), 'C': (50, 5), 'D': (50, 5),
        })
        s = PKClusterSampler(labels, clusters, P=2, K=3,
                             num_batches_per_epoch=5, seed=42)
        s.set_epoch(0)
        b0 = list(s)
        s.set_epoch(1)
        b1 = list(s)
        assert b0 != b1


class TestNoClusterIds:
    def test_works_with_none_cluster_ids(self):
        """If cluster_ids is None, every sample is a singleton cluster (pure PK)."""
        labels = ['A'] * 20 + ['B'] * 20 + ['C'] * 20
        sampler = PKClusterSampler(labels, None, P=2, K=4,
                                   num_batches_per_epoch=10, seed=3)
        for batch in sampler:
            assert len(batch) == 8
            counts = batch_class_distribution(batch, labels)
            assert len(counts) == 2
            assert all(v == 4 for v in counts.values())
