"""Tests for cipher.data.splits."""

from cipher.data.splits import create_stratified_split


class TestStratifiedSplit:
    def test_all_ids_assigned(self):
        """Every ID should appear in exactly one split."""
        ids = list(range(100))
        labels = [i % 5 for i in ids]
        splits = create_stratified_split(ids, labels, seed=42)

        all_assigned = set(splits['train'] + splits['val'] + splits['test'])
        assert all_assigned == set(ids)

    def test_no_overlap(self):
        """No ID should appear in more than one split."""
        ids = list(range(100))
        labels = [i % 5 for i in ids]
        splits = create_stratified_split(ids, labels, seed=42)

        train = set(splits['train'])
        val = set(splits['val'])
        test = set(splits['test'])
        assert len(train & val) == 0
        assert len(train & test) == 0
        assert len(val & test) == 0

    def test_reproducible(self):
        """Same seed should produce same split."""
        ids = list(range(50))
        labels = [i % 3 for i in ids]

        s1 = create_stratified_split(ids, labels, seed=99)
        s2 = create_stratified_split(ids, labels, seed=99)
        assert s1['train'] == s2['train']
        assert s1['val'] == s2['val']
        assert s1['test'] == s2['test']

    def test_different_seeds(self):
        """Different seeds should (usually) produce different splits."""
        ids = list(range(50))
        labels = [i % 3 for i in ids]

        s1 = create_stratified_split(ids, labels, seed=1)
        s2 = create_stratified_split(ids, labels, seed=2)
        assert s1['train'] != s2['train']

    def test_small_class(self):
        """Classes with <= 2 members go entirely to train."""
        ids = ['a', 'b', 'c', 'd', 'e']
        labels = ['rare', 'rare', 'common', 'common', 'common']
        splits = create_stratified_split(ids, labels, seed=42)

        # Both 'rare' items should be in train
        assert 'a' in splits['train']
        assert 'b' in splits['train']

    def test_approximate_ratios(self):
        """With enough data, ratios should be roughly correct."""
        ids = list(range(200))
        labels = [i % 4 for i in ids]
        splits = create_stratified_split(ids, labels, train_ratio=0.7,
                                          val_ratio=0.15, seed=42)

        n = len(ids)
        train_frac = len(splits['train']) / n
        # Allow wide margin since stratification adjusts ratios
        assert 0.55 < train_frac < 0.85
