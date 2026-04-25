"""Tests for cipher.data.splits."""

import numpy as np

from cipher.data.splits import create_stratified_split, create_canonical_split


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


class TestCanonicalSplit:
    """Mirrors the old klebsiella `create_canonical_split.py` behavior:
    one shared split for both heads, primary K (with O-fallback when null)
    as the cluster key.
    """

    def _build_inputs(self):
        # 12 proteins. K classes: K1, K2, null. O classes: O1, O2.
        # Mix of K-only, O-only, and both.
        md5s = [f'm{i}' for i in range(12)]
        k_classes = ['K1', 'K2', 'null']
        o_classes = ['O1', 'O2']
        # one-hot K labels (label_strategy=single_label style)
        # m0-m4: K1.  m5-m6: K2.  m7-m9: null K (with valid O).  m10-m11: K1 only
        k_idx = [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 0, 0]
        # O labels: m0-m4 O1, m5-m6 O1, m7-m9 O2 (these provide O fallback),
        # m10-m11 null O — but we don't have a null O column here so 0 vector
        o_idx = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1]
        n = len(md5s)
        k_labels = np.zeros((n, len(k_classes)), dtype=np.float32)
        o_labels = np.zeros((n, len(o_classes)), dtype=np.float32)
        for i, ki in enumerate(k_idx):
            k_labels[i, ki] = 1.0
        for i, oi in enumerate(o_idx):
            if oi >= 0:
                o_labels[i, oi] = 1.0
        return md5s, k_labels, o_labels, k_classes, o_classes

    def test_no_overlap(self):
        md5s, k, o, kc, oc = self._build_inputs()
        s = create_canonical_split(md5s, k, o, kc, oc, seed=0)
        assert set(s['train']) & set(s['val']) == set()
        assert set(s['train']) & set(s['test']) == set()
        assert set(s['val']) & set(s['test']) == set()

    def test_reproducible(self):
        md5s, k, o, kc, oc = self._build_inputs()
        a = create_canonical_split(md5s, k, o, kc, oc, seed=42)
        b = create_canonical_split(md5s, k, o, kc, oc, seed=42)
        assert a == b

    def test_null_K_fallback_to_O(self):
        """K-null + O-valid proteins (m7-m9) must be placed via O cluster,
        not all dumped into a single 'null' bucket. They should appear in
        the union of train/val/test with their O cluster (O2 here)."""
        md5s, k, o, kc, oc = self._build_inputs()
        s = create_canonical_split(md5s, k, o, kc, oc, seed=0)
        all_assigned = set(s['train']) | set(s['val']) | set(s['test'])
        # m7-m9 should be assigned (via O fallback)
        for m in ('m7', 'm8', 'm9'):
            assert m in all_assigned, f'{m} (null K, valid O) was not split-assigned'

    def test_no_K_no_O_dropped(self):
        """A protein with no K and no O label is dropped (no cluster)."""
        md5s = ['a', 'b', 'c']
        k_classes = ['K1']
        o_classes = ['O1']
        k = np.zeros((3, 1), dtype=np.float32)
        o = np.zeros((3, 1), dtype=np.float32)
        k[0, 0] = 1.0  # only 'a' has a K label
        s = create_canonical_split(md5s, k, o, k_classes, o_classes, seed=0)
        all_assigned = set(s['train']) | set(s['val']) | set(s['test'])
        # 'a' has K → must be in some split; 'b','c' have neither → dropped
        assert 'a' in all_assigned
        assert 'b' not in all_assigned
        assert 'c' not in all_assigned
