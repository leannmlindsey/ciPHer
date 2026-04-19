"""Tests for Predictor.score_pair (z-score normalization across K and O)."""

import numpy as np
import pytest
from cipher.evaluation.predictor import Predictor


class FakePredictor(Predictor):
    """A test fixture: returns fixed predictions per call."""

    def __init__(self, k_probs_list, o_probs_list,
                 k_classes=None, o_classes=None):
        self._k_list = k_probs_list  # one dict per protein
        self._o_list = o_probs_list
        self._k_classes = k_classes or sorted(k_probs_list[0].keys())
        self._o_classes = o_classes or sorted(o_probs_list[0].keys())
        self._idx = 0

    @property
    def embedding_type(self):
        return 'fake'

    @property
    def k_classes(self):
        return self._k_classes

    @property
    def o_classes(self):
        return self._o_classes

    def predict_protein(self, embedding):
        i = self._idx
        self._idx += 1
        return {
            'k_probs': self._k_list[i],
            'o_probs': self._o_list[i],
        }


class TestZScoreNormalization:
    def test_uniform_predictions_give_zero(self):
        """Uniform distribution -> all z-scores are 0 -> degenerate, returns 0."""
        # All K-types equally likely
        k = {f'K{i}': 0.01 for i in range(100)}
        o = {f'O{i}': 0.05 for i in range(20)}
        pred = FakePredictor([k], [o])

        score = pred.score_pair([np.zeros(10)], 'K0', 'O0')
        # Both have zero std → both return 0.0 → max is 0
        assert score == 0.0

    def test_zscore_makes_k_and_o_comparable(self):
        """K head with 100 classes vs O with 20: a K_prob of 0.5 (50x random)
        should outscore an O_prob of 0.5 (10x random) because K is more
        impressive given more classes to compete with."""
        # K: target K0 has high prob, others uniform
        k = {f'K{i}': 0.005 for i in range(100)}
        k['K0'] = 0.5  # very high relative to baseline 0.01
        # O: target O0 has same absolute prob but in a much smaller class space
        o = {f'O{i}': 0.025 for i in range(20)}
        o['O0'] = 0.5  # high but baseline is 0.05

        pred = FakePredictor([k], [o])
        # Compute z-scores manually to verify
        k_vals = np.array(list(k.values()))
        o_vals = np.array(list(o.values()))
        k_z = (0.5 - k_vals.mean()) / k_vals.std()
        o_z = (0.5 - o_vals.mean()) / o_vals.std()

        score = pred.score_pair([np.zeros(10)], 'K0', 'O0')
        assert score == max(k_z, o_z)
        # K should win because larger class space → higher z
        assert k_z > o_z

    def test_raw_mode_uses_raw_probs(self):
        k = {'K0': 0.3, 'K1': 0.7}
        o = {'O0': 0.5, 'O1': 0.5}
        pred = FakePredictor([k], [o])
        pred.score_normalization = 'raw'

        score = pred.score_pair([np.zeros(10)], 'K0', 'O0')
        # raw: max(0.3, 0.5) = 0.5
        assert score == pytest.approx(0.5)

    def test_max_over_proteins(self):
        """Multiple proteins → should pick the highest-scoring one."""
        # protein 1: weak signal
        k1 = {f'K{i}': 0.1 for i in range(10)}
        k1['K0'] = 0.15
        o1 = {f'O{i}': 0.1 for i in range(10)}
        o1['O0'] = 0.15
        # protein 2: strong K signal
        k2 = {f'K{i}': 0.05 for i in range(10)}
        k2['K0'] = 0.55
        o2 = {f'O{i}': 0.1 for i in range(10)}
        o2['O0'] = 0.1

        pred = FakePredictor([k1, k2], [o1, o2])
        score = pred.score_pair([np.zeros(10), np.zeros(10)], 'K0', 'O0')

        # Protein 2's K-z should dominate
        k2_vals = np.array(list(k2.values()))
        expected = (0.55 - k2_vals.mean()) / k2_vals.std()
        assert score == pytest.approx(expected, rel=1e-5)

    def test_no_proteins_returns_none(self):
        pred = FakePredictor([{'K0': 1.0}], [{'O0': 1.0}])
        assert pred.score_pair([], 'K0', 'O0') is None

    def test_unknown_target_class_skips(self):
        """If target class isn't in probs, that head doesn't contribute."""
        k = {'K0': 0.5, 'K1': 0.5}
        o = {'O0': 0.5, 'O1': 0.5}
        pred = FakePredictor([k], [o])

        # host_k='K_unknown' not in K classes, but O0 is
        score = pred.score_pair([np.zeros(10)], 'K_unknown', 'O0')
        # Should still return O's z-score (not None)
        assert score is not None

    def test_both_targets_unknown_returns_none(self):
        k = {'K0': 0.5, 'K1': 0.5}
        o = {'O0': 0.5, 'O1': 0.5}
        pred = FakePredictor([k], [o])

        score = pred.score_pair([np.zeros(10)], 'K_unknown', 'O_unknown')
        assert score is None

    def test_null_serotypes_treated_as_unknown(self):
        k = {'K0': 0.5, 'K1': 0.5}
        o = {'O0': 0.5, 'O1': 0.5}
        pred = FakePredictor([k], [o])

        # 'N/A' is a null label
        assert pred.score_pair([np.zeros(10)], 'N/A', 'N/A') is None
