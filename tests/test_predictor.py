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


class TestHeadMode:
    """Tests for the head_mode attribute that gates K/O participation."""

    def _build_predictor(self):
        # K head wins under normal scoring; O head would win if forced alone.
        k = {'K0': 0.9, 'K1': 0.05, 'K2': 0.05}
        o = {'O0': 0.95, 'O1': 0.05}
        return FakePredictor([k] * 3, [o] * 3)

    def test_both_is_default_and_matches_prior_behavior(self):
        pred = self._build_predictor()
        assert pred.head_mode == 'both'
        score_both = pred.score_pair([np.zeros(10)], 'K0', 'O0')

        # Restore default and recompute — should be identical.
        pred2 = self._build_predictor()
        pred2.head_mode = 'both'
        score_explicit = pred2.score_pair([np.zeros(10)], 'K0', 'O0')
        assert score_both == score_explicit

    def test_k_only_uses_only_k_head(self):
        pred = self._build_predictor()
        pred.head_mode = 'k_only'
        # Should score K0 via z-score of {0.9, 0.05, 0.05}, ignoring O entirely.
        # Because has_o is forced False, passing a garbage host_o shouldn't matter.
        score_k = pred.score_pair([np.zeros(10)], 'K0', 'O_NONEXISTENT')
        pred2 = self._build_predictor()
        pred2.head_mode = 'k_only'
        score_k_valid = pred2.score_pair([np.zeros(10)], 'K0', 'O0')
        assert score_k == score_k_valid  # O-side inputs don't affect result

    def test_o_only_uses_only_o_head(self):
        pred = self._build_predictor()
        pred.head_mode = 'o_only'
        score_o = pred.score_pair([np.zeros(10)], 'K_NONEXISTENT', 'O0')
        pred2 = self._build_predictor()
        pred2.head_mode = 'o_only'
        score_o_valid = pred2.score_pair([np.zeros(10)], 'K0', 'O0')
        assert score_o == score_o_valid

    def test_k_only_ignores_bad_o_probs(self):
        """If K has a correct match and O has uniform garbage, k_only should
        still give a decisive score; 'both' might be muddied by O noise."""
        k = {'K0': 0.9, 'K1': 0.05, 'K2': 0.05}
        o_garbage = {'O0': 0.5, 'O1': 0.5}  # uniform, useless
        pred = FakePredictor([k], [o_garbage])
        pred.head_mode = 'k_only'
        s = pred.score_pair([np.zeros(10)], 'K0', 'O0')
        assert s is not None
        assert s > 0  # K z-score of 0.9 against 0.05/0.05 is strongly positive

    def test_o_only_returns_none_if_host_o_missing(self):
        """In o_only mode, we must have a valid host_o; absence is fatal."""
        k = {'K0': 0.9, 'K1': 0.1}
        o = {'O0': 0.5, 'O1': 0.5}
        pred = FakePredictor([k], [o])
        pred.head_mode = 'o_only'
        assert pred.score_pair([np.zeros(10)], 'K0', None) is None
        assert pred.score_pair([np.zeros(10)], 'K0', 'N/A') is None

    def test_k_only_returns_none_if_host_k_missing(self):
        k = {'K0': 0.9, 'K1': 0.1}
        o = {'O0': 0.5, 'O1': 0.5}
        pred = FakePredictor([k], [o])
        pred.head_mode = 'k_only'
        assert pred.score_pair([np.zeros(10)], None, 'O0') is None
        assert pred.score_pair([np.zeros(10)], 'N/A', 'O0') is None

    def test_invalid_head_mode_raises(self):
        pred = self._build_predictor()
        pred.head_mode = 'garbage'
        with pytest.raises(ValueError, match='Unknown head_mode'):
            pred.score_pair([np.zeros(10)], 'K0', 'O0')
