"""Tests for cipher.evaluation.metrics."""

import pytest
from cipher.evaluation.metrics import hr_at_k, mrr, hr_curve


class TestHrAtK:
    def test_all_hits(self):
        assert hr_at_k([1, 1, 1], k=1) == 1.0

    def test_no_hits(self):
        assert hr_at_k([5, 10, 20], k=3) == 0.0

    def test_partial(self):
        # 2 of 4 ranks are <= 3
        assert hr_at_k([1, 3, 5, 10], k=3) == pytest.approx(0.5)

    def test_boundary(self):
        # rank exactly equal to k counts as a hit
        assert hr_at_k([5], k=5) == 1.0
        assert hr_at_k([6], k=5) == 0.0

    def test_empty(self):
        assert hr_at_k([], k=5) == 0.0

    def test_single_rank(self):
        assert hr_at_k([1], k=1) == 1.0
        assert hr_at_k([2], k=1) == 0.0


class TestMrr:
    def test_basic(self):
        # MRR = mean(1/1, 1/2, 1/4) = (1 + 0.5 + 0.25) / 3
        assert mrr([1, 2, 4]) == pytest.approx((1 + 0.5 + 0.25) / 3)

    def test_all_rank_one(self):
        assert mrr([1, 1, 1]) == 1.0

    def test_empty(self):
        assert mrr([]) == 0.0

    def test_single(self):
        assert mrr([3]) == pytest.approx(1.0 / 3)


class TestHrCurve:
    def test_shape(self):
        curve = hr_curve([1, 3, 5], max_k=10)
        assert len(curve) == 10
        assert set(curve.keys()) == set(range(1, 11))

    def test_monotonic(self):
        """HR@k should be non-decreasing as k increases."""
        curve = hr_curve([1, 3, 5, 10], max_k=20)
        values = [curve[k] for k in range(1, 21)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]

    def test_values(self):
        curve = hr_curve([1, 5, 10], max_k=10)
        assert curve[1] == pytest.approx(1.0 / 3)  # only rank 1 <= 1
        assert curve[5] == pytest.approx(2.0 / 3)  # ranks 1, 5 <= 5
        assert curve[10] == 1.0                      # all <= 10
