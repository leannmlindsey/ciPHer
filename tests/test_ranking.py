"""Tests for ranking utilities (tie handling, etc.)."""

import pytest
from cipher.evaluation.ranking import _ranks_with_ties


class TestRanksWithTies:
    def test_no_ties(self):
        scored = [('a', 0.9), ('b', 0.5), ('c', 0.1)]
        ranks = _ranks_with_ties(scored, tie_method='competition')
        assert ranks == {'a': 1, 'b': 2, 'c': 3}

    def test_competition_ranking(self):
        # 3 tied at top, then one lower
        scored = [('a', 0.9), ('b', 0.9), ('c', 0.9), ('d', 0.5)]
        ranks = _ranks_with_ties(scored, tie_method='competition')
        # All 3 tied get rank 1, next gets rank 4 (gap)
        assert ranks == {'a': 1, 'b': 1, 'c': 1, 'd': 4}

    def test_arbitrary_ranking(self):
        scored = [('a', 0.9), ('b', 0.9), ('c', 0.9), ('d', 0.5)]
        ranks = _ranks_with_ties(scored, tie_method='arbitrary')
        assert ranks == {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    def test_competition_multiple_tie_groups(self):
        # 2 tied at top, then 3 tied in middle, then 1 alone
        scored = [('a', 0.9), ('b', 0.9), ('c', 0.5), ('d', 0.5),
                  ('e', 0.5), ('f', 0.1)]
        ranks = _ranks_with_ties(scored, tie_method='competition')
        # a, b -> 1; c, d, e -> 3 (skipping 2); f -> 6
        assert ranks == {'a': 1, 'b': 1, 'c': 3, 'd': 3, 'e': 3, 'f': 6}

    def test_all_tied(self):
        scored = [('a', 0.5), ('b', 0.5), ('c', 0.5)]
        ranks = _ranks_with_ties(scored, tie_method='competition')
        assert ranks == {'a': 1, 'b': 1, 'c': 1}

    def test_empty(self):
        assert _ranks_with_ties([], tie_method='competition') == {}

    def test_single_item(self):
        ranks = _ranks_with_ties([('a', 0.5)], tie_method='competition')
        assert ranks == {'a': 1}
