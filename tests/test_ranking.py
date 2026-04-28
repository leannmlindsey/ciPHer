"""Tests for ranking utilities (tie handling, fixed-denominator HR@k, etc.)."""

import numpy as np
import pytest
from cipher.evaluation.predictor import Predictor
from cipher.evaluation.ranking import _ranks_with_ties, evaluate_rankings


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


# ============================================================
# evaluate_rankings — fixed-denominator HR@k (changed 2026-04-27)
# ============================================================

class _FakePredictor(Predictor):
    """Predictor that scores only hosts whose K is in `known_k_classes` and
    O is in `known_o_classes`. Hosts whose serotype is null or out-of-vocab
    will cause score_pair to return None — exercising the unscorable path.

    Score for a known target class is just the protein's prob for that
    class (no z-score; deterministic + monotonic per class).
    """

    score_normalization = 'raw'

    def __init__(self, k_probs, o_probs, known_k, known_o):
        self._k_probs = dict(k_probs)
        self._o_probs = dict(o_probs)
        self._known_k = list(known_k)
        self._known_o = list(known_o)

    @property
    def embedding_type(self):
        return 'fake'

    @property
    def k_classes(self):
        return self._known_k

    @property
    def o_classes(self):
        return self._known_o

    def predict_protein(self, embedding):
        return {'k_probs': dict(self._k_probs), 'o_probs': dict(self._o_probs)}


def _write_dataset(tmp_path, dataset_name, rows, phage_proteins):
    """Write a minimal validation dataset directory.

    rows: list of (host_id, host_assembly, host_K, host_O, phage_id, label)
    phage_proteins: list of (phage_id, protein_id)
    """
    ds_dir = tmp_path / dataset_name
    meta = ds_dir / 'metadata'
    meta.mkdir(parents=True)

    with open(meta / 'interaction_matrix.tsv', 'w') as f:
        f.write('host_id\thost_assembly\thost_K\thost_O\tphage_id\tlabel\n')
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')

    with open(meta / 'phage_protein_mapping.csv', 'w') as f:
        f.write('phage_id,protein_id\n')
        for phage, prot in phage_proteins:
            f.write(f'{phage},{prot}\n')

    return ds_dir


class TestEvaluateRankingsFixedDenominator:
    """Verifies the 2026-04-27 fix: every positive (phage, host) pair
    contributes to n_pairs and HR@k, even when the model can't score it.
    Unscorable positives (host_K is null, or host_K is outside the
    model's class vocabulary, etc.) count as misses."""

    def test_unscorable_positive_counted_as_miss(self, tmp_path):
        # Phage A has two positives (HK1, HK2) and one negative (HK3) —
        # all scorable. Predictor strongly prefers HK1's K-type.
        # Phage B has one positive (HNULL) whose serotype is null —
        # unscorable. Phage B's positive must still count as a miss.
        rows = [
            ('HK1', 'A1', 'K1',   'O1',   'phageA', 1),
            ('HK2', 'A2', 'K2',   'O2',   'phageA', 1),
            ('HK3', 'A3', 'K3',   'O3',   'phageA', 0),
            ('HNULL', 'A4', 'null', 'null', 'phageB', 1),
        ]
        proteins = [('phageA', 'pA'), ('phageB', 'pB')]
        ds = _write_dataset(tmp_path, 'TEST', rows, proteins)

        # Predictor: K1 has highest prob, others lower; O similar
        pred = _FakePredictor(
            k_probs={'K1': 0.9, 'K2': 0.5, 'K3': 0.1},
            o_probs={'O1': 0.9, 'O2': 0.5, 'O3': 0.1},
            known_k=['K1', 'K2', 'K3'],
            known_o=['O1', 'O2', 'O3'],
        )

        emb_dict = {'mdA': np.zeros(4), 'mdB': np.zeros(4)}
        pid_md5 = {'pA': 'mdA', 'pB': 'mdB'}

        r = evaluate_rankings(pred, 'TEST', str(ds), emb_dict, pid_md5,
                              max_k=10)

        rh = r['rank_hosts']
        # 3 positive (phage, host) pairs total: (A,HK1), (A,HK2), (B,HNULL)
        assert rh['n_pairs'] == 3, 'n_pairs must include unscorable positives'
        assert rh['n_pairs_scored'] == 2, 'only 2 positives are scorable'
        # HR@1 = 1/3 (HK1 is rank-1; HK2 is rank-2; HNULL is unscorable)
        assert rh['hr_at_k'][1] == pytest.approx(1 / 3)
        # HR@2 = 2/3 (HK1 + HK2 in top 2; HNULL still unscorable)
        assert rh['hr_at_k'][2] == pytest.approx(2 / 3)
        # HR@10 = 2/3 still — HNULL never lands in top-k for any finite k
        assert rh['hr_at_k'][10] == pytest.approx(2 / 3)

    def test_out_of_vocab_k_counts_as_miss(self, tmp_path):
        # Phage has positive whose host_K isn't in the predictor's K_classes.
        # That positive still counts toward n_pairs; HR@k must reflect the miss.
        rows = [
            ('H1', 'A1', 'K1',     'O1',     'phage1', 1),
            ('H2', 'A2', 'K_RARE', 'O_RARE', 'phage1', 1),
        ]
        proteins = [('phage1', 'p1')]
        ds = _write_dataset(tmp_path, 'TEST', rows, proteins)

        pred = _FakePredictor(
            k_probs={'K1': 0.9},
            o_probs={'O1': 0.9},
            known_k=['K1'],
            known_o=['O1'],
        )
        emb_dict = {'md1': np.zeros(4)}
        pid_md5 = {'p1': 'md1'}

        r = evaluate_rankings(pred, 'TEST', str(ds), emb_dict, pid_md5,
                              max_k=20)

        rh = r['rank_hosts']
        assert rh['n_pairs'] == 2
        assert rh['n_pairs_scored'] == 1
        # H1 ranks 1; H2 unscorable. HR@1 = 1/2, HR@20 = 1/2.
        assert rh['hr_at_k'][1] == pytest.approx(0.5)
        assert rh['hr_at_k'][20] == pytest.approx(0.5)

    def test_phage_with_no_scorable_proteins(self, tmp_path):
        # Phage has positives but no proteins with embeddings.
        # All its positives should count as misses, not be silently dropped.
        rows = [
            ('H1', 'A1', 'K1', 'O1', 'phage_noemb', 1),
            ('H2', 'A2', 'K2', 'O2', 'phage_noemb', 1),
        ]
        proteins = [('phage_noemb', 'p_missing')]
        ds = _write_dataset(tmp_path, 'TEST', rows, proteins)

        pred = _FakePredictor(
            k_probs={'K1': 0.9, 'K2': 0.5},
            o_probs={'O1': 0.9, 'O2': 0.5},
            known_k=['K1', 'K2'],
            known_o=['O1', 'O2'],
        )
        emb_dict = {}        # no embedding available
        pid_md5 = {}         # no md5 mapping

        r = evaluate_rankings(pred, 'TEST', str(ds), emb_dict, pid_md5,
                              max_k=10)

        rh = r['rank_hosts']
        assert rh['n_pairs'] == 2
        assert rh['n_pairs_scored'] == 0
        for k in (1, 5, 10):
            assert rh['hr_at_k'][k] == 0.0, (
                f'HR@{k} must be 0 when no positives are scorable')

    def test_mrr_treats_unscorable_as_zero_contribution(self, tmp_path):
        # Two positives: one at rank 1 (1/1=1.0 contribution), one unscorable
        # (1/inf=0.0 contribution). MRR over 2 positives = 0.5.
        rows = [
            ('H1', 'A1', 'K1',   'O1',   'phage1', 1),
            ('H2', 'A2', 'null', 'null', 'phage1', 1),
            ('H3', 'A3', 'K3',   'O3',   'phage1', 0),
        ]
        proteins = [('phage1', 'p1')]
        ds = _write_dataset(tmp_path, 'TEST', rows, proteins)

        pred = _FakePredictor(
            k_probs={'K1': 0.9, 'K3': 0.1},
            o_probs={'O1': 0.9, 'O3': 0.1},
            known_k=['K1', 'K3'],
            known_o=['O1', 'O3'],
        )
        emb_dict = {'md1': np.zeros(4)}
        pid_md5 = {'p1': 'md1'}

        r = evaluate_rankings(pred, 'TEST', str(ds), emb_dict, pid_md5,
                              max_k=20)

        rh = r['rank_hosts']
        assert rh['n_pairs'] == 2
        # MRR = (1.0 + 0.0) / 2 = 0.5
        assert rh['mrr'] == pytest.approx(0.5)
