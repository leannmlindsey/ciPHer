"""Tests for cipher.data.serotypes."""

from cipher.data.serotypes import is_null, normalize_k_type, normalize_k_to_short


class TestIsNull:
    def test_null_values(self):
        for val in ['null', 'N/A', 'Unknown', '', 'NA', 'n/a']:
            assert is_null(val), f'{val!r} should be null'

    def test_real_values(self):
        for val in ['K1', 'KL107', 'O1', 'O3ab']:
            assert not is_null(val), f'{val!r} should not be null'


class TestNormalizeKType:
    def test_k_to_kl(self):
        assert normalize_k_type('K1') == 'KL1'
        assert normalize_k_type('K23') == 'KL23'
        assert normalize_k_type('K107') == 'KL107'

    def test_kl_unchanged(self):
        assert normalize_k_type('KL1') == 'KL1'
        assert normalize_k_type('KL107') == 'KL107'

    def test_null_passthrough(self):
        assert normalize_k_type('N/A') == 'N/A'
        assert normalize_k_type('null') == 'null'


class TestNormalizeKToShort:
    def test_kl_to_k(self):
        assert normalize_k_to_short('KL1') == 'K1'
        assert normalize_k_to_short('KL23') == 'K23'
        assert normalize_k_to_short('KL82') == 'K82'

    def test_high_kl_unchanged(self):
        # KL107+ have no short equivalent
        assert normalize_k_to_short('KL107') == 'KL107'
        assert normalize_k_to_short('KL186') == 'KL186'

    def test_null_passthrough(self):
        assert normalize_k_to_short('N/A') == 'N/A'
