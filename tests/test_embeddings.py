"""Tests for cipher.data.embeddings."""

import os
import tempfile

import numpy as np
import pytest

from cipher.data.embeddings import load_embeddings, load_embeddings_concat


@pytest.fixture
def two_npz(tmp_path):
    """Two NPZ files keyed by the same MD5s, with different vector dims."""
    md5s = ['aaa', 'bbb', 'ccc']
    dim1, dim2 = 4, 7
    d1 = {m: np.arange(dim1, dtype=np.float32) + i for i, m in enumerate(md5s)}
    d2 = {m: np.arange(dim2, dtype=np.float32) * 10 + i for i, m in enumerate(md5s)}
    p1 = tmp_path / 'emb1.npz'
    p2 = tmp_path / 'emb2.npz'
    np.savez(p1, **d1)
    np.savez(p2, **d2)
    return str(p1), str(p2), d1, d2, md5s, dim1, dim2


def _noop(*args, **kwargs):
    pass


class TestLoadEmbeddingsConcat:
    def test_concatenates_in_order(self, two_npz):
        p1, p2, d1, d2, md5s, dim1, dim2 = two_npz
        out = load_embeddings_concat(p1, p2, log=_noop)
        assert set(out.keys()) == set(md5s)
        for m in md5s:
            expected = np.concatenate([d1[m], d2[m]])
            np.testing.assert_array_equal(out[m], expected)
            assert out[m].shape == (dim1 + dim2,)

    def test_md5_filter_limits_output(self, two_npz):
        p1, p2, _, _, _, _, _ = two_npz
        out = load_embeddings_concat(p1, p2, md5_filter={'bbb'}, log=_noop)
        assert set(out.keys()) == {'bbb'}

    def test_missing_md5_in_second_file_raises(self, tmp_path):
        p1 = tmp_path / 'emb1.npz'
        p2 = tmp_path / 'emb2.npz'
        np.savez(p1, aaa=np.ones(3, dtype=np.float32),
                     bbb=np.ones(3, dtype=np.float32))
        np.savez(p2, aaa=np.ones(2, dtype=np.float32))  # missing 'bbb'
        with pytest.raises(ValueError, match='coverage mismatch'):
            load_embeddings_concat(str(p1), str(p2),
                                   md5_filter={'aaa', 'bbb'}, log=_noop)

    def test_order_matters_in_output(self, two_npz):
        """file_1 comes first in the concatenated vector."""
        p1, p2, d1, d2, md5s, dim1, dim2 = two_npz
        out_12 = load_embeddings_concat(p1, p2, log=_noop)
        out_21 = load_embeddings_concat(p2, p1, log=_noop)
        # Same total dim but ordering flipped
        m = md5s[0]
        assert out_12[m].shape == out_21[m].shape == (dim1 + dim2,)
        np.testing.assert_array_equal(out_12[m][:dim1], d1[m])
        np.testing.assert_array_equal(out_21[m][:dim2], d2[m])
