"""Unit tests for models/light_attention_binary.

Tests only the pieces that can fail silently: the collate_fn's padding/mask
correctness, ConvolutionalAttention's mask invariance (padding a sequence
should not change the output), the wrapper model's output shape, the BCE
loss reduction, and the predictor's output structure.
"""

import os
import sys

import numpy as np
import pytest
import torch

# Make `models/light_attention_binary/model.py` importable by tests.
_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'light_attention_binary')
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from model import (  # noqa: E402
    ConvolutionalAttention,
    LightAttentionBinary,
    PerResidueDataset,
    TransformerPooler,
    crop_c_terminal,
    per_residue_collate_fn,
    bce_loss,
)
from predict import LightAttentionBinaryPredictor  # noqa: E402
from train import check_embedding_coverage, load_embeddings_or_bins  # noqa: E402


# ---------------------------------------------------------------------------
# per_residue_collate_fn
# ---------------------------------------------------------------------------

class TestPerResidueCollate:
    def test_pads_to_batch_max_length(self):
        embs = [torch.ones(3, 4), torch.ones(5, 4) * 2, torch.ones(2, 4) * 3]
        labels = [torch.zeros(7) for _ in embs]
        batch = list(zip(embs, labels))

        padded, masks, labels_out = per_residue_collate_fn(batch)

        assert padded.shape == (3, 5, 4)
        assert masks.shape == (3, 5)
        assert labels_out.shape == (3, 7)

    def test_masks_are_one_at_real_positions(self):
        embs = [torch.ones(3, 4), torch.ones(5, 4), torch.ones(2, 4)]
        labels = [torch.zeros(2) for _ in embs]
        _, masks, _ = per_residue_collate_fn(list(zip(embs, labels)))

        # real positions = original length, padded = 0
        expected = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
        ], dtype=torch.float32)
        assert torch.equal(masks, expected)

    def test_preserves_original_data_in_real_positions(self):
        e1 = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        e2 = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 100
        labels = [torch.zeros(1), torch.zeros(1)]

        padded, _, _ = per_residue_collate_fn(list(zip([e1, e2], labels)))
        assert torch.equal(padded[0, :2], e1)
        assert torch.equal(padded[1, :4], e2)
        # Padding positions stay zero
        assert torch.all(padded[0, 2:] == 0)


# ---------------------------------------------------------------------------
# ConvolutionalAttention
# ---------------------------------------------------------------------------

class TestConvolutionalAttention:
    def test_output_shape(self):
        ca = ConvolutionalAttention(embed_dim=16, width=5)
        x = torch.randn(2, 30, 16)
        out = ca(x)
        assert out.shape == (2, 16)

    def test_mask_invariance(self):
        """Zero-padding a sequence + setting the mask to 0 at pad positions
        must give the same output as running on the unpadded sequence."""
        torch.manual_seed(0)
        ca = ConvolutionalAttention(embed_dim=8, width=5)
        ca.eval()

        L_real = 10
        x_real = torch.randn(1, L_real, 8)

        # Run unpadded (no mask needed, but pass mask=all-ones for parity)
        mask_real = torch.ones(1, L_real)
        with torch.no_grad():
            out_real = ca(x_real, mask_real)

        # Run with extra zero-padding + mask zeros at padded positions
        L_padded = 17
        x_padded = torch.zeros(1, L_padded, 8)
        x_padded[:, :L_real] = x_real
        mask_padded = torch.zeros(1, L_padded)
        mask_padded[:, :L_real] = 1.0
        with torch.no_grad():
            out_padded = ca(x_padded, mask_padded)

        # Outputs must match to floating-point tolerance. If they don't, the
        # pooler is leaking information from padded positions.
        torch.testing.assert_close(out_real, out_padded, rtol=1e-5, atol=1e-5)

    def test_attention_weights_sum_to_one_over_real_positions(self):
        """Softmax over masked logits → weights must be 0 on padding and
        sum to 1 on real positions."""
        ca = ConvolutionalAttention(embed_dim=8, width=3)
        ca.eval()

        x = torch.randn(2, 12, 8)
        mask = torch.zeros(2, 12)
        mask[0, :5] = 1.0
        mask[1, :9] = 1.0

        with torch.no_grad():
            _, attn = ca(x, mask, return_attns=True)
        # attn has shape [B, D, L]; sum over L per (B, D) should be ~1
        sums = attn.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-5)

        # Weights on padded positions should be 0
        assert torch.all(attn[0, :, 5:] == 0)
        assert torch.all(attn[1, :, 9:] == 0)


# ---------------------------------------------------------------------------
# LightAttentionBinary
# ---------------------------------------------------------------------------

class TestLightAttentionBinary:
    def test_forward_shape(self):
        model = LightAttentionBinary(embed_dim=32, num_classes=10, pooler_cnn_width=5)
        model.eval()
        x = torch.randn(3, 20, 32)
        mask = torch.ones(3, 20)
        with torch.no_grad():
            logits = model(x, mask)
        assert logits.shape == (3, 10)

    def test_num_classes_below_two_raises(self):
        with pytest.raises(ValueError, match='num_classes'):
            LightAttentionBinary(embed_dim=16, num_classes=1)

    def test_dataset_roundtrip(self):
        embs = [np.random.rand(4, 8).astype(np.float32),
                np.random.rand(6, 8).astype(np.float32)]
        labels = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        ds = PerResidueDataset(embs, labels)
        assert len(ds) == 2
        e0, l0 = ds[0]
        assert e0.shape == (4, 8)
        assert torch.equal(l0, torch.tensor([1.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# bce_loss
# ---------------------------------------------------------------------------

class TestBceLoss:
    def test_matches_sum_then_mean_reduction(self):
        """Verify: bce_loss == sum(BCE(per-element), dim=1).mean()."""
        torch.manual_seed(1)
        logits = torch.randn(5, 7)
        targets = (torch.rand(5, 7) > 0.5).float()

        per_elem = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        expected = per_elem.sum(dim=1).mean()
        torch.testing.assert_close(bce_loss(logits, targets), expected)

    def test_differs_from_default_mean_reduction(self):
        """Sanity check the divergence from PyTorch's default reduction='mean'.
        The default divides by B*C; ours divides by C-sum then averages over B,
        so the two values differ by a factor of `num_classes`."""
        logits = torch.randn(4, 6)
        targets = torch.ones(4, 6)
        default_mean = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='mean')
        ours = bce_loss(logits, targets)
        # ours == default_mean * num_classes
        torch.testing.assert_close(ours, default_mean * 6)


# ---------------------------------------------------------------------------
# LightAttentionBinaryPredictor
# ---------------------------------------------------------------------------

class TestPredictor:
    def _build_predictor(self, num_k=4, num_o=3, embed_dim=16):
        torch.manual_seed(7)
        k_model = LightAttentionBinary(embed_dim=embed_dim, num_classes=num_k,
                                       pooler_cnn_width=3).eval()
        o_model = LightAttentionBinary(embed_dim=embed_dim, num_classes=num_o,
                                       pooler_cnn_width=3).eval()
        k_classes = [f'K{i}' for i in range(num_k)]
        o_classes = [f'O{i}' for i in range(num_o)]
        return LightAttentionBinaryPredictor(
            k_model, o_model, k_classes, o_classes,
            embedding_type_name='esm2_650m_full',
            device=torch.device('cpu'),
        )

    def test_predict_protein_returns_expected_structure(self):
        pred = self._build_predictor(num_k=4, num_o=3)
        emb = np.random.rand(10, 16).astype(np.float32)
        out = pred.predict_protein(emb)

        assert set(out.keys()) == {'k_probs', 'o_probs'}
        assert set(out['k_probs'].keys()) == {'K0', 'K1', 'K2', 'K3'}
        assert set(out['o_probs'].keys()) == {'O0', 'O1', 'O2'}

    def test_predict_protein_outputs_sigmoid_probs(self):
        """All outputs must be in [0, 1] since we always sigmoid."""
        pred = self._build_predictor()
        emb = np.random.rand(8, 16).astype(np.float32)
        out = pred.predict_protein(emb)
        all_probs = list(out['k_probs'].values()) + list(out['o_probs'].values())
        assert all(0.0 <= p <= 1.0 for p in all_probs)

    def test_predict_protein_rejects_1d_embedding(self):
        """Per-residue predictor must not accept pooled (D,)-shape input."""
        pred = self._build_predictor()
        with pytest.raises(ValueError, match='per-residue'):
            pred.predict_protein(np.random.rand(16).astype(np.float32))

    def test_embedding_type_property(self):
        pred = self._build_predictor()
        assert pred.embedding_type == 'esm2_650m_full'


# ---------------------------------------------------------------------------
# check_embedding_coverage (data-loader guard)
# ---------------------------------------------------------------------------

class TestEmbeddingCoverageGuard:
    def test_full_coverage_returns_all(self):
        md5s = ['a', 'b', 'c', 'd']
        emb = {m: np.zeros((5, 8)) for m in md5s}
        valid = check_embedding_coverage(md5s, emb, 0.5, 'k/train', '/fake.npz')
        assert valid == md5s

    def test_just_above_threshold_passes(self):
        md5s = ['a', 'b', 'c', 'd']
        emb = {'a': np.zeros((5, 8)), 'b': np.zeros((5, 8)),
               'c': np.zeros((5, 8))}  # 75% coverage
        valid = check_embedding_coverage(md5s, emb, 0.5, 'k/train', '/fake.npz')
        assert set(valid) == {'a', 'b', 'c'}

    def test_below_threshold_raises(self):
        """Recreates the 14%-coverage failure we hit 2026-04-21."""
        md5s = [f'm{i}' for i in range(100)]
        emb = {f'm{i}': np.zeros((5, 8)) for i in range(14)}  # 14% coverage
        with pytest.raises(ValueError, match='Embedding coverage too low'):
            check_embedding_coverage(md5s, emb, 0.5, 'k/train', '/fake.npz')

    def test_error_message_includes_counts_and_path(self):
        md5s = [f'm{i}' for i in range(10)]
        emb = {'m0': np.zeros((5, 8))}
        with pytest.raises(ValueError) as exc:
            check_embedding_coverage(md5s, emb, 0.5, 'o/train',
                                     '/some/embeddings.npz')
        msg = str(exc.value)
        assert 'o/train' in msg
        assert '1/10' in msg
        assert '/some/embeddings.npz' in msg

    def test_empty_split_does_not_raise(self):
        """Empty val/test splits shouldn't trip the guard — coverage
        calculation is 0/0 which we treat as pass-through."""
        valid = check_embedding_coverage([], {}, 0.5, 'k/val', '/fake.npz')
        assert valid == []

    def test_threshold_zero_disables_guard(self):
        """Setting min_coverage=0 should pass even with empty emb_dict."""
        md5s = ['a', 'b', 'c']
        valid = check_embedding_coverage(md5s, {}, 0.0, 'k/train', '/fake.npz')
        assert valid == []


# ---------------------------------------------------------------------------
# load_embeddings_or_bins (directory of NPZs vs single file)
# ---------------------------------------------------------------------------

class TestLoadEmbeddingsOrBins:
    def _silent(self, *args, **kwargs):
        pass

    def test_single_file_path_unchanged(self, tmp_path):
        """Passing a .npz file should fall through to load_embeddings."""
        path = tmp_path / 'emb.npz'
        np.savez(path, a=np.ones((3, 4), dtype=np.float32),
                       b=np.zeros((2, 4), dtype=np.float32))
        out = load_embeddings_or_bins(str(path))
        assert set(out.keys()) == {'a', 'b'}
        np.testing.assert_array_equal(out['a'], np.ones((3, 4), dtype=np.float32))

    def test_directory_loads_all_bins(self, tmp_path):
        """Loading from a directory should union all *.npz files."""
        bin_dir = tmp_path / 'bins'
        bin_dir.mkdir()
        np.savez(bin_dir / 'maxlen128.npz', a=np.ones((2, 3), dtype=np.float32))
        np.savez(bin_dir / 'maxlen256.npz', b=np.ones((4, 3), dtype=np.float32) * 2)
        np.savez(bin_dir / 'maxlen512.npz', c=np.ones((8, 3), dtype=np.float32) * 3)
        out = load_embeddings_or_bins(str(bin_dir), log=self._silent)
        assert set(out.keys()) == {'a', 'b', 'c'}
        np.testing.assert_array_equal(out['b'], np.ones((4, 3), dtype=np.float32) * 2)

    def test_md5_filter_reduces_memory(self, tmp_path):
        """md5_filter should drop unwanted keys before they enter the dict."""
        bin_dir = tmp_path / 'bins'
        bin_dir.mkdir()
        np.savez(bin_dir / 'b1.npz',
                 keep1=np.ones((2, 3), dtype=np.float32),
                 drop1=np.zeros((2, 3), dtype=np.float32))
        np.savez(bin_dir / 'b2.npz',
                 keep2=np.ones((2, 3), dtype=np.float32) * 5,
                 drop2=np.zeros((2, 3), dtype=np.float32))
        out = load_embeddings_or_bins(str(bin_dir),
                                       md5_filter={'keep1', 'keep2'},
                                       log=self._silent)
        assert set(out.keys()) == {'keep1', 'keep2'}

    def test_duplicate_keys_first_wins(self, tmp_path):
        """If two bins share an MD5 (shouldn't happen post-extraction but
        could after a partial resume), keep the first occurrence and skip
        the second silently."""
        bin_dir = tmp_path / 'bins'
        bin_dir.mkdir()
        np.savez(bin_dir / 'aaa.npz', dup=np.ones((2, 3), dtype=np.float32))
        np.savez(bin_dir / 'bbb.npz', dup=np.ones((2, 3), dtype=np.float32) * 99)
        out = load_embeddings_or_bins(str(bin_dir), log=self._silent)
        # First-sorted bin (aaa.npz) wins
        np.testing.assert_array_equal(out['dup'], np.ones((2, 3), dtype=np.float32))

    def test_empty_directory_raises(self, tmp_path):
        empty = tmp_path / 'no_bins'
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match='No .npz files'):
            load_embeddings_or_bins(str(empty), log=self._silent)


# ---------------------------------------------------------------------------
# C-terminal cropping
# ---------------------------------------------------------------------------

class TestCTerminalCrop:
    def test_crop_keeps_last_n(self):
        emb = torch.arange(50, dtype=torch.float32).reshape(10, 5)
        out = crop_c_terminal(emb, 4)
        assert out.shape == (4, 5)
        # last 4 rows (residues 6..9) preserved verbatim
        torch.testing.assert_close(out, emb[6:])

    def test_crop_none_passthrough(self):
        emb = torch.zeros(20, 8)
        assert crop_c_terminal(emb, None).shape == (20, 8)
        assert crop_c_terminal(emb, 0).shape == (20, 8)
        assert crop_c_terminal(emb, -1).shape == (20, 8)

    def test_crop_larger_than_sequence(self):
        """If N > L, return the whole protein unchanged."""
        emb = torch.zeros(5, 8)
        assert crop_c_terminal(emb, 100).shape == (5, 8)

    def test_dataset_applies_crop(self):
        """PerResidueDataset(c_terminal_crop=N) should crop on construction."""
        embs = [np.random.rand(20, 4).astype(np.float32),
                np.random.rand(50, 4).astype(np.float32),
                np.random.rand(3, 4).astype(np.float32)]
        labels = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        ds = PerResidueDataset(embs, labels, c_terminal_crop=10)
        # Long proteins cropped to 10; short protein (L=3) untouched
        assert ds.embeddings[0].shape == (10, 4)
        assert ds.embeddings[1].shape == (10, 4)
        assert ds.embeddings[2].shape == (3, 4)


# ---------------------------------------------------------------------------
# TransformerPooler
# ---------------------------------------------------------------------------

class TestTransformerPooler:
    def test_output_shape(self):
        pooler = TransformerPooler(embed_dim=16, num_heads=2, num_layers=1, max_len=64)
        x = torch.randn(2, 30, 16)
        mask = torch.ones(2, 30)
        out = pooler(x, mask)
        assert out.shape == (2, 16)

    def test_truncates_above_max_len(self):
        """If sequence is longer than max_len, the pooler keeps the
        C-terminal max_len residues so we don't crash."""
        pooler = TransformerPooler(embed_dim=8, num_heads=2, num_layers=1, max_len=20)
        x = torch.randn(1, 50, 8)
        mask = torch.ones(1, 50)
        out = pooler(x, mask)
        assert out.shape == (1, 8)

    def test_lab_model_with_transformer_pooler(self):
        """End-to-end: LightAttentionBinary(pooler_type='transformer') runs."""
        m = LightAttentionBinary(
            embed_dim=16, num_classes=10, pooler_type='transformer',
            transformer_num_heads=2, transformer_num_layers=1,
            transformer_max_len=64,
        )
        m.eval()
        x = torch.randn(3, 20, 16)
        mask = torch.ones(3, 20)
        with torch.no_grad():
            logits = m(x, mask)
        assert logits.shape == (3, 10)

    def test_unknown_pooler_type_raises(self):
        with pytest.raises(ValueError, match='pooler_type'):
            LightAttentionBinary(embed_dim=16, num_classes=5, pooler_type='bogus')
