"""Prediction interface for trained LightAttentionBinary models.

Loads trained K and O heads from an experiment directory and provides the
standard Predictor interface for use with cipher-evaluate.

Note: predict_protein expects a per-residue embedding of shape (L, D), not
a pre-pooled vector. The ConvolutionalAttention pooler runs inline as part
of the forward pass.
"""

import json
import os

import numpy as np
import torch

from cipher.evaluation.predictor import Predictor
from model import LightAttentionBinary


class LightAttentionBinaryPredictor(Predictor):
    """Predictor backed by trained K and O LightAttentionBinary heads."""

    def __init__(self, k_model, o_model, k_classes, o_classes,
                 embedding_type_name, device):
        self._k_model = k_model
        self._o_model = o_model
        self._k_classes = k_classes
        self._o_classes = o_classes
        self._embedding_type = embedding_type_name
        self._device = device

    @property
    def embedding_type(self):
        return self._embedding_type

    @property
    def k_classes(self):
        return self._k_classes

    @property
    def o_classes(self):
        return self._o_classes

    def predict_protein(self, embedding):
        """Predict K and O probabilities for a single protein.

        Args:
            embedding: numpy array of shape (L, D) -- per-residue embeddings.

        Returns:
            {'k_probs': {k_type: prob, ...}, 'o_probs': {o_type: prob, ...}}
        """
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(
                f'LightAttentionBinary expects per-residue embeddings of '
                f'shape (L, D); got {emb.shape}.'
            )
        L, _ = emb.shape
        x = torch.from_numpy(emb).unsqueeze(0).to(self._device)      # [1, L, D]
        masks = torch.ones(1, L, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            k_logits = self._k_model(x, masks)
            o_logits = self._o_model(x, masks)

        # Always sigmoid -- every head was trained with BCE-per-class.
        k_probs_arr = torch.sigmoid(k_logits).cpu().numpy()[0]
        o_probs_arr = torch.sigmoid(o_logits).cpu().numpy()[0]

        k_probs = {k: float(p) for k, p in zip(self._k_classes, k_probs_arr)}
        o_probs = {o: float(p) for o, p in zip(self._o_classes, o_probs_arr)}

        return {'k_probs': k_probs, 'o_probs': o_probs}


def _load_head(model_dir, device):
    """Load a trained head from `model_dir` (expects config.json + best_model.pt)."""
    with open(os.path.join(model_dir, 'config.json')) as f:
        config = json.load(f)

    model = LightAttentionBinary(
        embed_dim=config['embed_dim'],
        num_classes=config['num_classes'],
        pooler_cnn_width=config.get('pooler_cnn_width', 9),
        dropout=config.get('dropout', 0.1),
    )

    state_dict = torch.load(
        os.path.join(model_dir, 'best_model.pt'),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config


def get_predictor(experiment_dir):
    """Load trained K and O heads and return a Predictor."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    k_model, k_config = _load_head(
        os.path.join(experiment_dir, 'model_k'), device)
    o_model, o_config = _load_head(
        os.path.join(experiment_dir, 'model_o'), device)

    embedding_type = 'esm2_650m_full'
    exp_json = os.path.join(experiment_dir, 'experiment.json')
    if os.path.exists(exp_json):
        with open(exp_json) as f:
            meta = json.load(f)
        embedding_type = (meta.get('config', {})
                          .get('data', {})
                          .get('embedding_type', embedding_type))

    return LightAttentionBinaryPredictor(
        k_model=k_model,
        o_model=o_model,
        k_classes=k_config['classes'],
        o_classes=o_config['classes'],
        embedding_type_name=embedding_type,
        device=device,
    )
