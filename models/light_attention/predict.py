"""Prediction interface for trained Light Attention models.

Loads trained K and O LightAttentionClassifier heads and exposes the standard
cipher.evaluation.Predictor interface. predict_protein() accepts a single
variable-length (L, D) embedding, runs it through LA pooling + classifier,
and returns per-class probabilities.
"""

import json
import os

import numpy as np
import torch

from cipher.evaluation.predictor import Predictor
from model import LightAttentionClassifier


class LightAttentionPredictor(Predictor):
    """Predictor for trained Light Attention K and O heads."""

    def __init__(self, k_model, o_model, k_classes, o_classes,
                 k_strategy, o_strategy, embedding_type_name, device):
        self._k_model = k_model
        self._o_model = o_model
        self._k_classes = k_classes
        self._o_classes = o_classes
        self._k_strategy = k_strategy
        self._o_strategy = o_strategy
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
        """Run LA pooling + classifier on a single (L, D) embedding."""
        if embedding.ndim != 2:
            raise ValueError(
                f'Light Attention expects a 2D (L, D) embedding; '
                f'got shape {embedding.shape}.')

        emb_tensor = torch.as_tensor(embedding, dtype=torch.float32, device=self._device)
        # Build (1, D, L) input and an all-ones mask — single proteins have no
        # padding, so every position is valid.
        x = emb_tensor.unsqueeze(0).permute(0, 2, 1)
        mask = torch.ones((1, emb_tensor.shape[0]), dtype=torch.bool, device=self._device)

        with torch.no_grad():
            k_logits = self._k_model(x, mask)
            o_logits = self._o_model(x, mask)

        softmax_strategies = {'single_label', 'weighted_soft'}

        if self._k_strategy in softmax_strategies:
            k_probs_arr = torch.softmax(k_logits, dim=1).cpu().numpy()[0]
        else:
            k_probs_arr = torch.sigmoid(k_logits).cpu().numpy()[0]

        if self._o_strategy in softmax_strategies:
            o_probs_arr = torch.softmax(o_logits, dim=1).cpu().numpy()[0]
        else:
            o_probs_arr = torch.sigmoid(o_logits).cpu().numpy()[0]

        k_probs = {k: float(p) for k, p in zip(self._k_classes, k_probs_arr)}
        o_probs = {o: float(p) for o, p in zip(self._o_classes, o_probs_arr)}

        return {'k_probs': k_probs, 'o_probs': o_probs}


def _load_head(model_dir, device):
    """Load a trained head from a directory containing config.json + best_model.pt."""
    with open(os.path.join(model_dir, 'config.json')) as f:
        config = json.load(f)

    model = LightAttentionClassifier(
        embedding_dim=config['embedding_dim'],
        num_classes=config['num_classes'],
        kernel_size=config.get('la_kernel_size', 9),
        conv_dropout=config.get('la_conv_dropout', 0.25),
        hidden_dims=config['hidden_dims'],
        se_dim=config['se_dim'],
        classifier_dropout=config['dropout'],
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

    embedding_type = 'esm2_650m_seg4'
    exp_json = os.path.join(experiment_dir, 'experiment.json')
    if os.path.exists(exp_json):
        with open(exp_json) as f:
            meta = json.load(f)
        embedding_type = (meta.get('config', {})
                          .get('data', {})
                          .get('embedding_type', embedding_type))

    return LightAttentionPredictor(
        k_model=k_model,
        o_model=o_model,
        k_classes=k_config['classes'],
        o_classes=o_config['classes'],
        k_strategy=k_config.get('strategy', 'single_label'),
        o_strategy=o_config.get('strategy', 'single_label'),
        embedding_type_name=embedding_type,
        device=device,
    )
