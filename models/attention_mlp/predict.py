"""Prediction interface for trained AttentionMLP models.

Loads trained K and O heads and provides the standard Predictor interface
for use with cipher-evaluate.
"""

import json
import os

import numpy as np
import torch

from cipher.evaluation.predictor import Predictor
from model import AttentionMLP


class AttentionMLPPredictor(Predictor):
    """Predictor that uses trained K and O AttentionMLP heads."""

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
        """Predict K and O probabilities for a single protein embedding.

        Args:
            embedding: numpy array of shape (embedding_dim,)

        Returns:
            {'k_probs': {k_type: prob, ...}, 'o_probs': {o_type: prob, ...}}
        """
        x = torch.FloatTensor(embedding).unsqueeze(0).to(self._device)

        with torch.no_grad():
            k_logits = self._k_model(x)
            o_logits = self._o_model(x)

        # Convert logits to probabilities.
        # Strategies that use softmax during training:
        #   single_label, weighted_soft (both compete between classes)
        # Strategies that use sigmoid (independent per-class):
        #   multi_label, multi_label_threshold, weighted_multi_label
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
    """Load a trained model head from a directory.

    Args:
        model_dir: path containing config.json and best_model.pt

    Returns:
        (model, config_dict)
    """
    with open(os.path.join(model_dir, 'config.json')) as f:
        config = json.load(f)

    model = AttentionMLP(
        input_dim=config['input_dim'],
        num_classes=config['num_classes'],
        hidden_dims=config['hidden_dims'],
        se_dim=config['se_dim'],
        dropout=config['dropout'],
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
    """Load trained K and O models and return a Predictor.

    Args:
        experiment_dir: path to experiment directory containing
                        model_k/ and model_o/ subdirectories

    Returns:
        AttentionMLPPredictor instance
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    k_model, k_config = _load_head(
        os.path.join(experiment_dir, 'model_k'), device)
    o_model, o_config = _load_head(
        os.path.join(experiment_dir, 'model_o'), device)

    # Determine embedding type from experiment config
    embedding_type = 'esm2_650m'  # default
    exp_json = os.path.join(experiment_dir, 'experiment.json')
    if os.path.exists(exp_json):
        with open(exp_json) as f:
            meta = json.load(f)
        embedding_type = (meta.get('config', {})
                          .get('data', {})
                          .get('embedding_type', embedding_type))

    return AttentionMLPPredictor(
        k_model=k_model,
        o_model=o_model,
        k_classes=k_config['classes'],
        o_classes=o_config['classes'],
        k_strategy=k_config.get('strategy', 'single_label'),
        o_strategy=o_config.get('strategy', 'single_label'),
        embedding_type_name=embedding_type,
        device=device,
    )
