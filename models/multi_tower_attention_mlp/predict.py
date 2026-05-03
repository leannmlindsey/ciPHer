"""Predictor + standalone eval helpers for MultiTowerAttentionMLP.

The standard cipher.evaluation.runner expects ONE embedding NPZ per model.
Multi-tower needs THREE embeddings per protein at inference time, so the
typical approach (override `predict_protein` to take a single vector)
doesn't fit cleanly.

We therefore expose two surfaces:
  1. `MultiTowerPredictor` -- a Predictor subclass that accepts a
     pre-concatenated single vector (tower-A vec || tower-B vec ||
     tower-C vec). Useful if a composite NPZ is built upstream.
  2. `predict_proteins_from_3_npz()` -- the practical path: load 3 val
     NPZs separately, intersect MD5s, run the model, return per-protein
     {k_probs, o_probs}. Called by our custom strict-eval script.
"""

import json
import os

import numpy as np
import torch

try:
    from cipher.evaluation.predictor import Predictor
except Exception:
    Predictor = object  # type: ignore

from model import MultiTowerAttentionMLP, DEFAULT_TOWER_HIDDEN_DIMS


def _load_model(experiment_dir, device):
    """Reconstruct MultiTowerAttentionMLP from saved config.json + best_model.pt."""
    with open(os.path.join(experiment_dir, 'config.json')) as f:
        cfg = json.load(f)
    model = MultiTowerAttentionMLP(
        input_dims=cfg['input_dims'],
        num_K=cfg['num_K'],
        num_O=cfg['num_O'],
        tower_hidden_dims=tuple(cfg.get('tower_hidden_dims',
                                        DEFAULT_TOWER_HIDDEN_DIMS)),
        se_dim=cfg.get('se_dim', 640),
        dropout=cfg.get('dropout', 0.1),
    )
    state = torch.load(
        os.path.join(experiment_dir, 'best_model.pt'),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg


class MultiTowerPredictor(Predictor):  # type: ignore[misc]
    """Predictor that accepts a single concatenated vector (A || B || C).
    Splits internally based on input_dims saved in config.json."""

    def __init__(self, model, cfg, device):
        self._model = model
        self._cfg = cfg
        self._device = device
        self._K_classes = cfg['K_classes']
        self._O_classes = cfg['O_classes']
        self._input_dims = cfg['input_dims']

    @property
    def embedding_type(self):
        return 'multi_tower_concat'

    @property
    def k_classes(self):
        return self._K_classes

    @property
    def o_classes(self):
        return self._O_classes

    def predict_protein(self, embedding):
        emb = np.asarray(embedding, dtype=np.float32)
        expected = sum(self._input_dims)
        if emb.ndim != 1 or emb.shape[0] != expected:
            raise ValueError(
                f'MultiTowerPredictor expects a 1D concat vector of len '
                f'{expected}; got shape {emb.shape}')
        # Split A || B || C
        offsets = [0]
        for d in self._input_dims:
            offsets.append(offsets[-1] + d)
        xs = [
            torch.from_numpy(emb[offsets[i]:offsets[i + 1]]).unsqueeze(0).to(self._device)
            for i in range(len(self._input_dims))
        ]
        with torch.no_grad():
            logK, logO = self._model(xs)
        k_probs = torch.sigmoid(logK).cpu().numpy()[0]
        o_probs = torch.sigmoid(logO).cpu().numpy()[0]
        return {
            'k_probs': dict(zip(self._K_classes, k_probs.tolist())),
            'o_probs': dict(zip(self._O_classes, o_probs.tolist())),
        }


def get_predictor(experiment_dir):
    """Standard cipher.evaluation.runner entry point. Requires a composite
    NPZ keyed by MD5 with concatenated A||B||C vectors as input."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = _load_model(experiment_dir, device)
    return MultiTowerPredictor(model, cfg, device)


def predict_proteins_from_3_npz(experiment_dir, val_emb_paths,
                                 device=None):
    """Load 3 val NPZs separately, intersect MD5s, run the model, return
    per-MD5 (k_probs_dict, o_probs_dict). Path used by our custom
    strict-eval script -- avoids needing a pre-concatenated val NPZ.

    Args:
        experiment_dir: dir with best_model.pt and config.json
        val_emb_paths: list of 3 paths to validation NPZs (in tower order)

    Returns:
        dict {md5: (k_probs, o_probs)}
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = _load_model(experiment_dir, device)
    K_classes = cfg['K_classes']
    O_classes = cfg['O_classes']

    if len(val_emb_paths) != 3:
        raise ValueError(f'Need 3 val NPZ paths; got {len(val_emb_paths)}')

    dicts = [dict(np.load(p)) for p in val_emb_paths]
    common = set(dicts[0]).intersection(dicts[1]).intersection(dicts[2])
    out = {}
    with torch.no_grad():
        for md5 in common:
            xs = [
                torch.from_numpy(np.asarray(d[md5], dtype=np.float32))
                     .unsqueeze(0).to(device)
                for d in dicts
            ]
            logK, logO = model(xs)
            k_probs = torch.sigmoid(logK).cpu().numpy()[0]
            o_probs = torch.sigmoid(logO).cpu().numpy()[0]
            out[md5] = (
                dict(zip(K_classes, k_probs.tolist())),
                dict(zip(O_classes, o_probs.tolist())),
            )
    return out
