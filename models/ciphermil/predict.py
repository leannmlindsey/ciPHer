"""ciphermil Predictor — implements cipher's evaluation interface.

Key difference from per-protein models (attention_mlp / binary_mlp): MIL
operates on bags (whole phages), not single proteins. So we OVERRIDE
score_pair (the bag-level scoring entry point) instead of relying on
predict_protein.

predict_protein is required by the abstract base class but isn't
meaningful for MIL (one protein in isolation has no MIL bag context).
We implement it as a no-op returning empty probs — the cipher
ranking pipeline calls score_pair (which we override), so
predict_protein is never actually invoked during evaluation.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from cipher.evaluation.predictor import Predictor

# Add this model's dir to sys.path for `import model`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import AttentionMIL


def _load_head(head_dir, device):
    """Load a trained MIL head from <experiment>/model_{k,o}/."""
    with open(os.path.join(head_dir, 'config.json')) as f:
        config = json.load(f)

    model = AttentionMIL(
        input_dim=config['input_dim'],
        num_classes=config['num_classes'],
        L=config['L'], D=config['D'], K=config['K'],
        dropout=config['dropout'],
    )
    state = torch.load(os.path.join(head_dir, 'best_model.pt'),
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, config


class CipherMILPredictor(Predictor):
    """Bag-level MIL Predictor. Overrides score_pair for whole-bag forward."""

    def __init__(self, k_model, o_model, k_classes, o_classes,
                 embedding_type, device):
        self._k_model = k_model
        self._o_model = o_model
        self._k_classes = k_classes
        self._o_classes = o_classes
        self._k_class_to_idx = {c: i for i, c in enumerate(k_classes)}
        self._o_class_to_idx = {c: i for i, c in enumerate(o_classes)}
        self._embedding_type = embedding_type
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
        """Required by abstract base — but never called for MIL.

        score_pair (which is overridden below) does the bag-level forward
        directly; the per-protein API is only meaningful for non-MIL
        architectures.
        """
        return {'k_probs': {}, 'o_probs': {}}

    def score_pair(self, protein_embeddings, host_k, host_o):
        """Score a phage-host pair via bag-level MIL forward.

        Args:
            protein_embeddings: list of np.ndarray, one per phage protein
                (the bag). Variable length.
            host_k: K-type of the host
            host_o: O-type of the host

        Returns:
            float score (higher = more likely to interact) or None if
            unscorable. Score follows cipher's standard convention:
            zscore-then-max across K and O heads (gated by head_mode).
        """
        if not protein_embeddings:
            return None

        # Whole-bag forward — single torch op for all proteins of this phage.
        bag = torch.tensor(np.stack(protein_embeddings),
                           dtype=torch.float32, device=self._device)

        with torch.no_grad():
            k_probs = None
            if self._k_model is not None:
                k_logits, _ = self._k_model(bag)
                k_probs = torch.softmax(k_logits, dim=1).cpu().numpy().squeeze()
            o_probs = None
            if self._o_model is not None:
                o_logits, _ = self._o_model(bag)
                o_probs = torch.softmax(o_logits, dim=1).cpu().numpy().squeeze()

        # head_mode gates which head participates (cipher convention).
        has_k = (k_probs is not None and host_k in self._k_class_to_idx
                 and self.head_mode in ('both', 'k_only'))
        has_o = (o_probs is not None and host_o in self._o_class_to_idx
                 and self.head_mode in ('both', 'o_only'))

        if not has_k and not has_o:
            return None

        # zscore each head's distribution then max-over-heads. Standard
        # cipher score_pair convention so MIL scores are commensurate
        # with attention_mlp / binary_mlp / LA scores in the harvest.
        if self.score_normalization == 'zscore':
            scores = []
            if has_k:
                k_mu, k_sigma = k_probs.mean(), k_probs.std() + 1e-9
                k_z = (k_probs[self._k_class_to_idx[host_k]] - k_mu) / k_sigma
                scores.append(float(k_z))
            if has_o:
                o_mu, o_sigma = o_probs.mean(), o_probs.std() + 1e-9
                o_z = (o_probs[self._o_class_to_idx[host_o]] - o_mu) / o_sigma
                scores.append(float(o_z))
            return max(scores) if scores else None
        else:  # 'raw'
            scores = []
            if has_k:
                scores.append(float(k_probs[self._k_class_to_idx[host_k]]))
            if has_o:
                scores.append(float(o_probs[self._o_class_to_idx[host_o]]))
            return max(scores) if scores else None


def get_predictor(experiment_dir):
    """Load trained MIL heads and return a CipherMILPredictor."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    k_dir = os.path.join(experiment_dir, 'model_k')
    o_dir = os.path.join(experiment_dir, 'model_o')
    k_model, k_cfg = (None, {})
    o_model, o_cfg = (None, {})
    if os.path.isdir(k_dir) and os.path.exists(os.path.join(k_dir, 'config.json')):
        k_model, k_cfg = _load_head(k_dir, device)
    if os.path.isdir(o_dir) and os.path.exists(os.path.join(o_dir, 'config.json')):
        o_model, o_cfg = _load_head(o_dir, device)
    if k_model is None and o_model is None:
        raise FileNotFoundError(
            f'Neither model_k/ nor model_o/ found in {experiment_dir}')

    # Embedding type from experiment.json
    embedding_type = 'prott5_xl'  # safe default
    exp_json = os.path.join(experiment_dir, 'experiment.json')
    if os.path.exists(exp_json):
        with open(exp_json) as f:
            meta = json.load(f)
        embedding_type = (meta.get('config', {})
                          .get('data', {})
                          .get('embedding_type', embedding_type))

    return CipherMILPredictor(
        k_model=k_model,
        o_model=o_model,
        k_classes=k_cfg.get('classes', []),
        o_classes=o_cfg.get('classes', []),
        embedding_type=embedding_type,
        device=device,
    )
