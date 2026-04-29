"""Dual-experiment-dir, dual-embedding predictor.

Combines a K head from one trained experiment with an O head from a
DIFFERENT trained experiment, where each head may have been trained on
its own embedding family (e.g. K on ProtT5 mean, O on ESM-2 650M mean).

Use case: our strong K head (trained with one recipe + embedding) plus
the old strong O head (trained with the old single_label recipe on
ESM-2 mean). Each head was the best at its own job; this predictor
runs both at inference time and combines their outputs via the standard
score_pair logic — but feeds each head from its own embedding dict.

Loading:
    from cipher.evaluation.dual_head_predictor import build_dual_head_predictor
    predictor = build_dual_head_predictor(
        k_experiment_dir=<path to K-trained experiment>,
        o_experiment_dir=<path to O-trained experiment>,
        k_emb_dict=<dict of {md5: K-head-embedding}>,
        o_emb_dict=<dict of {md5: O-head-embedding}>,
    )

The two source predictors are loaded via the standard `get_predictor`
entrypoint of each model. We then steal each one's K (or O) head and
its associated metadata (classes, strategy, embedding_type), and ignore
the other head from each source.
"""

import importlib.util
import os
import sys

import numpy as np

from cipher.evaluation.predictor import Predictor


# Match find_predict_module in runner.py (avoids circular import)
def _load_predict_module(experiment_dir):
    # Try: <experiment_dir>/predict.py first, then walk up to find one.
    candidates = [
        os.path.join(experiment_dir, 'predict.py'),
    ]
    # Common pattern: experiments/<model>/<run>/  → models/<model>/predict.py
    parts = os.path.abspath(experiment_dir).split(os.sep)
    if 'experiments' in parts:
        i = parts.index('experiments')
        if i + 1 < len(parts):
            model_name = parts[i + 1]
            # find repo root: parent of 'experiments'
            repo_root = os.sep.join(parts[:i])
            candidates.append(os.path.join(repo_root, 'models', model_name, 'predict.py'))
    for cand in candidates:
        if os.path.exists(cand):
            return cand, os.path.dirname(cand)
    raise FileNotFoundError(
        f'No predict.py found for experiment dir {experiment_dir}; '
        f'tried: {candidates}')


def _load_source_predictor(experiment_dir):
    """Load a fully-functional Predictor from one experiment dir."""
    predict_path, model_dir = _load_predict_module(experiment_dir)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    spec = importlib.util.spec_from_file_location(
        f'_dual_predict_{os.path.basename(experiment_dir)}', predict_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'get_predictor'):
        raise AttributeError(
            f'{predict_path} must define get_predictor(experiment_dir)')
    return module.get_predictor(os.path.abspath(experiment_dir))


class DualHeadPredictor(Predictor):
    """K head from one experiment + O head from another, with per-head embeddings.

    The two source predictors are full Predictor instances. We delegate
    the per-head probability computation to each source's
    `predict_protein()` and steal only the relevant head's output
    (k_probs from the K source, o_probs from the O source).

    `predict_protein(emb)` is intentionally not the right entrypoint
    here — there is no single "the embedding" for a dual-embedding
    predictor. Calling it raises NotImplementedError. The runner
    invokes `score_phage_md5s` (the polymorphic ranking entrypoint),
    which we override below to do dual lookups.
    """

    def __init__(self, k_source, o_source, k_emb_dict, o_emb_dict,
                 score_normalization='zscore'):
        self._k_source = k_source
        self._o_source = o_source
        self._k_embs = k_emb_dict
        self._o_embs = o_emb_dict
        self.score_normalization = score_normalization

    @property
    def embedding_type(self):
        # Two different embedding types; report the combined string.
        return f'dual({self._k_source.embedding_type}+{self._o_source.embedding_type})'

    @property
    def k_classes(self):
        return self._k_source.k_classes

    @property
    def o_classes(self):
        return self._o_source.o_classes

    def predict_protein(self, embedding):  # pragma: no cover
        raise NotImplementedError(
            'DualHeadPredictor uses per-head embeddings; '
            'call score_phage_md5s instead of predict_protein.')

    def _predict_dual(self, k_emb, o_emb):
        """Run K head on k_emb and O head on o_emb; merge probs."""
        k_out = self._k_source.predict_protein(k_emb)
        o_out = self._o_source.predict_protein(o_emb)
        return {
            'k_probs': k_out.get('k_probs', {}),
            'o_probs': o_out.get('o_probs', {}),
        }

    def score_phage_md5s(self, prot_md5s, host_k, host_o, emb_dict=None,
                          **kwargs):
        """Score a phage-host pair using per-head embedding dicts.

        Looks up each protein's MD5 in BOTH internal dicts. Only proteins
        present in both contribute (we need a K and an O embedding for
        each protein). The shared `emb_dict` arg from the caller is
        ignored — it would be a single shared dict, but we have two.
        """
        from cipher.data.serotypes import is_null

        has_k = host_k is not None and not is_null(host_k)
        has_o = host_o is not None and not is_null(host_o)
        if not has_k and not has_o:
            return None

        # Pair up K and O embeddings per md5. Skip md5s missing from
        # either dict — we can't dual-score those.
        pairs = []
        for m in prot_md5s:
            ke = self._k_embs.get(m)
            oe = self._o_embs.get(m)
            if ke is not None and oe is not None:
                pairs.append((ke, oe))

        if not pairs:
            return None

        best_score = -np.inf
        for k_emb, o_emb in pairs:
            preds = self._predict_dual(k_emb, o_emb)
            k_score = self._head_score(preds['k_probs'], host_k) if has_k else None
            o_score = self._head_score(preds['o_probs'], host_o) if has_o else None
            valid = [s for s in (k_score, o_score) if s is not None]
            if not valid:
                continue
            protein_score = max(valid)
            if protein_score > best_score:
                best_score = protein_score

        if best_score == -np.inf:
            return None
        return float(best_score)


def build_dual_head_predictor(k_experiment_dir, o_experiment_dir,
                              k_emb_dict, o_emb_dict,
                              score_normalization='zscore'):
    """Convenience constructor — loads both source predictors and assembles.

    Args:
        k_experiment_dir: experiment dir whose K head we want.
        o_experiment_dir: experiment dir whose O head we want. May be
            the same as k_experiment_dir for "single source, two
            embeddings" use cases (uncommon).
        k_emb_dict: {md5: ndarray} for K head input.
        o_emb_dict: {md5: ndarray} for O head input.
        score_normalization: 'zscore' or 'raw'; controls combine logic.

    Returns:
        DualHeadPredictor instance.
    """
    k_source = _load_source_predictor(k_experiment_dir)
    o_source = _load_source_predictor(o_experiment_dir)
    return DualHeadPredictor(
        k_source=k_source, o_source=o_source,
        k_emb_dict=k_emb_dict, o_emb_dict=o_emb_dict,
        score_normalization=score_normalization,
    )
