"""Standard prediction interface that all models must implement."""

from abc import ABC, abstractmethod

import numpy as np


class Predictor(ABC):
    """Standard interface for phage-host interaction prediction.

    Every experiment must provide a predict.py that implements this class.
    The evaluation runner loads the predictor and calls score_pair() to
    compute rankings.

    Subclasses MUST implement:
        - predict_protein(): per-protein K/O probabilities
        - embedding_type: what type of input the model expects

    Subclasses MAY override:
        - score_pair(): custom scoring logic (default: max-over-proteins,
          max-over-KO with z-score normalization across K and O)
    """

    # How to combine K and O scores for the same protein.
    # 'zscore' (default): z-score each head's output before max — accounts
    #                     for the fact that K head has many more classes
    #                     than O head, so raw probabilities aren't comparable.
    # 'raw':              max of raw probabilities (legacy; biased toward
    #                     whichever head has fewer classes).
    score_normalization = 'zscore'

    # Which head(s) participate in scoring. Set via cipher-evaluate --head-mode.
    # 'both'    (default): use K and O heads as before; max-over-heads-per-protein.
    # 'k_only':            ignore the O head entirely, rank on K alone.
    # 'o_only':            ignore the K head entirely, rank on O alone.
    #
    # Per-dataset best mode is determined by the dataset's phage breadth and
    # ranking direction — see notes/findings/2026-04-23_head_eval_phage_breadth.md
    # for the empirical pattern across datasets and models.
    head_mode = 'both'

    @property
    @abstractmethod
    def embedding_type(self):
        """Return the embedding type this model expects.

        Returns one of: 'esm2_650m', 'esm2_3b', 'kmer3', 'kmer_murphy8_5',
        or any string matching a key in the embeddings config.
        """

    @property
    def k_classes(self):
        """Return list of K-type class names the model can predict."""
        return []

    @property
    def o_classes(self):
        """Return list of O-type class names the model can predict."""
        return []

    @abstractmethod
    def predict_protein(self, embedding):
        """Predict K and O probabilities for a single protein.

        Args:
            embedding: numpy array, shape depends on embedding_type

        Returns:
            dict with:
                'k_probs': dict {k_type: probability} for all K-types
                'o_probs': dict {o_type: probability} for all O-types
                           (empty dict if model is K-only)
        """

    def score_phage_md5s(self, prot_md5s, host_k, host_o, emb_dict, **kwargs):
        """Score a phage-host pair given protein MD5s + a shared embedding dict.

        Default implementation: look up each MD5's embedding in the shared
        emb_dict, then defer to `score_pair`. Subclasses that need
        per-head embedding routing (e.g. dual-embedding predictors that
        feed K and O heads from different embedding spaces) override
        this method to ignore the shared emb_dict and use their own
        per-head dicts.

        Args:
            prot_md5s: list of MD5 hex digests, one per phage protein.
            host_k: K-type of host (or None / null sentinel).
            host_o: O-type of host (or None / null sentinel).
            emb_dict: shared {md5: embedding_array} as loaded by the
                runner. Single-embedding predictors look up here; dual-
                embedding predictors ignore this.
            **kwargs: forward-compatibility — unknown kwargs are
                accepted but ignored. Keeps the door open for future
                signatures (e.g. host MD5, phage MD5) without breaking.

        Returns:
            float score (higher = more likely to interact) or None.
        """
        prot_embs = [emb_dict[m] for m in prot_md5s if m in emb_dict]
        if not prot_embs:
            return None
        return self.score_pair(prot_embs, host_k, host_o)

    def score_pair(self, protein_embeddings, host_k, host_o):
        """Score a phage-host pair.

        Bio-motivated scoring: host has two locks (K, O), phage has multiple
        keys (proteins). Score = max over proteins of max(K_score, O_score).

        K and O probabilities come from heads with very different class counts
        (~158 K vs ~22 O), so raw probabilities aren't directly comparable.
        By default, z-score normalization is applied per-head per-protein
        before taking max(K, O), making the two heads comparable in magnitude.

        Args:
            protein_embeddings: list of numpy arrays (one per protein)
            host_k: string K-type of host (or None if unknown)
            host_o: string O-type of host (or None if unknown)

        Returns:
            float score (higher = more likely to interact), or None if
            scoring is not possible (no embeddings, no serotype match)
        """
        from cipher.data.serotypes import is_null

        if not protein_embeddings:
            return None

        has_k = host_k is not None and not is_null(host_k)
        has_o = host_o is not None and not is_null(host_o)

        # head_mode further gates which head participates. 'k_only' disables
        # the O branch entirely; 'o_only' disables K.
        if self.head_mode == 'k_only':
            has_o = False
        elif self.head_mode == 'o_only':
            has_k = False
        elif self.head_mode != 'both':
            raise ValueError(
                f'Unknown head_mode: {self.head_mode!r}. '
                f'Use "both", "k_only", or "o_only".')

        if not has_k and not has_o:
            return None

        best_score = -np.inf

        for emb in protein_embeddings:
            preds = self.predict_protein(emb)
            k_probs = preds.get('k_probs', {})
            o_probs = preds.get('o_probs', {})

            k_score = self._head_score(k_probs, host_k) if has_k else None
            o_score = self._head_score(o_probs, host_o) if has_o else None

            valid = [s for s in (k_score, o_score) if s is not None]
            if not valid:
                continue

            protein_score = max(valid)
            if protein_score > best_score:
                best_score = protein_score

        if best_score == -np.inf:
            return None
        return float(best_score)

    def _head_score(self, probs_dict, target_class):
        """Score one head's prediction for the target class.

        Applies the configured normalization (z-score or raw).

        Args:
            probs_dict: {class_name: probability} from predict_protein
            target_class: the class we want a score for

        Returns:
            float score, or None if target_class not in probs_dict
        """
        if not probs_dict or target_class not in probs_dict:
            return None

        target_p = probs_dict[target_class]

        if self.score_normalization == 'raw':
            return target_p

        if self.score_normalization == 'zscore':
            # z-score relative to the head's distribution for this protein
            vals = np.asarray(list(probs_dict.values()), dtype=np.float64)
            mu = vals.mean()
            sigma = vals.std()
            if sigma < 1e-12:
                return 0.0  # degenerate distribution; no information
            return float((target_p - mu) / sigma)

        raise ValueError(
            f'Unknown score_normalization: {self.score_normalization!r}. '
            f'Use "zscore" or "raw".')
