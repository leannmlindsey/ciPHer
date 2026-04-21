"""Predict stub — contrastive_encoder is a feature transformer, not a classifier.

cipher-evaluate needs a Predictor. This module satisfies the interface but
raises with a clear message pointing users at the drop-in NPZ + downstream
attention_mlp workflow.
"""

import os


class _ContrastiveEncoderNotAPredictor:
    embedding_type = 'contrastive_encoder'

    def __init__(self, run_dir):
        self.run_dir = run_dir

    def predict_protein(self, embedding):
        raise NotImplementedError(_MSG.format(self.run_dir))

    def score_pair(self, *args, **kwargs):
        raise NotImplementedError(_MSG.format(self.run_dir))


_MSG = (
    "\n"
    "contrastive_encoder runs do not classify directly — they produce a\n"
    "learned embedding NPZ. To evaluate, train a downstream classifier on\n"
    "the generated NPZs:\n"
    "\n"
    "  cipher-train --model attention_mlp \\\n"
    "      --embedding_file {0}/contrastive_train_md5.npz \\\n"
    "      --val_embedding_file {0}/contrastive_val_md5.npz \\\n"
    "      [ ... same filter/sampling flags you used for the encoder ... ]\n"
    "\n"
    "Then cipher-evaluate the attention_mlp run as usual.\n"
)


def get_predictor(run_dir):
    # Validate the run dir exists so user sees the right error ordering.
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(run_dir)
    return _ContrastiveEncoderNotAPredictor(run_dir)
