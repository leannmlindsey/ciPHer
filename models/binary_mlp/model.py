"""BinaryOneVsRestMLP: one binary classifier per serotype, sharing a trunk.

The motivating literature: PhageHostLearn (Boeckaerts) trains an independent
binary classifier per K-type with xgboost, rather than a single multi-label
softmax/sigmoid. This MLP variant ports that structural decision while
letting us share a feature trunk so we can train end-to-end on 160k-d kmer
inputs without one-MLP-per-class blowing up to billions of parameters.

Architecture for one head (K or O):

    Input (e.g. 160000-d kmer_aa20_k4)
    -> Shared trunk: BatchNorm + Linear(160k, h0) + ReLU + Dropout
                     + Linear(h0, h1) + ReLU + Dropout       [trunk_dims]
    -> n_classes parallel per-class heads, each:
       Linear(h1, head_h) + ReLU + Dropout + Linear(head_h, 1)
    -> Concat per-class logits => (batch, n_classes)

The per-class heads are independent — different weights per K-class —
so each class gets dedicated capacity to learn its decision boundary
on top of generic kmer features extracted by the shared trunk. This is
the structural change vs `attention_mlp`'s single Linear(h, n_classes)
output head: same number of trunk params, but each class gets a 2-layer
head instead of a 1-layer linear projection.

Training: BCE-with-logits over the (batch, n_classes) output, with
per-class pos_weight to handle K-class imbalance. Same loss surface
as `attention_mlp`'s `multi_label_threshold` strategy; the change is
the model's per-class capacity, not the loss.

Inference: per-class sigmoid -> probability per K-class -> max over
proteins of zscore(K) (the standard cipher score_pair pipeline).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class BinaryOneVsRestMLP(nn.Module):
    """Shared trunk + per-class binary heads."""

    def __init__(self, input_dim, num_classes,
                 trunk_dims=(1024, 256), head_hidden=64,
                 dropout=0.1, use_input_batchnorm=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Shared trunk — does the bulk of feature extraction.
        trunk = []
        if use_input_batchnorm:
            # 160k-d kmer input is sparse with high dynamic range; an input
            # BN gives the trunk a chance to whiten it before the first
            # large Linear blows up.
            trunk.append(nn.BatchNorm1d(input_dim))
        prev = input_dim
        for h in trunk_dims:
            trunk.append(nn.Linear(prev, h))
            trunk.append(nn.BatchNorm1d(h))
            trunk.append(nn.ReLU())
            trunk.append(nn.Dropout(dropout))
            prev = h
        self.trunk = nn.Sequential(*trunk)
        self.trunk_out_dim = prev

        # Per-class heads — each class gets its own 2-layer MLP.
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.trunk_out_dim, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
            for _ in range(num_classes)
        ])

    def forward(self, x):
        """Return logits, shape (batch, n_classes).

        Each per-class head produces 1 logit; concatenated.
        """
        h = self.trunk(x)  # (batch, trunk_out_dim)
        # Stack per-class logits along dim=1.
        per_class_logits = [head(h) for head in self.heads]  # list of (batch, 1)
        return torch.cat(per_class_logits, dim=1)  # (batch, n_classes)


class SerotypeDataset(Dataset):
    """Dataset for single-head serotype prediction (K or O)."""

    def __init__(self, embeddings, labels):
        """
        Args:
            embeddings: numpy array (N, embedding_dim)
            labels: numpy array (N, n_classes)
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def get_loss_fn(strategy, class_weights=None):
    """BCE-with-logits per-class. `class_weights` is the per-class pos_weight
    tensor (length n_classes). Same semantics as attention_mlp's
    `multi_label_threshold` loss; binary_mlp doesn't support single_label or
    weighted_soft because the per-class head structure doesn't softmax-compete
    across classes."""
    if strategy in ('multi_label', 'multi_label_threshold', 'weighted_multi_label'):
        return nn.BCEWithLogitsLoss(pos_weight=class_weights)
    if strategy == 'single_label':
        raise ValueError(
            "binary_mlp doesn't support strategy='single_label'. The whole "
            "point of one-vs-rest is to NOT softmax-compete across classes; "
            "use attention_mlp if you want single_label.")
    raise ValueError(f'Unknown strategy: {strategy}')


def compute_metrics(logits, labels, strategy, threshold=0.5):
    """Same metric set as attention_mlp's multi-label branch.

    For per-class binary heads, sigmoid + threshold is the natural decision;
    f1_micro / f1_macro / top1_acc all defined the same way as multi-label.
    """
    metrics = {}
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    binary_labels = (labels > 0).float()

    # Sample-averaged
    tp_s = (preds * binary_labels).sum(dim=1)
    fp_s = (preds * (1 - binary_labels)).sum(dim=1)
    fn_s = ((1 - preds) * binary_labels).sum(dim=1)
    prec_s = tp_s / (tp_s + fp_s + 1e-8)
    rec_s = tp_s / (tp_s + fn_s + 1e-8)
    f1_s = 2 * prec_s * rec_s / (prec_s + rec_s + 1e-8)
    metrics['f1_micro'] = float(f1_s.mean())
    metrics['precision'] = float(prec_s.mean())
    metrics['recall'] = float(rec_s.mean())

    # Per-class macro
    tp_c = (preds * binary_labels).sum(dim=0)
    fp_c = (preds * (1 - binary_labels)).sum(dim=0)
    fn_c = ((1 - preds) * binary_labels).sum(dim=0)
    prec_c = tp_c / (tp_c + fp_c + 1e-8)
    rec_c = tp_c / (tp_c + fn_c + 1e-8)
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-8)
    has_pos = binary_labels.sum(dim=0) > 0
    if has_pos.any():
        metrics['f1_macro'] = float(f1_c[has_pos].mean())
        metrics['precision_macro'] = float(prec_c[has_pos].mean())
        metrics['recall_macro'] = float(rec_c[has_pos].mean())
    else:
        metrics['f1_macro'] = 0.0
        metrics['precision_macro'] = 0.0
        metrics['recall_macro'] = 0.0

    # top1_acc: argmax over per-class probabilities — does it land on a
    # positive label? Useful for ranking-style sanity checks even though
    # binary-head training doesn't compete across classes during loss.
    top1 = logits.argmax(dim=1)
    top1_pos = labels[torch.arange(len(labels)), top1] > 0
    metrics['top1_acc'] = float(top1_pos.float().mean())

    metrics['exact_match'] = float(
        (preds == binary_labels).all(dim=1).float().mean())
    return metrics
