"""Light Attention + per-class binary classifier.

Ported from `reference/src/utils/modeling_utils.py` (ConvolutionalAttention,
SimpleMLP) and `reference/src/models.py` (SequenceClassificationModel) in the
PPAM-TDM PHROG-category classifier. The architecture is preserved verbatim;
only the training target changes (K-type or O-type serotypes instead of
PHROG categories). Two independent instances of this model are trained per
experiment, one per head (K, O).

Architecture:
    Input: per-residue embeddings [B, L, D] (e.g. ESM-2 650M, D=1280)
    -> ConvolutionalAttention pooler (two Conv1d(width=9) branches: softmax
       attention weights + max-pool values, concat -> Linear)
    -> SimpleMLP (Linear(D -> D/2, weight_norm) -> ReLU -> Dropout
                  -> Linear(D/2 -> num_classes, weight_norm))
    -> Logits [B, num_classes]

Trained with BCEWithLogitsLoss on one-hot targets so each output logit is an
independent per-class binary head (max-over-classes is only applied at
inference time via sigmoid, not via softmax during training).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data import Dataset


class SimpleMLP(nn.Module):
    """Two-layer MLP with weight_norm. Verbatim from PPAM-TDM reference."""

    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None),
        )

    def forward(self, x):
        return self.mlp(x)


class ConvolutionalAttention(nn.Module):
    """Light-attention pooler. Verbatim from PPAM-TDM reference.

    Two Conv1d branches:
        a = softmax over sequence length (masked) -- attention weights
        v = per-position value vectors
    Output = Linear([max_pool(v), sum(a * v)]) -- shape [B, D].
    """

    def __init__(self, embed_dim=1280, width=9):
        super().__init__()
        self._ca_w1 = nn.Conv1d(embed_dim, embed_dim, width, padding=width // 2)
        self._ca_w2 = nn.Conv1d(embed_dim, embed_dim, width, padding=width // 2)
        self._ca_mlp = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, embeddings, masks=None, return_attns=False):
        """
        Args:
            embeddings: Tensor [B, L, D]
            masks: Tensor [B, L] with 1 for real residues, 0 for padding
        """
        _temp = embeddings.permute(0, 2, 1)  # [B, D, L]
        a = self._ca_w1(_temp)               # [B, D, L]
        v = self._ca_w2(_temp)               # [B, D, L]

        if masks is not None:
            attn_mask = masks.unsqueeze(1).to(dtype=a.dtype)  # [B, 1, L]
            a = a.masked_fill(attn_mask == 0, float('-inf'))
        a = a.softmax(dim=-1)

        v_sum = (a * v).sum(dim=-1)  # [B, D]

        if masks is not None:
            v_masked = v.masked_fill(attn_mask == 0, float('-1e4'))
            v_max = v_masked.max(dim=-1).values
        else:
            v_max = v.max(dim=-1).values

        pooled_output = self._ca_mlp(torch.cat([v_max, v_sum], dim=1))  # [B, D]

        if return_attns:
            return pooled_output, a
        return pooled_output


class LightAttentionBinary(nn.Module):
    """Light Attention + SimpleMLP head for per-class binary classification.

    One shared ConvolutionalAttention pooler feeds one SimpleMLP with
    num_classes output logits. At training time BCEWithLogitsLoss treats each
    logit as an independent per-class binary head. This matches
    SequenceClassificationModel from the reference, minus the num_labels==2
    reduction (we always have num_classes >> 2).
    """

    def __init__(self, embed_dim, num_classes, pooler_cnn_width=9, dropout=0.1):
        super().__init__()
        if num_classes < 2:
            raise ValueError('num_classes must be >= 2')
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.pooler = ConvolutionalAttention(embed_dim, width=pooler_cnn_width)
        self.classifier = SimpleMLP(embed_dim, embed_dim // 2, num_classes, dropout=dropout)

    def forward(self, embeddings, masks=None, return_attns=False):
        pooled = self.pooler(embeddings, masks, return_attns=return_attns)
        if return_attns:
            pooled, attns = pooled
            return self.classifier(pooled), attns
        return self.classifier(pooled)


class PerResidueDataset(Dataset):
    """Variable-length per-residue embeddings paired with multi-hot labels.

    Args:
        embeddings: list of arrays, each shape (L_i, D)
        labels:     array (N, num_classes), multi-hot or one-hot
    """

    def __init__(self, embeddings, labels):
        self.embeddings = [torch.as_tensor(e, dtype=torch.float32) for e in embeddings]
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def per_residue_collate_fn(batch):
    """Pad variable-length per-residue embeddings to the batch max length.

    Returns:
        padded: [B, L_max, D]
        masks:  [B, L_max] with 1 for real residues, 0 for padding
        labels: [B, num_classes]
    """
    embs, labels = zip(*batch)
    max_len = max(e.shape[0] for e in embs)
    embed_dim = embs[0].shape[1]
    B = len(embs)

    padded = torch.zeros(B, max_len, embed_dim, dtype=torch.float32)
    masks = torch.zeros(B, max_len, dtype=torch.float32)
    for i, e in enumerate(embs):
        L = e.shape[0]
        padded[i, :L] = e
        masks[i, :L] = 1.0

    labels = torch.stack(labels)
    return padded, masks, labels


def bce_loss(logits, targets):
    """Colleague's per-class BCE loss: sum across classes, mean across batch.

    Note: differs from PyTorch's default BCEWithLogitsLoss(reduction='mean'),
    which divides by B * num_classes. This form keeps the per-sample loss on
    the same scale as a multi-class cross-entropy, matching the reference
    implementation (training.py:88).
    """
    per_elem = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return per_elem.sum(dim=1).mean()


def compute_metrics(logits, labels, threshold=0.5):
    """Multi-label metrics for BCE-trained K or O head.

    Returns dict with top1_acc, f1_micro, f1_macro, precision_macro,
    recall_macro, exact_match. Matches attention_mlp.compute_metrics for the
    multi_label_threshold case so existing early-stopping code works.
    """
    metrics = {}
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    binary_labels = (labels > 0).float()

    tp_sample = (preds * binary_labels).sum(dim=1)
    fp_sample = (preds * (1 - binary_labels)).sum(dim=1)
    fn_sample = ((1 - preds) * binary_labels).sum(dim=1)
    precision_sample = tp_sample / (tp_sample + fp_sample + 1e-8)
    recall_sample = tp_sample / (tp_sample + fn_sample + 1e-8)
    f1_sample = 2 * precision_sample * recall_sample / (precision_sample + recall_sample + 1e-8)

    metrics['f1_micro'] = float(f1_sample.mean())
    metrics['precision'] = float(precision_sample.mean())
    metrics['recall'] = float(recall_sample.mean())

    tp_class = (preds * binary_labels).sum(dim=0)
    fp_class = (preds * (1 - binary_labels)).sum(dim=0)
    fn_class = ((1 - preds) * binary_labels).sum(dim=0)
    precision_class = tp_class / (tp_class + fp_class + 1e-8)
    recall_class = tp_class / (tp_class + fn_class + 1e-8)
    f1_class = 2 * precision_class * recall_class / (precision_class + recall_class + 1e-8)

    has_pos = binary_labels.sum(dim=0) > 0
    if has_pos.any():
        metrics['f1_macro'] = float(f1_class[has_pos].mean())
        metrics['precision_macro'] = float(precision_class[has_pos].mean())
        metrics['recall_macro'] = float(recall_class[has_pos].mean())
    else:
        metrics['f1_macro'] = 0.0
        metrics['precision_macro'] = 0.0
        metrics['recall_macro'] = 0.0

    top1_pred = logits.argmax(dim=1)
    top1_has_weight = labels[torch.arange(len(labels)), top1_pred] > 0
    metrics['top1_acc'] = float(top1_has_weight.float().mean())
    metrics['exact_match'] = float((preds == binary_labels).all(dim=1).float().mean())

    return metrics
