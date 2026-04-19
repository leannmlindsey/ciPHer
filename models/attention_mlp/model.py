"""AttentionMLP model for serotype prediction.

Single-head MLP with squeeze-and-excitation (SE) attention for predicting
K-type or O-type serotypes from protein embeddings. Two instances are
trained: one for K-type, one for O-type.

Architecture:
    Input (1280-d ESM-2 embedding)
    -> SE attention (bottleneck with learnable gate)
    -> MLP (configurable hidden layers with BatchNorm, ReLU, Dropout)
    -> Output logits (n_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class AttentionMLP(nn.Module):
    """Single-head MLP with SE attention for serotype prediction."""

    def __init__(self, input_dim, num_classes,
                 hidden_dims=(1280, 640, 320, 160), se_dim=640,
                 gate_init=0.5, dropout=0.1):
        super().__init__()

        self.use_attention = se_dim > 0

        if self.use_attention:
            self.feature_attention = nn.Sequential(
                nn.Linear(input_dim, se_dim),
                nn.LayerNorm(se_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(se_dim, input_dim),
                nn.Sigmoid()
            )
            self.attention_gate = nn.Parameter(torch.tensor(gate_init))

        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.output_head = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x, return_attention=False):
        if self.use_attention:
            attention_weights = self.feature_attention(x)
            x = x * (1 - self.attention_gate) + x * attention_weights * self.attention_gate
            attention_output = attention_weights
        else:
            attention_output = None

        out = x
        for layer in self.layers:
            out = layer(out)

        logits = self.output_head(out)

        if return_attention:
            return logits, attention_output
        return logits


class SerotypeDataset(Dataset):
    """Dataset for single-head serotype prediction."""

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
    """Build the training loss function for the given label strategy.

    Strategies:
        single_label           - one-hot labels, CrossEntropyLoss (softmax)
        multi_label            - binary labels, BCEWithLogitsLoss (per-class)
        multi_label_threshold  - binary labels (thresholded), BCE (same as multi_label)
        weighted_soft          - fractional labels, KL-divergence (softmax)
        weighted_multi_label   - fractional labels, BCE (per-class, no competition)

    Args:
        strategy: one of the above
        class_weights: optional tensor of per-class weights

    Returns:
        loss function(logits, labels) -> scalar
    """
    if strategy in ('multi_label', 'multi_label_threshold', 'weighted_multi_label'):
        # BCE on binary or soft [0,1] targets. PyTorch's BCEWithLogitsLoss
        # handles soft targets natively (no modification needed).
        return nn.BCEWithLogitsLoss(pos_weight=class_weights)
    elif strategy == 'single_label':
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        def single_label_loss(logits, one_hot_labels):
            targets = one_hot_labels.argmax(dim=1)
            return ce_loss(logits, targets)
        return single_label_loss
    elif strategy == 'weighted_soft':
        def kl_div_loss(logits, soft_targets):
            log_probs = F.log_softmax(logits, dim=1)
            targets = soft_targets.clamp(min=1e-8)
            return F.kl_div(log_probs, targets, reduction='batchmean')
        return kl_div_loss
    else:
        raise ValueError(f'Unknown strategy: {strategy}')


def compute_metrics(logits, labels, strategy, threshold=0.5):
    """Compute evaluation metrics for a batch of predictions.

    Args:
        logits: tensor (N, n_classes) raw model output
        labels: tensor (N, n_classes) true labels (binary or soft)
        strategy: one of the 5 label strategies
        threshold: decision threshold for sigmoid-based strategies

    Returns:
        dict of metric_name -> float
    """
    metrics = {}

    if strategy == 'single_label':
        true_class = labels.argmax(dim=1)
        pred_class = logits.argmax(dim=1)
        metrics['top1_acc'] = float((pred_class == true_class).float().mean())

        top5_preds = logits.topk(min(5, logits.shape[1]), dim=1).indices
        top5_correct = sum(
            true_class[i] in top5_preds[i] for i in range(len(true_class)))
        metrics['top5_acc'] = float(top5_correct / len(true_class))

    elif strategy in ('multi_label', 'multi_label_threshold',
                      'weighted_multi_label'):
        # All three use sigmoid output and independent per-class predictions.
        # For F1/precision/recall we binarize the targets at > 0 so that
        # soft weighted_multi_label labels work too.
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        binary_labels = (labels > 0).float()

        # --- Sample-averaged metrics (legacy, kept as f1_micro for compat) ---
        tp_sample = (preds * binary_labels).sum(dim=1)
        fp_sample = (preds * (1 - binary_labels)).sum(dim=1)
        fn_sample = ((1 - preds) * binary_labels).sum(dim=1)

        precision_sample = tp_sample / (tp_sample + fp_sample + 1e-8)
        recall_sample = tp_sample / (tp_sample + fn_sample + 1e-8)
        f1_sample = 2 * precision_sample * recall_sample / (precision_sample + recall_sample + 1e-8)

        metrics['f1_micro'] = float(f1_sample.mean())
        metrics['precision'] = float(precision_sample.mean())
        metrics['recall'] = float(recall_sample.mean())

        # --- Per-class (macro) metrics ---
        # TP/FP/FN summed per class across all samples
        tp_class = (preds * binary_labels).sum(dim=0)         # shape: (n_classes,)
        fp_class = (preds * (1 - binary_labels)).sum(dim=0)
        fn_class = ((1 - preds) * binary_labels).sum(dim=0)

        precision_class = tp_class / (tp_class + fp_class + 1e-8)
        recall_class = tp_class / (tp_class + fn_class + 1e-8)
        f1_class = 2 * precision_class * recall_class / (precision_class + recall_class + 1e-8)

        # Only average over classes that have at least 1 positive sample
        has_pos = binary_labels.sum(dim=0) > 0
        if has_pos.any():
            metrics['f1_macro'] = float(f1_class[has_pos].mean())
            metrics['precision_macro'] = float(precision_class[has_pos].mean())
            metrics['recall_macro'] = float(recall_class[has_pos].mean())
        else:
            metrics['f1_macro'] = 0.0
            metrics['precision_macro'] = 0.0
            metrics['recall_macro'] = 0.0

        # top1_acc: is the argmax prediction in the positive-label set?
        top1_pred = logits.argmax(dim=1)
        top1_has_weight = labels[torch.arange(len(labels)), top1_pred] > 0
        metrics['top1_acc'] = float(top1_has_weight.float().mean())

        metrics['exact_match'] = float(
            (preds == binary_labels).all(dim=1).float().mean())

    elif strategy == 'weighted_soft':
        probs = F.softmax(logits, dim=1)

        top1_pred = logits.argmax(dim=1)
        top1_has_weight = labels[torch.arange(len(labels)), top1_pred] > 0
        metrics['top1_acc'] = float(top1_has_weight.float().mean())

        log_probs = F.log_softmax(logits, dim=1)
        targets = labels.clamp(min=1e-8)
        kl = F.kl_div(log_probs, targets, reduction='batchmean')
        metrics['kl_div'] = float(kl)

        cos_sim = F.cosine_similarity(probs, labels, dim=1)
        metrics['cos_sim'] = float(cos_sim.mean())

    else:
        raise ValueError(f'Unknown strategy: {strategy}')

    return metrics
