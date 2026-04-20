"""Light Attention pooling + AttentionMLP classifier for serotype prediction.

Replaces mean-pooling of variable-length protein embeddings with a learned
pooling mechanism (Stärk et al. 2021). Two parallel 1D convolutions over the
sequence produce a value stream and an attention stream; the final fixed-length
representation concatenates an attention-weighted sum and a max-pool of the
value features. A SE-attention MLP head identical to `models/attention_mlp`
consumes the pooled vector and outputs K or O logits.

Input per protein is a variable-length (L, D) matrix. Works for any embedding
that keeps per-residue or per-segment structure (e.g. ESM-2 seg4: L=4; ESM-2
per-residue: L=protein length).

Reference implementation:
    https://github.com/HannesStark/protein-localization/blob/main/models/light_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class LightAttentionPooling(nn.Module):
    """Variable-length (L, D) embeddings -> fixed-length 2*D vector."""

    def __init__(self, embedding_dim=1280, kernel_size=9, conv_dropout=0.25):
        super().__init__()
        self.feature_convolution = nn.Conv1d(
            embedding_dim, embedding_dim, kernel_size,
            stride=1, padding=kernel_size // 2,
        )
        self.attention_convolution = nn.Conv1d(
            embedding_dim, embedding_dim, kernel_size,
            stride=1, padding=kernel_size // 2,
        )
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)

        self.embedding_dim = embedding_dim
        self.output_dim = 2 * embedding_dim

    def forward(self, x, mask):
        """
        Args:
            x:    [batch, embedding_dim, seq_len]
            mask: [batch, seq_len] — True for valid positions, False for padding
        Returns:
            [batch, 2 * embedding_dim]
        """
        o = self.feature_convolution(x)
        o = self.dropout(o)

        attention = self.attention_convolution(x)
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)
        o2, _ = torch.max(o, dim=-1)

        return torch.cat([o1, o2], dim=-1)


class AttentionMLP(nn.Module):
    """SE-attention MLP head (copy of models/attention_mlp/model.py:AttentionMLP).

    Kept local to avoid cross-model-directory imports. If divergence between
    this copy and the attention_mlp version becomes a problem, extract to
    src/cipher/.
    """

    def __init__(self, input_dim, num_classes,
                 hidden_dims=(2560, 1280, 640, 320, 160), se_dim=640,
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


class LightAttentionClassifier(nn.Module):
    """LA pooling + AttentionMLP head, trained end-to-end."""

    def __init__(self, embedding_dim, num_classes, kernel_size=9,
                 conv_dropout=0.25,
                 hidden_dims=(2560, 1280, 640, 320, 160),
                 se_dim=640, classifier_dropout=0.1):
        super().__init__()
        self.pooling = LightAttentionPooling(
            embedding_dim=embedding_dim,
            kernel_size=kernel_size,
            conv_dropout=conv_dropout,
        )
        self.classifier = AttentionMLP(
            input_dim=self.pooling.output_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            se_dim=se_dim,
            dropout=classifier_dropout,
        )

    def forward(self, x, mask):
        pooled = self.pooling(x, mask)
        return self.classifier(pooled)


class PerResidueDataset(Dataset):
    """Variable-length (L, D) embeddings with one-hot / binary labels."""

    def __init__(self, embeddings_list, labels):
        """
        Args:
            embeddings_list: list of numpy arrays, each shape (L_i, D)
            labels: numpy array (N, n_classes)
        """
        self.embeddings = embeddings_list
        self.labels = torch.FloatTensor(labels)
        self.lengths = [e.shape[0] for e in embeddings_list]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb = torch.FloatTensor(self.embeddings[idx])  # (L, D)
        return emb, self.labels[idx], self.lengths[idx]


def collate_per_residue(batch):
    """Pad variable-length (L, D) tensors to batch max length.

    Returns:
        x:    [batch, D, max_len] — padded and permuted for Conv1d
        y:    [batch, n_classes]
        mask: [batch, max_len] — True for valid positions
    """
    embeddings, labels, lengths = zip(*batch)
    padded = pad_sequence(embeddings, batch_first=True)  # [B, max_len, D]
    x = padded.permute(0, 2, 1)                           # [B, D, max_len]

    y = torch.stack(labels)
    lengths = torch.tensor(lengths)
    max_len = padded.shape[1]
    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    return x, y, mask


# ----------------------------------------------------------------------------
# Losses and metrics — copied from models/attention_mlp/model.py to keep this
# model self-contained. Any change to the shared loss/metric semantics should
# be mirrored here (or hoisted to src/cipher/).
# ----------------------------------------------------------------------------

def get_loss_fn(strategy, class_weights=None):
    if strategy in ('multi_label', 'multi_label_threshold', 'weighted_multi_label'):
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
