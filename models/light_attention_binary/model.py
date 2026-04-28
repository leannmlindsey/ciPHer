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


class TransformerPooler(nn.Module):
    """CLS-token transformer pooler over per-residue embeddings.

    Replaces the ConvolutionalAttention pooler with a stack of
    nn.TransformerEncoderLayer blocks (multi-head self-attention + FFN).
    A learnable CLS token is prepended to the sequence; the encoder output
    at position 0 is returned as the pooled [B, D] vector.

    Pre-norm layer order is used (norm_first=True) for stable training
    from scratch -- standard for vision transformers and modern protein
    models. Learnable absolute position embeddings up to max_len.
    """

    def __init__(self, embed_dim, num_heads=8, num_layers=2,
                 dim_feedforward=None, dropout=0.1, max_len=2048):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 2 * embed_dim
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, embeddings, masks=None, return_attns=False):
        """
        Args:
            embeddings: [B, L, D]
            masks: [B, L] with 1 for real residues, 0 for padding
            return_attns: ignored (encoder doesn't expose per-layer weights cleanly);
                included for API compatibility with ConvolutionalAttention.

        Returns:
            pooled: [B, D]
        """
        B, L, _ = embeddings.shape
        if L > self.max_len:
            embeddings = embeddings[:, -self.max_len:, :]
            if masks is not None:
                masks = masks[:, -self.max_len:]
            L = self.max_len

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, embeddings], dim=1)
        x = x + self.pos_embed[:, :L + 1, :]

        key_padding_mask = None
        if masks is not None:
            cls_mask = torch.ones(B, 1, dtype=masks.dtype, device=masks.device)
            full_mask = torch.cat([cls_mask, masks], dim=1)
            key_padding_mask = (full_mask == 0)

        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        pooled = self.norm(out[:, 0])
        if return_attns:
            return pooled, None
        return pooled


class LightAttentionBinary(nn.Module):
    """Pooler + SimpleMLP head for per-class binary classification.

    One shared pooler feeds one SimpleMLP with num_classes output logits.
    BCEWithLogitsLoss treats each logit as an independent binary head.

    Pooler is selectable via `pooler_type`:
        'conv_attn' (default) -- ConvolutionalAttention from the PPAM-TDM
            reference. Two Conv1d(width=W) + softmax-attention + max-pool.
        'transformer' -- TransformerPooler with CLS token. More expressive
            but more parameters; may help when the K-discriminating signal
            lives in a few specific residues that softmax-attention cannot
            isolate.
    """

    def __init__(self, embed_dim, num_classes, pooler_type='conv_attn',
                 pooler_cnn_width=9, transformer_num_heads=8,
                 transformer_num_layers=2, transformer_max_len=2048,
                 dropout=0.1):
        super().__init__()
        if num_classes < 2:
            raise ValueError('num_classes must be >= 2')
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.pooler_type = pooler_type
        if pooler_type == 'conv_attn':
            self.pooler = ConvolutionalAttention(embed_dim, width=pooler_cnn_width)
        elif pooler_type == 'transformer':
            self.pooler = TransformerPooler(
                embed_dim,
                num_heads=transformer_num_heads,
                num_layers=transformer_num_layers,
                max_len=transformer_max_len,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"pooler_type must be 'conv_attn' or 'transformer'; got {pooler_type!r}")
        self.classifier = SimpleMLP(embed_dim, embed_dim // 2, num_classes, dropout=dropout)

    def forward(self, embeddings, masks=None, return_attns=False):
        pooled = self.pooler(embeddings, masks, return_attns=return_attns)
        if return_attns:
            pooled, attns = pooled
            return self.classifier(pooled), attns
        return self.classifier(pooled)


def crop_c_terminal(emb, n_residues):
    """Keep the last `n_residues` rows of a (L, D) tensor / array.

    Returns the input unchanged if `n_residues` is None, <= 0, or >= L.
    Used to focus the pooler on the C-terminal binding domain of
    tailspike-like RBPs, where K-type-discriminating residues are
    concentrated.
    """
    if n_residues is None or n_residues <= 0:
        return emb
    L = emb.shape[0]
    if n_residues >= L:
        return emb
    return emb[-n_residues:]


class PerResidueDataset(Dataset):
    """Variable-length per-residue embeddings paired with multi-hot labels.

    Args:
        embeddings: list of arrays, each shape (L_i, D)
        labels: array (N, num_classes), multi-hot or one-hot
        c_terminal_crop: if set, keep only the last N residues of each
            embedding. None or <= 0 means no cropping (default).
    """

    def __init__(self, embeddings, labels, c_terminal_crop=None):
        if c_terminal_crop is not None and c_terminal_crop > 0:
            embeddings = [crop_c_terminal(e, c_terminal_crop) for e in embeddings]
        self.embeddings = [torch.as_tensor(e, dtype=torch.float32) for e in embeddings]
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.float32)
        self.c_terminal_crop = c_terminal_crop

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
