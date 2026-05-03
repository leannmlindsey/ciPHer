"""Multi-tower (mid-fusion) AttentionMLP for K/O serotype prediction.

Three identical-architecture towers, each consuming a different protein
embedding (ProtT5-XL seg8 / ESM-2 3B mean / kmer_aa20_k4), produce 320-d
latents; concat into 960-d; two heads (K, O) sit on top of the joint.

Why mid-fusion rather than concat-and-MLP: in plain concat the sparse
kmer block (~160k-d, ~0.2% density) gets ignored in the shared first
Linear because the dense pLM block dominates gradient magnitudes (see
agent 1's 2026-05-03 diagnosis). Each tower having its own SE-attention
+ first-layer projection lets the kmer signal reach the join already
densified.

Architecture per tower (matches existing models/attention_mlp AttentionMLP
minus its final classification head; emits the 320-d penultimate
representation):

    Input (D-d)
      -> SE attention (bottleneck via se_dim, learnable gate)
      -> Linear(D -> 1280) -> BN -> ReLU -> Dropout
      -> Linear(1280 -> 640) -> BN -> ReLU -> Dropout
      -> Linear(640 -> 320) -> BN -> ReLU -> Dropout
      -> 320-d latent

Joint:
    [320 || 320 || 320] = 960-d
      -> Linear(960 -> num_K)   K head
      -> Linear(960 -> num_O)   O head
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# Tower hidden dims minus the final 160 layer of the original AttentionMLP.
# 320-d is large enough to retain useful signal but small enough that the
# joint Linear(960 -> num_classes) doesn't overfit on the intersected
# training set (~10-25K rows).
DEFAULT_TOWER_HIDDEN_DIMS = (1280, 640, 320)


class AttentionMLPTower(nn.Module):
    """One AttentionMLP-shaped tower. Emits hidden_dims[-1] features
    (the penultimate representation of the original AttentionMLP -- no
    classification head).
    """

    def __init__(self, input_dim, hidden_dims=DEFAULT_TOWER_HIDDEN_DIMS,
                 se_dim=640, gate_init=0.5, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.use_attention = se_dim > 0

        if self.use_attention:
            self.feature_attention = nn.Sequential(
                nn.Linear(input_dim, se_dim),
                nn.LayerNorm(se_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(se_dim, input_dim),
                nn.Sigmoid(),
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

    @property
    def output_dim(self):
        return self.hidden_dims[-1]

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

        if return_attention:
            return out, attention_output
        return out


class MultiTowerAttentionMLP(nn.Module):
    """Three AttentionMLPTower instances feeding two classification heads.

    Towers are listed in a fixed order (matches the order of inputs in
    forward and dataset). num_K and num_O can differ; both heads share
    the 960-d joint representation.
    """

    def __init__(self, input_dims, num_K, num_O,
                 tower_hidden_dims=DEFAULT_TOWER_HIDDEN_DIMS,
                 se_dim=640, dropout=0.1):
        super().__init__()
        if len(input_dims) != 3:
            raise ValueError(
                f'Expected 3 input dims (one per tower); got {len(input_dims)}')
        self.towers = nn.ModuleList([
            AttentionMLPTower(d, hidden_dims=tower_hidden_dims,
                              se_dim=se_dim, dropout=dropout)
            for d in input_dims
        ])
        self.joint_dim = sum(t.output_dim for t in self.towers)
        self.num_K = num_K
        self.num_O = num_O
        self.head_K = nn.Linear(self.joint_dim, num_K)
        self.head_O = nn.Linear(self.joint_dim, num_O)

    def forward(self, xs):
        """
        Args:
            xs: list of 3 tensors, shape (B, input_dims[i]) each.

        Returns:
            (logits_K, logits_O), each (B, num_K) / (B, num_O).
        """
        if len(xs) != 3:
            raise ValueError(f'Expected 3 inputs; got {len(xs)}')
        latents = [tower(x) for tower, x in zip(self.towers, xs)]
        joint = torch.cat(latents, dim=1)
        return self.head_K(joint), self.head_O(joint)


class MultiTowerDataset(Dataset):
    """Multi-tower dataset over MD5s present in all three embedding sources.

    Args:
        md5s: list of MD5 hashes (one row each); should be the intersected
              set produced by the training-data prep pipeline.
        emb_dicts: list of 3 dicts, each {md5: 1-D embedding array}.
        k_labels: array (N, num_K) multi-hot or one-hot.
        o_labels: array (N, num_O) multi-hot or one-hot.
    """

    def __init__(self, md5s, emb_dicts, k_labels, o_labels):
        if len(emb_dicts) != 3:
            raise ValueError(f'Need 3 emb_dicts; got {len(emb_dicts)}')
        self.md5s = list(md5s)
        # Materialise the per-tower input arrays in MD5 order.
        # Each is a 2D float32 array (N, D_tower).
        self._xs = [
            np.stack([np.asarray(d[m], dtype=np.float32) for m in self.md5s])
            for d in emb_dicts
        ]
        self.k_labels = torch.as_tensor(np.asarray(k_labels),
                                        dtype=torch.float32)
        self.o_labels = torch.as_tensor(np.asarray(o_labels),
                                        dtype=torch.float32)

    def input_dims(self):
        return [x.shape[1] for x in self._xs]

    def __len__(self):
        return len(self.md5s)

    def __getitem__(self, idx):
        xs = [torch.from_numpy(x[idx]) for x in self._xs]
        return xs, self.k_labels[idx], self.o_labels[idx]


def multi_tower_collate(batch):
    """DataLoader collate: stacks the 3-tuple inputs by tower."""
    xs_per_sample, k_labels, o_labels = zip(*batch)
    n_towers = len(xs_per_sample[0])
    xs_stacked = [torch.stack([s[i] for s in xs_per_sample], dim=0)
                  for i in range(n_towers)]
    k_labels = torch.stack(k_labels, dim=0)
    o_labels = torch.stack(o_labels, dim=0)
    return xs_stacked, k_labels, o_labels


def bce_loss(logits, targets):
    """BCEWithLogitsLoss with sum-over-classes then mean-over-batch reduction.

    Same form as light_attention_binary; keeps loss magnitude on the same
    scale as multi-class CE rather than dividing by num_classes.
    """
    per_elem = F.binary_cross_entropy_with_logits(logits, targets,
                                                   reduction='none')
    return per_elem.sum(dim=1).mean()


def compute_metrics(logits, labels, threshold=0.5):
    """Multi-label metrics: f1_micro, f1_macro, precision, recall, top1_acc,
    exact_match. Same shape as the per-head metrics in the existing models
    so harvest_results.py picks them up.
    """
    metrics = {}
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    binary_labels = (labels > 0).float()

    tp_sample = (preds * binary_labels).sum(dim=1)
    fp_sample = (preds * (1 - binary_labels)).sum(dim=1)
    fn_sample = ((1 - preds) * binary_labels).sum(dim=1)
    p = tp_sample / (tp_sample + fp_sample + 1e-8)
    r = tp_sample / (tp_sample + fn_sample + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    metrics['f1_micro'] = float(f1.mean())
    metrics['precision'] = float(p.mean())
    metrics['recall'] = float(r.mean())

    tp_class = (preds * binary_labels).sum(dim=0)
    fp_class = (preds * (1 - binary_labels)).sum(dim=0)
    fn_class = ((1 - preds) * binary_labels).sum(dim=0)
    p_c = tp_class / (tp_class + fp_class + 1e-8)
    r_c = tp_class / (tp_class + fn_class + 1e-8)
    f1_c = 2 * p_c * r_c / (p_c + r_c + 1e-8)
    has_pos = binary_labels.sum(dim=0) > 0
    if has_pos.any():
        metrics['f1_macro'] = float(f1_c[has_pos].mean())
        metrics['precision_macro'] = float(p_c[has_pos].mean())
        metrics['recall_macro'] = float(r_c[has_pos].mean())
    else:
        metrics['f1_macro'] = 0.0
        metrics['precision_macro'] = 0.0
        metrics['recall_macro'] = 0.0

    top1 = logits.argmax(dim=1)
    metrics['top1_acc'] = float(
        (labels[torch.arange(len(labels)), top1] > 0).float().mean()
    )
    metrics['exact_match'] = float(
        (preds == binary_labels).all(dim=1).float().mean())
    return metrics
