"""Contrastive encoder backbone + ArcFace angular-margin heads.

Architecture
------------
```
input embedding (D_in)
    -> Linear -> BatchNorm -> ReLU -> Dropout   (repeated for hidden_dims)
    -> Linear -> L2-normalize                   representation z (output_dim)

Two ArcFaceHead modules consume z:
    ArcFaceHead(K-types) -> CE loss on K
    ArcFaceHead(O-types) -> CE loss on O

The backbone is shared; at inference the encoder produces L2-normalized
representations that replace the input embeddings for downstream tasks.
ArcFace heads are discarded at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveEncoder(nn.Module):
    """MLP backbone that maps input embeddings to L2-normalized output.

    Output dim defaults to the input dim so the resulting NPZ is a drop-in
    replacement for the source embeddings.
    """

    def __init__(self, input_dim, hidden_dims=(1280, 1024, 1024),
                 output_dim=None, dropout=0.1):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.body = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x):
        z = self.body(x)
        return F.normalize(z, p=2, dim=-1)


class ArcFaceHead(nn.Module):
    """ArcFace angular-margin classification head (Deng et al., 2019).

    For each class k we learn a centroid w_k on the unit hypersphere. For
    an input z (already L2-normalized), cos(theta_k) = z . w_k is the
    raw cosine logit. An additive margin m is applied to the *ground-truth*
    class's angle, so the model must separate classes by at least that
    angle to drive the softmax probability of the correct class high:

        cos(theta_y + m)    for y == label
        cos(theta_k)        otherwise

    Logits are multiplied by a scale s (typical 30) before softmax.

    Parameters
    ----------
    feat_dim : int
        dimensionality of the representation z.
    num_classes : int
        number of classes (K-types or O-types).
    margin : float
        additive angular margin in radians (default 0.5).
    scale : float
        output scale factor (default 30).
    ignore_index : int
        label value to mask out of the loss (default -1). Useful for
        samples that lack a label on this head.
    """

    def __init__(self, feat_dim, num_classes, margin=0.5, scale=30.0,
                 ignore_index=-1):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.ignore_index = ignore_index
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, z, labels=None):
        """Compute logits and (optionally) the CE loss.

        Args
        ----
        z : (batch, feat_dim), expected L2-normalized
        labels : (batch,) long, or None for inference-only

        Returns
        -------
        logits : (batch, num_classes) — scaled cosine, for ranking
        loss : scalar Tensor if labels given, else None
        """
        W = F.normalize(self.weight, p=2, dim=-1)
        cos_theta = (z @ W.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None:
            return self.scale * cos_theta, None

        # Build the scalar loss over valid (non-ignored) entries only.
        valid = labels != self.ignore_index
        if not valid.any():
            return self.scale * cos_theta, z.new_zeros(())

        v_labels = labels[valid]
        v_cos = cos_theta[valid]

        # Apply angular margin to the ground-truth class
        theta = torch.acos(v_cos)                                   # (b, K)
        gt_theta = theta.gather(1, v_labels.unsqueeze(1)).squeeze(1)  # (b,)
        gt_cos_margined = torch.cos(gt_theta + self.margin)         # (b,)

        logits = v_cos.clone()
        logits.scatter_(1, v_labels.unsqueeze(1),
                        gt_cos_margined.unsqueeze(1))
        logits = self.scale * logits

        loss = F.cross_entropy(logits, v_labels)
        # Return full-batch logits (no margin applied) for downstream diagnostics.
        return self.scale * cos_theta, loss


class ContrastiveModel(nn.Module):
    """Convenience wrapper bundling the encoder with K + O ArcFace heads."""

    def __init__(self, input_dim, n_k_classes, n_o_classes,
                 hidden_dims=(1280, 1024, 1024), output_dim=None,
                 dropout=0.1, margin=0.5, scale=30.0):
        super().__init__()
        self.encoder = ContrastiveEncoder(
            input_dim, hidden_dims, output_dim, dropout)
        feat = self.encoder.output_dim
        self.head_k = ArcFaceHead(feat, n_k_classes, margin, scale)
        self.head_o = ArcFaceHead(feat, n_o_classes, margin, scale)

    def forward(self, x, k_labels=None, o_labels=None):
        z = self.encoder(x)
        k_logits, k_loss = self.head_k(z, k_labels)
        o_logits, o_loss = self.head_o(z, o_labels)
        return {
            'z': z,
            'k_logits': k_logits, 'k_loss': k_loss,
            'o_logits': o_logits, 'o_loss': o_loss,
        }
