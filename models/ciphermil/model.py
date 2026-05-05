"""ciphermil — attention-based multiple instance learning for phage→host
prediction, ported from EvoMIL (Liu et al. 2023, github.com/liudan111/EvoMIL).

EvoMIL framing: a virus = a bag of protein embeddings. The attention pool
weighs each protein's contribution and produces ONE bag-level prediction
per virus. We adapt that to phage→host serotype prediction:

    bag      = a phage = list of its protein embeddings (variable length)
    label    = a single K-type or O-type (single-label softmax — one
               serotype per phage, the mode of its hosts' K/O calls)
    score    = bag-level softmax distribution over K (or O) classes

Training is per-head: ONE separate MIL model for the K head, ONE separate
for the O head. They don't share a trunk (per leann's 2026-05-05 design
call). Per the cipher Predictor interface, score_pair is overridden to
do a bag-level forward pass, NOT predict_protein.

Architecture (per head):
    Input bag:    (n_proteins, input_dim)         e.g. n × 1024 for ProtT5-XL mean
    -> feature_extractor:  Linear(input_dim, L) + ReLU      L=800 (EvoMIL default)
    -> H:         (n_proteins, L)
    -> attention: Linear(L, D) + Tanh + Linear(D, K)         D=128, K=1
    -> A:         softmax over n_proteins → (1, n_proteins)
    -> M:         A @ H = (1, L)        bag-level rep
    -> classifier: Linear(L, n_classes)
    -> logits:    (1, n_classes)
    -> softmax →  bag-level probability distribution over classes

Single-label softmax + cross-entropy loss. Different from cipher's
existing multi-label sigmoid+BCE convention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    """Standard Ilse-2018 attention MIL with multi-class softmax classifier.

    Operates on ONE bag at a time (one phage). For batched training,
    iterate over bags — bag size is variable so batching across bags
    requires padding or per-bag iteration.
    """

    def __init__(self, input_dim, num_classes, L=800, D=128, K=1, dropout=0.1):
        """
        Args:
            input_dim: dim of each protein embedding (1024 for ProtT5-XL mean,
                1280 for ESM-2-650M mean, 2560 for ESM-2-3B mean,
                160000 for kmer_aa20_k4)
            num_classes: number of output classes (~156 for K after
                min_class_samples=25 filter; ~22 for O)
            L: feature dim after the first linear (EvoMIL default 800)
            D: attention bottleneck dim (EvoMIL default 128)
            K: number of attention heads. EvoMIL uses 1; we follow.
            dropout: applied on the trunk's hidden activations
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.L = L
        self.D = D
        self.K = K

        # Per-protein feature extractor — projects each instance independently.
        # EvoMIL hardcoded an extra ReLU MLP block; we keep the simple
        # Linear+ReLU+Dropout trunk since cipher's pre-trained embeddings
        # are already informative without further nonlinear extraction.
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, L),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention module — Ilse 2018 form: Linear -> Tanh -> Linear.
        # Outputs unnormalized attention scores per instance.
        self.attention = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Linear(D, K),
        )

        # Classifier on the bag-level pooled representation.
        # No sigmoid — output is logits; softmax happens in the loss
        # (CrossEntropyLoss does logits internally).
        self.classifier = nn.Linear(L * K, num_classes)

    def forward(self, x):
        """Single-bag forward.

        Args:
            x: (n_proteins, input_dim) — variable-length bag of protein embeddings

        Returns:
            logits: (1, num_classes)  — bag-level logits (apply softmax for probs)
            attention_weights: (K, n_proteins) — attention over instances,
                useful for inspection / interpretation
        """
        # Per-protein features.
        H = self.feature_extractor(x)         # (n, L)

        # Attention scores per instance, softmax-normalized over instances.
        A = self.attention(H)                 # (n, K)
        A = torch.transpose(A, 1, 0)          # (K, n)
        A = F.softmax(A, dim=1)               # softmax over n instances

        # Pooled bag-level representation.
        M = torch.mm(A, H)                    # (K, L)
        M = M.view(1, -1)                     # (1, K*L)

        # Bag-level classification logits.
        logits = self.classifier(M)           # (1, num_classes)
        return logits, A
